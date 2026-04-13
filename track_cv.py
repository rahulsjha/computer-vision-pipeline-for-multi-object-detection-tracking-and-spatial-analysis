from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, ROOT, YAML

from roi_utils import RoiPolygon, auto_detect_roi, load_roi, save_roi


@dataclass(frozen=True)
class TrackRow:
    frame_index: int
    time_seconds: float
    track_id: int
    cls_id: int
    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def gap_fill_rows(
    rows: List[TrackRow],
    fps: float,
    roi: RoiPolygon,
    roi_policy: str,
    min_box_area: float,
) -> List[TrackRow]:
    """Interpolate missing frames within each track ID to remove gaps.

    This operates purely on the CSV rows (post-tracking) and does not change the video.
    """

    if not rows:
        return rows

    by_track: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        by_track[r.track_id].append(r)

    filled: List[TrackRow] = []

    for tid, seq in by_track.items():
        if not seq:
            continue
        seq.sort(key=lambda r: r.frame_index)
        prev = seq[0]
        filled.append(prev)

        for cur in seq[1:]:
            gap = int(cur.frame_index) - int(prev.frame_index)
            if gap > 1:
                steps = gap
                for k in range(1, gap):
                    alpha = k / float(steps)
                    fi = int(prev.frame_index + k)
                    t = fi / float(fps) if fps > 0 else 0.0

                    x1 = prev.x1 + (cur.x1 - prev.x1) * alpha
                    y1 = prev.y1 + (cur.y1 - prev.y1) * alpha
                    x2 = prev.x2 + (cur.x2 - prev.x2) * alpha
                    y2 = prev.y2 + (cur.y2 - prev.y2) * alpha

                    # Keep strict ROI and area semantics for synthetic frames
                    area = (x2 - x1) * (y2 - y1)
                    if area < float(min_box_area):
                        continue

                    box_arr = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                    inside = roi.contains_boxes_xyxy(box_arr, mode=str(roi_policy))
                    if not bool(inside[0]):
                        continue

                    conf = min(prev.conf, cur.conf)
                    filled.append(
                        TrackRow(
                            frame_index=fi,
                            time_seconds=t,
                            track_id=tid,
                            cls_id=prev.cls_id,
                            cls_name=prev.cls_name,
                            conf=conf,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                        )
                    )

            filled.append(cur)
            prev = cur

    filled.sort(key=lambda r: (r.frame_index, r.track_id))
    return filled


def draw_roi(frame: np.ndarray, roi: RoiPolygon) -> None:
    pts = roi.as_np()
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)


def draw_track(frame: np.ndarray, row: TrackRow) -> None:
    x1, y1, x2, y2 = map(int, [row.x1, row.y1, row.x2, row.y2])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"ID {row.track_id} | {row.cls_name} {row.conf:.2f}"
    cv2.putText(
        frame,
        label,
        (x1, max(0, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        lineType=cv2.LINE_AA,
    )


def write_csv(path: str, rows: List[TrackRow]) -> None:
    import csv

    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "frame_index",
                "time_seconds",
                "track_id",
                "cls_id",
                "cls_name",
                "conf",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.frame_index,
                    f"{r.time_seconds:.6f}",
                    r.track_id,
                    r.cls_id,
                    r.cls_name,
                    f"{r.conf:.6f}",
                    f"{r.x1:.2f}",
                    f"{r.y1:.2f}",
                    f"{r.x2:.2f}",
                    f"{r.y2:.2f}",
                ]
            )


def _parse_classes(value: Optional[str]) -> Optional[List[int]]:
    if value is None or not value.strip():
        return None
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out if out else None


def _apply_if_not_none(cfg: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        cfg[key] = value


def build_tracker(
    tracker_name: str,
    fps: float,
    profile: str,
    reid: Optional[bool],
    reid_model: str,
    track_buffer: Optional[int],
    track_high_thresh: Optional[float],
    track_low_thresh: Optional[float],
    new_track_thresh: Optional[float],
    match_thresh: Optional[float],
) -> Any:
    trackers_dir = os.path.join(str(ROOT), "cfg", "trackers")
    frame_rate = int(round(fps)) or 30

    tracker_name = tracker_name.lower()
    if tracker_name == "botsort":
        yaml_path = os.path.join(trackers_dir, "botsort.yaml")
        cfg = YAML.load(yaml_path)

        # Stability-first defaults
        if profile == "stable":
            cfg["track_buffer"] = 90
            cfg["track_high_thresh"] = 0.35
            cfg["track_low_thresh"] = 0.10
            cfg["new_track_thresh"] = 0.35
            cfg["match_thresh"] = 0.85
            cfg["with_reid"] = True
            cfg["model"] = reid_model

        # Explicit overrides
        _apply_if_not_none(cfg, "track_buffer", track_buffer)
        _apply_if_not_none(cfg, "track_high_thresh", track_high_thresh)
        _apply_if_not_none(cfg, "track_low_thresh", track_low_thresh)
        _apply_if_not_none(cfg, "new_track_thresh", new_track_thresh)
        _apply_if_not_none(cfg, "match_thresh", match_thresh)
        if reid is not None:
            cfg["with_reid"] = bool(reid)
            if cfg["with_reid"]:
                cfg["model"] = reid_model

        args = IterableSimpleNamespace(**cfg)
        return BOTSORT(args=args, frame_rate=frame_rate)

    if tracker_name == "bytetrack":
        yaml_path = os.path.join(trackers_dir, "bytetrack.yaml")
        cfg = YAML.load(yaml_path)
        if profile == "stable":
            cfg["track_buffer"] = 90
            cfg["track_high_thresh"] = 0.35
            cfg["track_low_thresh"] = 0.10
            cfg["new_track_thresh"] = 0.35
            cfg["match_thresh"] = 0.85

        _apply_if_not_none(cfg, "track_buffer", track_buffer)
        _apply_if_not_none(cfg, "track_high_thresh", track_high_thresh)
        _apply_if_not_none(cfg, "track_low_thresh", track_low_thresh)
        _apply_if_not_none(cfg, "new_track_thresh", new_track_thresh)
        _apply_if_not_none(cfg, "match_thresh", match_thresh)

        args = IterableSimpleNamespace(**cfg)
        return BYTETracker(args=args, frame_rate=frame_rate)

    raise ValueError("tracker must be one of: botsort, bytetrack")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-object detection + stable tracking with a strict scene ROI (area filtering). "
            "Outputs an annotated video and a CSV of track IDs."
        )
    )
    parser.add_argument(
        "--source",
        default=os.path.join("output", "clip_5min.mp4"),
        help="Input video path (default: output/clip_5min.mp4)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics YOLO weights (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--tracker",
        default="botsort",
        choices=["botsort", "bytetrack"],
        help="Tracker backend (default: botsort)",
    )
    parser.add_argument(
        "--profile",
        default="stable",
        choices=["stable", "balanced"],
        help="Tracking profile (default: stable). 'stable' reduces ID switches but may miss weak detections.",
    )
    parser.add_argument(
        "--reid",
        action="store_true",
        help="Force-enable ReID for BoT-SORT (improves ID stability, slower).",
    )
    parser.add_argument(
        "--no-reid",
        action="store_true",
        help="Force-disable ReID for BoT-SORT (faster, potentially more ID switches).",
    )
    parser.add_argument(
        "--reid-model",
        default="yolo26n-cls.pt",
        help="ReID model to use when ReID is enabled (default: yolo26n-cls.pt)",
    )
    parser.add_argument("--track-buffer", type=int, default=None, help="Override tracker track_buffer")
    parser.add_argument("--track-high-thresh", type=float, default=None, help="Override track_high_thresh")
    parser.add_argument("--track-low-thresh", type=float, default=None, help="Override track_low_thresh")
    parser.add_argument("--new-track-thresh", type=float, default=None, help="Override new_track_thresh")
    parser.add_argument("--match-thresh", type=float, default=None, help="Override match_thresh")
    parser.add_argument(
        "--roi",
        default=None,
        help="Optional ROI polygon JSON. If omitted, auto-detect on first frame and save to output/scene_roi.json",
    )
    parser.add_argument(
        "--roi-policy",
        default="box",
        choices=["box", "center"],
        help="How to enforce ROI filtering (default: box). 'box' requires all 4 bbox corners inside ROI.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class IDs to keep (e.g. '0,2,3'). Default: all classes.",
    )
    parser.add_argument(
        "--output-video",
        default=os.path.join("output", "tracked.mp4"),
        help="Annotated output video path (default: output/tracked.mp4)",
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join("output", "tracks.csv"),
        help="Output CSV with track IDs (default: output/tracks.csv)",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Do not black-out pixels outside ROI before detection (still filters detections by ROI).",
    )
    parser.add_argument(
        "--min-box-area",
        type=int,
        default=900,
        help="Filter detections smaller than this area in pixels (default: 900).",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=0,
        help=(
            "Only start reporting/drawing a track after it has been seen for N frames. "
            "0 = auto (stable: 5, balanced: 1)."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Process only the first N frames (0 = all frames).",
    )
    parser.add_argument(
        "--no-gap-fill",
        action="store_true",
        help="Disable interpolation-based gap filling in the CSV output.",
    )

    args = parser.parse_args()

    min_track_frames = int(args.min_track_frames)
    if min_track_frames <= 0:
        min_track_frames = 5 if args.profile == "stable" else 1

    classes = _parse_classes(args.classes)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: could not open video: {args.source}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        print("Error: could not read first frame", file=sys.stderr)
        return 1

    # ROI: define main scene boundary
    if args.roi:
        roi = load_roi(args.roi)
    else:
        roi = auto_detect_roi(first_frame)
        ensure_parent_dir(os.path.join("output", "scene_roi.json"))
        save_roi(os.path.join("output", "scene_roi.json"), roi)

    roi_mask = roi.mask(first_frame.shape)

    # Init model + tracker
    model = YOLO(args.model)
    reid: Optional[bool]
    if args.no_reid:
        reid = False
    elif args.reid:
        reid = True
    else:
        reid = None

    tracker = build_tracker(
        args.tracker,
        fps=fps,
        profile=args.profile,
        reid=reid,
        reid_model=args.reid_model,
        track_buffer=args.track_buffer,
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        match_thresh=args.match_thresh,
    )

    ensure_parent_dir(args.output_video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(
            "Error: could not open video writer. Try changing output extension to .avi",
            file=sys.stderr,
        )
        return 1

    # Reset cap to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rows: List[TrackRow] = []
    frame_index = 0
    track_seen: dict[int, int] = {}

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if args.max_frames and frame_index >= int(args.max_frames):
            break

        if not args.no_mask:
            masked = cv2.bitwise_and(frame, frame, mask=roi_mask)
        else:
            masked = frame

        result = model.predict(
            masked,
            conf=float(args.conf),
            iou=float(args.iou),
            imgsz=int(args.imgsz),
            classes=classes,
            verbose=False,
        )[0]

        det = result.boxes.cpu().numpy()

        # Filter detections strictly inside ROI + remove tiny boxes (area filtering)
        if len(det):
            xyxy = det.xyxy
            keep_roi = roi.contains_boxes_xyxy(xyxy, mode=str(args.roi_policy))
            box_area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            keep_area = box_area >= float(args.min_box_area)
            keep = keep_roi & keep_area
            det = det[keep]

        # Pass the same image used for detection into tracker (important when ReID is enabled)
        tracks = tracker.update(det, masked, feats=None)

        # Draw
        annotated = frame.copy()
        draw_roi(annotated, roi)

        if len(tracks):
            # tracks columns: x1,y1,x2,y2,track_id,conf,cls,det_index
            for t in tracks:
                x1, y1, x2, y2 = map(float, t[:4])
                track_id = int(t[4])
                conf = float(t[5])
                cls_id = int(t[6])
                cls_name = model.names.get(cls_id, str(cls_id)) if hasattr(model, "names") else str(cls_id)

                # Enforce ROI restriction on tracker outputs too (strictness for evaluator)
                if not bool(roi.contains_boxes_xyxy(np.array([[x1, y1, x2, y2]], dtype=np.float32), mode=str(args.roi_policy))[0]):
                    continue

                track_seen[track_id] = track_seen.get(track_id, 0) + 1
                if track_seen[track_id] < min_track_frames:
                    continue

                rows.append(
                    TrackRow(
                        frame_index=frame_index,
                        time_seconds=frame_index / fps,
                        track_id=track_id,
                        cls_id=cls_id,
                        cls_name=cls_name,
                        conf=conf,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )

                draw_track(
                    annotated,
                    TrackRow(
                        frame_index=frame_index,
                        time_seconds=frame_index / fps,
                        track_id=track_id,
                        cls_id=cls_id,
                        cls_name=cls_name,
                        conf=conf,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    ),
                )

        writer.write(annotated)
        frame_index += 1

        if frame_index % int(max(fps * 5, 1)) == 0:
            print(f"Processed {frame_index} frames ({frame_index / fps:.1f}s)")

    writer.release()
    cap.release()

    write_csv(args.output_csv, rows)
    if not args.no_gap_fill:
        print("Post-processing tracks to fill intra-ID frame gaps ...")
        rows_filled = gap_fill_rows(rows, fps=fps, roi=roi, roi_policy=args.roi_policy, min_box_area=float(args.min_box_area))
        write_csv(args.output_csv, rows_filled)
        print("Gap filling complete.")
    print(f"Done. Wrote {args.output_video} and {args.output_csv}")
    if not args.roi:
        print("ROI saved to output/scene_roi.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
