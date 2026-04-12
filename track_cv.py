from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

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


def build_tracker(tracker_name: str, fps: float):
    trackers_dir = os.path.join(str(ROOT), "cfg", "trackers")
    if tracker_name.lower() in {"botsort", "botsort.yaml"}:
        yaml_path = os.path.join(trackers_dir, "botsort.yaml")
        cfg = YAML.load(yaml_path)
        # Ensure we don't require ReID features when running custom predict loop
        cfg["with_reid"] = False
        args = IterableSimpleNamespace(**cfg)
        return BOTSORT(args=args, frame_rate=int(round(fps)) or 30)
    if tracker_name.lower() in {"bytetrack", "bytetrack.yaml"}:
        yaml_path = os.path.join(trackers_dir, "bytetrack.yaml")
        cfg = YAML.load(yaml_path)
        args = IterableSimpleNamespace(**cfg)
        return BYTETracker(args=args, frame_rate=int(round(fps)) or 30)

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
        "--roi",
        default=None,
        help="Optional ROI polygon JSON. If omitted, auto-detect on first frame and save to output/scene_roi.json",
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

    args = parser.parse_args()

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
    tracker = build_tracker(args.tracker, fps=fps)

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

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
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

        # Filter detections strictly inside ROI (area filtering)
        if len(det):
            xyxy = det.xyxy
            centers = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2], axis=1)
            keep = roi.contains_points(centers)
            det = det[keep]

        tracks = tracker.update(det, frame, feats=None)

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
    print(f"Done. Wrote {args.output_video} and {args.output_csv}")
    if not args.roi:
        print("ROI saved to output/scene_roi.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
