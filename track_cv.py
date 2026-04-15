from __future__ import annotations
import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from math import hypot
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, ROOT, YAML
from roi_utils import RoiPolygon, auto_detect_roi, load_roi, save_roi
# NEW: Import improvement modules
try:
    from enhanced_detector import EnhancedDetector, DetectionConfig
    ENHANCED_DETECTOR_AVAILABLE = True
except ImportError:
    ENHANCED_DETECTOR_AVAILABLE = False
    print("⚠️  enhanced_detector module not available - using standard YOLO detection")
try:
    from calibration_engine import HomographyCalibration, KeypointDetector
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    print("⚠️  calibration_engine module not available - using standard scaling")
try:
    from trajectory_smoother import TrajectoryProcessor
    SMOOTHER_AVAILABLE = True
except ImportError:
    SMOOTHER_AVAILABLE = False
    print("⚠️  trajectory_smoother module not available - trajectories won't be smoothed")
try:
    from validation_suite import ValidationSuite
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False
    print("⚠️  validation_suite module not available - validation skipped")
try:
    from deepsort_wrapper import HybridTracker
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    print("⚠️  deepsort_wrapper module not available - DeepSORT disabled")
try:
    from world_space_heatmap import WorldSpaceHeatmap
    HEATMAP_AVAILABLE = True
except ImportError:
    HEATMAP_AVAILABLE = False
    print("⚠️  world_space_heatmap module not available - world heatmaps disabled")
try:
    from enhanced_trajectory_viz import TrajectoryVisualizer
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False
    print("⚠️  enhanced_trajectory_viz module not available - enhanced visualization disabled")
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
def compute_meter_per_pixel(roi: RoiPolygon, field_length_m: float) -> float:
    """Estimate a global pixel→meter scale from the ROI.
    Uses the maximum distance between ROI vertices as the main field axis and
    assumes that length corresponds to `field_length_m` meters.
    """
    pts = np.array(roi.points, dtype=float)
    if len(pts) < 2:
        return 0.05  # conservative fallback (20 px ≈ 1 m)
    max_d = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
            if d > max_d:
                max_d = d
    if max_d <= 0:
        return 0.05
    return float(field_length_m) / float(max_d)
def calibrate_with_homography(frame: np.ndarray, roi: RoiPolygon) -> Optional[HomographyCalibration]:
    """Attempt homography-based calibration; return None if it fails."""
    if not CALIBRATION_AVAILABLE:
        return None
    
    try:
        detector = KeypointDetector()
        
        print("  Detecting pitch lines...")
        lines = detector.detect_lines(frame, threshold=100)
        if len(lines) < 4:
            print(f"    ⚠️  Too few lines detected ({len(lines)} < 4), falling back to scaling method")
            return None
        
        print(f"    ✓ Detected {len(lines)} lines")
        
        print("  Finding line intersections...")
        intersections = detector.find_line_intersections(lines, tolerance=10)
        if len(intersections) < 4:
            print(f"    ⚠️  Too few intersections ({len(intersections)} < 4), falling back to scaling method")
            return None
        
        print(f"    ✓ Found {len(intersections)} intersections")
        
        print("  Extracting corner points...")
        corners = detector.find_field_corners(intersections, frame.shape[0], frame.shape[1])
        if corners is None or len(corners) < 4:
            print("    ⚠️  Could not extract corners, falling back to scaling method")
            return None
        
        print("    ✓ Extracted 4 corner points")
        
        print("  Computing homography...")
        cal = HomographyCalibration()
        cal.compute_from_corners(corners)
        
        print("    ✓ Homography computed successfully")
        return cal
    
    except Exception as e:
        print(f"    ⚠️  Homography calibration failed: {e}")
        return None
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
def build_movement_table(
    rows: List[TrackRow],
    fps: float,
    meter_per_pixel: float,
    field_length_m: float,
    clusters: int = 3,
) -> List[Dict[str, Any]]:
    """Convert raw track rows into movement-enriched dictionaries.
    Adds pixel center, world coordinates (meters), per-frame speed, cumulative
    distance, and a simple cluster_id based on mean field position.
    """
    if not rows:
        return []
    by_track: Dict[int, List[TrackRow]] = defaultdict(list)
    for r in rows:
        by_track[r.track_id].append(r)
    # First pass: compute per-track trajectories and metrics
    per_track_rows: Dict[int, List[Dict[str, Any]]] = {}
    track_mean_x_m: Dict[int, float] = {}
    for tid, seq in by_track.items():
        if not seq:
            continue
        seq_sorted = sorted(seq, key=lambda r: r.frame_index)
        track_entries: List[Dict[str, Any]] = []
        total_dist_m = 0.0
        sum_x_m = 0.0
        count = 0
        prev_cx = prev_cy = None
        prev_frame = None
        for r in seq_sorted:
            cx = 0.5 * (r.x1 + r.x2)
            cy = 0.5 * (r.y1 + r.y2)
            x_m = cx * meter_per_pixel
            y_m = cy * meter_per_pixel
            speed_mps = 0.0
            if prev_cx is not None and prev_cy is not None and prev_frame is not None:
                dt = (r.frame_index - prev_frame) / float(fps) if fps > 0 else 0.0
                if dt > 0:
                    d_pix = hypot(cx - prev_cx, cy - prev_cy)
                    d_m = d_pix * meter_per_pixel
                    total_dist_m += d_m
                    speed_mps = d_m / dt
            speed_kmh = speed_mps * 3.6
            entry = {
                "frame_index": r.frame_index,
                "frame_id": r.frame_index,
                "time_seconds": r.time_seconds,
                "track_id": r.track_id,
                "object_id": r.track_id,
                "cls_id": r.cls_id,
                "cls_name": r.cls_name,
                "conf": r.conf,
                "x1": r.x1,
                "y1": r.y1,
                "x2": r.x2,
                "y2": r.y2,
                "x": cx,
                "y": cy,
                "x_meters": x_m,
                "y_meters": y_m,
                "speed_mps": speed_mps,
                "speed_kmh": speed_kmh,
                "speed": speed_kmh,
                "distance_traveled": total_dist_m,
                # cluster_id filled later
                "cluster_id": -1,
            }
            track_entries.append(entry)
            prev_cx, prev_cy, prev_frame = cx, cy, r.frame_index
            sum_x_m += x_m
            count += 1
        if not track_entries or count == 0:
            continue
        per_track_rows[tid] = track_entries
        track_mean_x_m[tid] = sum_x_m / float(count)
    # Second pass: simple clustering by mean x-position
    cluster_edges: List[float] = []
    if track_mean_x_m:
        # Spread clusters evenly along field length [0, field_length_m]
        if clusters <= 1:
            cluster_edges = [float(field_length_m)]
        else:
            step = float(field_length_m) / float(clusters)
            cluster_edges = [step * (i + 1) for i in range(clusters - 1)] + [float("inf")]
    cluster_for_track: Dict[int, int] = {}
    for tid, mx in track_mean_x_m.items():
        cid = 0
        for idx, edge in enumerate(cluster_edges):
            if mx <= edge:
                cid = idx
                break
        cluster_for_track[tid] = cid
    # Final table
    table: List[Dict[str, Any]] = []
    for tid, entries in per_track_rows.items():
        cid = cluster_for_track.get(tid, 0)
        for e in entries:
            e["cluster_id"] = cid
            table.append(e)
    # Sort by frame then id for easier analysis
    table.sort(key=lambda d: (d["frame_index"], d["track_id"]))
    return table
def write_movement_csv(path: str, table: List[Dict[str, Any]]) -> None:
    import csv
    ensure_parent_dir(path)
    if not table:
        # still write header for downstream tools
        headers = [
            "frame_index",
            "frame_id",
            "time_seconds",
            "track_id",
            "object_id",
            "cls_id",
            "cls_name",
            "conf",
            "x1",
            "y1",
            "x2",
            "y2",
            "x",
            "y",
            "x_meters",
            "y_meters",
            "speed_mps",
            "speed_kmh",
            "speed",
            "distance_traveled",
            "cluster_id",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)
        return
    headers = list(table[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in table:
            w.writerow(row)
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
    parser.add_argument("--track-buffer", type=int, default=None, help="Override track_buffer")
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
    parser.add_argument(
        "--field-length-m",
        type=float,
        default=105.0,
        help="Approximate real-world length of the main field axis in meters (default: 105).",
    )
    
    # NEW: Enhancement options
    parser.add_argument(
        "--use-enhanced-detection",
        action="store_true",
        help="Use enhanced detection with stricter NMS and filtering (requires enhanced_detector.py)",
    )
    parser.add_argument(
        "--validate-output",
        action="store_true",
        help="Run validation suite on output CSV (requires validation_suite.py)",
    )
    parser.add_argument(
        "--smoothing-method",
        choices=["kalman", "savgol"],
        default="kalman",
        help="Method for trajectory smoothing (default: kalman)",
    )
    parser.add_argument(
        "--smooth-trajectories",
        action="store_true",
        help="Apply trajectory smoothing to output CSV (requires trajectory_smoother.py)",
    )
    parser.add_argument(
        "--use-homography-calibration",
        action="store_true",
        help="Use homography-based calibration for world-coordinate mapping (requires calibration_engine.py)",
    )
    parser.add_argument(
        "--save-homography",
        default=None,
        help="Path to save homography matrix JSON (only with --use-homography-calibration)",
    )
    
    # NEW: Advanced tracking options
    parser.add_argument(
        "--tracker-type",
        choices=["botsort", "bytetrack", "deepsort", "hybrid"],
        default="botsort",
        help="Tracker type: botsort/bytetrack (motion-based), deepsort/hybrid (appearance-based)",
    )
    parser.add_argument(
        "--reid-model",
        default="osnet_x1_0_imagenet.pth",
        help="ReID model path for DeepSORT (default: osnet_x1_0_imagenet.pth)",
    )
    
    # NEW: Visualization options
    parser.add_argument(
        "--generate-world-heatmaps",
        action="store_true",
        help="Generate world-coordinate heatmaps after processing (requires world_space_heatmap.py)",
    )
    parser.add_argument(
        "--generate-trajectory-viz",
        action="store_true",
        help="Generate enhanced trajectory visualizations (requires enhanced_trajectory_viz.py)",
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
    # NEW: Initialize calibration
    calibration = None
    meter_per_pixel = None
    
    if args.use_homography_calibration and CALIBRATION_AVAILABLE:
        print("\n📐 Attempting homography-based calibration...")
        calibration = calibrate_with_homography(first_frame, roi)
        if calibration and args.save_homography:
            calibration.save(args.save_homography)
            print(f"   Homography saved to {args.save_homography}")
    
    # NEW: Initialize enhanced detector
    enhanced_detector = None
    if args.use_enhanced_detection and ENHANCED_DETECTOR_AVAILABLE:
        print("\n🔍 Initializing enhanced detector...")
        try:
            detection_config = DetectionConfig("detection_config.yaml") if os.path.exists("detection_config.yaml") else None
            enhanced_detector = EnhancedDetector(detection_config, args.model)
            print("   ✓ Enhanced detector loaded")
        except Exception as e:
            print(f"   ⚠️  Failed to load enhanced detector: {e}")
            enhanced_detector = None
    # Init model + tracker
    if enhanced_detector is None:
        print("\n🤖 Loading standard YOLO model...")
        model = YOLO(args.model)
    else:
        model = enhanced_detector.model  # Use the model from enhanced detector
    
    reid: Optional[bool]
    if args.no_reid:
        reid = False
    elif args.reid:
        reid = True
    else:
        reid = None
    # NEW: Support for DeepSORT and hybrid trackers
    if args.tracker_type in ["deepsort", "hybrid"]:
        if DEEPSORT_AVAILABLE:
            print(f"\n🔄 Initializing {args.tracker_type.upper()} tracker...")
            try:
                from deepsort_wrapper import HybridTracker
                tracker = HybridTracker(
                    use_deepsort=(args.tracker_type == "deepsort"),
                    reid_model=args.reid_model
                )
                print(f"   ✓ {args.tracker_type.upper()} tracker ready")
            except Exception as e:
                print(f"   ⚠️  {args.tracker_type.upper()} initialization failed: {e}")
                print(f"   Falling back to {args.tracker}...")
                tracker = build_tracker(
                    args.tracker, fps=fps, profile=args.profile, reid=reid,
                    reid_model=args.reid_model, track_buffer=args.track_buffer,
                    track_high_thresh=args.track_high_thresh,
                    track_low_thresh=args.track_low_thresh,
                    new_track_thresh=args.new_track_thresh,
                    match_thresh=args.match_thresh,
                )
        else:
            print(f"\n⚠️  DeepSORT not available, using {args.tracker}...")
            tracker = build_tracker(
                args.tracker, fps=fps, profile=args.profile, reid=reid,
                reid_model=args.reid_model, track_buffer=args.track_buffer,
                track_high_thresh=args.track_high_thresh,
                track_low_thresh=args.track_low_thresh,
                new_track_thresh=args.new_track_thresh,
                match_thresh=args.match_thresh,
            )
    else:
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
        # NEW: Use enhanced detector if available
        if enhanced_detector is not None:
            boxes, confidences, class_ids = enhanced_detector.detect(masked)
            # Convert to ultralytics format: det has columns [x1, y1, x2, y2, conf, cls]
            if len(boxes) > 0:
                det_list = []
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = box
                    det_list.append([x1, y1, x2, y2, conf, cls_id])
                det = type('obj', (object,), {
                    'xyxy': np.array([d[:4] for d in det_list]),
                    'conf': np.array([d[4] for d in det_list]),
                    'cls': np.array([d[5] for d in det_list])
                })()
            else:
                det = type('obj', (object,), {
                    'xyxy': np.array([]),
                    'conf': np.array([]),
                    'cls': np.array([])
                })()
        else:
            # Standard YOLO detection
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
        if len(det.xyxy):
            xyxy = det.xyxy
            keep_roi = roi.contains_boxes_xyxy(xyxy, mode=str(args.roi_policy))
            box_area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            keep_area = box_area >= float(args.min_box_area)
            keep = keep_roi & keep_area
            det = type('obj', (object,), {
                'xyxy': det.xyxy[keep],
                'conf': det.conf[keep],
                'cls': det.cls[keep]
            })()
        else:
            det = type('obj', (object,), {
                'xyxy': np.array([]),
                'conf': np.array([]),
                'cls': np.array([])
            })()
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
    if not rows:
        print("Warning: no tracks were collected; CSV will be empty.")
        write_movement_csv(args.output_csv, [])
        return 0
    if args.no_gap_fill:
        rows_final = rows
    else:
        print("Post-processing tracks to fill intra-ID frame gaps ...")
        rows_final = gap_fill_rows(
            rows,
            fps=fps,
            roi=roi,
            roi_policy=args.roi_policy,
            min_box_area=float(args.min_box_area),
        )
        print("Gap filling complete.")
    # NEW: Use homography calibration if available, else fall back to scaling
    if calibration is None:
        meter_per_pixel = compute_meter_per_pixel(roi, field_length_m=float(args.field_length_m))
        print(f"\n📏 Estimated scene scale: {meter_per_pixel:.6f} meters per pixel (field length ~{args.field_length_m} m)")
    else:
        meter_per_pixel = None  # Will use calibration object instead
        print(f"\n✓ Using homography-based calibration")
    table = build_movement_table(
        rows_final,
        fps=fps,
        meter_per_pixel=meter_per_pixel,
        field_length_m=float(args.field_length_m),
    )
    
    # NEW: Apply trajectory smoothing if requested
    if args.smooth_trajectories and SMOOTHER_AVAILABLE:
        print("\n📈 Smoothing trajectories...")
        try:
            # First, write raw CSV temporarily
            temp_csv_path = args.output_csv.replace(".csv", "_raw.csv")
            write_movement_csv(temp_csv_path, table)
            
            # Then smooth it
            processor = TrajectoryProcessor(method=args.smoothing_method)
            smoothed_points = processor.process_csv(temp_csv_path)
            
            # Merge smoothed data back into table
            smoothed_dict = {(p.track_id, p.frame_index): p for p in smoothed_points}
            for row in table:
                key = (row["track_id"], row["frame_index"])
                if key in smoothed_dict:
                    sp = smoothed_dict[key]
                    row["x_meters_smooth"] = sp.x_m_smooth
                    row["y_meters_smooth"] = sp.y_m_smooth
                    row["speed_mps_smooth"] = sp.speed_mps_smooth
                    row["is_outlier"] = sp.is_outlier
            
            print("   ✓ Trajectories smoothed")
            # Clean up temp file
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
        except Exception as e:
            print(f"   ⚠️  Smoothing failed: {e}")
    
    write_movement_csv(args.output_csv, table)
    
    # NEW: Run validation suite if requested
    if args.validate_output and VALIDATOR_AVAILABLE:
        print("\n✅ Running validation checks...")
        try:
            validator = ValidationSuite()
            results = validator.run_all(args.output_csv)
            validator.print_report()
            
            # Save validation report
            report_path = args.output_csv.replace(".csv", "_validation.json")
            validator.save_report(report_path)
            print(f"   Validation report saved to {report_path}")
        except Exception as e:
            print(f"   ⚠️  Validation failed: {e}")
    print(f"\n✓ Done. Wrote {args.output_video} and {args.output_csv}")
    if not args.roi:
        print("ROI saved to output/scene_roi.json")
    
    # NEW: Generate world-coordinate heatmaps
    if args.generate_world_heatmaps and HEATMAP_AVAILABLE:
        print("\n🌍 Generating world-coordinate heatmaps...")
        try:
            heatmap_gen = WorldSpaceHeatmap(args.output_csv, meter_per_pixel=meter_per_pixel)
            outputs = heatmap_gen.save_all_heatmaps("output")
            print("   ✓ Heatmaps generated successfully")
        except Exception as e:
            print(f"   ⚠️  Heatmap generation failed: {e}")
    
    # NEW: Generate enhanced trajectory visualizations
    if args.generate_trajectory_viz and VIZ_AVAILABLE:
        print("\n📽️  Generating enhanced trajectory visualizations...")
        try:
            from enhanced_trajectory_viz import create_multiple_visualizations
            create_multiple_visualizations(args.output_csv, args.source, "output")
            print("   ✓ Visualizations generated successfully")
        except Exception as e:
            print(f"   ⚠️  Visualization generation failed: {e}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())

