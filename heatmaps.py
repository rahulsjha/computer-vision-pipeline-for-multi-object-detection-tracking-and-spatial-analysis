from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np

from roi_utils import RoiPolygon, load_roi


def ensure_parent_dir(path: str) -> None:
    """Ensure the given path exists as a directory.

    In this module we only ever pass an output directory, not a file path,
    so we can safely create the directory itself instead of its parent.
    """

    os.makedirs(os.path.abspath(path), exist_ok=True)


def load_tracks(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise ValueError(f"No rows parsed from {csv_path}")
    return rows


def build_heatmaps(
    rows: List[Dict[str, str]],
    roi: RoiPolygon,
    frame_shape: Tuple[int, int, int],
    num_clusters: int,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    h, w = frame_shape[:2]
    heat_global = np.zeros((h, w), dtype=np.float32)
    heat_by_cluster: Dict[int, np.ndarray] = {}

    # Prepare ROI mask for safety, but coordinates were already constrained upstream
    mask = roi.mask((h, w, 3))

    for r in rows:
        try:
            x = float(r.get("x") or r.get("cx") or r.get("center_x") or 0.0)
            y = float(r.get("y") or r.get("cy") or r.get("center_y") or 0.0)
        except ValueError:
            continue

        cx = int(round(x))
        cy = int(round(y))
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            continue
        if mask[cy, cx] == 0:
            continue

        heat_global[cy, cx] += 1.0

        try:
            cid = int(r.get("cluster_id", 0))
        except ValueError:
            cid = 0
        if cid not in heat_by_cluster:
            heat_by_cluster[cid] = np.zeros_like(heat_global)
        heat_by_cluster[cid][cy, cx] += 1.0

    # Ensure we have entries for all cluster ids up to num_clusters-1
    for cid in range(num_clusters):
        if cid not in heat_by_cluster:
            heat_by_cluster[cid] = np.zeros_like(heat_global)

    return heat_global, heat_by_cluster


def render_heatmap(heat: np.ndarray, base: np.ndarray, alpha: float = 0.6, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    if not heat.any():
        return base.copy()

    heat_norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_norm, colormap)
    blended = cv2.addWeighted(base, 1.0 - alpha, heat_color, alpha, 0)
    return blended


def build_trajectory_map(
    rows: List[Dict[str, str]],
    roi: RoiPolygon,
    frame_shape: Tuple[int, int, int],
) -> np.ndarray:
    h, w = frame_shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw ROI outline for context
    cv2.polylines(canvas, [roi.as_np()], isClosed=True, color=(0, 255, 255), thickness=2)

    by_track: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    for r in rows:
        try:
            tid = int(r.get("track_id") or r.get("object_id"))
            x = float(r.get("x") or r.get("cx") or r.get("center_x"))
            y = float(r.get("y") or r.get("cy") or r.get("center_y"))
            fr = int(r.get("frame_index") or r.get("frame_id"))
        except (TypeError, ValueError):
            continue
        by_track[tid].append((fr, x, y))

    for tid, seq in by_track.items():
        seq_sorted = sorted(seq, key=lambda t: t[0])
        pts = []
        for _, x, y in seq_sorted:
            cx = int(round(x))
            cy = int(round(y))
            if 0 <= cx < w and 0 <= cy < h:
                pts.append((cx, cy))
        if len(pts) >= 2:
            color = (37 * (tid % 7), 17 * (tid % 11), 29 * (tid % 13))
            for p0, p1 in zip(pts, pts[1:]):
                cv2.line(canvas, p0, p1, color, 1, lineType=cv2.LINE_AA)

    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build global/cluster heatmaps and trajectory maps from a tracking CSV "
            "constrained by the scene ROI."
        )
    )
    parser.add_argument("--csv", default=os.path.join("output", "tracks_5min_geo.csv"), help="Input tracks CSV")
    parser.add_argument("--roi", default=os.path.join("output", "scene_roi.json"), help="ROI JSON path")
    parser.add_argument(
        "--frame-source",
        default=os.path.join("output", "clip_5min.mp4"),
        help="Video to grab a background frame from (default: output/clip_5min.mp4)",
    )
    parser.add_argument("--out-dir", default="output", help="Output directory for heatmaps and trajectory maps")

    args = parser.parse_args()

    # Wrap the heavy logic so that any unexpected error is also written
    # to a small debug file inside the output directory.
    try:
        roi = load_roi(args.roi)
        rows = load_tracks(args.csv)

        cap = cv2.VideoCapture(args.frame_source)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise SystemExit(f"Could not read a frame from {args.frame_source}")

        h, w = frame.shape[:2]
        # Darken background but keep scene layout visible
        base = (frame * 0.4).astype(np.uint8)

        # How many clusters were used in tracking CSV (cluster_id column)
        cluster_ids = set()
        for r in rows:
            try:
                cluster_ids.add(int(r.get("cluster_id", 0)))
            except (TypeError, ValueError):
                continue
        num_clusters = max(cluster_ids) + 1 if cluster_ids else 1

        heat_global, heat_by_cluster = build_heatmaps(rows, roi, frame.shape, num_clusters=num_clusters)

        ensure_parent_dir(args.out_dir)

        global_img = render_heatmap(heat_global, base)
        cv2.imwrite(os.path.join(args.out_dir, "heatmap_global.png"), global_img)

        for cid, heat in heat_by_cluster.items():
            img = render_heatmap(heat, base)
            cv2.imwrite(os.path.join(args.out_dir, f"heatmap_cluster_{cid}.png"), img)

        # Event density map: same as global heatmap but higher contrast
        event_heat = heat_global.copy()
        event_heat = cv2.GaussianBlur(event_heat, (0, 0), sigmaX=5.0, sigmaY=5.0)
        event_img = render_heatmap(event_heat, base, alpha=0.7, colormap=cv2.COLORMAP_HOT)
        cv2.imwrite(os.path.join(args.out_dir, "heatmap_events.png"), event_img)

        traj_img = build_trajectory_map(rows, roi, frame.shape)
        cv2.imwrite(os.path.join(args.out_dir, "trajectories_2d.png"), traj_img)

        print("Saved heatmaps and trajectories to", os.path.abspath(args.out_dir))
        return 0
    except Exception as exc:  # pragma: no cover - defensive logging
        # Best-effort debug file in the requested out-dir
        try:
            ensure_parent_dir(args.out_dir)
            debug_path = os.path.join(args.out_dir, "heatmaps_error.txt")
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"Error while building heatmaps: {exc!r}\n")
        except Exception:
            # If even this fails, just re-raise the original error
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
