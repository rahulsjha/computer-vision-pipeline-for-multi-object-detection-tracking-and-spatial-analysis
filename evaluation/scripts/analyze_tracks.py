from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class TrackMetrics:
    rows: int
    unique_frames: int
    duration_s: Optional[float]
    unique_track_ids: int
    avg_tracks_per_frame: float
    max_tracks_in_frame: int

    roi_center_out_pct: float
    roi_any_corner_out_pct: float

    avg_track_length_frames: float
    median_track_length_frames: int
    tracks_with_gaps: int
    worst_max_gap_frames: int
    tracks_len_le_5: int

    total_matches: int
    switches: int
    switch_rate: float

    class_counts: dict[str, int]


def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def _extract_roi_points(obj: Any) -> list[tuple[float, float]]:
    if isinstance(obj, list):
        candidate = obj
    elif isinstance(obj, dict):
        # Common schemas: {"points": [{x,y}, ...]}, {"roi": ...}, etc.
        for key in ["points", "roi", "polygon", "poly", "vertices"]:
            if key in obj and isinstance(obj[key], list):
                candidate = obj[key]
                break
        else:
            # Fallback: first list-of-points value
            candidate = None
            for v in obj.values():
                if isinstance(v, list) and v and isinstance(v[0], (dict, list, tuple)):
                    candidate = v
                    break
            if candidate is None:
                raise ValueError("Could not find ROI points list in ROI JSON")
    else:
        raise ValueError("Unsupported ROI JSON format")

    pts: list[tuple[float, float]] = []
    for p in candidate:
        if isinstance(p, dict) and "x" in p and "y" in p:
            pts.append((float(p["x"]), float(p["y"])) )
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            pts.append((float(p[0]), float(p[1])))

    if len(pts) < 3:
        raise ValueError("ROI must have at least 3 points")
    return pts


def _point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float, eps: float = 1e-9) -> bool:
    # Collinearity via cross-product
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    if min(ax, bx) - eps <= px <= max(ax, bx) + eps and min(ay, by) - eps <= py <= max(ay, by) + eps:
        return True
    return False


def point_in_poly(px: float, py: float, poly: list[tuple[float, float]]) -> bool:
    """Ray casting, inclusive of boundary."""
    inside = False
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]

        if _point_on_segment(px, py, ax, ay, bx, by):
            return True

        # Intersect edge with horizontal ray to +inf
        intersects = (ay > py) != (by > py)
        if not intersects:
            continue
        dy = by - ay
        if abs(dy) < 1e-12:
            continue
        x_int = (bx - ax) * (py - ay) / dy + ax
        if x_int >= px:
            inside = not inside
    return inside


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_metrics(csv_path: Path, roi_path: Path) -> TrackMetrics:
    roi_obj = json.loads(roi_path.read_text(encoding="utf-8"))
    poly = _extract_roi_points(roi_obj)

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = [c.strip() for c in (reader.fieldnames or [])]
        rows = list(reader)

    if not rows:
        raise ValueError("CSV has no rows")

    def pick(candidates: Iterable[str]) -> Optional[str]:
        mapping = {c.lower(): c for c in fieldnames}
        for c in candidates:
            if c.lower() in mapping:
                return mapping[c.lower()]
        return None

    col_frame = pick(["frame_index", "frame", "frame_id", "frameid", "frame_idx", "frameindex", "idx", "f"])
    col_id = pick(["track_id", "trackid", "id", "track"])
    col_time = pick(["time_seconds", "time", "t", "timestamp", "seconds", "sec", "time_s"])
    col_cls_id = pick(["cls_id", "class_id", "class", "cls"])
    col_cls_name = pick(["cls_name", "class_name", "label", "name"])

    bbox_cols = {
        "x1": pick(["x1", "left", "xmin"]),
        "y1": pick(["y1", "top", "ymin"]),
        "x2": pick(["x2", "right", "xmax"]),
        "y2": pick(["y2", "bottom", "ymax"]),
    }

    missing = []
    if not col_frame:
        missing.append("frame_index")
    if not col_id:
        missing.append("track_id")
    for k, v in bbox_cols.items():
        if not v:
            missing.append(k)
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(missing)}; found: {fieldnames}")

    records: list[tuple[int, int, float, float, float, float, Optional[float], str]] = []
    class_counts: Counter[str] = Counter()

    for r in rows:
        fr = _to_int(r.get(col_frame))
        tid = _to_int(r.get(col_id))
        x1 = _to_float(r.get(bbox_cols["x1"]))
        y1 = _to_float(r.get(bbox_cols["y1"]))
        x2 = _to_float(r.get(bbox_cols["x2"]))
        y2 = _to_float(r.get(bbox_cols["y2"]))
        tm = _to_float(r.get(col_time)) if col_time else None

        cls_name = ""
        if col_cls_name and r.get(col_cls_name) is not None:
            cls_name = str(r.get(col_cls_name)).strip()
        elif col_cls_id and r.get(col_cls_id) is not None:
            cls_name = str(r.get(col_cls_id)).strip()

        if fr is None or tid is None or None in (x1, y1, x2, y2):
            continue

        x1f, x2f = (x1, x2) if x1 <= x2 else (x2, x1)
        y1f, y2f = (y1, y2) if y1 <= y2 else (y2, y1)
        records.append((fr, tid, x1f, y1f, x2f, y2f, tm, cls_name))
        if cls_name:
            class_counts[cls_name] += 1

    if not records:
        raise ValueError("No valid parsed rows")

    frames = [fr for fr, *_ in records]
    track_ids = [tid for _, tid, *_ in records]

    unique_frames = sorted(set(frames))
    unique_tracks = sorted(set(track_ids))

    by_frame: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    by_track: dict[int, list[int]] = defaultdict(list)

    out_center = 0
    out_corner = 0

    for fr, tid, x1, y1, x2, y2, tm, cls_name in records:
        by_frame[fr].append((tid, (x1, y1, x2, y2)))
        by_track[tid].append(fr)

        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if not point_in_poly(cx, cy, poly):
            out_center += 1

        corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
        if any(not point_in_poly(px, py, poly) for px, py in corners):
            out_corner += 1

    n_rows = len(records)
    n_frames = len(unique_frames)
    n_tracks = len(unique_tracks)

    duration_s: Optional[float] = None
    if col_time:
        times = [tm for *_, tm, _cls in records if tm is not None]
        duration_s = max(times) if times else None

    avg_tracks_per_frame = (n_rows / n_frames) if n_frames else 0.0
    max_tracks_in_frame = max((len(v) for v in by_frame.values()), default=0)

    pct_center_out = 100.0 * out_center / n_rows if n_rows else 0.0
    pct_corner_out = 100.0 * out_corner / n_rows if n_rows else 0.0

    # Track stats
    track_stats: list[tuple[int, int, int]] = []  # (len, gaps, max_gap)
    tracks_le_5 = 0
    for tid, frs in by_track.items():
        s = sorted(set(frs))
        length = len(s)
        gaps = 0
        max_gap = 0
        for a, b in zip(s, s[1:]):
            d = b - a
            if d > 1:
                gaps += 1
                max_gap = max(max_gap, d - 1)
        if length <= 5:
            tracks_le_5 += 1
        track_stats.append((length, gaps, max_gap))

    if track_stats:
        lengths = [t[0] for t in track_stats]
        gaps_list = [t[1] for t in track_stats]
        maxg_list = [t[2] for t in track_stats]
        avg_len = sum(lengths) / len(lengths)
        med_len = sorted(lengths)[len(lengths) // 2]
        tracks_with_gaps = sum(1 for g in gaps_list if g > 0)
        worst_max_gap = max(maxg_list)
    else:
        avg_len = 0.0
        med_len = 0
        tracks_with_gaps = 0
        worst_max_gap = 0

    # Approx ID switches (greedy IoU matching)
    unique_frames_set = set(unique_frames)
    iou_thr = 0.3
    switches = 0
    total_matches = 0

    for f0 in unique_frames:
        f1 = f0 + 1
        if f1 not in unique_frames_set:
            continue
        a = by_frame.get(f0, [])
        b = by_frame.get(f1, [])
        if not a or not b:
            continue

        pairs: list[tuple[float, int, int]] = []
        for i, (id_a, box_a) in enumerate(a):
            for j, (id_b, box_b) in enumerate(b):
                v = iou(box_a, box_b)
                if v >= iou_thr:
                    pairs.append((v, i, j))
        if not pairs:
            continue
        pairs.sort(reverse=True, key=lambda x: x[0])

        used_i: set[int] = set()
        used_j: set[int] = set()
        for v, i, j in pairs:
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            total_matches += 1
            if a[i][0] != b[j][0]:
                switches += 1

    switch_rate = (switches / total_matches) if total_matches else 0.0

    return TrackMetrics(
        rows=n_rows,
        unique_frames=n_frames,
        duration_s=duration_s,
        unique_track_ids=n_tracks,
        avg_tracks_per_frame=float(avg_tracks_per_frame),
        max_tracks_in_frame=int(max_tracks_in_frame),
        roi_center_out_pct=float(pct_center_out),
        roi_any_corner_out_pct=float(pct_corner_out),
        avg_track_length_frames=float(avg_len),
        median_track_length_frames=int(med_len),
        tracks_with_gaps=int(tracks_with_gaps),
        worst_max_gap_frames=int(worst_max_gap),
        tracks_len_le_5=int(tracks_le_5),
        total_matches=int(total_matches),
        switches=int(switches),
        switch_rate=float(switch_rate),
        class_counts={k: int(v) for k, v in class_counts.items()},
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute ROI + tracking stability metrics from a tracks CSV")
    ap.add_argument("--csv", required=True, help="Path to tracks CSV")
    ap.add_argument("--roi", required=True, help="Path to ROI JSON")
    ap.add_argument("--out", required=True, help="Output JSON report path")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    roi_path = Path(args.roi)
    out_path = Path(args.out)

    m = compute_metrics(csv_path=csv_path, roi_path=roi_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(m), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
