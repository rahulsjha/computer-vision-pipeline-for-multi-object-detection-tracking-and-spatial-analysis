from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional


def _to_int(x: Any) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


@dataclass
class GeoCheckReport:
    path: str
    rows: int
    unique_object_ids: int
    duration_s: Optional[float]
    fps_est: Optional[float]

    missing_required_columns: list[str]

    # Scale sanity
    meter_per_pixel_x_median: Optional[float]
    meter_per_pixel_y_median: Optional[float]
    meter_per_pixel_x_p95_abs_dev: Optional[float]
    meter_per_pixel_y_p95_abs_dev: Optional[float]

    # Movement sanity
    speed_mps_min: Optional[float]
    speed_mps_median: Optional[float]
    speed_mps_p95: Optional[float]
    speed_mps_max: Optional[float]
    negative_speed_rows: int

    distance_nonmonotonic_objects: int


def _median(vals: list[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    return s[len(s) // 2]


def _pctl(vals: list[float], p: float) -> Optional[float]:
    if not vals:
        return None
    if p <= 0:
        return min(vals)
    if p >= 100:
        return max(vals)
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    i = int(math.floor(k))
    j = int(math.ceil(k))
    if i == j:
        return s[i]
    a = s[i]
    b = s[j]
    return a + (b - a) * (k - i)


def check_geo_csv(csv_path: Path) -> GeoCheckReport:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = [c.strip() for c in (reader.fieldnames or [])]

    required = [
        "frame_id",
        "object_id",
        "x",
        "y",
        "x_meters",
        "y_meters",
        "speed",
        "distance_traveled",
        "cluster_id",
    ]
    missing_required = [c for c in required if c not in fieldnames]

    if not rows:
        return GeoCheckReport(
            path=str(csv_path),
            rows=0,
            unique_object_ids=0,
            duration_s=None,
            fps_est=None,
            missing_required_columns=missing_required,
            meter_per_pixel_x_median=None,
            meter_per_pixel_y_median=None,
            meter_per_pixel_x_p95_abs_dev=None,
            meter_per_pixel_y_p95_abs_dev=None,
            speed_mps_min=None,
            speed_mps_median=None,
            speed_mps_p95=None,
            speed_mps_max=None,
            negative_speed_rows=0,
            distance_nonmonotonic_objects=0,
        )

    # Parse basics
    obj_ids = set()
    times: list[float] = []

    # Scale distributions
    mpp_x: list[float] = []
    mpp_y: list[float] = []

    # Speed distributions
    speed_mps: list[float] = []
    negative_speed_rows = 0

    # Distance monotonicity check per object
    last_dist: dict[int, float] = {}
    nonmono_objs: set[int] = set()

    # FPS estimate: use median delta time per delta frame
    dt_over_df: list[float] = []

    prev_by_frame: dict[int, float] = {}

    for r in rows:
        oid = _to_int(r.get("object_id"))
        if oid is not None:
            obj_ids.add(oid)

        t = _to_float(r.get("time_seconds"))
        if t is not None:
            times.append(t)

        x = _to_float(r.get("x"))
        y = _to_float(r.get("y"))
        xm = _to_float(r.get("x_meters"))
        ym = _to_float(r.get("y_meters"))
        if x is not None and xm is not None and abs(x) > 1e-9:
            mpp_x.append(xm / x)
        if y is not None and ym is not None and abs(y) > 1e-9:
            mpp_y.append(ym / y)

        sp = _to_float(r.get("speed_mps"))
        if sp is None:
            sp = _to_float(r.get("speed"))
        if sp is not None:
            speed_mps.append(sp)
            if sp < -1e-6:
                negative_speed_rows += 1

        dist = _to_float(r.get("distance_traveled"))
        if oid is not None and dist is not None:
            prev = last_dist.get(oid)
            if prev is not None and dist + 1e-6 < prev:
                nonmono_objs.add(oid)
            last_dist[oid] = dist

        fr = _to_int(r.get("frame_id"))
        if fr is None:
            fr = _to_int(r.get("frame_index"))
        if fr is not None and t is not None:
            # store first seen time for each frame
            if fr not in prev_by_frame:
                prev_by_frame[fr] = t

    # FPS estimate
    if prev_by_frame:
        frames_sorted = sorted(prev_by_frame.keys())
        for a, b in zip(frames_sorted, frames_sorted[1:]):
            df = b - a
            if df <= 0:
                continue
            ta = prev_by_frame[a]
            tb = prev_by_frame[b]
            dt = tb - ta
            if dt > 0:
                dt_over_df.append(dt / df)

    fps_est = None
    if dt_over_df:
        dt_med = _median(dt_over_df)
        if dt_med and dt_med > 0:
            fps_est = 1.0 / dt_med

    duration_s = max(times) if times else None

    mppx_med = _median(mpp_x)
    mppy_med = _median(mpp_y)

    mppx_dev = None
    if mppx_med is not None and mpp_x:
        devs = [abs(v - mppx_med) for v in mpp_x]
        mppx_dev = _pctl(devs, 95.0)

    mppy_dev = None
    if mppy_med is not None and mpp_y:
        devs = [abs(v - mppy_med) for v in mpp_y]
        mppy_dev = _pctl(devs, 95.0)

    return GeoCheckReport(
        path=str(csv_path),
        rows=len(rows),
        unique_object_ids=len(obj_ids),
        duration_s=duration_s,
        fps_est=fps_est,
        missing_required_columns=missing_required,
        meter_per_pixel_x_median=mppx_med,
        meter_per_pixel_y_median=mppy_med,
        meter_per_pixel_x_p95_abs_dev=mppx_dev,
        meter_per_pixel_y_p95_abs_dev=mppy_dev,
        speed_mps_min=min(speed_mps) if speed_mps else None,
        speed_mps_median=_median(speed_mps),
        speed_mps_p95=_pctl(speed_mps, 95.0),
        speed_mps_max=max(speed_mps) if speed_mps else None,
        negative_speed_rows=negative_speed_rows,
        distance_nonmonotonic_objects=len(nonmono_objs),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate geometry+movement CSV schema and sanity")
    ap.add_argument("--csv", required=True, help="Path to tracks_*_geo.csv")
    ap.add_argument("--out", default=str(Path("evaluation") / "results" / "geo_check_report.json"), help="Output JSON report")
    args = ap.parse_args()

    report = check_geo_csv(Path(args.csv))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    # Minimal console summary
    print(f"Rows: {report.rows} | objects: {report.unique_object_ids} | duration_s: {report.duration_s}")
    print(f"Missing required columns: {report.missing_required_columns}")
    print(
        "Scale (m/px) median: "
        f"x={report.meter_per_pixel_x_median} (p95|dev|={report.meter_per_pixel_x_p95_abs_dev}), "
        f"y={report.meter_per_pixel_y_median} (p95|dev|={report.meter_per_pixel_y_p95_abs_dev})"
    )
    print(
        "Speed m/s: "
        f"min={report.speed_mps_min} med={report.speed_mps_median} p95={report.speed_mps_p95} max={report.speed_mps_max} "
        f"neg_rows={report.negative_speed_rows}"
    )
    print(f"Distance non-monotonic objects: {report.distance_nonmonotonic_objects}")
    print(f"fps_est: {report.fps_est}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
