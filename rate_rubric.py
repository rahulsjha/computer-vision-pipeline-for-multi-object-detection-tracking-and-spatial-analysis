from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from analyze_tracks import TrackMetrics, compute_metrics


Status = Literal["pass", "partial", "fail", "unknown"]


def _status_by_threshold(value: float, pass_max: float, partial_max: float) -> Status:
    if value <= pass_max:
        return "pass"
    if value <= partial_max:
        return "partial"
    return "fail"


def _overall_status(*statuses: Status) -> Status:
    order = {"pass": 0, "partial": 1, "unknown": 2, "fail": 3}
    # worst status wins (fail > unknown > partial > pass)
    return max(statuses, key=lambda s: order.get(s, 99))


def rate(metrics: TrackMetrics) -> dict[str, Any]:
    # 1) Detection
    # Note: without ground truth, we can only provide proxy checks.
    det_has_rows = metrics.rows > 0 and metrics.unique_frames > 0
    det_has_tracks = metrics.unique_track_ids > 0 and metrics.avg_tracks_per_frame > 0.05
    det_has_classes = len(metrics.class_counts) > 0

    # If there are detections but no class labels, still allow partial.
    detection_status: Status
    if det_has_rows and det_has_tracks:
        detection_status = "pass" if det_has_classes else "partial"
    else:
        detection_status = "fail"

    # 1b) Area filtering (detections outside active area)
    # Use strict criterion: any corner outside ROI.
    roi_filter_status = _status_by_threshold(metrics.roi_any_corner_out_pct, pass_max=0.1, partial_max=1.0)

    # 2) Tracking (CRITICAL)
    # Proxy stability metrics (heuristic; no GT):
    tracks = max(metrics.unique_track_ids, 1)
    gaps_ratio = metrics.tracks_with_gaps / tracks
    short_ratio = metrics.tracks_len_le_5 / tracks

    # Switch rate is computed over IoU>=0.3 matches between consecutive frames.
    switch_status = _status_by_threshold(metrics.switch_rate * 100.0, pass_max=1.0, partial_max=3.0)
    gap_status = _status_by_threshold(gaps_ratio * 100.0, pass_max=5.0, partial_max=15.0)
    short_status = _status_by_threshold(short_ratio * 100.0, pass_max=20.0, partial_max=40.0)

    # Persistence proxy: median/avg length.
    # If median is extremely small, tracking is unstable even if switch heuristic is low.
    if metrics.median_track_length_frames >= 30 and metrics.avg_track_length_frames >= 45:
        persistence_status: Status = "pass"
    elif metrics.median_track_length_frames >= 10 and metrics.avg_track_length_frames >= 20:
        persistence_status = "partial"
    else:
        persistence_status = "fail"

    tracking_status = _overall_status(switch_status, gap_status, short_status, persistence_status)

    # 3) Scene constraint (Area Filtering)
    # (a) Define boundaries: ROI exists (implicit here because metrics computed).
    # (b) Restrict detections within it: same check as roi_filter_status.
    scene_constraint_status = roi_filter_status

    notes: list[str] = []
    if not det_has_classes:
        notes.append(
            "Detection: class labels missing/empty in CSV; cannot verify 'all relevant entities' without a class taxonomy + ground truth."
        )
    notes.append(
        "Tracking stability uses heuristics (IoU-based match switch rate, track gaps, short tracks). For submission-grade validation, compare against ground truth or add re-identification evaluation."
    )

    return {
        "overall": _overall_status(detection_status, roi_filter_status, tracking_status, scene_constraint_status),
        "detection": {
            "status": detection_status,
            "has_rows": det_has_rows,
            "has_tracks": det_has_tracks,
            "has_class_labels": det_has_classes,
        },
        "area_filtering": {
            "status": roi_filter_status,
            "roi_any_corner_out_pct": metrics.roi_any_corner_out_pct,
            "roi_center_out_pct": metrics.roi_center_out_pct,
        },
        "tracking": {
            "status": tracking_status,
            "persistence_status": persistence_status,
            "switch_status": switch_status,
            "gap_status": gap_status,
            "short_status": short_status,
            "switch_rate": metrics.switch_rate,
            "total_matches": metrics.total_matches,
            "switches": metrics.switches,
            "tracks_with_gaps": metrics.tracks_with_gaps,
            "worst_max_gap_frames": metrics.worst_max_gap_frames,
            "tracks_len_le_5": metrics.tracks_len_le_5,
            "avg_track_length_frames": metrics.avg_track_length_frames,
            "median_track_length_frames": metrics.median_track_length_frames,
        },
        "scene_constraint": {
            "status": scene_constraint_status,
            "restricted_within_area": roi_filter_status,
        },
        "metrics": asdict(metrics),
        "notes": notes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Rate pipeline output against rubric (Detection/Tracking/Scene filtering)")
    ap.add_argument("--csv", required=True, help="Tracks CSV (e.g., output/tracks_30s_strict.csv)")
    ap.add_argument("--roi", required=True, help="ROI JSON (e.g., output/scene_roi.json)")
    ap.add_argument("--out", default=str(Path("output") / "rubric_scorecard.json"), help="Output JSON path")
    args = ap.parse_args()

    m = compute_metrics(csv_path=Path(args.csv), roi_path=Path(args.roi))
    report = rate(m)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Minimal console scorecard
    print(f"Overall: {report['overall']}")
    print(f"Detection: {report['detection']['status']}")
    print(
        "Area filtering (strict corners): "
        f"{report['area_filtering']['status']} (corner-out={report['area_filtering']['roi_any_corner_out_pct']:.3f}%)"
    )
    print(
        "Tracking: "
        f"{report['tracking']['status']} (switch_rate={report['tracking']['switch_rate']*100.0:.2f}%, "
        f"tracks_with_gaps={report['tracking']['tracks_with_gaps']}/{report['metrics']['unique_track_ids']})"
    )
    print(f"Scene constraint: {report['scene_constraint']['status']}")
    print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
