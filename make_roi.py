from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from roi_utils import auto_detect_roi, save_roi


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-detect ROI polygon from first video frame")
    ap.add_argument("--video", required=True, help="Path to a video file")
    ap.add_argument(
        "--out",
        default=str(Path("output") / "scene_roi.json"),
        help="Output ROI JSON path (default: output/scene_roi.json)",
    )
    args = ap.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Could not read first frame")

    roi = auto_detect_roi(frame)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_roi(str(out_path), roi)
    print(f"ROI saved to {out_path}")
    print(f"Points: {roi.points}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
