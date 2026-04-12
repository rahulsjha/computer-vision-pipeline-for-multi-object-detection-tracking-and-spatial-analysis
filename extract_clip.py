import argparse
import os
import sys
from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class ClipSpec:
    start_seconds: float
    duration_seconds: float


def parse_time_to_seconds(value: str) -> float:
    """Parse seconds as float or HH:MM:SS(.ms)."""
    value = value.strip()
    if not value:
        raise ValueError("Empty time value")

    if ":" not in value:
        return float(value)

    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid time format: {value!r}. Use seconds or HH:MM:SS")

    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    if minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60:
        raise ValueError(f"Invalid time value: {value!r}")

    return hours * 3600 + minutes * 60 + seconds


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def extract_clip(input_path: str, output_path: str, spec: ClipSpec) -> None:
    if spec.start_seconds < 0:
        raise ValueError("start time must be >= 0")
    if spec.duration_seconds <= 0:
        raise ValueError("duration must be > 0")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        # Some files/codecs don't report FPS. Use a conservative default.
        fps = 30.0
        print("Warning: could not read FPS; defaulting to 30", file=sys.stderr)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Could not read video dimensions")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start_frame = int(round(spec.start_seconds * fps))
    frames_to_write = int(round(spec.duration_seconds * fps))
    end_frame_exclusive = start_frame + frames_to_write

    if total_frames > 0 and start_frame >= total_frames:
        raise ValueError(
            f"Start time {spec.start_seconds}s is beyond video length (~{total_frames / fps:.2f}s)"
        )

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ensure_parent_dir(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            "Could not open output writer. Try a different output extension or codec. "
            "For example, use .avi on some systems."
        )

    written = 0
    while written < frames_to_write:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

        # Light progress logging every ~5 seconds of video
        if written % max(int(fps * 5), 1) == 0:
            secs = written / fps
            print(f"Wrote {secs:.0f}s / {spec.duration_seconds:.0f}s")

    writer.release()
    cap.release()

    actual_seconds = written / fps
    if written == 0:
        raise RuntimeError("No frames were written (check start time and codecs)")

    # Friendly summary
    if total_frames > 0:
        end_frame_actual = start_frame + written
        print(
            f"Done. Wrote {written} frames (~{actual_seconds:.2f}s) "
            f"from frame {start_frame} to {end_frame_actual - 1} (requested end < {end_frame_exclusive})."
        )
    else:
        print(f"Done. Wrote {written} frames (~{actual_seconds:.2f}s).")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a clip from a video (default: first 5 minutes) using OpenCV."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="videoplayback.mp4",
        help="Input video path (default: videoplayback.mp4)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join("output", "clip_5min.mp4"),
        help="Output clip path (default: output/clip_5min.mp4)",
    )
    parser.add_argument(
        "--start",
        default="0",
        help="Start time in seconds or HH:MM:SS (default: 0)",
    )
    parser.add_argument(
        "--duration",
        default="300",
        help="Duration in seconds or HH:MM:SS (default: 300 = 5 minutes)",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        start_seconds = parse_time_to_seconds(args.start)
        duration_seconds = parse_time_to_seconds(args.duration)
        extract_clip(
            input_path=args.input,
            output_path=args.output,
            spec=ClipSpec(start_seconds=start_seconds, duration_seconds=duration_seconds),
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
