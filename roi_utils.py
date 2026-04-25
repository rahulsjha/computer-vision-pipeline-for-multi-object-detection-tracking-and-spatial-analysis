from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


def _normalize_points(points: Iterable[Sequence[float]]) -> List[Point]:
    normalized: List[Point] = []
    for point in points:
        if len(point) != 2:
            raise ValueError("ROI points must be 2D coordinates")
        normalized.append((int(round(float(point[0]))), int(round(float(point[1])))))
    if len(normalized) < 3:
        raise ValueError("ROI polygon requires at least 3 points")
    return normalized


@dataclass(frozen=True)
class RoiPolygon:
    points: List[Point]

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", _normalize_points(self.points))

    def as_np(self) -> np.ndarray:
        return np.asarray(self.points, dtype=np.int32)

    def mask(self, frame_shape: Tuple[int, ...]) -> np.ndarray:
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [self.as_np()], 255)
        return mask

    def contains_point(self, x: float, y: float) -> bool:
        polygon = self.as_np().astype(np.float32)
        return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0


def _fallback_roi(frame: np.ndarray) -> RoiPolygon:
    height, width = frame.shape[:2]
    inset_x = max(int(width * 0.03), 1)
    inset_y = max(int(height * 0.03), 1)
    return RoiPolygon(
        [
            (inset_x, inset_y),
            (width - inset_x, inset_y),
            (width - inset_x, height - inset_y),
            (inset_x, height - inset_y),
        ]
    )


def auto_detect_roi(frame: np.ndarray) -> RoiPolygon:
    if frame is None or frame.size == 0:
        raise ValueError("Frame is empty")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (25, 25, 25), (95, 255, 255))

    kernel = np.ones((7, 7), dtype=np.uint8)
    cleaned = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _fallback_roi(frame)

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < frame.shape[0] * frame.shape[1] * 0.05:
        return _fallback_roi(frame)

    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(polygon) < 4:
        rect = cv2.minAreaRect(contour)
        polygon = cv2.boxPoints(rect)
        polygon = np.asarray(polygon, dtype=np.float32)
    else:
        polygon = polygon.reshape(-1, 2)

    return RoiPolygon(polygon.tolist())


def save_roi(path: str, roi: RoiPolygon) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"points": [[x, y] for x, y in roi.points]}
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_roi(path: str) -> RoiPolygon:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        points = payload.get("points")
    else:
        points = payload

    if not isinstance(points, list):
        raise ValueError(f"Unsupported ROI file format: {path}")

    return RoiPolygon(points)