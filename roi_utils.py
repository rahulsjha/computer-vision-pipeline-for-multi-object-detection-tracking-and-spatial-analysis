from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


@dataclass(frozen=True)
class RoiPolygon:
    points: List[Point]

    def as_np(self) -> np.ndarray:
        return np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))

    def contains_points(self, xy: np.ndarray) -> np.ndarray:
        """Return a boolean mask for which points are inside/on the polygon.

        Args:
            xy: array of shape (N, 2)
        """
        poly = np.array(self.points, dtype=np.int32)
        out = np.zeros((len(xy),), dtype=bool)
        for i, (x, y) in enumerate(xy.astype(float)):
            out[i] = cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
        return out

    def mask(self, frame_shape: Sequence[int]) -> np.ndarray:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [self.as_np()], 255)
        return mask


def save_roi(path: str, roi: RoiPolygon) -> None:
    payload = {"points": [{"x": int(x), "y": int(y)} for x, y in roi.points]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_roi(path: str) -> RoiPolygon:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    points = payload.get("points")
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("ROI JSON must contain at least 3 points under key 'points'")
    out: List[Point] = []
    for p in points:
        out.append((int(p["x"]), int(p["y"])))
    return RoiPolygon(points=out)


def _order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    # pts: (N,2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    return pts[order]


def auto_detect_roi(frame: np.ndarray) -> RoiPolygon:
    """Heuristic: detect the main scene boundary on the first frame.

    If it fails, returns the full-frame rectangle.
    """
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return RoiPolygon(points=[(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)])

    # Largest contour by area
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 0.05 * (w * h):
        # Too small to be meaningful; fallback to full frame
        return RoiPolygon(points=[(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)])

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    pts = approx.reshape(-1, 2) if approx is not None and len(approx) >= 3 else None
    if pts is None or len(pts) < 3:
        return RoiPolygon(points=[(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)])

    # If too many points, simplify via convex hull then approx again
    if len(pts) > 12:
        hull = cv2.convexHull(pts)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        pts = approx.reshape(-1, 2)

    # Clip points to image bounds
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    # Make ordering stable
    pts = _order_points_clockwise(pts.astype(np.float32)).astype(np.int32)

    return RoiPolygon(points=[(int(x), int(y)) for x, y in pts])
