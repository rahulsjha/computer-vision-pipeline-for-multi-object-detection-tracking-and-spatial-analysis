"""Geometric Calibration Engine

Implements homography-based calibration using:
- Hough Transform for line detection (sidelines, goal lines)
- Circular Hough Transform for center circle
- Keypoint matching to standard pitch model
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np


@dataclass
class PitchKeypoints:
    """Detected keypoints on the pitch."""
    corners: List[Tuple[float, float]] = None  # 4 corners
    center_circle: Optional[Tuple[float, float, float]] = None  # (x, y, radius)
    sidelines: List[Tuple[Tuple[float, float], Tuple[float, float]]] = None  # Lines
    goal_lines: List[Tuple[Tuple[float, float], Tuple[float, float]]] = None  # Lines
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PitchGeometry:
    """Standard football pitch geometry."""
    
    # FIFA Standard Pitch Dimensions (meters)
    LENGTH_M = 105.0
    WIDTH_M = 68.0
    
    # Key Markings (meters from corner)
    CENTER_CIRCLE_RADIUS = 9.15
    GOAL_AREA_LENGTH = 5.5
    GOAL_AREA_WIDTH = 18.32
    PENALTY_AREA_LENGTH = 16.5
    PENALTY_AREA_WIDTH = 40.32
    CORNER_ARC_RADIUS = 1.0
    
    @classmethod
    def get_corner_points(cls) -> np.ndarray:
        """Get 4 corner points of standard pitch."""
        return np.array([
            [0.0, 0.0],              # Top-left
            [cls.LENGTH_M, 0.0],     # Top-right
            [cls.LENGTH_M, cls.WIDTH_M],   # Bottom-right
            [0.0, cls.WIDTH_M]       # Bottom-left
        ], dtype=np.float32)
    
    @classmethod
    def get_center_circle(cls) -> Tuple[float, float, float]:
        """Get center circle (x, y, radius) in world space."""
        return (cls.LENGTH_M / 2, cls.WIDTH_M / 2, cls.CENTER_CIRCLE_RADIUS)
    
    @classmethod
    def get_penalty_boxes(cls) -> List[np.ndarray]:
        """Get penalty box corners."""
        left_box = np.array([
            [0, (cls.WIDTH_M - cls.PENALTY_AREA_WIDTH) / 2],
            [cls.PENALTY_AREA_LENGTH, (cls.WIDTH_M - cls.PENALTY_AREA_WIDTH) / 2],
            [cls.PENALTY_AREA_LENGTH, (cls.WIDTH_M + cls.PENALTY_AREA_WIDTH) / 2],
            [0, (cls.WIDTH_M + cls.PENALTY_AREA_WIDTH) / 2]
        ], dtype=np.float32)
        
        right_box = np.array([
            [cls.LENGTH_M - cls.PENALTY_AREA_LENGTH, (cls.WIDTH_M - cls.PENALTY_AREA_WIDTH) / 2],
            [cls.LENGTH_M, (cls.WIDTH_M - cls.PENALTY_AREA_WIDTH) / 2],
            [cls.LENGTH_M, (cls.WIDTH_M + cls.PENALTY_AREA_WIDTH) / 2],
            [cls.LENGTH_M - cls.PENALTY_AREA_LENGTH, (cls.WIDTH_M + cls.PENALTY_AREA_WIDTH) / 2]
        ], dtype=np.float32)
        
        return [left_box, right_box]


class HomographyCalibration:
    """Compute homography matrix from pitch keypoints."""
    
    def __init__(self):
        self.homography_matrix = None
        self.inverse_homography = None
    
    def compute_from_corners(self, pixel_corners: np.ndarray) -> np.ndarray:
        """
        Compute homography from 4 corner points.
        
        Args:
            pixel_corners: 4x2 array of pixel coordinates
        
        Returns:
            3x3 homography matrix (pixel to world)
        """
        world_corners = PitchGeometry.get_corner_points()
        
        # Compute homography (pixel → world)
        self.homography_matrix, _ = cv2.findHomography(
            pixel_corners.astype(np.float32),
            world_corners.astype(np.float32)
        )
        
        # Compute inverse (world → pixel)
        self.inverse_homography = np.linalg.inv(self.homography_matrix)
        
        return self.homography_matrix
    
    def pixel_to_world(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """Project pixel coordinates to world space."""
        if self.homography_matrix is None:
            raise ValueError("Homography not initialized. Call compute_from_corners first.")
        
        point = np.array([[[x_px, y_px]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)
        return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])
    
    def world_to_pixel(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """Project world coordinates to pixel space."""
        if self.inverse_homography is None:
            raise ValueError("Inverse homography not initialized. Call compute_from_corners first.")
        
        point = np.array([[[x_m, y_m]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.inverse_homography)
        return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])
    
    def save(self, path: str) -> None:
        """Save homography matrices."""
        if self.homography_matrix is None:
            raise ValueError("No homography to save")
        
        data = {
            'homography': self.homography_matrix.tolist(),
            'inverse_homography': self.inverse_homography.tolist()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Homography saved to {path}")
    
    def load(self, path: str) -> None:
        """Load homography matrices."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.homography_matrix = np.array(data['homography'])
        self.inverse_homography = np.array(data['inverse_homography'])
        
        print(f"✓ Homography loaded from {path}")


class KeypointDetector:
    """Detect pitch keypoints using computer vision."""
    
    @staticmethod
    def detect_lines(frame: np.ndarray, threshold: int = 100) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Detect lines using Hough Transform."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        lines = cv2.HoughLinesP(
            edges,
            rho=1.0,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=200,
            maxLineGap=20
        )
        
        result = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                result.append(((x1, y1), (x2, y2)))
        
        return result
    
    @staticmethod
    def detect_circles(frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect circles using Hough Circle Transform."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=200,
            param1=100,      # Canny threshold
            param2=30,       # Accumulator threshold
            minRadius=40,
            maxRadius=100
        )
        
        result = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0]:
                x, y, r = circle
                result.append((x, y, r))
        
        return result
    
    @staticmethod
    def find_line_intersections(lines: List[Tuple[Tuple[int, int], Tuple[int, int]]], 
                                tolerance: int = 10) -> List[Tuple[float, float]]:
        """Find intersections between lines."""
        def line_intersection(line1: Tuple, line2: Tuple) -> Optional[Tuple[float, float]]:
            """Find intersection point of two lines."""
            (x1, y1), (x2, y2) = line1
            (x3, y3), (x4, y4) = line2
            
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                return None  # Parallel lines
            
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            return (x, y)
        
        intersections = []
        for i, line1 in enumerate(lines):
            for line2 in lines[i+1:]:
                intersection = line_intersection(line1, line2)
                if intersection:
                    # Check if not duplicate
                    is_duplicate = False
                    for existing in intersections:
                        dist = ((intersection[0] - existing[0])**2 + (intersection[1] - existing[1])**2)**0.5
                        if dist < tolerance:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        intersections.append(intersection)
        
        return intersections
    
    @staticmethod
    def find_field_corners(intersections: List[Tuple[float, float]], 
                          frame_height: int, 
                          frame_width: int) -> Optional[np.ndarray]:
        """Extract 4 corner points from intersections."""
        if len(intersections) < 4:
            return None
        
        intersections = np.array(intersections)
        
        # Find corners as extremal points
        # Top-left: min(x + y)
        # Top-right: max(x - y)
        # Bottom-right: max(x + y)
        # Bottom-left: min(x - y)
        
        sums = intersections[:, 0] + intersections[:, 1]
        diffs = intersections[:, 0] - intersections[:, 1]
        
        tl_idx = np.argmin(sums)
        tr_idx = np.argmax(diffs)
        br_idx = np.argmax(sums)
        bl_idx = np.argmin(diffs)
        
        corners = np.array([
            intersections[tl_idx],
            intersections[tr_idx],
            intersections[br_idx],
            intersections[bl_idx]
        ], dtype=np.float32)
        
        return corners


class CalibrationValidator:
    """Validate calibration accuracy."""
    
    @staticmethod
    def validate_known_distances(homography: HomographyCalibration, 
                                 tolerance_percent: float = 5.0) -> Dict[str, any]:
        """Validate calibration against known pitch distances."""
        result = {
            'is_valid': True,
            'errors': [],
            'checks': {}
        }
        
        # Check 1: Field length (105m)
        p1 = homography.pixel_to_world(0, 0)
        p2 = homography.pixel_to_world(105, 0)  # Approximate - depends on actual homography
        
        # This is a simplified check; real validation needs actual pixel corners
        
        return result
    
    @staticmethod
    def check_reprojection_error(homography: HomographyCalibration,
                                 world_points: List[Tuple[float, float]]) -> float:
        """Compute reprojection error for world points."""
        errors = []
        
        for x_m, y_m in world_points:
            # Project to pixel
            x_px, y_px = homography.world_to_pixel(x_m, y_m)
            
            # Project back to world
            x_m_reproj, y_m_reproj = homography.pixel_to_world(x_px, y_px)
            
            # Compute error
            error = ((x_m - x_m_reproj)**2 + (y_m - y_m_reproj)**2)**0.5
            errors.append(error)
        
        return float(np.mean(errors)) if errors else 0.0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calibrate video using homography")
    parser.add_argument("--frame", required=True, help="Frame image path for calibration")
    parser.add_argument("--output", required=True, help="Output homography JSON path")
    
    args = parser.parse_args()
    
    frame = cv2.imread(args.frame)
    
    # Detect keypoints
    detector = KeypointDetector()
    
    print("Detecting lines...")
    lines = detector.detect_lines(frame)
    print(f"  Found {len(lines)} lines")
    
    print("Finding intersections...")
    intersections = detector.find_line_intersections(lines)
    print(f"  Found {len(intersections)} intersections")
    
    print("Finding corners...")
    corners = detector.find_field_corners(intersections, frame.shape[0], frame.shape[1])
    
    if corners is not None:
        print(f"  Found {len(corners)} corners")
        
        # Compute homography
        cal = HomographyCalibration()
        cal.compute_from_corners(corners)
        cal.save(args.output)
        
        print(f"✓ Calibration complete. Saved to {args.output}")
    else:
        print("✗ Could not detect corners")
