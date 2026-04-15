"""Trajectory Smoothing and Speed Validation Module

Implements multiple smoothing strategies:
- Kalman Filter (velocity model)
- Savitzky-Golay Filter (polynomial smoothing)
- Exponential Moving Average

Provides speed validation and sanity checks.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.signal import savgol_filter


@dataclass
class TrajectoryPoint:
    """Single trajectory point with optional smoothed variants."""
    
    frame_index: int
    time_seconds: float
    track_id: int
    
    # Original coordinates
    x_px: float
    y_px: float
    
    # World coordinates
    x_m: float
    y_m: float
    
    # Speed (original)
    speed_mps: float
    speed_kmh: float
    distance_traveled: float
    
    # Smoothed variants (added post-processing)
    x_px_smooth: Optional[float] = None
    y_px_smooth: Optional[float] = None
    x_m_smooth: Optional[float] = None
    y_m_smooth: Optional[float] = None
    speed_mps_smooth: Optional[float] = None
    distance_smooth: Optional[float] = None
    
    # Validation flags
    is_outlier: bool = False
    speed_violation: str = ""  # Empty if valid, else reason
    confidence_score: float = 1.0  # [0, 1] smoothing confidence


class KalmanTrajectoryFilter:
    """Per-track Kalman Filter for trajectory smoothing."""
    
    def __init__(self, process_variance: float = 0.1, measurement_variance: float = 1.0):
        """
        Args:
            process_variance: Q - model uncertainty
            measurement_variance: R - measurement uncertainty
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.filters: Dict[int, KalmanFilter] = {}
    
    def _create_filter(self, track_id: int) -> KalmanFilter:
        """Create a 2D Kalman filter for a track."""
        kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, vx, y, vy], Measurement: [x, y]
        kf.F = np.array([[1, 1, 0, 0],     # State transition
                         [0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],     # Measurement function
                         [0, 0, 1, 0]])
        kf.R = np.eye(2) * self.measurement_variance  # Measurement noise
        kf.Q = np.eye(4) * self.process_variance     # Process noise
        kf.P = np.eye(4)                   # Covariance matrix
        kf.x = np.array([0, 0, 0, 0])     # Initial state
        return kf
    
    def smooth(self, track_points: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """Apply Kalman smoothing to a track."""
        if not track_points:
            return track_points
        
        track_id = track_points[0].track_id
        kf = self._create_filter(track_id)
        
        smoothed_points = []
        for point in track_points:
            # Predict
            kf.predict()
            
            # Update with measurement
            z = np.array([point.x_m, point.y_m])
            kf.update(z)
            
            # Extract smoothed estimate
            x_smooth, _, y_smooth, _ = kf.x
            
            # Create new point with smoothed values
            smoothed_point = TrajectoryPoint(**asdict(point))
            smoothed_point.x_m_smooth = float(x_smooth)
            smoothed_point.y_m_smooth = float(y_smooth)
            
            smoothed_points.append(smoothed_point)
        
        return smoothed_points


class TrajectoryValidator:
    """Validates trajectories for physical realism."""
    
    def __init__(self, max_speed_mps: float = 10.0, max_acceleration: float = 5.0):
        """
        Args:
            max_speed_mps: Maximum realistic speed (m/s)
            max_acceleration: Maximum realistic acceleration (m/s²)
        """
        self.max_speed_mps = max_speed_mps
        self.max_acceleration = max_acceleration
    
    def validate_point(self, point: TrajectoryPoint, previous_point: Optional[TrajectoryPoint] = None) -> Tuple[bool, str]:
        """
        Validate a single point.
        
        Returns:
            (is_valid, reason_if_invalid)
        """
        # Check speed bounds
        if point.speed_mps < 0:
            return False, f"Negative speed: {point.speed_mps} m/s"
        
        if point.speed_mps > self.max_speed_mps:
            return False, f"Speed too high: {point.speed_mps:.2f} m/s (max: {self.max_speed_mps})"
        
        # Check acceleration (if previous point exists)
        if previous_point is not None and previous_point.time_seconds > 0:
            dt = point.time_seconds - previous_point.time_seconds
            if dt > 0:
                dv = point.speed_mps - previous_point.speed_mps
                acceleration = dv / dt
                
                if abs(acceleration) > self.max_acceleration:
                    return False, f"Acceleration too high: {acceleration:.2f} m/s² (max: {self.max_acceleration})"
        
        # Check coordinate bounds (assuming 105m × 68m pitch)
        if not (0 <= point.x_m <= 105):
            return False, f"X out of bounds: {point.x_m:.1f}m (should be 0-105)"
        
        if not (0 <= point.y_m <= 68):
            return False, f"Y out of bounds: {point.y_m:.1f}m (should be 0-68)"
        
        return True, ""
    
    def detect_outliers_zscore(self, points: List[TrajectoryPoint], column: str = "speed_mps", threshold: float = 3.0) -> List[bool]:
        """Detect outliers using Z-score method."""
        values = np.array([getattr(p, column) for p in points])
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return [False] * len(points)
        
        z_scores = np.abs((values - mean) / std)
        return (z_scores > threshold).tolist()


class SavitzkyGolayTrajectoryFilter:
    """Polygon fitting based smoothing (alternative to Kalman)."""
    
    def __init__(self, window_length: int = 5, polyorder: int = 2):
        """
        Args:
            window_length: Window size (must be odd)
            polyorder: Polynomial order (1=linear, 2=quadratic, etc.)
        """
        self.window_length = window_length if window_length % 2 == 1 else window_length + 1
        self.polyorder = polyorder
    
    def smooth(self, track_points: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """Apply Savitzky-Golay smoothing to a track."""
        if len(track_points) < self.window_length:
            return track_points  # Not enough points to smooth
        
        # Extract coordinates
        x_values = np.array([p.x_m for p in track_points])
        y_values = np.array([p.y_m for p in track_points])
        
        # Apply smoothing
        try:
            x_smooth = savgol_filter(x_values, self.window_length, self.polyorder)
            y_smooth = savgol_filter(y_values, self.window_length, self.polyorder)
        except ValueError:
            # Window too large or other issue
            return track_points
        
        # Compute smoothed speeds
        dx = np.diff(x_smooth)
        dy = np.diff(y_smooth)
        distances = np.sqrt(dx**2 + dy**2)
        
        # Reconstruct points with smoothed values
        smoothed_points = []
        cumulative_distance = 0.0
        
        for i, point in enumerate(track_points):
            smoothed_point = TrajectoryPoint(**asdict(point))
            smoothed_point.x_m_smooth = float(x_smooth[i])
            smoothed_point.y_m_smooth = float(y_smooth[i])
            
            if i > 0:
                cumulative_distance += distances[i - 1]
                # Approximate speed from smoothed coordinates (need FPS)
                dt = (point.time_seconds - track_points[i-1].time_seconds) or 0.04  # 25 FPS default
                if dt > 0:
                    distance_delta = distances[i - 1]
                    smoothed_point.speed_mps_smooth = distance_delta / dt
            
            smoothed_point.distance_smooth = cumulative_distance
            smoothed_points.append(smoothed_point)
        
        return smoothed_points


class TrajectoryProcessor:
    """Orchestrates smoothing and validation."""
    
    def __init__(self, fps: float = 25.0, max_speed_mps: float = 10.0, smoothing_method: str = "kalman"):
        """
        Args:
            fps: Frames per second
            max_speed_mps: Maximum realistic speed
            smoothing_method: "kalman" or "savgol"
        """
        self.fps = fps
        self.max_speed_mps = max_speed_mps
        self.smoothing_method = smoothing_method
        
        self.validator = TrajectoryValidator(max_speed_mps=max_speed_mps)
        
        if smoothing_method == "kalman":
            self.smoother = KalmanTrajectoryFilter()
        else:
            self.smoother = SavitzkyGolayTrajectoryFilter(window_length=5, polyorder=2)
    
    def process_csv(self, csv_path: str, output_path: Optional[str] = None) -> List[Dict]:
        """
        Load CSV, process trajectories, and optionally save.
        
        Args:
            csv_path: Input CSV file
            output_path: Output CSV path (if None, not saved)
        
        Returns:
            List of processed track dictionaries
        """
        # Read CSV
        rows: Dict[int, List[TrajectoryPoint]] = defaultdict(list)
        
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    point = TrajectoryPoint(
                        frame_index=int(row['frame_index']),
                        time_seconds=float(row['time_seconds']),
                        track_id=int(row['track_id']),
                        x_px=float(row['x']),
                        y_px=float(row['y']),
                        x_m=float(row['x_meters']),
                        y_m=float(row['y_meters']),
                        speed_mps=float(row['speed_mps']),
                        speed_kmh=float(row['speed_kmh']),
                        distance_traveled=float(row['distance_traveled']),
                    )
                    rows[point.track_id].append(point)
                except (KeyError, ValueError) as e:
                    print(f"Warning: Skipping row due to error: {e}")
                    continue
        
        # Process each track
        all_points = []
        for track_id, points in rows.items():
            # Smooth trajectories
            if self.smoothing_method == "kalman":
                smoothed = self.smoother.smooth(points)
            else:
                smoothed = self.smoother.smooth(points)
            
            # Validate points
            for i, point in enumerate(smoothed):
                prev_point = smoothed[i-1] if i > 0 else None
                is_valid, reason = self.validator.validate_point(point, prev_point)
                
                if not is_valid:
                    point.is_outlier = True
                    point.speed_violation = reason
                
                all_points.append(point)
        
        # Save output if requested
        if output_path:
            self._save_enhanced_csv(all_points, output_path)
        
        return [asdict(p) for p in all_points]
    
    def _save_enhanced_csv(self, points: List[TrajectoryPoint], output_path: str) -> None:
        """Save enhanced CSV with smoothed values and validation flags."""
        if not points:
            return
        
        # Prepare fieldnames
        fieldnames = [
            'frame_index', 'time_seconds', 'track_id',
            'x_px', 'y_px', 'x_m', 'y_m',
            'speed_mps', 'speed_kmh', 'distance_traveled',
            'x_px_smooth', 'y_px_smooth', 'x_m_smooth', 'y_m_smooth',
            'speed_mps_smooth', 'distance_smooth',
            'is_outlier', 'speed_violation', 'confidence_score'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for point in points:
                row = {
                    'frame_index': point.frame_index,
                    'time_seconds': round(point.time_seconds, 4),
                    'track_id': point.track_id,
                    'x_px': round(point.x_px, 2),
                    'y_px': round(point.y_px, 2),
                    'x_m': round(point.x_m, 2),
                    'y_m': round(point.y_m, 2),
                    'speed_mps': round(point.speed_mps, 4),
                    'speed_kmh': round(point.speed_kmh, 4),
                    'distance_traveled': round(point.distance_traveled, 2),
                    'x_px_smooth': round(point.x_px_smooth, 2) if point.x_px_smooth is not None else '',
                    'y_px_smooth': round(point.y_px_smooth, 2) if point.y_px_smooth is not None else '',
                    'x_m_smooth': round(point.x_m_smooth, 2) if point.x_m_smooth is not None else '',
                    'y_m_smooth': round(point.y_m_smooth, 2) if point.y_m_smooth is not None else '',
                    'speed_mps_smooth': round(point.speed_mps_smooth, 4) if point.speed_mps_smooth is not None else '',
                    'distance_smooth': round(point.distance_smooth, 2) if point.distance_smooth is not None else '',
                    'is_outlier': point.is_outlier,
                    'speed_violation': point.speed_violation,
                    'confidence_score': round(point.confidence_score, 3),
                }
                writer.writerow(row)
        
        print(f"✓ Enhanced CSV saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Smooth trajectories and validate speeds")
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--method", default="kalman", choices=["kalman", "savgol"], help="Smoothing method")
    parser.add_argument("--max-speed", type=float, default=10.0, help="Max realistic speed (m/s)")
    parser.add_argument("--fps", type=float, default=25.0, help="Video FPS")
    
    args = parser.parse_args()
    
    processor = TrajectoryProcessor(
        fps=args.fps,
        max_speed_mps=args.max_speed,
        smoothing_method=args.method
    )
    
    processor.process_csv(args.csv, args.output)
    print(f"✓ Processing complete. Output: {args.output}")
