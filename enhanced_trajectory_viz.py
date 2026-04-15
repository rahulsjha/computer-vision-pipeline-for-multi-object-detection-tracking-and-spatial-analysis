"""
Enhanced Trajectory Visualization
Advanced visualization with per-player/team filtering, motion arrows, and heatmaps.
"""

import os
import cv2
import numpy as np
import csv
import logging
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """Create enhanced trajectory visualizations with filtering and annotations."""
    
    def __init__(
        self,
        csv_path: str,
        video_path: str,
        output_path: str = "output/trajectory_viz.mp4",
        meter_per_pixel: float = 0.05,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ):
        """
        Initialize trajectory visualizer.
        
        Args:
            csv_path: Path to tracking CSV
            video_path: Path to input video (for frame reference)
            output_path: Path to output video
            meter_per_pixel: Calibration factor
            pitch_length_m: Pitch length
            pitch_width_m: Pitch width
        """
        self.csv_path = csv_path
        self.video_path = video_path
        self.output_path = output_path
        self.meter_per_pixel = meter_per_pixel
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        
        # Colors for different teams (detected by position clustering)
        self.color_team_1 = (0, 255, 0)      # Green (left team)
        self.color_team_2 = (255, 0, 0)      # Red (right team)
        self.color_referee = (255, 255, 255) # White
        
        self.data = self._load_csv()
        self.trajectories = self._build_trajectories()
        self.video = None
        self.fps = 25.0
        self.width = 1280
        self.height = 720
    
    def _load_csv(self) -> List[Dict]:
        """Load tracking data from CSV."""
        data = []
        
        if not os.path.exists(self.csv_path):
            logger.error(f"CSV not found: {self.csv_path}")
            return data
        
        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        data.append({
                            "frame": int(row.get("frame_index", 0)),
                            "track_id": int(row.get("track_id", 0)),
                            "x_m": float(row.get("x_meters", 0)),
                            "y_m": float(row.get("y_meters", 0)),
                            "x_px": float(row.get("x", 0)),
                            "y_px": float(row.get("y", 0)),
                            "speed_mps": float(row.get("speed_mps", 0)),
                            "conf": float(row.get("conf", 0)),
                        })
                    except (ValueError, KeyError):
                        continue
            
            logger.info(f"Loaded {len(data)} tracking points")
            return data
        
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return data
    
    def _build_trajectories(self) -> Dict[int, List[Dict]]:
        """Group tracking points by track ID."""
        trajectories = defaultdict(list)
        
        for point in self.data:
            trajectories[point["track_id"]].append(point)
        
        # Sort by frame
        for track_id in trajectories:
            trajectories[track_id].sort(key=lambda p: p["frame"])
        
        logger.info(f"Built trajectories for {len(trajectories)} tracks")
        return trajectories
    
    def classify_team(self, track_id: int) -> str:
        """
        Classify track as team 1, team 2, or referee based on position.
        
        Simple heuristic: tracks mostly on left side = team 1, right side = team 2
        """
        if track_id not in self.trajectories:
            return "unknown"
        
        positions = self.trajectories[track_id]
        mean_x = np.mean([p["x_m"] for p in positions])
        
        # Left half = team 1, right half = team 2
        if mean_x < self.pitch_length_m / 2:
            return "team_1"
        elif mean_x > self.pitch_length_m / 2:
            return "team_2"
        else:
            return "unknown"
    
    def filter_tracks(
        self,
        team: Optional[str] = None,
        min_length: int = 0,
        min_confidence: float = 0.0,
    ) -> Set[int]:
        """
        Filter tracks based on criteria.
        
        Args:
            team: "team_1", "team_2", or None for all
            min_length: Minimum track length in frames
            min_confidence: Minimum average confidence
        
        Returns:
            Set of track IDs matching criteria
        """
        filtered = set()
        
        for track_id, positions in self.trajectories.items():
            # Length filter
            if len(positions) < min_length:
                continue
            
            # Confidence filter
            avg_conf = np.mean([p["conf"] for p in positions])
            if avg_conf < min_confidence:
                continue
            
            # Team filter
            if team is not None:
                track_team = self.classify_team(track_id)
                if track_team != team:
                    continue
            
            filtered.add(track_id)
        
        logger.info(f"Filtered to {len(filtered)} tracks (team={team}, min_length={min_length})")
        return filtered
    
    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get color for a track based on team classification."""
        team = self.classify_team(track_id)
        
        if team == "team_1":
            return self.color_team_1
        elif team == "team_2":
            return self.color_team_2
        else:
            return self.color_referee
    
    def draw_trajectory(
        self,
        frame: np.ndarray,
        track_id: int,
        positions: List[Dict],
        draw_arrows: bool = True,
        arrow_interval: int = 5,
    ) -> np.ndarray:
        """
        Draw trajectory on frame.
        
        Args:
            frame: Frame to draw on
            track_id: Track ID
            positions: List of position dictionaries
            draw_arrows: Draw motion arrows
            arrow_interval: Draw arrow every N points
        
        Returns:
            Frame with drawn trajectory
        """
        color = self.get_track_color(track_id)
        
        if len(positions) < 2:
            return frame
        
        # Draw trajectory line
        points = [(int(p["x_px"]), int(p["y_px"])) for p in positions]
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 2)
        
        # Draw motion arrows
        if draw_arrows:
            for i in range(0, len(points) - arrow_interval, arrow_interval):
                p1 = points[i]
                p2 = points[i + arrow_interval]
                
                # Draw arrow
                self._draw_arrow(frame, p1, p2, color, thickness=2)
        
        # Draw start and end points
        cv2.circle(frame, points[0], 5, (0, 255, 255), -1)  # Yellow start
        cv2.circle(frame, points[-1], 5, color, -1)  # Team color end
        
        # Add track ID label at end
        cv2.putText(frame, str(track_id), points[-1], cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 2)
        
        return frame
    
    def _draw_arrow(
        self,
        frame: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
    ):
        """Draw arrow from p1 to p2."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.sqrt(dx**2 + dy**2)
        
        if length < 2:
            return
        
        # Normalize
        dx /= length
        dy /= length
        
        # Draw main line
        cv2.line(frame, p1, p2, color, thickness)
        
        # Draw arrowhead
        angle = np.arctan2(dy, dx)
        arrow_len = min(15, int(length * 0.3))
        
        p3 = (int(p2[0] - arrow_len * np.cos(angle - np.pi / 6)),
              int(p2[1] - arrow_len * np.sin(angle - np.pi / 6)))
        p4 = (int(p2[0] - arrow_len * np.cos(angle + np.pi / 6)),
              int(p2[1] - arrow_len * np.sin(angle + np.pi / 6)))
        
        cv2.line(frame, p2, p3, color, thickness)
        cv2.line(frame, p2, p4, color, thickness)
    
    def draw_pitch_markings(self, frame: np.ndarray) -> np.ndarray:
        """Draw pitch markings on frame."""
        h, w = frame.shape[:2]
        color = (100, 100, 100)
        thickness = 1
        
        # Center circle
        center = (w // 2, h // 2)
        radius = int((9.15 / self.pitch_length_m) * w)
        cv2.circle(frame, center, radius, color, thickness)
        
        # Center line
        cv2.line(frame, (w // 2, 0), (w // 2, h), color, thickness)
        
        # Goal areas
        goal_w = int((16.5 / self.pitch_length_m) * w)
        goal_h = int((40.32 / self.pitch_width_m) * h)
        
        cv2.rectangle(frame, (0, center[1] - goal_h // 2), 
                     (goal_w, center[1] + goal_h // 2), color, thickness)
        cv2.rectangle(frame, (w - goal_w, center[1] - goal_h // 2), 
                     (w, center[1] + goal_h // 2), color, thickness)
        
        return frame
    
    def visualize_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        tracked_tracks: Optional[Set[int]] = None,
        draw_pitch: bool = True,
        max_trail_length: int = 30,
    ) -> np.ndarray:
        """
        Visualize a single frame with trajectories.
        
        Args:
            frame: Input frame
            frame_idx: Frame index
            tracked_tracks: Set of track IDs to draw (None = all)
            draw_pitch: Draw pitch markings
            max_trail_length: Maximum trajectory length to draw
        
        Returns:
            Visualized frame
        """
        frame = frame.copy()
        
        if draw_pitch:
            frame = self.draw_pitch_markings(frame)
        
        # Draw all trajectories up to this frame
        for track_id, positions in self.trajectories.items():
            # Filter by tracked_tracks if provided
            if tracked_tracks is not None and track_id not in tracked_tracks:
                continue
            
            # Get positions up to this frame
            frame_positions = [p for p in positions if p["frame"] <= frame_idx]
            
            if frame_positions:
                # Limit trail length
                if len(frame_positions) > max_trail_length:
                    frame_positions = frame_positions[-max_trail_length:]
                
                frame = self.draw_trajectory(frame, track_id, frame_positions)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def generate_visualization(
        self,
        team: Optional[str] = None,
        min_track_length: int = 5,
        max_frames: Optional[int] = None,
    ):
        """
        Generate visualization video.
        
        Args:
            team: Filter to specific team ("team_1", "team_2", or None)
            min_track_length: Minimum track length in frames
            max_frames: Maximum frames to process (None = all)
        """
        # Get filtered tracks
        filtered_tracks = self.filter_tracks(
            team=team,
            min_length=min_track_length,
        )
        
        if not filtered_tracks:
            logger.warning("No tracks to visualize after filtering")
            return
        
        # Open input video for frame dimensions
        if not os.path.exists(self.video_path):
            logger.warning(f"Video not found: {self.video_path}, using default dimensions")
            cap = None
        else:
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, 
                                (self.width, self.height))
        
        if not writer.isOpened():
            logger.error(f"Failed to open video writer for {self.output_path}")
            return
        
        # Get frame range
        max_frame = max(
            [max(p["frame"] for p in self.trajectories[tid]) 
             for tid in filtered_tracks if self.trajectories[tid]]
        )
        
        if max_frames:
            max_frame = min(max_frame, max_frames)
        
        logger.info(f"Generating visualization for frames 0-{max_frame}")
        
        # Generate frames
        for frame_idx in range(max_frame + 1):
            # Create blank frame
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Visualize
            frame = self.visualize_frame(frame, frame_idx, filtered_tracks)
            
            writer.write(frame)
            
            if frame_idx % 100 == 0:
                logger.info(f"  Processed frame {frame_idx}/{max_frame}")
        
        writer.release()
        if cap:
            cap.release()
        
        logger.info(f"✓ Saved visualization to {self.output_path}")


def create_multiple_visualizations(
    csv_path: str,
    video_path: str,
    output_dir: str = "output",
):
    """
    Create multiple visualizations (overall, team 1, team 2).
    
    Args:
        csv_path: Path to tracking CSV
        video_path: Path to input video
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Overall trajectory visualization
    logger.info("Generating overall trajectory visualization...")
    viz_all = TrajectoryVisualizer(
        csv_path, video_path,
        output_path=os.path.join(output_dir, "trajectory_viz_all.mp4")
    )
    viz_all.generate_visualization(team=None, min_track_length=5, max_frames=500)
    
    # Team 1 visualization
    logger.info("Generating team 1 visualization...")
    viz_team1 = TrajectoryVisualizer(
        csv_path, video_path,
        output_path=os.path.join(output_dir, "trajectory_viz_team1.mp4")
    )
    viz_team1.generate_visualization(team="team_1", min_track_length=5, max_frames=500)
    
    # Team 2 visualization
    logger.info("Generating team 2 visualization...")
    viz_team2 = TrajectoryVisualizer(
        csv_path, video_path,
        output_path=os.path.join(output_dir, "trajectory_viz_team2.mp4")
    )
    viz_team2.generate_visualization(team="team_2", min_track_length=5, max_frames=500)
    
    logger.info("✓ All visualizations generated")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    csv_path = "output/tracks_5min_geo.csv"
    video_path = "output/clip_5min.mp4"
    
    if os.path.exists(csv_path):
        logger.info("Generating enhanced trajectory visualizations...")
        create_multiple_visualizations(csv_path, video_path)
    else:
        logger.error(f"CSV not found: {csv_path}")
