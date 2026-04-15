"""
World-Space Heatmap Generator
Converts pixel-based heatmaps to world-coordinate space with advanced visualization.
Includes zone-based clustering, team separation, and event detection.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import cv2
import csv
from collections import defaultdict

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available - K-means clustering disabled")

logger = logging.getLogger(__name__)


@dataclass
class WorldCoordinate:
    """World-space position (meters)."""
    x_m: float
    y_m: float
    
    def to_pixel(self, meter_per_pixel: float) -> Tuple[int, int]:
        """Convert to pixel coordinates."""
        return (int(self.x_m / meter_per_pixel), int(self.y_m / meter_per_pixel))
    
    def distance_to(self, other: "WorldCoordinate") -> float:
        """Euclidean distance to another coordinate."""
        return np.sqrt((self.x_m - other.x_m)**2 + (self.y_m - other.y_m)**2)


@dataclass
class PitchZone:
    """Pitch zone definition."""
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    color: Tuple[int, int, int]  # BGR
    
    def contains(self, coord: WorldCoordinate) -> bool:
        """Check if coordinate is in this zone."""
        return (self.x_min <= coord.x_m <= self.x_max and 
                self.y_min <= coord.y_m <= self.y_max)


class WorldSpaceHeatmap:
    """Generate heatmaps in world-coordinate space (105m x 68m pitch)."""
    
    # Standard pitch zones (meters)
    STANDARD_ZONES = [
        PitchZone("Opponent Penalty Box", 80.0, 105.0, 20.4, 47.6, (0, 100, 255)),  # Orange
        PitchZone("Opponent Wing", 60.0, 80.0, 0.0, 68.0, (255, 100, 0)),            # Cyan
        PitchZone("Center", 40.0, 65.0, 20.0, 48.0, (0, 255, 0)),                    # Green
        PitchZone("Own Wing", 25.0, 45.0, 0.0, 68.0, (100, 0, 255)),                 # Magenta
        PitchZone("Own Penalty Box", 0.0, 25.0, 20.4, 47.6, (255, 0, 0)),            # Blue
    ]
    
    def __init__(
        self,
        csv_path: str,
        meter_per_pixel: float = 0.05,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
        num_clusters: int = 5,
    ):
        """
        Initialize world-space heatmap generator.
        
        Args:
            csv_path: Path to tracking CSV with x_meters, y_meters columns
            meter_per_pixel: Conversion factor (pixels to meters)
            pitch_length_m: Pitch length in meters
            pitch_width_m: Pitch width in meters
            num_clusters: Number of K-means clusters for zone detection
        """
        self.csv_path = csv_path
        self.meter_per_pixel = meter_per_pixel
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.num_clusters = num_clusters
        
        # Heatmap resolution (1px = 0.5m, so 210x136 for 105x68m pitch)
        self.heatmap_resolution = 0.5  # meters per pixel in heatmap
        self.heatmap_width = int(pitch_length_m / self.heatmap_resolution)
        self.heatmap_height = int(pitch_width_m / self.heatmap_resolution)
        
        self.data = self._load_tracking_data()
        self.zones = self.STANDARD_ZONES
    
    def _load_tracking_data(self) -> List[Dict[str, Any]]:
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
                            "track_id": int(row.get("track_id", 0)),
                            "frame_index": int(row.get("frame_index", 0)),
                            "x_m": float(row.get("x_meters", 0)),
                            "y_m": float(row.get("y_meters", 0)),
                            "speed_mps": float(row.get("speed_mps", 0)),
                            "confidence": float(row.get("conf", 0)),
                        })
                    except (ValueError, KeyError):
                        continue
            
            logger.info(f"Loaded {len(data)} tracking points from {self.csv_path}")
            return data
        
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return data
    
    def generate_presence_heatmap(self) -> np.ndarray:
        """
        Generate presence heatmap (how much time each zone is occupied).
        
        Returns:
            Heatmap array of shape (height, width)
        """
        heatmap = np.zeros((self.heatmap_height, self.heatmap_width), dtype=np.float32)
        
        if not self.data:
            logger.warning("No data to generate presence heatmap")
            return heatmap
        
        for point in self.data:
            x_m, y_m = point["x_m"], point["y_m"]
            
            # Clamp to pitch boundaries
            x_m = max(0, min(x_m, self.pitch_length_m))
            y_m = max(0, min(y_m, self.pitch_width_m))
            
            # Convert to heatmap coordinates
            hx = int(x_m / self.heatmap_resolution)
            hy = int(y_m / self.heatmap_resolution)
            
            # Add Gaussian kernel around this point for smoothing
            self._add_gaussian_kernel(heatmap, hx, hy, sigma=2.0, weight=1.0)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        
        return heatmap
    
    def generate_speed_heatmap(self) -> np.ndarray:
        """
        Generate speed heatmap (average speed in each zone).
        
        Returns:
            Heatmap array of shape (height, width)
        """
        heatmap = np.zeros((self.heatmap_height, self.heatmap_width), dtype=np.float32)
        counts = np.zeros_like(heatmap)
        
        if not self.data:
            return heatmap
        
        for point in self.data:
            x_m, y_m = point["x_m"], point["y_m"]
            speed = point["speed_mps"]
            
            # Clamp to pitch
            x_m = max(0, min(x_m, self.pitch_length_m))
            y_m = max(0, min(y_m, self.pitch_width_m))
            
            hx = int(x_m / self.heatmap_resolution)
            hy = int(y_m / self.heatmap_resolution)
            
            if 0 <= hx < self.heatmap_width and 0 <= hy < self.heatmap_height:
                heatmap[hy, hx] += speed
                counts[hy, hx] += 1
        
        # Average speed per cell
        mask = counts > 0
        heatmap[mask] = heatmap[mask] / counts[mask]
        
        # Normalize to 0-255
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        
        return heatmap
    
    def generate_zone_heatmap(self) -> np.ndarray:
        """
        Generate zone-based heatmap with K-means clustering.
        
        Returns:
            Heatmap showing which zones are most active
        """
        if not SKLEARN_AVAILABLE or len(self.data) < self.num_clusters:
            logger.warning("Cannot generate zone heatmap (insufficient data or sklearn not available)")
            return np.zeros((self.heatmap_height, self.heatmap_width), dtype=np.uint8)
        
        # Extract coordinates
        coords = np.array([[p["x_m"], p["y_m"]] for p in self.data])
        
        # K-means clustering
        try:
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(coords)
            
            # Create heatmap with cluster centers
            heatmap = np.zeros((self.heatmap_height, self.heatmap_width), dtype=np.uint8)
            
            for cluster_id, center in enumerate(kmeans.cluster_centers_):
                x_m, y_m = center
                hx = int(x_m / self.heatmap_resolution)
                hy = int(y_m / self.heatmap_resolution)
                
                # Draw cluster region
                cluster_intensity = int(255 * (cluster_id + 1) / self.num_clusters)
                self._add_gaussian_kernel(heatmap, hx, hy, sigma=5.0, weight=cluster_intensity)
            
            return heatmap
        
        except Exception as e:
            logger.error(f"Zone clustering failed: {e}")
            return np.zeros((self.heatmap_height, self.heatmap_width), dtype=np.uint8)
    
    def generate_activity_by_zone(self) -> Dict[str, int]:
        """
        Calculate activity level for each predefined zone.
        
        Returns:
            Dictionary mapping zone name to point count
        """
        zone_activity = {zone.name: 0 for zone in self.zones}
        
        for point in self.data:
            coord = WorldCoordinate(point["x_m"], point["y_m"])
            for zone in self.zones:
                if zone.contains(coord):
                    zone_activity[zone.name] += 1
                    break
        
        return zone_activity
    
    def _add_gaussian_kernel(
        self,
        heatmap: np.ndarray,
        cx: int,
        cy: int,
        sigma: float = 2.0,
        weight: float = 1.0,
    ):
        """Add Gaussian kernel to heatmap for smoothing."""
        h, w = heatmap.shape
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        gaussian = weight * np.exp(-(x*x + y*y) / (2 * sigma**2))
        heatmap[:] = np.maximum(heatmap, gaussian)
    
    def visualize_heatmap(
        self,
        heatmap: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Convert grayscale heatmap to colored visualization.
        
        Args:
            heatmap: Grayscale heatmap (0-255)
            colormap: OpenCV colormap (COLORMAP_JET, COLORMAP_HOT, etc.)
        
        Returns:
            BGR colored image
        """
        if heatmap.max() == 0:
            heatmap = np.ones_like(heatmap) * 50
        
        colored = cv2.applyColorMap(heatmap, colormap)
        return colored
    
    def draw_pitch_grid(self, image: np.ndarray, line_color: Tuple[int, int, int] = (100, 100, 100)):
        """
        Draw pitch markings on image.
        
        Args:
            image: Image to draw on
            line_color: Line color (BGR)
        
        Returns:
            Image with pitch grid
        """
        h, w = image.shape[:2]
        thickness = 1
        
        # Pitch dimensions in pixels
        center_x = w // 2
        center_y = h // 2
        
        # Center circle
        radius_pixels = int((9.15 / self.pitch_length_m) * w)
        cv2.circle(image, (center_x, center_y), radius_pixels, line_color, thickness)
        
        # Center line
        cv2.line(image, (center_x, 0), (center_x, h), line_color, thickness)
        
        # Goal area markers
        goal_area_w = int((16.5 / self.pitch_length_m) * w)
        goal_area_h = int((40.32 / self.pitch_width_m) * h)
        
        # Own goal area
        cv2.rectangle(image, (0, center_y - goal_area_h // 2), 
                     (goal_area_w, center_y + goal_area_h // 2), line_color, thickness)
        
        # Opponent goal area
        cv2.rectangle(image, (w - goal_area_w, center_y - goal_area_h // 2), 
                     (w, center_y + goal_area_h // 2), line_color, thickness)
        
        return image
    
    def save_all_heatmaps(self, output_dir: str = "output"):
        """Generate and save all heatmap visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Presence heatmap
        logger.info("Generating presence heatmap...")
        presence = self.generate_presence_heatmap()
        presence_viz = self.visualize_heatmap(presence, cv2.COLORMAP_JET)
        presence_viz = self.draw_pitch_grid(presence_viz)
        cv2.imwrite(os.path.join(output_dir, "heatmap_presence_world.png"), presence_viz)
        logger.info(f"  ✓ Saved to {output_dir}/heatmap_presence_world.png")
        
        # Speed heatmap
        logger.info("Generating speed heatmap...")
        speed = self.generate_speed_heatmap()
        speed_viz = self.visualize_heatmap(speed, cv2.COLORMAP_HOT)
        speed_viz = self.draw_pitch_grid(speed_viz)
        cv2.imwrite(os.path.join(output_dir, "heatmap_speed_world.png"), speed_viz)
        logger.info(f"  ✓ Saved to {output_dir}/heatmap_speed_world.png")
        
        # Zone heatmap
        logger.info("Generating zone heatmap...")
        zones = self.generate_zone_heatmap()
        zones_viz = self.visualize_heatmap(zones, cv2.COLORMAP_VIRIDIS)
        zones_viz = self.draw_pitch_grid(zones_viz)
        cv2.imwrite(os.path.join(output_dir, "heatmap_zones_world.png"), zones_viz)
        logger.info(f"  ✓ Saved to {output_dir}/heatmap_zones_world.png")
        
        # Zone activity report
        logger.info("Calculating zone activity...")
        activity = self.generate_activity_by_zone()
        activity_path = os.path.join(output_dir, "zone_activity_report.json")
        with open(activity_path, 'w') as f:
            json.dump(activity, f, indent=2)
        logger.info(f"  ✓ Saved activity report to {activity_path}")
        
        return {
            "presence": os.path.join(output_dir, "heatmap_presence_world.png"),
            "speed": os.path.join(output_dir, "heatmap_speed_world.png"),
            "zones": os.path.join(output_dir, "heatmap_zones_world.png"),
            "activity": activity_path,
        }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test
    csv_path = "output/tracks_5min_geo.csv"
    if os.path.exists(csv_path):
        logger.info(f"Generating world-space heatmaps from {csv_path}...")
        heatmap_gen = WorldSpaceHeatmap(csv_path)
        outputs = heatmap_gen.save_all_heatmaps()
        logger.info("✓ All heatmaps generated successfully")
    else:
        logger.error(f"CSV not found: {csv_path}")
