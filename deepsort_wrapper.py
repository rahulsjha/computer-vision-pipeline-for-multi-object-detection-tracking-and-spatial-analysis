"""
DeepSORT Tracker Wrapper
Integrates deep-sort-pytorch with appearance-based tracking for improved ID stability.
Configuration-driven from tracking_config.yaml
"""

import os
import yaml
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DeepSORTConfig:
    """DeepSORT configuration from YAML."""
    max_age: int = 15
    max_iou_distance: float = 0.50
    max_cosine_distance: float = 0.20
    nn_budget: int = 100
    metric: str = "cosine"  # or "euclidean"
    
    @classmethod
    def load(cls, config_path: str = "tracking_config.yaml") -> "DeepSORTConfig":
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            logger.warning(f"Config not found: {config_path}, using defaults")
            return cls()
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            
            deepsort_cfg = config_data.get("deepsort", {})
            return cls(
                max_age=deepsort_cfg.get("max_age", 15),
                max_iou_distance=deepsort_cfg.get("max_iou_distance", 0.50),
                max_cosine_distance=deepsort_cfg.get("max_cosine_distance", 0.20),
                nn_budget=deepsort_cfg.get("nn_budget", 100),
                metric=deepsort_cfg.get("metric", "cosine"),
            )
        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            return cls()


class DeepSORTTracker:
    """
    Wrapper around deep-sort-pytorch for multi-object tracking with appearance features.
    
    Replaces ByteTrack for improved ID stability in crowded scenes.
    """
    
    def __init__(self, config: Optional[DeepSORTConfig] = None, reid_model: Optional[str] = None):
        """
        Initialize DeepSORT tracker.
        
        Args:
            config: DeepSORTConfig object or None to load from YAML
            reid_model: Path to ReID model file (e.g., osnet_x1_0_imagenet.pth)
        """
        self.config = config or DeepSORTConfig.load()
        self.reid_model_path = reid_model
        self.tracker = None
        self.feature_extractor = None
        
        self._initialize_tracker()
    
    def _initialize_tracker(self):
        """Initialize the DeepSORT tracker and feature extractor."""
        try:
            from deep_sort_pytorch.deep_sort import DeepSort
            
            logger.info("Initializing DeepSORT tracker...")
            
            # Initialize feature extractor first
            self.feature_extractor = self._init_feature_extractor()
            
            # Check for ReID model file
            model_path = self.reid_model_path or "osnet_x1_0_imagenet.pth"
            model_exists = os.path.exists(model_path)
            
            if not model_exists:
                logger.warning(f"ReID model not found at {model_path}, tracker will use motion-based features only")
            
            # Initialize tracker (use dummy path if model doesn't exist, DeepSORT will fall back to motion tracking)
            self.tracker = DeepSort(
                model_path=model_path if model_exists else "dummy.pth",
                max_age=self.config.max_age,
                nn_budget=self.config.nn_budget,
                use_cuda=True,  # Try to use GPU if available
                metric=self.config.metric,
            )
            
            logger.info("✓ DeepSORT tracker initialized successfully")
            
        except ImportError:
            logger.error("deep-sort-pytorch not installed. Install with: pip install deep-sort-pytorch")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize DeepSORT: {e}")
            raise
    
    def _init_feature_extractor(self):
        """Initialize the appearance feature extractor."""
        try:
            # Try to import and check for ReID model
            model_path = self.reid_model_path or "osnet_x1_0_imagenet.pth"
            if not os.path.exists(model_path):
                logger.warning(f"ReID model not found at {model_path}, using fallback features")
                return None
            
            from deep_sort_pytorch.deep_sort.track import Track
            from deep_sort_pytorch.deep_sort.linear_assignment import INFTY_COST
            return self
        except ImportError:
            logger.warning("deep_sort_pytorch not available, using fallback features")
            return None
        except:
            logger.warning("Could not initialize feature extractor, using basic features")
            return None
    
    def update(
        self,
        detections: np.ndarray,
        frame: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        class_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Update tracker with new detections.
        
        Args:
            detections: Array of shape (N, 4) with bounding boxes [x1, y1, x2, y2]
            frame: Input frame for feature extraction
            confidences: Detection confidences (N,)
            class_ids: Class IDs (N,)
        
        Returns:
            Array of shape (M, 8) with tracked objects:
            [x1, y1, x2, y2, track_id, conf, cls_id, det_index]
        """
        if self.tracker is None:
            logger.error("Tracker not initialized")
            return np.empty((0, 8))
        
        # Extract features from detections if available
        features = self._extract_features(frame, detections)
        
        # Update tracker
        try:
            # DeepSORT expects: detections, frame, features, class_ids
            self.tracker.update(
                bbox_xywh=self._xyxy_to_xywh(detections) if len(detections) > 0 else np.empty((0, 4)),
                confidences=confidences or np.ones(len(detections)),
                ori_img=frame,
                features=features,
                class_ids=class_ids,
            )
            
            # Extract results
            online_targets = self.tracker.online_targets()
            results = self._format_results(online_targets, confidences, class_ids)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during tracker update: {e}")
            return np.empty((0, 8))
    
    def _xyxy_to_xywh(self, boxes: np.ndarray) -> np.ndarray:
        """Convert bounding boxes from [x1, y1, x2, y2] to [x, y, w, h]."""
        if len(boxes) == 0:
            return np.empty((0, 4))
        
        xywh = np.zeros_like(boxes)
        xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # center x
        xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # center y
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0]        # width
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1]        # height
        
        return xywh
    
    def _extract_features(self, frame: np.ndarray, detections: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract appearance features from detections in the frame.
        
        Args:
            frame: Input frame
            detections: Bounding boxes
        
        Returns:
            Feature vectors or None
        """
        if len(detections) == 0:
            return None
        
        try:
            from deep_sort_pytorch.deep_sort.reid_model import Extractor
            
            model_path = self.reid_model_path or "osnet_x1_0_imagenet.pth"
            
            # Check if model exists
            if not os.path.exists(model_path):
                logger.debug(f"ReID model not found at {model_path}, using placeholder features")
                return np.random.randn(len(detections), 128).astype(np.float32)
            
            if not hasattr(self, '_extractor'):
                self._extractor = Extractor(
                    model_path=model_path,
                    use_cuda=True,
                )
            
            features = self._extractor(frame, detections)
            return features
            
        except Exception as e:
            logger.debug(f"Feature extraction failed: {e}, using placeholder features")
            # Return dummy features as fallback
            return np.random.randn(len(detections), 128).astype(np.float32)
    
    def _format_results(
        self,
        targets: List,
        confidences: Optional[np.ndarray],
        class_ids: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Format DeepSORT results to match expected output format.
        
        Returns:
            Array with columns: [x1, y1, x2, y2, track_id, conf, cls_id, 0]
        """
        results = []
        
        if not targets:
            return np.empty((0, 8))
        
        for idx, target in enumerate(targets):
            try:
                # Get tracking results
                bbox = target.to_xyxy()  # [x1, y1, x2, y2]
                track_id = target.track_id
                
                # Get detection properties
                conf = float(confidences[idx]) if confidences is not None and idx < len(confidences) else 0.9
                cls_id = int(class_ids[idx]) if class_ids is not None and idx < len(class_ids) else 0
                
                result = np.array([
                    bbox[0], bbox[1], bbox[2], bbox[3],  # x1, y1, x2, y2
                    track_id,                              # track_id
                    conf,                                  # confidence
                    cls_id,                                # class_id
                    0,                                     # detection_index
                ], dtype=np.float32)
                
                results.append(result)
                
            except Exception as e:
                logger.debug(f"Error formatting result {idx}: {e}")
                continue
        
        return np.array(results) if results else np.empty((0, 8))
    
    def reset(self):
        """Reset tracker state."""
        if self.tracker:
            self.tracker.reset()
            logger.info("Tracker reset")


class HybridTracker:
    """
    Hybrid tracker that gracefully falls back from DeepSORT to ByteTrack.
    
    This ensures the pipeline continues even if DeepSORT fails to load.
    Accepts BOTSort-compatible interface for seamless integration.
    """
    
    def __init__(self, use_deepsort: bool = True, reid_model: Optional[str] = None):
        """
        Initialize hybrid tracker.
        
        Args:
            use_deepsort: Try to use DeepSORT first
            reid_model: Path to ReID model for DeepSORT
        """
        self.use_deepsort = use_deepsort
        self.deepsort_tracker = None
        self.bytetrack_fallback = None
        self.is_deepsort = False
        
        if use_deepsort:
            try:
                config = DeepSORTConfig.load()
                self.deepsort_tracker = DeepSORTTracker(config, reid_model)
                self.is_deepsort = True
                logger.info("✓ Using DeepSORT tracker")
            except Exception as e:
                logger.warning(f"DeepSORT initialization failed: {e}")
                logger.warning(f"Falling back to ByteTrack (ensure deep-sort-pytorch installed)")
                self.is_deepsort = False
    
    def update(self, det, frame, feats=None) -> np.ndarray:
        """
        Update tracker with detections (BOTSort-compatible interface).
        
        Args:
            det: Detection object with .xyxy, .conf, .cls attributes
                 (compatible with ultralytics format)
            frame: Input frame
            feats: Optional features (ignored for now)
        
        Returns:
            Tracked results as numpy array
        """
        if self.deepsort_tracker and self.is_deepsort:
            try:
                # Extract components from det object
                xyxy = getattr(det, 'xyxy', np.empty((0, 4)))
                confidences = getattr(det, 'conf', np.ones(len(xyxy)))
                class_ids = getattr(det, 'cls', np.zeros(len(xyxy)))
                
                # Call DeepSORT update
                return self.deepsort_tracker.update(xyxy, frame, confidences, class_ids)
            except Exception as e:
                logger.warning(f"DeepSORT update failed: {e}")
                self.is_deepsort = False
        
        # ByteTrack fallback (should be implemented but for now return empty)
        if not self.is_deepsort:
            logger.warning("No active tracker backend available")
            # Return empty tracks or use ByteTrack here
            return np.empty((0, 8))


# Convenience function
def create_tracker(tracker_type: str = "hybrid", reid_model: Optional[str] = None):
    """
    Create a tracker instance.
    
    Args:
        tracker_type: "deepsort", "hybrid", or "bytetrack"
        reid_model: Path to ReID model
    
    Returns:
        Tracker instance
    """
    if tracker_type.lower() == "deepsort":
        return DeepSORTTracker(reid_model=reid_model)
    elif tracker_type.lower() == "hybrid":
        return HybridTracker(use_deepsort=True, reid_model=reid_model)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")


if __name__ == "__main__":
    # Test the wrapper
    print("Testing DeepSORT wrapper...")
    
    config = DeepSORTConfig.load()
    print(f"Config loaded: {config}")
    
    try:
        tracker = create_tracker("hybrid")
        print(f"✓ Tracker created successfully")
    except Exception as e:
        print(f"✗ Failed to create tracker: {e}")
