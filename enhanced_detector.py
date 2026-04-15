"""Enhanced Detection Module with Configuration

Implements improved detection quality with:
- Configurable confidence thresholds
- Strict NMS tuning
- Area-based filtering
- YOLOv9 support
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class DetectionConfig:
    """Load and manage detection configuration."""
    
    def __init__(self, config_path: str = "detection_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load YAML configuration."""
        if not Path(self.config_path).exists():
            return self._default_config()
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def _default_config() -> dict:
        """Return default configuration if file not found."""
        return {
            'detection': {
                'model': 'yolov8n.pt',
                'confidence_threshold': 0.60,
                'iou_threshold': 0.45,
                'bbox_area': {'min_px': 2500, 'max_px': 921600},
                'classes_to_detect': [0],
            }
        }
    
    def get(self, key: str, default=None):
        """Get config value by dot notation (e.g., 'detection.confidence_threshold')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value else default


class EnhancedDetector:
    """Detection with quality improvements."""
    
    def __init__(self, config: Optional[DetectionConfig] = None, model_path: str = "yolov8n.pt"):
        """
        Args:
            config: DetectionConfig object
            model_path: Path to YOLO model (can be overridden by config)
        """
        self.config = config or DetectionConfig()
        
        # Try to get model from config first, fall back to parameter
        config_model = self.config.get('detection.model')
        if config_model:
            model_path = config_model
        
        self.model = YOLO(model_path)
        self.model_path = model_path
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect objects with enhanced filtering.
        
        Returns:
            (boxes_xyxy, confidences, class_ids)
            boxes_xyxy: Nx4 array of [x1, y1, x2, y2]
            confidences: N array of confidence scores
            class_ids: N array of class IDs
        """
        conf_threshold = self.config.get('detection.confidence_threshold', 0.60)
        iou_threshold = self.config.get('detection.iou_threshold', 0.45)
        
        # Run inference
        results = self.model(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        if not results or len(results) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        
        # Extract detections
        result = results[0]
        boxes = result.boxes.xyxy.numpy()  # [x1, y1, x2, y2]
        confidences = result.boxes.conf.numpy()
        class_ids = result.boxes.cls.numpy().astype(int)
        
        # Apply filtering
        boxes, confidences, class_ids = self._apply_filters(
            frame, boxes, confidences, class_ids
        )
        
        return boxes, confidences, class_ids
    
    def _apply_filters(self, frame: np.ndarray, 
                      boxes: np.ndarray, 
                      confidences: np.ndarray, 
                      class_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply post-detection filters."""
        
        if len(boxes) == 0:
            return boxes, confidences, class_ids
        
        # 1. Area Filtering
        min_area = self.config.get('detection.bbox_area.min_px', 2500)
        max_area = self.config.get('detection.bbox_area.max_px', 921600)
        
        boxes, confidences, class_ids = self._filter_by_area(
            boxes, confidences, class_ids, min_area, max_area
        )
        
        # 2. Class Filtering
        allowed_classes = self.config.get('detection.classes_to_detect', [0])
        mask = np.isin(class_ids, allowed_classes)
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        # 3. Remove Duplicates (very high IoU overlaps)
        if len(boxes) > 0:
            boxes, confidences, class_ids = self._remove_duplicates(
                boxes, confidences, class_ids, iou_threshold=0.85
            )
        
        return boxes, confidences, class_ids
    
    @staticmethod
    def _filter_by_area(boxes: np.ndarray, confidences: np.ndarray, 
                       class_ids: np.ndarray,
                       min_area: float, max_area: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter detections by bounding box area."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        mask = (areas >= min_area) & (areas <= max_area)
        
        return boxes[mask], confidences[mask], class_ids[mask]
    
    @staticmethod
    def _remove_duplicates(boxes: np.ndarray, confidences: np.ndarray, 
                          class_ids: np.ndarray,
                          iou_threshold: float = 0.85) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove duplicate detections (very high IoU)."""
        if len(boxes) <= 1:
            return boxes, confidences, class_ids
        
        # Sort by confidence (descending)
        sort_idx = np.argsort(-confidences)
        boxes = boxes[sort_idx]
        confidences = confidences[sort_idx]
        class_ids = class_ids[sort_idx]
        
        keep_idx = []
        suppressed = set()
        
        for i in range(len(boxes)):
            if i in suppressed:
                continue
            
            keep_idx.append(i)
            
            # Compare with remaining boxes
            for j in range(i + 1, len(boxes)):
                if j in suppressed:
                    continue
                
                iou = EnhancedDetector._compute_iou(boxes[i], boxes[j])
                
                if iou > iou_threshold:
                    suppressed.add(j)
        
        keep_idx = np.array(keep_idx)
        return boxes[keep_idx], confidences[keep_idx], class_ids[keep_idx]
    
    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0


class YOLOv9Support:
    """YOLOv9 model support and comparison."""
    
    MODELS = {
        'yolov8n': {'size': '16MB', 'speed': 'Fastest', 'accuracy': 'Lower'},
        'yolov8s': {'size': '44MB', 'speed': 'Fast', 'accuracy': 'Medium'},
        'yolov8m': {'size': '95MB', 'speed': 'Medium', 'accuracy': 'Good'},
        'yolov9c': {'size': '25MB', 'speed': 'Fast', 'accuracy': 'Higher'},
        'yolov9m': {'size': '50MB', 'speed': 'Medium', 'accuracy': 'Highest'},
    }
    
    @staticmethod
    def get_model_info(model_name: str) -> dict:
        """Get model information."""
        return YOLOv9Support.MODELS.get(model_name, {})
    
    @staticmethod
    def compare_models() -> str:
        """Return comparison table."""
        lines = ["Model Comparison:\n"]
        lines.append("Model      | Size  | Speed      | Accuracy")
        lines.append("-" * 50)
        for model, info in YOLOv9Support.MODELS.items():
            lines.append(f"{model:10} | {info['size']:5} | {info['speed']:10} | {info['accuracy']}")
        return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced detection")
    parser.add_argument("--image", required=True, help="Image path")
    parser.add_argument("--config", default="detection_config.yaml", help="Config file")
    parser.add_argument("--model", default="yolov8n.pt", help="Model path")
    parser.add_argument("--compare", action="store_true", help="Show model comparison")
    
    args = parser.parse_args()
    
    if args.compare:
        print(YOLOv9Support.compare_models())
    else:
        config = DetectionConfig(args.config)
        detector = EnhancedDetector(config, args.model)
        
        frame = cv2.imread(args.image)
        boxes, confidences, class_ids = detector.detect(frame)
        
        print(f"Detections: {len(boxes)}")
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            print(f"  Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}], Conf: {conf:.2f}, Class: {cls_id}")
