"""Perception module for YOLO-based object detection.

This module provides the ObjectDetector class that wraps YOLOv11 for detecting
obstacles in video frames.
"""

import logging
from typing import List

import numpy as np
import torch
from ultralytics import YOLO  # type: ignore

from src.config import Config
from src.models import Detection


class ObjectDetector:
    """YOLO-based object detector for obstacle detection in video frames."""

    def __init__(self, config: Config):
        """Initialize detector with YOLO model and configuration.

        Args:
            config: System configuration containing model settings

        Raises:
            RuntimeError: If model fails to load
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Device selection with graceful fallback
        device = config.model.device  # type: ignore
        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning(
                "CUDA requested but not available, falling back to CPU"
            )
            device = "cpu"

        # Load YOLO model
        try:
            self.model = YOLO(config.model.name)  # type: ignore
            self.model.to(device)
            self.logger.info(f"Loaded {config.model.name} on {device}")  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

        # Pre-compute obstacle class set for O(1) filtering
        self.obstacle_classes = set(config.planning.obstacle_classes)  # type: ignore

    def detect_frame(self, frame: np.ndarray) -> List[Detection]:
        """Detect obstacles in a single frame.

        Args:
            frame: Input image as numpy array (BGR format from OpenCV)

        Returns:
            List of Detection objects for obstacles only (filtered by class)
            Returns empty list if detection fails or no obstacles found
        """
        try:
            # Run YOLO inference
            results = self.model.predict(
                frame,
                conf=self.config.model.confidence_threshold,  # type: ignore
                verbose=False  # Suppress YOLO's own logging
            )

            # Parse results to Detection objects
            detections = self._parse_yolo_results(results[0])

            # Filter to only obstacle classes
            obstacle_detections = self._filter_obstacle_classes(detections)

            self.logger.debug(
                f"Detected {len(detections)} objects, "
                f"{len(obstacle_detections)} obstacles"
            )

            return obstacle_detections

        except Exception as e:
            self.logger.warning(f"Frame detection failed: {e}")
            return []  # Graceful degradation

    def _parse_yolo_results(self, result) -> List[Detection]:  # type: ignore
        """Parse YOLO results into Detection objects.

        Args:
            result: YOLO result object from model.predict()

        Returns:
            List of Detection objects (all classes, unfiltered)
        """
        detections: List[Detection] = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy[i]

            # Validate bounding box
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                self.logger.debug(f"Skipping invalid box: ({x1},{y1},{x2},{y2})")
                continue

            class_id = cls[i]
            detections.append(Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                class_id=class_id,
                class_name=self.model.names[class_id],
                confidence=float(conf[i])
            ))

        return detections

    def _filter_obstacle_classes(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections to only configured obstacle classes.

        Args:
            detections: List of all detections

        Returns:
            List of detections with class names in obstacle_classes
        """
        return [d for d in detections if d.class_name in self.obstacle_classes]
