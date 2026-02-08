"""Coordinate transform module for spatial calculations.

This module provides the CoordinateTransform class that handles distance
calculations between detections and the agent, and coordinate space
transformations.
"""

import logging
import math
from typing import List, Optional, Tuple

from src.config import Config
from src.models import AgentState, Detection


class CoordinateTransform:
    """Spatial calculations and coordinate transformations for navigation."""

    def __init__(self, config: Config) -> None:
        """Initialize coordinate transform with configuration.

        Parameters
        ----------
        config : Config
            System configuration.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def calculate_distance(
        self, detection: Detection, agent: AgentState
    ) -> float:
        """Calculate Euclidean distance from detection center to agent position.

        Parameters
        ----------
        detection : Detection
            Detected obstacle with bounding box.
        agent : AgentState
            Current agent state with position.

        Returns
        -------
        float
            Euclidean distance in pixels.
        """
        cx, cy = detection.center
        dx = cx - agent.x
        dy = cy - agent.y
        return math.sqrt(dx**2 + dy**2)

    def find_nearest_obstacle(
        self, detections: List[Detection], agent: AgentState
    ) -> Optional[Tuple[float, Detection]]:
        """Find the nearest obstacle to the agent.

        Parameters
        ----------
        detections : List[Detection]
            List of detected obstacles.
        agent : AgentState
            Current agent state with position.

        Returns
        -------
        Optional[Tuple[float, Detection]]
            Tuple of (distance, detection) for the closest obstacle,
            or None if the list is empty.
        """
        if not detections:
            return None

        min_distance = float("inf")
        nearest: Optional[Detection] = None

        for detection in detections:
            distance = self.calculate_distance(detection, agent)
            if distance < min_distance:
                min_distance = distance
                nearest = detection

        return (min_distance, nearest)  # type: ignore[return-value]

    def image_to_world(self, x: float, y: float) -> Tuple[float, float]:
        """Convert image coordinates to world coordinates.

        Currently an identity transform (placeholder for future homography).

        Parameters
        ----------
        x : float
            X coordinate in image space.
        y : float
            Y coordinate in image space.

        Returns
        -------
        Tuple[float, float]
            World coordinates (x, y).
        """
        self.logger.debug("image_to_world: using identity transform (placeholder)")
        return (x, y)
