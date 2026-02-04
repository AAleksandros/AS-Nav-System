"""Data models for the autonomous navigation system.

This module defines the core data structures used throughout the navigation pipeline:
- Detection: Object detection results from YOLO
- State: Navigation state machine states
- Action: Control actions for the agent
- AgentState: Current state of the autonomous agent
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class State(Enum):
    """Navigation state machine states.

    Attributes:
        NAVIGATE: Safe to move forward, no obstacles in range
        AVOID: Obstacle within threshold, performing avoidance maneuvers
        STOP: Critical obstacle detected, emergency stop engaged
    """
    NAVIGATE = "navigate"
    AVOID = "avoid"
    STOP = "stop"


@dataclass
class Detection:
    """Object detection result from YOLO.

    Represents a single detected object with bounding box coordinates,
    class information, and confidence score.

    Attributes:
        x1: Top-left x coordinate (pixels)
        y1: Top-left y coordinate (pixels)
        x2: Bottom-right x coordinate (pixels)
        y2: Bottom-right y coordinate (pixels)
        class_id: COCO class ID
        class_name: Human-readable class name
        confidence: Detection confidence score (0.0 to 1.0)
    """
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    class_name: str
    confidence: float

    @property
    def center(self) -> Tuple[float, float]:
        """Calculate the center point of the bounding box.

        Returns:
            Tuple of (center_x, center_y) in pixels
        """
        center_x = (self.x1 + self.x2) / 2.0
        center_y = (self.y1 + self.y2) / 2.0
        return (center_x, center_y)

    @property
    def area(self) -> float:
        """Calculate the area of the bounding box.

        Returns:
            Area in square pixels
        """
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return width * height


@dataclass
class Action:
    """Control action for the autonomous agent.

    Represents a discrete action that the agent should execute,
    including movement speed and turning angle.

    Attributes:
        type: Action type - "move_forward", "turn_left", "turn_right", or "stop"
        speed: Movement speed in pixels per frame
        angle: Turning angle in degrees (positive = left, negative = right)
    """
    type: str
    speed: float
    angle: float = 0.0


@dataclass
class AgentState:
    """Current state of the autonomous agent.

    Tracks the agent's position, orientation, velocity, and movement history.

    Attributes:
        x: X coordinate in image space (pixels)
        y: Y coordinate in image space (pixels)
        heading: Direction of travel in degrees (0° = right, 90° = up, 180° = left, 270° = down)
        velocity: Current speed in pixels per frame
        trajectory: Historical positions as list of (x, y) tuples (max 100 points)
    """
    x: float
    y: float
    heading: float = 90.0  # Default: facing up
    velocity: float = 0.0
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
