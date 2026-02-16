"""Data models for the autonomous navigation system.

This module defines the core data structures used throughout the navigation pipeline:
- Detection: Object detection results from YOLO (legacy)
- State: Navigation state machine states
- Action: Control actions for the agent (legacy)
- AgentState: Current state of the autonomous agent
- Obstacle: Circular obstacle in 2D world
- ControlCommand: Continuous control command (heading, speed, angular velocity)
- SensorReading: Single ray-cast sensor measurement
- ForceVector: 2D force with source label
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


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


# --- New dataclasses for APF simulator ---


@dataclass
class Obstacle:
    """Circular obstacle in the 2D world.

    Attributes
    ----------
    x : float
        Center x coordinate (world units).
    y : float
        Center y coordinate (world units).
    radius : float
        Obstacle radius (world units).
    vx : float
        Velocity in x direction (world units per second). 0 for static.
    vy : float
        Velocity in y direction (world units per second). 0 for static.
    label : str
        Human-readable label for the obstacle.
    """

    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0
    label: str = "obstacle"

    @property
    def is_dynamic(self) -> bool:
        """Whether this obstacle is moving."""
        return self.vx != 0.0 or self.vy != 0.0

    @property
    def position(self) -> Tuple[float, float]:
        """Get obstacle center as tuple."""
        return (self.x, self.y)


@dataclass
class ControlCommand:
    """Continuous control command for the agent.

    Attributes
    ----------
    desired_heading : float
        Target heading in radians.
    desired_speed : float
        Target speed in world units per second.
    angular_velocity : float
        Angular velocity in radians per second (set by PID controller).
    """

    desired_heading: float
    desired_speed: float
    angular_velocity: float = 0.0


@dataclass
class SensorReading:
    """Single ray-cast sensor measurement.

    Attributes
    ----------
    angle : float
        Ray angle in radians (world frame).
    distance : float
        Measured distance to nearest obstacle (or max_range if no hit).
    hit : bool
        Whether the ray hit an obstacle.
    hit_point : Optional[Tuple[float, float]]
        World coordinates of the hit point, or None if no hit.
    """

    angle: float
    distance: float
    hit: bool
    hit_point: Optional[Tuple[float, float]] = None


@dataclass
class ForceVector:
    """2D force vector with source label.

    Attributes
    ----------
    fx : float
        Force component in x direction.
    fy : float
        Force component in y direction.
    source : str
        Label describing the force source (e.g., "attractive", "repulsive").
    """

    fx: float
    fy: float
    source: str = ""

    @property
    def magnitude(self) -> float:
        """Magnitude of the force vector."""
        return math.sqrt(self.fx * self.fx + self.fy * self.fy)

    @property
    def heading(self) -> float:
        """Heading of the force vector in radians."""
        return math.atan2(self.fy, self.fx)
