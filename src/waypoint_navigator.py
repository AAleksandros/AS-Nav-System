"""Waypoint-based navigation for goal-directed autonomous navigation.

Adds waypoint following capability to the navigation system, allowing agents
to navigate toward goals while avoiding obstacles.
"""

import logging
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np

from src.models import AgentState, Action


@dataclass
class Waypoint:
    """A waypoint defining a position the agent should reach.

    Attributes:
        x: X-coordinate in pixels
        y: Y-coordinate in pixels
        tolerance: Radius in pixels within which waypoint is considered reached
    """

    x: float
    y: float
    tolerance: float = 30.0

    def distance_to(self, agent_state: AgentState) -> float:
        """Calculate distance from agent to this waypoint.

        Args:
            agent_state: Current agent state

        Returns:
            Distance in pixels
        """
        dx = self.x - agent_state.x
        dy = self.y - agent_state.y
        return math.sqrt(dx * dx + dy * dy)

    def is_reached(self, agent_state: AgentState) -> bool:
        """Check if agent has reached this waypoint.

        Args:
            agent_state: Current agent state

        Returns:
            True if agent is within tolerance distance
        """
        return self.distance_to(agent_state) <= self.tolerance


class WaypointNavigator:
    """Waypoint-based navigation planner for goal-directed behavior."""

    def __init__(self, waypoints: List[Waypoint], turn_rate: float = 5.0, speed: float = 2.0):
        """Initialize waypoint navigator.

        Args:
            waypoints: List of waypoints to follow in order
            turn_rate: Maximum turn rate in degrees per frame
            speed: Movement speed in pixels per frame
        """
        self.waypoints = waypoints
        self.turn_rate = turn_rate
        self.speed = speed
        self.current_waypoint_idx = 0
        self.logger = logging.getLogger(__name__)

        if waypoints:
            self.logger.info(f"Initialized with {len(waypoints)} waypoints")
        else:
            self.logger.warning("Initialized with no waypoints")

    @property
    def current_waypoint(self) -> Optional[Waypoint]:
        """Get the current target waypoint.

        Returns:
            Current waypoint or None if all reached
        """
        if 0 <= self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return None

    @property
    def is_complete(self) -> bool:
        """Check if all waypoints have been reached.

        Returns:
            True if navigation is complete
        """
        return self.current_waypoint_idx >= len(self.waypoints)

    def calculate_desired_heading(self, agent_state: AgentState) -> Optional[float]:
        """Calculate desired heading toward current waypoint.

        Args:
            agent_state: Current agent state

        Returns:
            Desired heading in degrees, or None if no waypoint
        """
        waypoint = self.current_waypoint
        if waypoint is None:
            return None

        # Calculate angle to waypoint
        dx = waypoint.x - agent_state.x
        dy = waypoint.y - agent_state.y

        # atan2 returns angle in radians, convert to degrees
        # Note: atan2(dy, dx) gives angle from x-axis
        # Our coordinate system: 0°=right, 90°=up, 180°=left, 270°=down
        desired_heading = math.degrees(math.atan2(-dy, dx))  # Negative dy because y increases downward

        # Normalize to [0, 360)
        desired_heading = desired_heading % 360

        return desired_heading

    def get_turn_action(
        self,
        agent_state: AgentState,
        allow_forward: bool = True
    ) -> Action:
        """Get navigation action to reach current waypoint.

        Args:
            agent_state: Current agent state
            allow_forward: Whether forward movement is safe (no obstacles blocking)

        Returns:
            Navigation action (move_forward, turn_left, turn_right, or stop)
        """
        # Check if current waypoint is reached
        waypoint = self.current_waypoint
        if waypoint is None:
            return Action(type="stop", speed=0.0)  # All waypoints reached

        if waypoint.is_reached(agent_state):
            self.current_waypoint_idx += 1
            self.logger.info(
                f"Reached waypoint {self.current_waypoint_idx} at "
                f"({waypoint.x:.1f}, {waypoint.y:.1f})"
            )

            # Check next waypoint
            next_waypoint = self.current_waypoint
            if next_waypoint is None:
                self.logger.info("All waypoints reached!")
                return Action(type="stop", speed=0.0)

        # Calculate desired heading
        desired_heading = self.calculate_desired_heading(agent_state)
        if desired_heading is None:
            return Action(type="stop", speed=0.0)

        # Calculate heading error
        current_heading = agent_state.heading % 360
        heading_error = desired_heading - current_heading

        # Normalize error to [-180, 180]
        if heading_error > 180:
            heading_error -= 360
        elif heading_error < -180:
            heading_error += 360

        # Determine turn action
        if abs(heading_error) < self.turn_rate:
            # Heading is close enough, move forward if safe
            return Action(type="move_forward", speed=self.speed) if allow_forward else Action(type="stop", speed=0.0)
        elif heading_error > 0:
            # Need to turn left (counter-clockwise)
            return Action(type="turn_left", speed=self.speed, angle=self.turn_rate)
        else:
            # Need to turn right (clockwise)
            return Action(type="turn_right", speed=self.speed, angle=self.turn_rate)

    def get_progress_ratio(self) -> float:
        """Get navigation progress as ratio [0.0, 1.0].

        Returns:
            Progress ratio (0.0 = start, 1.0 = complete)
        """
        if not self.waypoints:
            return 1.0

        return self.current_waypoint_idx / len(self.waypoints)

    def get_status_string(self, agent_state: AgentState) -> str:
        """Get human-readable navigation status.

        Args:
            agent_state: Current agent state

        Returns:
            Status string for display
        """
        if self.is_complete:
            return "Navigation Complete"

        waypoint = self.current_waypoint
        if waypoint is None:
            return "No Waypoint"

        distance = waypoint.distance_to(agent_state)
        progress = self.get_progress_ratio()

        return (
            f"Waypoint {self.current_waypoint_idx + 1}/{len(self.waypoints)} | "
            f"Dist: {distance:.0f}px | Progress: {progress * 100:.0f}%"
        )


def create_waypoint_path(
    start: Tuple[float, float],
    end: Tuple[float, float],
    num_waypoints: int = 5,
    tolerance: float = 30.0
) -> List[Waypoint]:
    """Create a linear path of waypoints from start to end.

    Args:
        start: Starting (x, y) position
        end: Ending (x, y) position
        num_waypoints: Number of intermediate waypoints to create
        tolerance: Waypoint reach tolerance

    Returns:
        List of waypoints forming a path
    """
    waypoints = []

    for i in range(num_waypoints + 1):
        t = i / num_waypoints
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        waypoints.append(Waypoint(x, y, tolerance))

    return waypoints


def create_road_following_path(
    road_centerline: List[Tuple[float, float]],
    tolerance: float = 40.0
) -> List[Waypoint]:
    """Create waypoints following a road centerline.

    Args:
        road_centerline: List of (x, y) points defining road center
        tolerance: Waypoint reach tolerance

    Returns:
        List of waypoints following the road
    """
    return [Waypoint(x, y, tolerance) for x, y in road_centerline]
