"""Waypoint-based navigation for goal-directed autonomous navigation.

Provides waypoint sequencing for the APF simulator. The agent follows
waypoints in order, advancing to the next when within tolerance distance.
"""

import logging
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.models import AgentState


@dataclass
class Waypoint:
    """A waypoint defining a position the agent should reach.

    Attributes:
        x: X-coordinate in world units
        y: Y-coordinate in world units
        tolerance: Radius within which waypoint is considered reached
    """

    x: float
    y: float
    tolerance: float = 30.0

    def distance_to(self, agent_state: AgentState) -> float:
        """Calculate distance from agent to this waypoint.

        Args:
            agent_state: Current agent state

        Returns:
            Distance in world units
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

    def __init__(self, waypoints: List[Waypoint]) -> None:
        """Initialize waypoint navigator.

        Args:
            waypoints: List of waypoints to follow in order
        """
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        self.logger = logging.getLogger(__name__)
        self._prev_distance: Optional[float] = None
        self._min_approach_distance: float = float("inf")
        self._receding_count: int = 0

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
    def current_goal(self) -> Optional[Tuple[float, float]]:
        """Get the current goal position as (x, y) tuple.

        Returns:
            Current goal coordinates or None if all waypoints reached
        """
        wp = self.current_waypoint
        if wp is None:
            return None
        return (wp.x, wp.y)

    @property
    def is_complete(self) -> bool:
        """Check if all waypoints have been reached.

        Returns:
            True if navigation is complete
        """
        return self.current_waypoint_idx >= len(self.waypoints)

    @property
    def progress(self) -> float:
        """Get navigation progress as ratio [0.0, 1.0].

        Returns:
            Progress ratio (0.0 = start, 1.0 = complete)
        """
        if not self.waypoints:
            return 1.0
        return self.current_waypoint_idx / len(self.waypoints)

    def check_and_advance(self, x: float, y: float) -> bool:
        """Check if current waypoint is reached and advance if so.

        Uses two advancement criteria:
        1. Radius-based: agent is within waypoint tolerance (original).
        2. Overshoot: agent approached within 2x tolerance then receded
           for 2+ consecutive steps (prevents U-turns from near-misses).

        Args:
            x: Agent x position
            y: Agent y position

        Returns:
            True if a waypoint was just reached and advanced
        """
        wp = self.current_waypoint
        if wp is None:
            return False

        # Compute distance to current waypoint
        dx = wp.x - x
        dy = wp.y - y
        dist = math.sqrt(dx * dx + dy * dy)

        # Primary check: radius-based advancement (unchanged)
        if dist <= wp.tolerance:
            self.logger.info(
                "Reached waypoint %d/%d at (%.1f, %.1f)",
                self.current_waypoint_idx + 1,
                len(self.waypoints), wp.x, wp.y,
            )
            self.current_waypoint_idx += 1
            self._reset_tracking()
            return True

        # Secondary check: closest-approach overshoot detection
        self._min_approach_distance = min(self._min_approach_distance, dist)
        overshoot_tolerance = wp.tolerance * 2.0

        if self._prev_distance is not None:
            if dist > self._prev_distance:
                self._receding_count += 1
            else:
                self._receding_count = 0

            if (
                self._receding_count >= 2
                and self._min_approach_distance <= overshoot_tolerance
            ):
                self.logger.info(
                    "Overshoot-advanced waypoint %d/%d at (%.1f, %.1f)"
                    " (closest approach: %.1f)",
                    self.current_waypoint_idx + 1,
                    len(self.waypoints), wp.x, wp.y,
                    self._min_approach_distance,
                )
                self.current_waypoint_idx += 1
                self._reset_tracking()
                return True

        self._prev_distance = dist
        return False

    def _reset_tracking(self) -> None:
        """Reset overshoot tracking state for a new waypoint."""
        self._prev_distance = None
        self._min_approach_distance = float("inf")
        self._receding_count = 0
