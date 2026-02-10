"""Blended navigation planner combining waypoint-seeking and obstacle avoidance.

This module implements a navigation planner that smoothly blends headings between
waypoint-seeking direction and obstacle avoidance direction based on obstacle proximity.
"""

import logging
import math
from typing import List, Optional, Tuple

from src.config import Config
from src.models import AgentState, Detection, Action, State
from src.waypoint_navigator import WaypointNavigator


logger = logging.getLogger(__name__)


def blend_headings(h1: float, h2: float, weight: float) -> float:
    """Blend two headings using circular interpolation.

    Handles 0/360 degree wraparound by taking the shortest arc between headings.

    Args:
        h1: First heading in degrees [0, 360)
        h2: Second heading in degrees [0, 360)
        weight: Blend weight [0, 1] where 0 returns h1 and 1 returns h2

    Returns:
        Blended heading in degrees [0, 360)
    """
    # Calculate shortest angular difference
    diff = (h2 - h1 + 180.0) % 360.0 - 180.0

    # Interpolate along shortest arc
    result = (h1 + weight * diff) % 360.0

    return result


def heading_to_action(
    current_heading: float,
    desired_heading: float,
    speed: float,
    turn_rate: float,
    tolerance: float = 5.0,
) -> Action:
    """Convert heading error to navigation action.

    Args:
        current_heading: Current agent heading in degrees
        desired_heading: Desired target heading in degrees
        speed: Movement speed in pixels per frame
        turn_rate: Turn rate in degrees per frame
        tolerance: Heading alignment tolerance in degrees

    Returns:
        Navigation action (move_forward, turn_left, or turn_right)
    """
    # Calculate heading error with wraparound
    error = (desired_heading - current_heading + 180.0) % 360.0 - 180.0

    # If aligned within tolerance, move forward
    if abs(error) <= tolerance:
        return Action(type="move_forward", speed=speed)

    # Turn toward desired heading
    if error > 0:
        return Action(type="turn_left", speed=speed, angle=turn_rate)
    else:
        return Action(type="turn_right", speed=speed, angle=turn_rate)


class BlendedNavigationPlanner:
    """Navigation planner with blended waypoint-seeking and obstacle avoidance.

    Combines waypoint navigation with obstacle avoidance by smoothly blending
    headings based on obstacle proximity. Uses circular interpolation to handle
    heading wraparound correctly.

    Attributes:
        config: System configuration
        navigator: Waypoint navigator for goal-directed behavior
        obstacle_threshold: Distance threshold for obstacle detection
        critical_distance: Distance for emergency stop
        speed: Agent movement speed
        turn_rate: Agent turn rate
        heading_tolerance: Tolerance for heading alignment
    """

    def __init__(self, config: Config, navigator: WaypointNavigator):
        """Initialize blended navigation planner.

        Args:
            config: System configuration with planning and control parameters
            navigator: Waypoint navigator for goal-directed behavior
        """
        self.config = config
        self.navigator = navigator
        self.logger = logging.getLogger(__name__)

        # Extract planning parameters
        self.obstacle_threshold: float = (
            config.planning.obstacle_distance_threshold  # type: ignore[attr-defined]
        )
        self.critical_distance: float = (
            config.planning.critical_distance  # type: ignore[attr-defined]
        )

        # Extract control parameters
        self.speed: float = config.control.agent_speed  # type: ignore[attr-defined]
        self.turn_rate: float = config.control.turn_rate  # type: ignore[attr-defined]

        # Heading alignment tolerance
        self.heading_tolerance = 5.0

        self.logger.info(
            "BlendedNavigationPlanner initialized: "
            f"threshold={self.obstacle_threshold}, "
            f"critical={self.critical_distance}"
        )

    @property
    def is_goal_reached(self) -> bool:
        """Check if navigation goal has been reached.

        Returns:
            True if all waypoints have been reached
        """
        return self.navigator.is_complete

    def goal_status(self, agent: AgentState) -> str:
        """Get human-readable goal status string.

        Args:
            agent: Current agent state

        Returns:
            Status string describing navigation progress
        """
        return self.navigator.get_status_string(agent)

    def update(
        self, detections: List[Detection], agent: AgentState
    ) -> Tuple[State, Action]:
        """Update navigation state and compute action.

        Core decision logic:
        1. Goal reached -> STOP
        2. Obstacle < critical -> STOP
        3. Obstacle < threshold -> AVOID with blended heading
        4. No obstacle / far -> NAVIGATE with waypoint heading

        Args:
            detections: List of detected obstacles
            agent: Current agent state

        Returns:
            Tuple of (navigation_state, action)
        """
        # Let navigator check if current waypoint is reached
        # (this updates the waypoint index if reached)
        waypoint_action = self.navigator.get_turn_action(agent, allow_forward=True)

        # Check if goal is reached (after waypoint update)
        if self.is_goal_reached:
            self.logger.info("Goal reached - stopping")
            return State.STOP, Action(type="stop", speed=0.0)

        # Find closest obstacle
        closest_obstacle = self._find_closest_obstacle(detections, agent)
        closest_distance = (
            self._distance_to_obstacle(closest_obstacle, agent)
            if closest_obstacle
            else float("inf")
        )

        # Calculate desired waypoint heading
        waypoint_heading = self.navigator.calculate_desired_heading(agent)
        if waypoint_heading is None:
            # No waypoint available
            return State.STOP, Action(type="stop", speed=0.0)

        # Decision logic based on obstacle distance
        if closest_distance < self.critical_distance:
            # Critical obstacle - emergency stop
            self.logger.debug(f"Critical obstacle at {closest_distance:.1f}px - STOP")
            return State.STOP, Action(type="stop", speed=0.0)

        elif closest_distance < self.obstacle_threshold:
            # Obstacle within threshold - blend avoidance with waypoint heading
            avoidance_heading = self._compute_avoidance_heading(
                closest_obstacle, agent  # type: ignore[arg-type]
            )
            blend_weight = self._compute_blend_weight(closest_distance)

            # Blend headings: waypoint -> avoidance as weight increases
            desired_heading = blend_headings(
                waypoint_heading, avoidance_heading, blend_weight
            )

            self.logger.debug(
                f"AVOID: obstacle at {closest_distance:.1f}px, "
                f"waypoint={waypoint_heading:.0f}°, "
                f"avoidance={avoidance_heading:.0f}°, "
                f"blend={blend_weight:.2f}, "
                f"desired={desired_heading:.0f}°"
            )

            action = heading_to_action(
                agent.heading,
                desired_heading,
                self.speed,
                self.turn_rate,
                self.heading_tolerance,
            )

            return State.AVOID, action

        else:
            # No obstacle or far away - navigate toward waypoint
            self.logger.debug(f"NAVIGATE toward waypoint at {waypoint_heading:.0f}°")

            action = heading_to_action(
                agent.heading,
                waypoint_heading,
                self.speed,
                self.turn_rate,
                self.heading_tolerance,
            )

            return State.NAVIGATE, action

    def _find_closest_obstacle(
        self, detections: List[Detection], agent: AgentState
    ) -> Optional[Detection]:
        """Find the closest obstacle to the agent.

        Args:
            detections: List of detected obstacles
            agent: Current agent state

        Returns:
            Closest detection or None if no obstacles
        """
        if not detections:
            return None

        closest = min(
            detections, key=lambda det: self._distance_to_obstacle(det, agent)
        )
        return closest

    def _distance_to_obstacle(self, detection: Detection, agent: AgentState) -> float:
        """Calculate distance from agent to obstacle center.

        Args:
            detection: Obstacle detection
            agent: Current agent state

        Returns:
            Euclidean distance in pixels
        """
        obs_cx, obs_cy = detection.center
        dx = agent.x - obs_cx
        dy = agent.y - obs_cy
        return math.sqrt(dx * dx + dy * dy)

    def _compute_avoidance_heading(
        self, detection: Detection, agent: AgentState
    ) -> float:
        """Compute heading pointing away from obstacle center.

        Args:
            detection: Obstacle detection
            agent: Current agent state

        Returns:
            Avoidance heading in degrees [0, 360)
        """
        obs_cx, obs_cy = detection.center

        # Vector from obstacle to agent
        dx = agent.x - obs_cx
        dy = agent.y - obs_cy

        # Convert to heading (away from obstacle)
        # Note: -dy because y increases downward in image coordinates
        avoidance_heading = math.degrees(math.atan2(-dy, dx)) % 360.0

        return avoidance_heading

    def _compute_blend_weight(self, distance: float) -> float:
        """Compute blend weight based on obstacle distance.

        Linear ramp:
        - At threshold: weight = 0 (pure waypoint heading)
        - At critical: weight = 1 (pure avoidance heading)
        - Between: linear interpolation
        - Outside: clamped to [0, 1]

        Args:
            distance: Distance to closest obstacle

        Returns:
            Blend weight in [0, 1]
        """
        # Linear interpolation
        weight = (self.obstacle_threshold - distance) / (
            self.obstacle_threshold - self.critical_distance
        )

        # Clamp to [0, 1]
        return max(0.0, min(1.0, weight))
