"""Planning module for autonomous navigation.

This module provides the NavigationPlanner class that implements a state machine
for navigation decisions. It takes obstacle detections and agent state, calculates
distances using CoordinateTransform, and outputs navigation state and action.
"""

import logging
from typing import List, Tuple

from src.config import Config
from src.coordinate_transform import CoordinateTransform
from src.models import Action, AgentState, Detection, State


class NavigationPlanner:
    """State machine planner for obstacle-aware navigation.

    Determines navigation state (NAVIGATE/AVOID/STOP) based on obstacle
    proximity and generates appropriate control actions.

    Parameters
    ----------
    config : Config
        System configuration with planning thresholds and control params.
    """

    def __init__(self, config: Config) -> None:
        """Initialize planner with configuration.

        Parameters
        ----------
        config : Config
            System configuration.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.transform = CoordinateTransform(config)
        self.obstacle_threshold: float = (
            config.planning.obstacle_distance_threshold  # type: ignore[attr-defined]
        )
        self.critical_distance: float = (
            config.planning.critical_distance  # type: ignore[attr-defined]
        )
        self.speed: float = config.control.agent_speed  # type: ignore[attr-defined]
        self.turn_rate: float = config.control.turn_rate  # type: ignore[attr-defined]
        self.current_state = State.NAVIGATE

    def update(
        self, detections: List[Detection], agent: AgentState
    ) -> Tuple[State, Action]:
        """Determine navigation state and action for the current frame.

        Parameters
        ----------
        detections : List[Detection]
            Detected obstacles in the current frame.
        agent : AgentState
            Current agent position and state.

        Returns
        -------
        Tuple[State, Action]
            The new navigation state and the action to execute.
        """
        result = self.transform.find_nearest_obstacle(detections, agent)

        if result is None:
            new_state = State.NAVIGATE
            action = Action(type="move_forward", speed=self.speed)
        else:
            distance, nearest = result
            if distance < self.critical_distance:
                new_state = State.STOP
                action = Action(type="stop", speed=0.0)
            elif distance < self.obstacle_threshold:
                new_state = State.AVOID
                direction = self._calculate_avoidance_direction(nearest, agent)
                action = Action(
                    type=direction, speed=self.speed, angle=self.turn_rate,
                )
            else:
                new_state = State.NAVIGATE
                action = Action(type="move_forward", speed=self.speed)

        if new_state != self.current_state:
            self.logger.info(
                "State transition: %s -> %s",
                self.current_state.value,
                new_state.value,
            )
        self.current_state = new_state
        return (self.current_state, action)

    def _calculate_avoidance_direction(
        self, detection: Detection, agent: AgentState
    ) -> str:
        """Calculate which direction to turn to avoid an obstacle.

        Turns away from the obstacle: if it's to the left, turn right,
        and vice versa.

        Parameters
        ----------
        detection : Detection
            The obstacle to avoid.
        agent : AgentState
            Current agent state.

        Returns
        -------
        str
            Either "turn_left" or "turn_right".
        """
        if detection.center[0] < agent.x:
            return "turn_right"
        return "turn_left"
