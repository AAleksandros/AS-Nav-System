"""Control module for agent movement and trajectory tracking.

This module provides the AgentController class that executes navigation actions,
updates agent position/heading using movement physics, and maintains trajectory history.
"""

import logging
import math
from typing import List, Optional, Tuple

from src.config import Config
from src.models import Action, AgentState


class AgentController:
    """Controller for autonomous agent movement and trajectory tracking.

    Executes control actions (move_forward, turn_left, turn_right, stop) and
    maintains agent state including position, heading, velocity, and trajectory.

    Parameters
    ----------
    config : Config
        System configuration with control parameters.
    """

    def __init__(
        self,
        config: Config,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> None:
        """Initialize controller with configuration.

        Parameters
        ----------
        config : Config
            System configuration.
        frame_width : Optional[int]
            Optional frame width for boundary clamping. If None, no clamping.
        frame_height : Optional[int]
            Optional frame height for boundary clamping. If None, no clamping.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Extract control parameters from config
        self.speed: float = config.control.agent_speed  # type: ignore[attr-defined]
        self.turn_rate: float = config.control.turn_rate  # type: ignore[attr-defined]
        self.trajectory_length: int = (
            config.control.trajectory_length  # type: ignore[attr-defined]
        )

        # Store frame boundaries for clamping
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Initialize agent state
        start_x: float = config.control.start_x  # type: ignore[attr-defined]
        start_y: float = config.control.start_y  # type: ignore[attr-defined]
        start_heading: float = (
            config.control.start_heading  # type: ignore[attr-defined]
        )

        self.agent = AgentState(
            x=start_x,
            y=start_y,
            heading=start_heading,
            velocity=0.0,
            trajectory=[(start_x, start_y)],
        )

        self.logger.info(
            "AgentController initialized at (%.1f, %.1f), heading %.1f°%s",
            start_x,
            start_y,
            start_heading,
            f" with bounds {frame_width}x{frame_height}" if frame_width else "",
        )

    def execute_action(self, action: Action) -> None:
        """Execute a control action and update agent state.

        Updates agent position, heading, velocity, and trajectory based on
        the action type and parameters.

        Parameters
        ----------
        action : Action
            The control action to execute.
        """
        if action.type == "move_forward":
            self._move_forward(action.speed)
        elif action.type == "turn_left":
            self._turn_left(action.angle, action.speed)
        elif action.type == "turn_right":
            self._turn_right(action.angle, action.speed)
        elif action.type == "stop":
            self._stop()
        else:
            self.logger.warning("Unknown action type: %s", action.type)

    def _move_forward(self, speed: float) -> None:
        """Move agent forward based on current heading.

        Parameters
        ----------
        speed : float
            Movement speed in pixels per frame.
        """
        # Convert heading to radians for trig functions
        heading_rad = math.radians(self.agent.heading)

        # Calculate displacement using movement equations
        # 0° = right, 90° = up, 180° = left, 270° = down
        dx = speed * math.cos(heading_rad)
        dy = -speed * math.sin(heading_rad)  # Negative: y increases downward

        # Calculate new position
        new_x = self.agent.x + dx
        new_y = self.agent.y + dy

        # Apply boundary clamping if bounds are set
        if self.frame_width is not None:
            new_x = max(0.0, min(new_x, float(self.frame_width)))
        if self.frame_height is not None:
            new_y = max(0.0, min(new_y, float(self.frame_height)))

        # Update position
        self.agent = AgentState(
            x=new_x,
            y=new_y,
            heading=self.agent.heading,
            velocity=speed,
            trajectory=self.agent.trajectory,
        )

        # Append new position to trajectory
        self._update_trajectory()

    def _turn_left(self, angle: float, speed: float) -> None:
        """Turn agent left (increase heading) and move forward.

        Parameters
        ----------
        angle : float
            Turn angle in degrees.
        speed : float
            Movement speed in pixels per frame.
        """
        # Update heading (increase for left turn)
        new_heading = (self.agent.heading + angle) % 360.0

        self.agent = AgentState(
            x=self.agent.x,
            y=self.agent.y,
            heading=new_heading,
            velocity=self.agent.velocity,
            trajectory=self.agent.trajectory,
        )

        # Move forward with new heading
        self._move_forward(speed)

    def _turn_right(self, angle: float, speed: float) -> None:
        """Turn agent right (decrease heading) and move forward.

        Parameters
        ----------
        angle : float
            Turn angle in degrees.
        speed : float
            Movement speed in pixels per frame.
        """
        # Update heading (decrease for right turn)
        new_heading = (self.agent.heading - angle) % 360.0

        self.agent = AgentState(
            x=self.agent.x,
            y=self.agent.y,
            heading=new_heading,
            velocity=self.agent.velocity,
            trajectory=self.agent.trajectory,
        )

        # Move forward with new heading
        self._move_forward(speed)

    def _stop(self) -> None:
        """Stop agent movement (set velocity to 0)."""
        self.agent = AgentState(
            x=self.agent.x,
            y=self.agent.y,
            heading=self.agent.heading,
            velocity=0.0,
            trajectory=self.agent.trajectory,
        )

    def _update_trajectory(self) -> None:
        """Append current position to trajectory and limit to max length."""
        current_pos = (self.agent.x, self.agent.y)
        new_trajectory: List[Tuple[float, float]] = list(self.agent.trajectory)
        new_trajectory.append(current_pos)

        # Limit trajectory to max length (keep most recent positions)
        if len(new_trajectory) > self.trajectory_length:
            new_trajectory = new_trajectory[-self.trajectory_length :]

        self.agent = AgentState(
            x=self.agent.x,
            y=self.agent.y,
            heading=self.agent.heading,
            velocity=self.agent.velocity,
            trajectory=new_trajectory,
        )

    def get_agent_state(self) -> AgentState:
        """Get current agent state.

        Returns
        -------
        AgentState
            Current agent position, heading, velocity, and trajectory.
        """
        return self.agent
