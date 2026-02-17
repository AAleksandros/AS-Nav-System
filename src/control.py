"""Control module for agent movement and trajectory tracking.

This module provides the AgentController class that updates agent position
and heading using PID-based continuous heading control for the APF simulator pipeline.
"""

import logging
import math
from typing import List, Optional, Tuple

from src.config import Config
from src.models import AgentState, ControlCommand
from src.utils.math_utils import angle_difference, normalize_angle

# PIDController imported at type-check time; runtime import is optional
# to avoid circular imports.
try:
    from src.pid_controller import PIDController
except ImportError:  # pragma: no cover
    PIDController = None  # type: ignore[misc, assignment]


class AgentController:
    """Controller for autonomous agent movement and trajectory tracking.

    Uses a PID controller to track the desired heading from APF commands,
    then updates position based on desired speed and current heading.

    Parameters
    ----------
    config : Config
        System configuration with control parameters.
    pid : Optional[PIDController]
        PID controller for continuous heading control.
    """

    def __init__(
        self,
        config: Config,
        pid: Optional["PIDController"] = None,  # type: ignore[type-arg]
    ) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Extract control parameters from config
        self.trajectory_length: int = (
            config.control.trajectory_length  # type: ignore[attr-defined]
        )

        # PID controller for continuous update path
        self.pid = pid

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

        # Internal heading in radians for continuous update path
        self._heading_rad: float = math.radians(start_heading)

        self.logger.info(
            "AgentController initialized at (%.1f, %.1f), heading %.1f°",
            start_x,
            start_y,
            start_heading,
        )

    def update(self, command: ControlCommand, dt: float) -> None:
        """Update agent state using continuous PID heading control.

        Uses the PID controller to track the desired heading from the command,
        then updates position based on desired speed and current heading.
        Uses radians and y-up coordinate convention.

        Parameters
        ----------
        command : ControlCommand
            Desired heading and speed.
        dt : float
            Time step in seconds.
        """
        if self.pid is None:
            self.logger.warning("update() called without PID controller")
            return

        # Compute heading error and PID output
        error = angle_difference(command.desired_heading, self._heading_rad)
        omega = self.pid.compute(error, dt)

        # Update heading
        self._heading_rad = normalize_angle(self._heading_rad + omega * dt)

        # Update position (y-up convention)
        speed = command.desired_speed
        dx = speed * math.cos(self._heading_rad) * dt
        dy = speed * math.sin(self._heading_rad) * dt
        new_x = self.agent.x + dx
        new_y = self.agent.y + dy

        # Update agent state
        self.agent = AgentState(
            x=new_x,
            y=new_y,
            heading=self._heading_rad,
            velocity=speed,
            trajectory=self.agent.trajectory,
        )

        # Update trajectory
        self._update_trajectory()

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
