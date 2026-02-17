"""Unit tests for the control module."""

import logging
import math
from unittest.mock import Mock

import pytest

from src.models import AgentState, ControlCommand
from src.pid_controller import PIDController


@pytest.fixture
def sample_config():
    """Create a mock config with control parameters."""
    config = Mock()
    config.control.agent_speed = 2.0
    config.control.turn_rate = 15.0  # degrees
    config.control.start_x = 320.0
    config.control.start_y = 400.0
    config.control.start_heading = 90.0  # degrees (pointing up)
    config.control.trajectory_length = 100
    return config


class TestAgentControllerInit:
    """Tests for AgentController initialization."""

    def test_initialization(self, sample_config):
        """Verify config stored and initial agent state created."""
        from src.control import AgentController

        controller = AgentController(sample_config)

        assert controller.trajectory_length == 100
        assert controller.agent.x == 320.0
        assert controller.agent.y == 400.0
        assert controller.agent.heading == 90.0
        assert controller.agent.velocity == 0.0
        assert controller.agent.trajectory == [(320.0, 400.0)]


class TestGetAgentState:
    """Tests for get_agent_state method."""

    def test_get_agent_state_returns_copy(self, sample_config):
        """get_agent_state returns current agent state."""
        from src.control import AgentController

        controller = AgentController(sample_config)

        state = controller.get_agent_state()

        assert isinstance(state, AgentState)
        assert state.x == 320.0
        assert state.y == 400.0
        assert state.heading == 90.0
        assert state.velocity == 0.0
        assert state.trajectory == [(320.0, 400.0)]


# --- Fixtures for continuous update (APF simulator path) ---


@pytest.fixture
def sim_config():
    """Config for APF simulator mode controller."""
    config = Mock()
    config.control.agent_speed = 2.0
    config.control.turn_rate = 15.0
    config.control.start_x = 100.0
    config.control.start_y = 100.0
    config.control.start_heading = 0.0  # degrees (legacy init)
    config.control.trajectory_length = 500
    return config


@pytest.fixture
def sim_pid() -> PIDController:
    """PID controller for simulator tests."""
    return PIDController(
        kp=4.0, ki=0.1, kd=0.5, max_omega=4.0, integral_limit=2.0,
    )


# --- Tests: update() method ---


class TestUpdateMethod:
    """Tests for the continuous update(command, dt) method."""

    def test_update_moves_agent_forward(self, sim_config, sim_pid) -> None:
        """Agent should move forward in heading direction."""
        from src.control import AgentController

        controller = AgentController(sim_config, pid=sim_pid)
        # Set heading to 0 radians (east) via init
        controller._heading_rad = 0.0
        x_before = controller.agent.x

        cmd = ControlCommand(desired_heading=0.0, desired_speed=60.0)
        controller.update(cmd, dt=0.1)

        # Moving east: x should increase
        assert controller.agent.x > x_before
        assert controller.agent.y == pytest.approx(100.0, abs=0.5)

    def test_update_moves_north(self, sim_config, sim_pid) -> None:
        """Heading pi/2 (north) should increase y in y-up convention."""
        from src.control import AgentController

        controller = AgentController(sim_config, pid=sim_pid)
        controller._heading_rad = math.pi / 2
        y_before = controller.agent.y

        cmd = ControlCommand(
            desired_heading=math.pi / 2, desired_speed=60.0,
        )
        controller.update(cmd, dt=0.1)

        # Moving north (y-up): y should increase
        assert controller.agent.y > y_before

    def test_update_tracks_heading(self, sim_config, sim_pid) -> None:
        """PID should steer heading toward desired heading."""
        from src.control import AgentController

        controller = AgentController(sim_config, pid=sim_pid)
        controller._heading_rad = 0.0  # facing east

        # Desire heading north (pi/2)
        cmd = ControlCommand(desired_heading=math.pi / 2, desired_speed=30.0)

        # Run several steps
        for _ in range(50):
            controller.update(cmd, dt=0.033)

        # Heading should converge toward pi/2
        assert controller._heading_rad == pytest.approx(
            math.pi / 2, abs=0.1
        )

    def test_update_records_trajectory(self, sim_config, sim_pid) -> None:
        """Each update should append to trajectory."""
        from src.control import AgentController

        controller = AgentController(sim_config, pid=sim_pid)
        initial_len = len(controller.agent.trajectory)

        cmd = ControlCommand(desired_heading=0.0, desired_speed=30.0)
        for _ in range(5):
            controller.update(cmd, dt=0.033)

        assert len(controller.agent.trajectory) == initial_len + 5

    def test_update_trajectory_capped(self, sim_config, sim_pid) -> None:
        """Trajectory should not exceed trajectory_length."""
        from src.control import AgentController

        sim_config.control.trajectory_length = 10
        controller = AgentController(sim_config, pid=sim_pid)

        cmd = ControlCommand(desired_heading=0.0, desired_speed=30.0)
        for _ in range(20):
            controller.update(cmd, dt=0.033)

        assert len(controller.agent.trajectory) <= 10

    def test_update_velocity_set(self, sim_config, sim_pid) -> None:
        """Velocity on agent state should reflect desired speed."""
        from src.control import AgentController

        controller = AgentController(sim_config, pid=sim_pid)

        cmd = ControlCommand(desired_heading=0.0, desired_speed=45.0)
        controller.update(cmd, dt=0.033)

        assert controller.agent.velocity == pytest.approx(45.0)

    def test_update_zero_speed(self, sim_config, sim_pid) -> None:
        """Zero speed should not move agent."""
        from src.control import AgentController

        controller = AgentController(sim_config, pid=sim_pid)
        x_before = controller.agent.x
        y_before = controller.agent.y

        cmd = ControlCommand(desired_heading=0.0, desired_speed=0.0)
        controller.update(cmd, dt=0.1)

        assert controller.agent.x == pytest.approx(x_before)
        assert controller.agent.y == pytest.approx(y_before)

    def test_update_heading_wraps(self, sim_config, sim_pid) -> None:
        """Heading should stay normalized after update."""
        from src.control import AgentController

        controller = AgentController(sim_config, pid=sim_pid)
        controller._heading_rad = math.pi - 0.1  # near pi

        # Desire heading just past -pi (equivalent to near pi from other side)
        cmd = ControlCommand(
            desired_heading=-math.pi + 0.1, desired_speed=30.0,
        )
        controller.update(cmd, dt=0.033)

        # Heading should remain in [-pi, pi]
        assert -math.pi <= controller._heading_rad <= math.pi

    def test_update_pid_reset_on_new_pid(self, sim_config) -> None:
        """Setting a new PID resets controller state."""
        from src.control import AgentController

        pid1 = PIDController(kp=4.0, ki=0.1, kd=0.5, max_omega=4.0, integral_limit=2.0)
        controller = AgentController(sim_config, pid=pid1)

        cmd = ControlCommand(desired_heading=1.0, desired_speed=30.0)
        controller.update(cmd, dt=0.1)

        # PID should have accumulated some state
        assert pid1._integral != 0.0 or pid1._prev_error != 0.0

    def test_update_without_pid_logs_warning(
        self, sim_config, caplog
    ) -> None:
        """update() without PID should log warning and not move."""
        from src.control import AgentController

        controller = AgentController(sim_config)  # no pid
        x_before = controller.agent.x
        y_before = controller.agent.y

        cmd = ControlCommand(desired_heading=0.0, desired_speed=30.0)
        with caplog.at_level(logging.WARNING):
            controller.update(cmd, dt=0.033)

        assert "update() called without PID controller" in caplog.text
        assert controller.agent.x == x_before
        assert controller.agent.y == y_before
