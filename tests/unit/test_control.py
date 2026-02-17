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
    config.controller.max_lateral_accel = None  # no limit by default
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
    config.controller.max_lateral_accel = None  # no limit by default
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


class TestSpeedDependentTurnRate:
    """Tests for speed-dependent turn rate limiting (max_lateral_accel)."""

    def test_turn_rate_limited_at_high_speed(self, sim_config) -> None:
        """At high speed, omega should be clamped by max_lateral_accel / speed."""
        from src.control import AgentController

        max_lat = 50.0
        sim_config.controller.max_lateral_accel = max_lat
        pid = PIDController(kp=4.0, ki=0.0, kd=0.0, max_omega=6.0, integral_limit=2.0)
        controller = AgentController(sim_config, pid=pid)
        controller._heading_rad = 0.0  # facing east

        speed = 32.0  # cruise speed
        dt = 0.033
        # Large heading error to saturate PID
        cmd = ControlCommand(desired_heading=math.pi / 2, desired_speed=speed)
        controller.update(cmd, dt=dt)

        # Max heading change = (max_lat / speed) * dt
        max_delta = (max_lat / speed) * dt
        heading_change = abs(controller._heading_rad)
        assert heading_change <= max_delta + 1e-9

    def test_turn_rate_unlimited_at_low_speed(self, sim_config) -> None:
        """At very low speed, max_omega from PID is the binding constraint."""
        from src.control import AgentController

        max_lat = 50.0
        max_omega = 4.0
        sim_config.controller.max_lateral_accel = max_lat
        pid = PIDController(
            kp=4.0, ki=0.0, kd=0.0,
            max_omega=max_omega, integral_limit=2.0,
        )
        controller = AgentController(sim_config, pid=pid)
        controller._heading_rad = 0.0

        speed = 1.0  # very low speed → lat limit = 50 rad/s, PID cap = 4
        dt = 0.033
        cmd = ControlCommand(desired_heading=math.pi / 2, desired_speed=speed)
        controller.update(cmd, dt=dt)

        # Heading change should be limited by PID's max_omega, not lateral accel
        max_delta = max_omega * dt
        heading_change = abs(controller._heading_rad)
        assert heading_change <= max_delta + 1e-9

    def test_turn_rate_no_limit_when_none(self, sim_config) -> None:
        """When max_lateral_accel is None, omega is not additionally clipped."""
        from src.control import AgentController

        sim_config.controller.max_lateral_accel = None
        max_omega = 6.0
        pid = PIDController(
            kp=4.0, ki=0.0, kd=0.0,
            max_omega=max_omega, integral_limit=2.0,
        )
        controller = AgentController(sim_config, pid=pid)
        controller._heading_rad = 0.0

        speed = 32.0
        dt = 0.033
        cmd = ControlCommand(desired_heading=math.pi / 2, desired_speed=speed)
        controller.update(cmd, dt=dt)

        # Without lateral accel limit, heading change bounded only by PID max_omega
        max_delta = max_omega * dt
        heading_change = abs(controller._heading_rad)
        assert heading_change <= max_delta + 1e-9
        # Should be larger than what lateral accel would allow
        lat_limited_delta = (50.0 / speed) * dt  # hypothetical limit
        assert heading_change > lat_limited_delta

    def test_turning_radius_scales_with_speed(self, sim_config) -> None:
        """At 2x speed, heading change per step is roughly halved."""
        from src.control import AgentController

        max_lat = 50.0
        sim_config.controller.max_lateral_accel = max_lat
        dt = 0.033

        # Run at speed=20 (saturated PID → omega = 50/20 = 2.5)
        pid1 = PIDController(kp=40.0, ki=0.0, kd=0.0, max_omega=6.0, integral_limit=2.0)
        c1 = AgentController(sim_config, pid=pid1)
        c1._heading_rad = 0.0
        cmd1 = ControlCommand(desired_heading=math.pi / 2, desired_speed=20.0)
        c1.update(cmd1, dt=dt)
        delta1 = abs(c1._heading_rad)

        # Run at speed=40 (saturated PID → omega = 50/40 = 1.25)
        pid2 = PIDController(kp=40.0, ki=0.0, kd=0.0, max_omega=6.0, integral_limit=2.0)
        c2 = AgentController(sim_config, pid=pid2)
        c2._heading_rad = 0.0
        cmd2 = ControlCommand(desired_heading=math.pi / 2, desired_speed=40.0)
        c2.update(cmd2, dt=dt)
        delta2 = abs(c2._heading_rad)

        # delta2 should be ~half of delta1
        assert delta2 == pytest.approx(delta1 / 2, rel=0.05)
