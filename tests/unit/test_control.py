"""Unit tests for the control module."""

import logging
from unittest.mock import Mock

import pytest

from src.models import Action, AgentState


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

        assert controller.speed == 2.0
        assert controller.turn_rate == 15.0
        assert controller.trajectory_length == 100
        assert controller.agent.x == 320.0
        assert controller.agent.y == 400.0
        assert controller.agent.heading == 90.0
        assert controller.agent.velocity == 0.0
        assert controller.agent.trajectory == [(320.0, 400.0)]


class TestMoveForward:
    """Tests for move_forward action."""

    def test_move_forward_heading_0(self, sample_config):
        """Heading 0° (right): x increases, y unchanged."""
        from src.control import AgentController

        sample_config.control.start_heading = 0.0
        controller = AgentController(sample_config)

        action = Action(type="move_forward", speed=2.0)
        controller.execute_action(action)

        # dx = 2.0 * cos(0) = 2.0, dy = -2.0 * sin(0) = 0
        assert controller.agent.x == pytest.approx(322.0)
        assert controller.agent.y == pytest.approx(400.0)
        assert controller.agent.velocity == 2.0
        assert len(controller.agent.trajectory) == 2

    def test_move_forward_heading_90(self, sample_config):
        """Heading 90° (up): y decreases (negative dy), x unchanged."""
        from src.control import AgentController

        sample_config.control.start_heading = 90.0
        controller = AgentController(sample_config)

        action = Action(type="move_forward", speed=2.0)
        controller.execute_action(action)

        # dx = 2.0 * cos(90°) = 0, dy = -2.0 * sin(90°) = -2.0
        assert controller.agent.x == pytest.approx(320.0)
        assert controller.agent.y == pytest.approx(398.0)
        assert controller.agent.velocity == 2.0

    def test_move_forward_heading_180(self, sample_config):
        """Heading 180° (left): x decreases, y unchanged."""
        from src.control import AgentController

        sample_config.control.start_heading = 180.0
        controller = AgentController(sample_config)

        action = Action(type="move_forward", speed=2.0)
        controller.execute_action(action)

        # dx = 2.0 * cos(180°) = -2.0, dy = -2.0 * sin(180°) = 0
        assert controller.agent.x == pytest.approx(318.0)
        assert controller.agent.y == pytest.approx(400.0)

    def test_move_forward_heading_270(self, sample_config):
        """Heading 270° (down): y increases (positive dy), x unchanged."""
        from src.control import AgentController

        sample_config.control.start_heading = 270.0
        controller = AgentController(sample_config)

        action = Action(type="move_forward", speed=2.0)
        controller.execute_action(action)

        # dx = 2.0 * cos(270°) = 0, dy = -2.0 * sin(270°) = 2.0
        assert controller.agent.x == pytest.approx(320.0)
        assert controller.agent.y == pytest.approx(402.0)

    def test_move_forward_heading_45(self, sample_config):
        """Heading 45° (diagonal): both x and y change."""
        from src.control import AgentController

        sample_config.control.start_heading = 45.0
        controller = AgentController(sample_config)

        action = Action(type="move_forward", speed=2.0)
        controller.execute_action(action)

        # dx = 2.0 * cos(45°) ≈ 1.414, dy = -2.0 * sin(45°) ≈ -1.414
        assert controller.agent.x == pytest.approx(321.414, abs=0.01)
        assert controller.agent.y == pytest.approx(398.586, abs=0.01)


class TestTurnActions:
    """Tests for turn_left and turn_right actions."""

    def test_turn_left(self, sample_config):
        """Turn left increases heading by turn_rate."""
        from src.control import AgentController

        sample_config.control.start_heading = 90.0
        controller = AgentController(sample_config)

        action = Action(type="turn_left", speed=2.0, angle=15.0)
        controller.execute_action(action)

        # 90 + 15 = 105
        assert controller.agent.heading == pytest.approx(105.0)
        assert controller.agent.velocity == 2.0
        # Position should update based on NEW heading
        assert controller.agent.x != 320.0 or controller.agent.y != 400.0

    def test_turn_right(self, sample_config):
        """Turn right decreases heading by turn_rate."""
        from src.control import AgentController

        sample_config.control.start_heading = 90.0
        controller = AgentController(sample_config)

        action = Action(type="turn_right", speed=2.0, angle=15.0)
        controller.execute_action(action)

        # 90 - 15 = 75
        assert controller.agent.heading == pytest.approx(75.0)
        assert controller.agent.velocity == 2.0

    def test_turn_heading_wraps_at_360(self, sample_config):
        """Heading wraps around at 360° -> 0°."""
        from src.control import AgentController

        sample_config.control.start_heading = 350.0
        controller = AgentController(sample_config)

        action = Action(type="turn_left", speed=2.0, angle=15.0)
        controller.execute_action(action)

        # 350 + 15 = 365 -> wraps to 5
        assert controller.agent.heading == pytest.approx(5.0)

    def test_turn_heading_wraps_at_0(self, sample_config):
        """Heading wraps around at 0° -> 360°."""
        from src.control import AgentController

        sample_config.control.start_heading = 5.0
        controller = AgentController(sample_config)

        action = Action(type="turn_right", speed=2.0, angle=15.0)
        controller.execute_action(action)

        # 5 - 15 = -10 -> wraps to 350
        assert controller.agent.heading == pytest.approx(350.0)


class TestStopAction:
    """Tests for stop action."""

    def test_stop_sets_velocity_zero(self, sample_config):
        """Stop action sets velocity to 0 and doesn't change position."""
        from src.control import AgentController

        controller = AgentController(sample_config)

        # First move to set velocity
        move_action = Action(type="move_forward", speed=2.0)
        controller.execute_action(move_action)
        assert controller.agent.velocity == 2.0

        # Then stop
        stop_action = Action(type="stop", speed=0.0)
        x_before = controller.agent.x
        y_before = controller.agent.y
        controller.execute_action(stop_action)

        assert controller.agent.velocity == 0.0
        assert controller.agent.x == x_before
        assert controller.agent.y == y_before


class TestTrajectoryTracking:
    """Tests for trajectory management."""

    def test_trajectory_stores_positions(self, sample_config):
        """Each action appends position to trajectory."""
        from src.control import AgentController

        controller = AgentController(sample_config)

        # Initial position in trajectory
        assert len(controller.agent.trajectory) == 1

        # Execute 3 moves
        action = Action(type="move_forward", speed=2.0)
        for _ in range(3):
            controller.execute_action(action)

        # Should have 4 positions total (initial + 3 moves)
        assert len(controller.agent.trajectory) == 4
        # Each position should be different
        assert len(set(controller.agent.trajectory)) == 4

    def test_trajectory_limited_to_max_length(self, sample_config):
        """Trajectory never exceeds trajectory_length."""
        from src.control import AgentController

        sample_config.control.trajectory_length = 10
        controller = AgentController(sample_config)

        action = Action(type="move_forward", speed=2.0)

        # Execute 20 moves (more than max)
        for _ in range(20):
            controller.execute_action(action)

        # Should only keep last 10
        assert len(controller.agent.trajectory) == 10

    def test_trajectory_preserves_recent_positions(self, sample_config):
        """When limited, keeps most recent positions."""
        from src.control import AgentController

        sample_config.control.trajectory_length = 3
        controller = AgentController(sample_config)

        action = Action(type="move_forward", speed=1.0)

        # Execute 5 moves
        positions = [(controller.agent.x, controller.agent.y)]
        for _ in range(5):
            controller.execute_action(action)
            positions.append((controller.agent.x, controller.agent.y))

        # Trajectory should have last 3 positions
        assert len(controller.agent.trajectory) == 3
        # Should match last 3 from our record
        assert controller.agent.trajectory == positions[-3:]


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


class TestUnknownAction:
    """Tests for unknown action type handling."""

    def test_unknown_action_logs_warning(self, sample_config, caplog):
        """Unknown action type logs warning and doesn't crash."""
        from src.control import AgentController

        controller = AgentController(sample_config)
        x_before = controller.agent.x
        y_before = controller.agent.y

        # Create action with invalid type (bypassing type system for testing)
        action = Action(type="invalid_action", speed=2.0)  # type: ignore[arg-type]

        with caplog.at_level(logging.WARNING):
            controller.execute_action(action)

        # Should log warning
        assert "Unknown action type: invalid_action" in caplog.text

        # Position should not change
        assert controller.agent.x == x_before
        assert controller.agent.y == y_before


class TestBoundaryClamping:
    """Tests for boundary clamping functionality."""

    def test_no_clamping_without_bounds(self, sample_config):
        """Without frame bounds, agent can move freely."""
        from src.control import AgentController

        sample_config.control.start_x = 10.0
        sample_config.control.start_y = 10.0
        sample_config.control.start_heading = 180.0  # Left

        # Initialize without bounds
        controller = AgentController(sample_config)

        action = Action(type="move_forward", speed=20.0)
        controller.execute_action(action)

        # Should move left freely, going negative
        assert controller.agent.x < 0

    def test_clamping_at_left_boundary(self, sample_config):
        """Agent position clamped at x=0."""
        from src.control import AgentController

        sample_config.control.start_x = 10.0
        sample_config.control.start_y = 100.0
        sample_config.control.start_heading = 180.0  # Left

        # Initialize with bounds
        controller = AgentController(sample_config, frame_width=640, frame_height=480)

        action = Action(type="move_forward", speed=20.0)
        controller.execute_action(action)

        # Should be clamped at 0
        assert controller.agent.x == 0.0
        assert controller.agent.y == 100.0

    def test_clamping_at_right_boundary(self, sample_config):
        """Agent position clamped at x=frame_width."""
        from src.control import AgentController

        sample_config.control.start_x = 630.0
        sample_config.control.start_y = 100.0
        sample_config.control.start_heading = 0.0  # Right

        # Initialize with bounds
        controller = AgentController(sample_config, frame_width=640, frame_height=480)

        action = Action(type="move_forward", speed=20.0)
        controller.execute_action(action)

        # Should be clamped at 640
        assert controller.agent.x == 640.0
        assert controller.agent.y == 100.0

    def test_clamping_at_top_boundary(self, sample_config):
        """Agent position clamped at y=0."""
        from src.control import AgentController

        sample_config.control.start_x = 100.0
        sample_config.control.start_y = 10.0
        sample_config.control.start_heading = 90.0  # Up

        # Initialize with bounds
        controller = AgentController(sample_config, frame_width=640, frame_height=480)

        action = Action(type="move_forward", speed=20.0)
        controller.execute_action(action)

        # Should be clamped at 0
        assert controller.agent.x == 100.0
        assert controller.agent.y == 0.0

    def test_clamping_at_bottom_boundary(self, sample_config):
        """Agent position clamped at y=frame_height."""
        from src.control import AgentController

        sample_config.control.start_x = 100.0
        sample_config.control.start_y = 470.0
        sample_config.control.start_heading = 270.0  # Down

        # Initialize with bounds
        controller = AgentController(sample_config, frame_width=640, frame_height=480)

        action = Action(type="move_forward", speed=20.0)
        controller.execute_action(action)

        # Should be clamped at 480
        assert controller.agent.x == 100.0
        assert controller.agent.y == 480.0

    def test_clamping_diagonal_movement(self, sample_config):
        """Agent position clamped on both axes when moving diagonally."""
        from src.control import AgentController

        sample_config.control.start_x = 5.0
        sample_config.control.start_y = 5.0
        sample_config.control.start_heading = 135.0  # Up-left diagonal

        # Initialize with bounds
        controller = AgentController(sample_config, frame_width=640, frame_height=480)

        action = Action(type="move_forward", speed=20.0)
        controller.execute_action(action)

        # Both should be clamped at 0
        assert controller.agent.x == 0.0
        assert controller.agent.y == 0.0

    def test_backward_compatibility_no_bounds(self, sample_config):
        """Without specifying bounds, controller works as before."""
        from src.control import AgentController

        controller = AgentController(sample_config)

        # Should initialize without error
        assert controller.agent.x == 320.0
        assert controller.agent.y == 400.0

        # Should move freely
        action = Action(type="move_forward", speed=2.0)
        controller.execute_action(action)

        # Movement should work normally
        assert controller.agent.y < 400.0
