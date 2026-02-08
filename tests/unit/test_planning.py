"""Unit tests for the planning module."""

from unittest.mock import Mock, patch

import pytest

from src.models import AgentState, Detection, State


@pytest.fixture
def sample_config():
    """Create a mock config with planning and control parameters."""
    config = Mock()
    config.planning.obstacle_distance_threshold = 100
    config.planning.critical_distance = 50
    config.control.agent_speed = 2.0
    config.control.turn_rate = 5.0
    return config


@pytest.fixture
def sample_agent():
    """Create an agent at a known position."""
    return AgentState(x=320.0, y=400.0)


def _make_detection(x1: float, y1: float, x2: float, y2: float) -> Detection:
    """Helper to create a detection with default class info."""
    return Detection(
        x1=x1, y1=y1, x2=x2, y2=y2,
        class_id=0, class_name="person", confidence=0.9,
    )


class TestNavigationPlannerInit:
    """Tests for NavigationPlanner initialization."""

    @patch("src.planning.CoordinateTransform")
    def test_initialization(self, mock_ct_class, sample_config):
        """Verify config stored, initial state is NAVIGATE, transform created."""
        from src.planning import NavigationPlanner

        planner = NavigationPlanner(sample_config)

        assert planner.current_state == State.NAVIGATE
        assert planner.obstacle_threshold == 100
        assert planner.critical_distance == 50
        assert planner.speed == 2.0
        assert planner.turn_rate == 5.0
        mock_ct_class.assert_called_once_with(sample_config)


class TestUpdateStateTransitions:
    """Tests for the update method state transitions."""

    @patch("src.planning.CoordinateTransform")
    def test_navigate_no_obstacles(self, mock_ct_class, sample_config, sample_agent):
        """Empty detection list -> NAVIGATE + move_forward."""
        from src.planning import NavigationPlanner

        mock_ct_class.return_value.find_nearest_obstacle.return_value = None
        planner = NavigationPlanner(sample_config)

        state, action = planner.update([], sample_agent)

        assert state == State.NAVIGATE
        assert action.type == "move_forward"
        assert action.speed == 2.0

    @patch("src.planning.CoordinateTransform")
    def test_navigate_far_obstacles(self, mock_ct_class, sample_config, sample_agent):
        """Obstacles at >100px -> NAVIGATE + move_forward."""
        from src.planning import NavigationPlanner

        detection = _make_detection(0.0, 0.0, 20.0, 20.0)
        mock_ct_class.return_value.find_nearest_obstacle.return_value = (
            150.0, detection,
        )
        planner = NavigationPlanner(sample_config)

        state, action = planner.update([detection], sample_agent)

        assert state == State.NAVIGATE
        assert action.type == "move_forward"

    @patch("src.planning.CoordinateTransform")
    def test_avoid_moderate_distance(self, mock_ct_class, sample_config, sample_agent):
        """Obstacle at 75px (between critical and threshold) -> AVOID + turn."""
        from src.planning import NavigationPlanner

        # Obstacle to the left of agent (center_x < agent.x)
        detection = _make_detection(100.0, 350.0, 200.0, 450.0)  # center=(150, 400)
        mock_ct_class.return_value.find_nearest_obstacle.return_value = (
            75.0, detection,
        )
        planner = NavigationPlanner(sample_config)

        state, action = planner.update([detection], sample_agent)

        assert state == State.AVOID
        assert action.type in ("turn_left", "turn_right")

    @patch("src.planning.CoordinateTransform")
    def test_stop_critical_distance(self, mock_ct_class, sample_config, sample_agent):
        """Obstacle at 30px (< critical) -> STOP + stop action."""
        from src.planning import NavigationPlanner

        detection = _make_detection(300.0, 380.0, 340.0, 420.0)
        mock_ct_class.return_value.find_nearest_obstacle.return_value = (
            30.0, detection,
        )
        planner = NavigationPlanner(sample_config)

        state, action = planner.update([detection], sample_agent)

        assert state == State.STOP
        assert action.type == "stop"
        assert action.speed == 0.0

    @patch("src.planning.CoordinateTransform")
    def test_navigate_to_stop_direct(self, mock_ct_class, sample_config, sample_agent):
        """Obstacle at <50px skips AVOID, goes straight to STOP."""
        from src.planning import NavigationPlanner

        detection = _make_detection(300.0, 380.0, 340.0, 420.0)
        mock_transform = mock_ct_class.return_value

        # First call: no obstacles -> NAVIGATE
        mock_transform.find_nearest_obstacle.return_value = None
        planner = NavigationPlanner(sample_config)
        state, _ = planner.update([], sample_agent)
        assert state == State.NAVIGATE

        # Second call: critical distance -> STOP (skipping AVOID)
        mock_transform.find_nearest_obstacle.return_value = (30.0, detection)
        state, action = planner.update([detection], sample_agent)
        assert state == State.STOP
        assert action.type == "stop"

    @patch("src.planning.CoordinateTransform")
    def test_state_persists_between_calls(
        self, mock_ct_class, sample_config, sample_agent
    ):
        """Verify current_state updates across calls."""
        from src.planning import NavigationPlanner

        detection = _make_detection(100.0, 350.0, 200.0, 450.0)
        mock_transform = mock_ct_class.return_value
        planner = NavigationPlanner(sample_config)

        # First call: AVOID
        mock_transform.find_nearest_obstacle.return_value = (75.0, detection)
        state, _ = planner.update([detection], sample_agent)
        assert planner.current_state == State.AVOID

        # Second call: NAVIGATE (obstacles cleared)
        mock_transform.find_nearest_obstacle.return_value = None
        state, _ = planner.update([], sample_agent)
        assert planner.current_state == State.NAVIGATE


class TestAvoidanceDirection:
    """Tests for _calculate_avoidance_direction."""

    @patch("src.planning.CoordinateTransform")
    def test_obstacle_left_turns_right(self, mock_ct_class, sample_config):
        """Obstacle center_x < agent.x -> turn_right."""
        from src.planning import NavigationPlanner

        planner = NavigationPlanner(sample_config)
        agent = AgentState(x=320.0, y=400.0)
        # center_x = 150 < 320
        detection = _make_detection(100.0, 350.0, 200.0, 450.0)

        direction = planner._calculate_avoidance_direction(detection, agent)

        assert direction == "turn_right"

    @patch("src.planning.CoordinateTransform")
    def test_obstacle_right_turns_left(self, mock_ct_class, sample_config):
        """Obstacle center_x > agent.x -> turn_left."""
        from src.planning import NavigationPlanner

        planner = NavigationPlanner(sample_config)
        agent = AgentState(x=320.0, y=400.0)
        # center_x = 500 > 320
        detection = _make_detection(450.0, 350.0, 550.0, 450.0)

        direction = planner._calculate_avoidance_direction(detection, agent)

        assert direction == "turn_left"

    @patch("src.planning.CoordinateTransform")
    def test_obstacle_centered_turns_left(self, mock_ct_class, sample_config):
        """Obstacle center_x == agent.x -> turn_left (default)."""
        from src.planning import NavigationPlanner

        planner = NavigationPlanner(sample_config)
        agent = AgentState(x=320.0, y=400.0)
        # center_x = 320 == agent.x
        detection = _make_detection(300.0, 350.0, 340.0, 450.0)

        direction = planner._calculate_avoidance_direction(detection, agent)

        assert direction == "turn_left"


class TestActionValues:
    """Tests for correct action values from config."""

    @patch("src.planning.CoordinateTransform")
    def test_navigate_action_speed(self, mock_ct_class, sample_config, sample_agent):
        """move_forward action has correct speed from config."""
        from src.planning import NavigationPlanner

        mock_ct_class.return_value.find_nearest_obstacle.return_value = None
        planner = NavigationPlanner(sample_config)

        _, action = planner.update([], sample_agent)

        assert action.type == "move_forward"
        assert action.speed == 2.0
        assert action.angle == 0.0

    @patch("src.planning.CoordinateTransform")
    def test_avoid_action_values(self, mock_ct_class, sample_config, sample_agent):
        """Turn action has correct speed and angle from config."""
        from src.planning import NavigationPlanner

        # Obstacle to the right -> turn_left
        detection = _make_detection(450.0, 350.0, 550.0, 450.0)  # center=(500, 400)
        mock_ct_class.return_value.find_nearest_obstacle.return_value = (
            75.0, detection,
        )
        planner = NavigationPlanner(sample_config)

        _, action = planner.update([detection], sample_agent)

        assert action.type == "turn_left"
        assert action.speed == 2.0
        assert action.angle == 5.0
