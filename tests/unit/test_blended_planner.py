"""Tests for blended navigation planner module."""

import math
import pytest
from unittest.mock import Mock

from src.blended_planner import (
    BlendedNavigationPlanner,
    blend_headings,
    heading_to_action,
)
from src.models import AgentState, Detection, Action, State
from src.waypoint_navigator import Waypoint, WaypointNavigator


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = Mock()
    config.planning = Mock()
    config.planning.obstacle_distance_threshold = 100.0
    config.planning.critical_distance = 30.0
    config.control = Mock()
    config.control.agent_speed = 2.0
    config.control.turn_rate = 5.0
    return config


@pytest.fixture
def sample_navigator():
    """Create sample waypoint navigator."""
    waypoints = [
        Waypoint(x=100.0, y=100.0, tolerance=30.0),
        Waypoint(x=200.0, y=200.0, tolerance=30.0),
    ]
    return WaypointNavigator(waypoints, turn_rate=5.0, speed=2.0)


class TestBlendHeadings:
    """Tests for blend_headings function."""

    def test_blend_weight_zero_returns_first(self):
        """Weight 0 should return first heading."""
        result = blend_headings(45.0, 135.0, 0.0)
        assert abs(result - 45.0) < 0.01

    def test_blend_weight_one_returns_second(self):
        """Weight 1 should return second heading."""
        result = blend_headings(45.0, 135.0, 1.0)
        assert abs(result - 135.0) < 0.01

    def test_blend_midpoint(self):
        """Weight 0.5 should return midpoint."""
        result = blend_headings(0.0, 90.0, 0.5)
        assert abs(result - 45.0) < 0.01

    def test_blend_wraparound_350_to_10(self):
        """Test blending across 0/360 boundary (350° to 10°)."""
        result = blend_headings(350.0, 10.0, 0.5)
        # Midpoint should be 0° (or 360°), going through 355, 0, 5
        assert abs(result - 0.0) < 0.01 or abs(result - 360.0) < 0.01

    def test_blend_wraparound_10_to_350(self):
        """Test blending across 0/360 boundary (10° to 350°)."""
        result = blend_headings(10.0, 350.0, 0.5)
        # Midpoint should be 0° (or 360°), going backward through 5, 0, 355
        assert abs(result - 0.0) < 0.01 or abs(result - 360.0) < 0.01

    def test_blend_opposite_headings(self):
        """Test blending opposite headings."""
        result = blend_headings(0.0, 180.0, 0.5)
        # For exactly opposite headings, either direction is valid
        # Should be either 90° or 270° (depends on direction chosen)
        assert abs(result - 90.0) < 0.01 or abs(result - 270.0) < 0.01

    def test_blend_quarter_weight(self):
        """Test blending with weight 0.25."""
        result = blend_headings(0.0, 100.0, 0.25)
        assert abs(result - 25.0) < 0.01


class TestHeadingToAction:
    """Tests for heading_to_action function."""

    def test_aligned_returns_forward(self):
        """When aligned with desired heading, move forward."""
        action = heading_to_action(
            current_heading=90.0,
            desired_heading=90.0,
            speed=2.0,
            turn_rate=5.0,
            tolerance=5.0
        )

        assert action.type == "move_forward"
        assert action.speed == 2.0

    def test_small_error_returns_forward(self):
        """Small heading error within tolerance returns forward."""
        action = heading_to_action(
            current_heading=90.0,
            desired_heading=93.0,
            speed=2.0,
            turn_rate=5.0,
            tolerance=5.0
        )

        assert action.type == "move_forward"
        assert action.speed == 2.0

    def test_left_turn_required(self):
        """Positive heading error requires left turn."""
        action = heading_to_action(
            current_heading=90.0,
            desired_heading=110.0,
            speed=2.0,
            turn_rate=5.0,
            tolerance=5.0
        )

        assert action.type == "turn_left"
        assert action.speed == 2.0
        assert action.angle == 5.0

    def test_right_turn_required(self):
        """Negative heading error requires right turn."""
        action = heading_to_action(
            current_heading=90.0,
            desired_heading=70.0,
            speed=2.0,
            turn_rate=5.0,
            tolerance=5.0
        )

        assert action.type == "turn_right"
        assert action.speed == 2.0
        assert action.angle == 5.0

    def test_wraparound_left_turn(self):
        """Test left turn across 0/360 boundary."""
        action = heading_to_action(
            current_heading=350.0,
            desired_heading=10.0,
            speed=2.0,
            turn_rate=5.0,
            tolerance=5.0
        )

        # 350 -> 10 is +20 degrees (left turn)
        assert action.type == "turn_left"

    def test_wraparound_right_turn(self):
        """Test right turn across 0/360 boundary."""
        action = heading_to_action(
            current_heading=10.0,
            desired_heading=350.0,
            speed=2.0,
            turn_rate=5.0,
            tolerance=5.0
        )

        # 10 -> 350 is -20 degrees (right turn)
        assert action.type == "turn_right"


class TestBlendedNavigationPlanner:
    """Tests for BlendedNavigationPlanner class."""

    def test_initialization(self, mock_config, sample_navigator):
        """Test planner initialization."""
        planner = BlendedNavigationPlanner(mock_config, sample_navigator)

        assert planner.config == mock_config
        assert planner.navigator == sample_navigator
        assert planner.obstacle_threshold == 100.0
        assert planner.critical_distance == 30.0

    def test_no_obstacles_navigate_state(self, mock_config, sample_navigator):
        """With no obstacles, should return NAVIGATE state."""
        planner = BlendedNavigationPlanner(mock_config, sample_navigator)
        agent = AgentState(x=50.0, y=50.0, heading=45.0)

        state, action = planner.update([], agent)

        assert state == State.NAVIGATE
        assert action.type in ["move_forward", "turn_left", "turn_right"]

    def test_far_obstacle_navigate_state(self, mock_config, sample_navigator):
        """Far obstacle (beyond threshold) should return NAVIGATE."""
        planner = BlendedNavigationPlanner(mock_config, sample_navigator)
        agent = AgentState(x=50.0, y=50.0, heading=90.0)

        # Obstacle far away (distance > threshold)
        detections = [
            Detection(
                x1=50.0, y1=200.0, x2=100.0, y2=250.0,
                class_id=0, class_name="car", confidence=1.0
            )
        ]

        state, action = planner.update(detections, agent)

        assert state == State.NAVIGATE

    def test_close_obstacle_avoid_state(self, mock_config, sample_navigator):
        """Close obstacle (within threshold) should return AVOID."""
        planner = BlendedNavigationPlanner(mock_config, sample_navigator)
        agent = AgentState(x=50.0, y=50.0, heading=90.0)

        # Obstacle close by (distance < threshold)
        detections = [
            Detection(
                x1=40.0, y1=80.0, x2=60.0, y2=100.0,
                class_id=0, class_name="car", confidence=1.0
            )
        ]

        state, action = planner.update(detections, agent)

        assert state == State.AVOID

    def test_critical_obstacle_stop_state(self, mock_config, sample_navigator):
        """Critical obstacle (within critical distance) should return STOP."""
        planner = BlendedNavigationPlanner(mock_config, sample_navigator)
        agent = AgentState(x=50.0, y=50.0, heading=90.0)

        # Obstacle very close (distance < critical)
        detections = [
            Detection(
                x1=45.0, y1=55.0, x2=55.0, y2=65.0,
                class_id=0, class_name="car", confidence=1.0
            )
        ]

        state, action = planner.update(detections, agent)

        assert state == State.STOP
        assert action.type == "stop"
        assert action.speed == 0.0

    def test_goal_reached_stop(self, mock_config):
        """When goal is reached, should return STOP."""
        # Navigator with waypoint at agent's position
        waypoints = [Waypoint(x=50.0, y=50.0, tolerance=30.0)]
        navigator = WaypointNavigator(waypoints, speed=2.0)
        planner = BlendedNavigationPlanner(mock_config, navigator)

        agent = AgentState(x=55.0, y=55.0, heading=90.0)

        state, action = planner.update([], agent)

        assert state == State.STOP
        assert action.type == "stop"
        assert planner.is_goal_reached

    def test_compute_avoidance_heading_north_obstacle(self, mock_config, sample_navigator):
        """Obstacle to the north should produce southward avoidance heading."""
        planner = BlendedNavigationPlanner(mock_config, sample_navigator)

        # Agent at (100, 100), obstacle at (100, 50) - directly north
        detection = Detection(
            x1=90.0, y1=40.0, x2=110.0, y2=60.0,
            class_id=0, class_name="car", confidence=1.0
        )
        agent = AgentState(x=100.0, y=100.0, heading=90.0)

        avoidance_heading = planner._compute_avoidance_heading(detection, agent)

        # Should point away from obstacle (south = 270°)
        assert abs(avoidance_heading - 270.0) < 1.0

    def test_compute_avoidance_heading_east_obstacle(self, mock_config, sample_navigator):
        """Obstacle to the east should produce westward avoidance heading."""
        planner = BlendedNavigationPlanner(mock_config, sample_navigator)

        # Agent at (100, 100), obstacle at (150, 100) - directly east
        detection = Detection(
            x1=140.0, y1=90.0, x2=160.0, y2=110.0,
            class_id=0, class_name="car", confidence=1.0
        )
        agent = AgentState(x=100.0, y=100.0, heading=0.0)

        avoidance_heading = planner._compute_avoidance_heading(detection, agent)

        # Should point away from obstacle (west = 180°)
        assert abs(avoidance_heading - 180.0) < 1.0

    def test_compute_blend_weight_at_threshold(self, mock_config, sample_navigator):
        """At threshold distance, blend weight should be 0."""
        planner = BlendedNavigationPlanner(mock_config, sample_navigator)

        weight = planner._compute_blend_weight(100.0)  # At threshold

        assert abs(weight - 0.0) < 0.01

    def test_compute_blend_weight_at_critical(self, mock_config, sample_navigator):
        """At critical distance, blend weight should be 1."""
        planner = BlendedNavigationPlanner(mock_config, sample_navigator)

        weight = planner._compute_blend_weight(30.0)  # At critical

        assert abs(weight - 1.0) < 0.01

    def test_compute_blend_weight_midpoint(self, mock_config, sample_navigator):
        """Halfway between threshold and critical, weight should be 0.5."""
        planner = BlendedNavigationPlanner(mock_config, sample_navigator)

        # Midpoint: (100 + 30) / 2 = 65
        weight = planner._compute_blend_weight(65.0)

        assert abs(weight - 0.5) < 0.01

    def test_compute_blend_weight_clamped_below(self, mock_config, sample_navigator):
        """Weight clamped at 1.0 for distances below critical."""
        planner = BlendedNavigationPlanner(mock_config, sample_navigator)

        weight = planner._compute_blend_weight(10.0)  # Below critical

        assert abs(weight - 1.0) < 0.01

    def test_compute_blend_weight_clamped_above(self, mock_config, sample_navigator):
        """Weight clamped at 0.0 for distances above threshold."""
        planner = BlendedNavigationPlanner(mock_config, sample_navigator)

        weight = planner._compute_blend_weight(150.0)  # Above threshold

        assert abs(weight - 0.0) < 0.01

    def test_blended_heading_between_waypoint_and_avoidance(
        self, mock_config, sample_navigator
    ):
        """Blended heading should be between waypoint and avoidance headings."""
        planner = BlendedNavigationPlanner(mock_config, sample_navigator)

        # Agent heading toward waypoint (north), obstacle to the east
        agent = AgentState(x=100.0, y=150.0, heading=90.0)

        # Obstacle to the right (east)
        detections = [
            Detection(
                x1=130.0, y1=140.0, x2=150.0, y2=160.0,
                class_id=0, class_name="car", confidence=1.0
            )
        ]

        state, action = planner.update(detections, agent)

        # Should be AVOID state with blended heading
        assert state == State.AVOID

    def test_is_goal_reached_property(self, mock_config):
        """Test is_goal_reached property."""
        waypoints = [Waypoint(x=100.0, y=100.0, tolerance=30.0)]
        navigator = WaypointNavigator(waypoints, speed=2.0)
        planner = BlendedNavigationPlanner(mock_config, navigator)

        # Before reaching goal
        assert not planner.is_goal_reached

        # Reach the goal
        agent = AgentState(x=105.0, y=105.0, heading=90.0)
        planner.update([], agent)

        # After reaching goal
        assert planner.is_goal_reached

    def test_goal_status_string(self, mock_config, sample_navigator):
        """Test goal_status property returns navigator status."""
        planner = BlendedNavigationPlanner(mock_config, sample_navigator)

        agent = AgentState(x=50.0, y=50.0, heading=90.0)
        status = planner.goal_status(agent)

        assert isinstance(status, str)
        assert "Waypoint" in status
