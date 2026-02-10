"""Tests for waypoint navigation module."""

import math
import pytest
from src.models import AgentState, Action
from src.waypoint_navigator import Waypoint, WaypointNavigator


class TestWaypoint:
    """Tests for Waypoint class."""

    def test_distance_to(self):
        """Test distance calculation to agent."""
        waypoint = Waypoint(x=100.0, y=100.0)
        agent = AgentState(x=0.0, y=0.0, heading=0.0)

        expected = math.sqrt(100**2 + 100**2)
        assert abs(waypoint.distance_to(agent) - expected) < 0.01

    def test_is_reached_within_tolerance(self):
        """Test waypoint is reached when within tolerance."""
        waypoint = Waypoint(x=100.0, y=100.0, tolerance=30.0)
        agent = AgentState(x=110.0, y=110.0, heading=0.0)

        distance = math.sqrt((110-100)**2 + (110-100)**2)
        assert distance < 30.0
        assert waypoint.is_reached(agent)

    def test_is_not_reached_outside_tolerance(self):
        """Test waypoint is not reached when outside tolerance."""
        waypoint = Waypoint(x=100.0, y=100.0, tolerance=30.0)
        agent = AgentState(x=150.0, y=150.0, heading=0.0)

        assert not waypoint.is_reached(agent)


class TestWaypointNavigator:
    """Tests for WaypointNavigator class."""

    def test_init_with_waypoints(self):
        """Test initialization with waypoints."""
        waypoints = [Waypoint(100, 100), Waypoint(200, 200)]
        nav = WaypointNavigator(waypoints, turn_rate=5.0, speed=2.0)

        assert len(nav.waypoints) == 2
        assert nav.turn_rate == 5.0
        assert nav.speed == 2.0
        assert nav.current_waypoint_idx == 0

    def test_init_with_empty_waypoints(self):
        """Test initialization with no waypoints."""
        nav = WaypointNavigator([], turn_rate=5.0, speed=2.0)

        assert len(nav.waypoints) == 0
        assert nav.is_complete

    def test_current_waypoint_property(self):
        """Test current_waypoint property."""
        waypoints = [Waypoint(100, 100), Waypoint(200, 200)]
        nav = WaypointNavigator(waypoints, speed=2.0)

        assert nav.current_waypoint == waypoints[0]
        nav.current_waypoint_idx = 1
        assert nav.current_waypoint == waypoints[1]
        nav.current_waypoint_idx = 2
        assert nav.current_waypoint is None

    def test_is_complete_property(self):
        """Test is_complete property."""
        waypoints = [Waypoint(100, 100)]
        nav = WaypointNavigator(waypoints, speed=2.0)

        assert not nav.is_complete
        nav.current_waypoint_idx = 1
        assert nav.is_complete

    def test_calculate_desired_heading_east(self):
        """Test heading calculation for eastward waypoint."""
        waypoint = Waypoint(x=200.0, y=100.0)
        nav = WaypointNavigator([waypoint], speed=2.0)
        agent = AgentState(x=100.0, y=100.0, heading=90.0)

        heading = nav.calculate_desired_heading(agent)
        assert heading is not None
        assert abs(heading - 0.0) < 0.01  # 0° = right/east

    def test_calculate_desired_heading_north(self):
        """Test heading calculation for northward waypoint."""
        waypoint = Waypoint(x=100.0, y=50.0)
        nav = WaypointNavigator([waypoint], speed=2.0)
        agent = AgentState(x=100.0, y=100.0, heading=0.0)

        heading = nav.calculate_desired_heading(agent)
        assert heading is not None
        assert abs(heading - 90.0) < 0.01  # 90° = up/north

    def test_calculate_desired_heading_west(self):
        """Test heading calculation for westward waypoint."""
        waypoint = Waypoint(x=50.0, y=100.0)
        nav = WaypointNavigator([waypoint], speed=2.0)
        agent = AgentState(x=100.0, y=100.0, heading=90.0)

        heading = nav.calculate_desired_heading(agent)
        assert heading is not None
        assert abs(heading - 180.0) < 0.01  # 180° = left/west

    def test_calculate_desired_heading_south(self):
        """Test heading calculation for southward waypoint."""
        waypoint = Waypoint(x=100.0, y=200.0)
        nav = WaypointNavigator([waypoint], speed=2.0)
        agent = AgentState(x=100.0, y=100.0, heading=90.0)

        heading = nav.calculate_desired_heading(agent)
        assert heading is not None
        assert abs(heading - 270.0) < 0.01  # 270° = down/south

    def test_calculate_desired_heading_no_waypoint(self):
        """Test heading calculation with no active waypoint."""
        nav = WaypointNavigator([], speed=2.0)
        agent = AgentState(x=100.0, y=100.0, heading=90.0)

        heading = nav.calculate_desired_heading(agent)
        assert heading is None

    def test_get_turn_action_aligned_move_forward(self):
        """Test action when aligned with waypoint - should move forward."""
        waypoint = Waypoint(x=100.0, y=50.0, tolerance=10.0)
        nav = WaypointNavigator([waypoint], turn_rate=5.0, speed=2.5)
        agent = AgentState(x=100.0, y=100.0, heading=90.0)  # Facing up, waypoint is up

        action = nav.get_turn_action(agent, allow_forward=True)

        assert action.type == "move_forward"
        assert action.speed == 2.5

    def test_get_turn_action_turn_left(self):
        """Test action when need to turn left."""
        waypoint = Waypoint(x=50.0, y=100.0, tolerance=10.0)  # West of agent
        nav = WaypointNavigator([waypoint], turn_rate=5.0, speed=3.0)
        agent = AgentState(x=100.0, y=100.0, heading=90.0)  # Facing up

        action = nav.get_turn_action(agent, allow_forward=True)

        assert action.type == "turn_left"
        assert action.speed == 3.0
        assert action.angle == 5.0

    def test_get_turn_action_turn_right(self):
        """Test action when need to turn right."""
        waypoint = Waypoint(x=200.0, y=100.0, tolerance=10.0)  # East of agent
        nav = WaypointNavigator([waypoint], turn_rate=5.0, speed=3.0)
        agent = AgentState(x=100.0, y=100.0, heading=90.0)  # Facing up

        action = nav.get_turn_action(agent, allow_forward=True)

        assert action.type == "turn_right"
        assert action.speed == 3.0
        assert action.angle == 5.0

    def test_get_turn_action_waypoint_reached(self):
        """Test action when waypoint is reached."""
        waypoint = Waypoint(x=100.0, y=100.0, tolerance=30.0)
        nav = WaypointNavigator([waypoint], speed=2.0)
        agent = AgentState(x=105.0, y=105.0, heading=90.0)

        action = nav.get_turn_action(agent, allow_forward=True)

        assert action.type == "stop"
        assert action.speed == 0.0
        assert nav.is_complete

    def test_get_turn_action_all_waypoints_complete(self):
        """Test action when all waypoints reached."""
        nav = WaypointNavigator([], speed=2.0)
        agent = AgentState(x=100.0, y=100.0, heading=90.0)

        action = nav.get_turn_action(agent, allow_forward=True)

        assert action.type == "stop"
        assert action.speed == 0.0

    def test_get_turn_action_blocked_returns_stop(self):
        """Test action when blocked (allow_forward=False)."""
        waypoint = Waypoint(x=100.0, y=50.0, tolerance=10.0)
        nav = WaypointNavigator([waypoint], turn_rate=5.0, speed=2.0)
        agent = AgentState(x=100.0, y=100.0, heading=90.0)

        action = nav.get_turn_action(agent, allow_forward=False)

        assert action.type == "stop"
        assert action.speed == 0.0

    def test_get_progress_ratio_empty(self):
        """Test progress ratio with no waypoints."""
        nav = WaypointNavigator([], speed=2.0)
        assert nav.get_progress_ratio() == 1.0

    def test_get_progress_ratio_partial(self):
        """Test progress ratio at midpoint."""
        waypoints = [Waypoint(100, 100), Waypoint(200, 200)]
        nav = WaypointNavigator(waypoints, speed=2.0)

        assert nav.get_progress_ratio() == 0.0
        nav.current_waypoint_idx = 1
        assert nav.get_progress_ratio() == 0.5
        nav.current_waypoint_idx = 2
        assert nav.get_progress_ratio() == 1.0

    def test_get_status_string_complete(self):
        """Test status string when complete."""
        nav = WaypointNavigator([], speed=2.0)
        agent = AgentState(x=100.0, y=100.0, heading=90.0)

        status = nav.get_status_string(agent)
        assert status == "Navigation Complete"

    def test_get_status_string_active(self):
        """Test status string with active waypoint."""
        waypoints = [Waypoint(100, 100), Waypoint(200, 200)]
        nav = WaypointNavigator(waypoints, speed=2.0)
        agent = AgentState(x=50.0, y=50.0, heading=90.0)

        status = nav.get_status_string(agent)
        assert "Waypoint 1/2" in status
        assert "Dist:" in status
        assert "Progress:" in status
