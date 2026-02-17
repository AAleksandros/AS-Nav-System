"""Tests for waypoint navigation module."""

import math
from src.models import AgentState
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

        distance = math.sqrt((110 - 100) ** 2 + (110 - 100) ** 2)
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
        nav = WaypointNavigator(waypoints)

        assert len(nav.waypoints) == 2
        assert nav.current_waypoint_idx == 0
        assert not nav.is_complete

    def test_init_with_empty_waypoints(self):
        """Test initialization with no waypoints."""
        nav = WaypointNavigator([])

        assert len(nav.waypoints) == 0
        assert nav.is_complete

    def test_current_waypoint_property(self):
        """Test current_waypoint property returns correct waypoint."""
        waypoints = [Waypoint(100, 100), Waypoint(200, 200)]
        nav = WaypointNavigator(waypoints)

        assert nav.current_waypoint == waypoints[0]
        nav.current_waypoint_idx = 1
        assert nav.current_waypoint == waypoints[1]
        nav.current_waypoint_idx = 2
        assert nav.current_waypoint is None

    def test_current_goal_returns_tuple(self):
        """Test current_goal returns (x, y) tuple."""
        waypoints = [Waypoint(100.0, 200.0)]
        nav = WaypointNavigator(waypoints)

        goal = nav.current_goal
        assert goal == (100.0, 200.0)

    def test_current_goal_none_when_complete(self):
        """Test current_goal returns None when all waypoints reached."""
        nav = WaypointNavigator([])

        assert nav.current_goal is None

    def test_is_complete_property(self):
        """Test is_complete property."""
        waypoints = [Waypoint(100, 100)]
        nav = WaypointNavigator(waypoints)

        assert not nav.is_complete
        nav.current_waypoint_idx = 1
        assert nav.is_complete

    def test_progress_empty(self):
        """Test progress with no waypoints returns 1.0."""
        nav = WaypointNavigator([])
        assert nav.progress == 1.0

    def test_progress_partial(self):
        """Test progress at various stages."""
        waypoints = [Waypoint(100, 100), Waypoint(200, 200)]
        nav = WaypointNavigator(waypoints)

        assert nav.progress == 0.0
        nav.current_waypoint_idx = 1
        assert nav.progress == 0.5
        nav.current_waypoint_idx = 2
        assert nav.progress == 1.0

    def test_check_and_advance_reaches_waypoint(self):
        """Test check_and_advance advances when within tolerance."""
        waypoints = [Waypoint(100.0, 100.0, tolerance=30.0), Waypoint(200.0, 200.0)]
        nav = WaypointNavigator(waypoints)

        # Agent is close to first waypoint
        result = nav.check_and_advance(105.0, 105.0)
        assert result is True
        assert nav.current_waypoint_idx == 1

    def test_check_and_advance_not_reached(self):
        """Test check_and_advance does not advance when far away."""
        waypoints = [Waypoint(100.0, 100.0, tolerance=10.0)]
        nav = WaypointNavigator(waypoints)

        result = nav.check_and_advance(200.0, 200.0)
        assert result is False
        assert nav.current_waypoint_idx == 0

    def test_check_and_advance_all_complete(self):
        """Test check_and_advance returns False when all waypoints done."""
        nav = WaypointNavigator([])

        result = nav.check_and_advance(0.0, 0.0)
        assert result is False

    def test_check_and_advance_sequential(self):
        """Test advancing through multiple waypoints sequentially."""
        waypoints = [
            Waypoint(100.0, 100.0, tolerance=20.0),
            Waypoint(200.0, 200.0, tolerance=20.0),
        ]
        nav = WaypointNavigator(waypoints)

        # Reach first waypoint
        assert nav.check_and_advance(100.0, 100.0) is True
        assert nav.current_waypoint_idx == 1
        assert not nav.is_complete

        # Reach second waypoint
        assert nav.check_and_advance(200.0, 200.0) is True
        assert nav.current_waypoint_idx == 2
        assert nav.is_complete
        assert nav.current_goal is None


class TestOvershootDetection:
    """Tests for closest-approach waypoint advancement."""

    def test_overshoot_advances_waypoint(self):
        """Agent approaches within overshoot_tolerance then recedes."""
        wp = Waypoint(100.0, 100.0, tolerance=25.0)
        nav = WaypointNavigator([wp, Waypoint(200.0, 200.0, tolerance=25.0)])

        # Approach: distance decreasing toward waypoint
        nav.check_and_advance(160.0, 100.0)  # d=60
        nav.check_and_advance(145.0, 100.0)  # d=45 (within overshoot_tolerance=50)
        nav.check_and_advance(148.0, 100.0)  # d=48, receding step 1
        result = nav.check_and_advance(152.0, 100.0)  # d=52, receding step 2
        assert result is True
        assert nav.current_waypoint_idx == 1

    def test_no_overshoot_when_far(self):
        """Agent that never gets within overshoot_tolerance should not advance."""
        wp = Waypoint(100.0, 100.0, tolerance=25.0)
        nav = WaypointNavigator([wp])

        nav.check_and_advance(200.0, 100.0)  # d=100
        nav.check_and_advance(195.0, 100.0)  # d=95
        nav.check_and_advance(198.0, 100.0)  # receding step 1
        result = nav.check_and_advance(202.0, 100.0)  # receding step 2
        assert result is False
        assert nav.current_waypoint_idx == 0

    def test_overshoot_resets_on_advance(self):
        """Tracking state resets after waypoint advancement."""
        wps = [
            Waypoint(100.0, 100.0, tolerance=25.0),
            Waypoint(200.0, 200.0, tolerance=25.0),
        ]
        nav = WaypointNavigator(wps)

        # Reach first waypoint normally
        nav.check_and_advance(100.0, 100.0)
        assert nav.current_waypoint_idx == 1

        # Now tracking should be fresh for second waypoint
        # Agent far from second waypoint and receding should NOT advance
        nav.check_and_advance(300.0, 300.0)  # d=~141
        nav.check_and_advance(305.0, 305.0)  # receding
        result = nav.check_and_advance(310.0, 310.0)  # receding
        assert result is False

    def test_radius_check_still_works(self):
        """Original radius-based advancement still works."""
        wp = Waypoint(100.0, 100.0, tolerance=25.0)
        nav = WaypointNavigator([wp])

        result = nav.check_and_advance(110.0, 100.0)  # d=10, within tolerance
        assert result is True

    def test_single_recede_not_enough(self):
        """A single receding step should not trigger advancement."""
        wp = Waypoint(100.0, 100.0, tolerance=25.0)
        nav = WaypointNavigator([wp])

        nav.check_and_advance(140.0, 100.0)  # d=40 (within overshoot tol=50)
        nav.check_and_advance(145.0, 100.0)  # d=45, receding step 1
        # Only 1 receding step — not enough
        result = nav.check_and_advance(140.0, 100.0)  # approaching again
        assert result is False

    def test_overshoot_tolerance_is_twice_tolerance(self):
        """Overshoot tolerance should be 2x the waypoint tolerance."""
        wp = Waypoint(100.0, 100.0, tolerance=10.0)
        nav = WaypointNavigator([wp])

        # Approach to d=19 (within 2*10=20 overshoot tolerance)
        nav.check_and_advance(119.0, 100.0)  # d=19
        nav.check_and_advance(121.0, 100.0)  # receding
        result = nav.check_and_advance(123.0, 100.0)  # receding step 2
        assert result is True
