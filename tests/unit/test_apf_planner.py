"""Tests for the Artificial Potential Fields planner."""

import math
from typing import List
from unittest.mock import Mock

import pytest

from src.apf_planner import APFPlanner
from src.models import ForceVector, SensorReading, State
from src.utils.math_utils import normalize_angle


# --- Fixtures ---


@pytest.fixture
def planner() -> APFPlanner:
    """Default planner with standard gains."""
    return APFPlanner(
        k_att=1.0,
        k_rep=100.0,
        influence_range=100.0,
        goal_tolerance=15.0,
        max_speed=60.0,
        escape_threshold=2.0,
        escape_window=30,
        escape_force=50.0,
    )


@pytest.fixture
def no_escape_planner() -> APFPlanner:
    """Planner with escape disabled (high threshold)."""
    return APFPlanner(
        k_att=1.0,
        k_rep=100.0,
        influence_range=100.0,
        goal_tolerance=15.0,
        max_speed=60.0,
        escape_threshold=0.0,
        escape_window=30,
        escape_force=0.0,
    )


@pytest.fixture
def hit_reading_ahead() -> List[SensorReading]:
    """Sensor reading with obstacle hit straight ahead at distance 30."""
    return [
        SensorReading(angle=0.0, distance=30.0, hit=True, hit_point=(30.0, 0.0)),
    ]


@pytest.fixture
def hit_reading_close() -> List[SensorReading]:
    """Sensor reading with obstacle very close ahead at distance 5."""
    return [
        SensorReading(angle=0.0, distance=5.0, hit=True, hit_point=(5.0, 0.0)),
    ]


@pytest.fixture
def no_hit_readings() -> List[SensorReading]:
    """Sensor readings with no obstacles detected."""
    return [
        SensorReading(angle=0.0, distance=150.0, hit=False),
        SensorReading(angle=math.pi / 2, distance=150.0, hit=False),
        SensorReading(angle=math.pi, distance=150.0, hit=False),
    ]


# --- Tests: Init ---


class TestAPFPlannerInit:
    def test_basic_attributes(self, planner: APFPlanner) -> None:
        assert planner.k_att == 1.0
        assert planner.k_rep == 100.0
        assert planner.influence_range == 100.0
        assert planner.goal_tolerance == 15.0
        assert planner.max_speed == 60.0
        assert planner.force_smoothing == 0.3
        assert planner.symmetry_threshold == 0.3
        assert planner.symmetry_nudge_force == 15.0

    def test_from_config(self) -> None:
        config = Mock()
        config.planner.k_att = 2.0
        config.planner.k_rep = 200.0
        config.planner.influence_range = 80.0
        config.planner.goal_tolerance = 10.0
        config.planner.max_speed = 50.0
        config.planner.escape_threshold = 3.0
        config.planner.escape_window = 20
        config.planner.escape_force = 40.0
        config.planner.slow_down_distance = 60.0
        config.planner.cruise_fraction = 0.5
        config.planner.vortex_weight = 0.5
        config.planner.force_smoothing = 0.4
        config.planner.symmetry_threshold = 0.2
        config.planner.symmetry_nudge_force = 20.0
        config.planner.forward_half_angle = 1.57
        config.planner.rear_attenuation = 0.0
        config.planner.attenuation_power = 1.0
        config.planner.commitment_duration = 0
        config.planner.commitment_force = 0.0

        p = APFPlanner.from_config(config)
        assert p.k_att == 2.0
        assert p.k_rep == 200.0
        assert p.influence_range == 80.0
        assert p.goal_tolerance == 10.0
        assert p.max_speed == 50.0
        assert p.escape_threshold == 3.0
        assert p.escape_window == 20
        assert p.escape_force == 40.0
        assert p.slow_down_distance == 60.0
        assert p.cruise_fraction == 0.5
        assert p.vortex_weight == 0.5
        assert p.force_smoothing == pytest.approx(0.4)
        assert p.symmetry_threshold == pytest.approx(0.2)
        assert p.symmetry_nudge_force == pytest.approx(20.0)

    def test_position_history_initialized_empty(self, planner: APFPlanner) -> None:
        assert len(planner._position_history) == 0


# --- Tests: Attractive Force ---


class TestAttractiveForce:
    def test_force_points_toward_goal(self, planner: APFPlanner) -> None:
        """Attractive force should point from agent toward goal."""
        f = planner.attractive_force(0.0, 0.0, 100.0, 0.0)
        assert f.fx > 0  # goal is east
        assert f.fy == pytest.approx(0.0, abs=1e-10)
        assert f.source == "attractive"

    def test_force_magnitude_equals_k_att(self, planner: APFPlanner) -> None:
        """Attractive force should be a unit vector scaled by k_att."""
        f = planner.attractive_force(0.0, 0.0, 100.0, 0.0)
        assert f.magnitude == pytest.approx(planner.k_att)

    def test_force_direction_northeast(self, planner: APFPlanner) -> None:
        """Attractive force toward a NE goal should point NE."""
        f = planner.attractive_force(0.0, 0.0, 50.0, 50.0)
        assert f.fx > 0
        assert f.fy > 0
        assert f.heading == pytest.approx(math.pi / 4, abs=0.01)

    def test_force_at_goal_is_zero(self, planner: APFPlanner) -> None:
        """When agent is at the goal, attractive force should be zero."""
        f = planner.attractive_force(100.0, 100.0, 100.0, 100.0)
        assert f.magnitude == pytest.approx(0.0)

    def test_force_direction_south(self, planner: APFPlanner) -> None:
        """Force toward a goal directly south."""
        f = planner.attractive_force(0.0, 100.0, 0.0, 0.0)
        assert f.fx == pytest.approx(0.0, abs=1e-10)
        assert f.fy < 0  # pointing south


# --- Tests: Repulsive Force ---


class TestRepulsiveForce:
    def test_repulsive_force_from_hit(self, planner: APFPlanner) -> None:
        """Obstacle hit should produce repulsive force away from obstacle."""
        reading = SensorReading(
            angle=0.0, distance=30.0, hit=True, hit_point=(30.0, 0.0)
        )
        f = planner.repulsive_force(0.0, 0.0, reading)
        assert f.fx < 0  # pushes away from obstacle (west)
        assert f.source == "repulsive"

    def test_no_force_when_no_hit(self, planner: APFPlanner) -> None:
        """No hit means zero repulsive force."""
        reading = SensorReading(angle=0.0, distance=150.0, hit=False)
        f = planner.repulsive_force(0.0, 0.0, reading)
        assert f.magnitude == pytest.approx(0.0)

    def test_no_force_beyond_influence_range(self, planner: APFPlanner) -> None:
        """Obstacle beyond influence range produces zero force."""
        reading = SensorReading(
            angle=0.0, distance=150.0, hit=True, hit_point=(150.0, 0.0)
        )
        f = planner.repulsive_force(0.0, 0.0, reading)
        assert f.magnitude == pytest.approx(0.0)

    def test_force_increases_as_obstacle_closer(self, planner: APFPlanner) -> None:
        """Closer obstacles should produce stronger repulsive force."""
        far = SensorReading(
            angle=0.0, distance=80.0, hit=True, hit_point=(80.0, 0.0)
        )
        near = SensorReading(
            angle=0.0, distance=20.0, hit=True, hit_point=(20.0, 0.0)
        )
        f_far = planner.repulsive_force(0.0, 0.0, far)
        f_near = planner.repulsive_force(0.0, 0.0, near)
        assert f_near.magnitude > f_far.magnitude

    def test_force_direction_away_from_obstacle(self, planner: APFPlanner) -> None:
        """Repulsive force should point away from the hit point."""
        # Obstacle to the north
        reading = SensorReading(
            angle=math.pi / 2, distance=50.0, hit=True, hit_point=(0.0, 50.0)
        )
        f = planner.repulsive_force(0.0, 0.0, reading)
        assert f.fy < 0  # pushes south, away from obstacle

    def test_very_close_obstacle_clamped(self, planner: APFPlanner) -> None:
        """Very close obstacle (distance ~0) should not produce infinite force."""
        reading = SensorReading(
            angle=0.0, distance=0.5, hit=True, hit_point=(0.5, 0.0)
        )
        f = planner.repulsive_force(0.0, 0.0, reading)
        assert math.isfinite(f.fx)
        assert math.isfinite(f.fy)

    def test_agent_at_hit_point(self, planner: APFPlanner) -> None:
        """When agent is exactly at hit point, force uses ray angle for direction."""
        reading = SensorReading(
            angle=0.0, distance=1.0, hit=True, hit_point=(10.0, 20.0)
        )
        f = planner.repulsive_force(10.0, 20.0, reading)
        # Direction should be opposite of ray angle (0), so fx < 0
        assert f.fx < 0
        assert math.isfinite(f.magnitude)


# --- Tests: Compute (Total Force + State) ---


class TestCompute:
    def test_returns_force_and_state(
        self, planner: APFPlanner, no_hit_readings: List[SensorReading]
    ) -> None:
        result = planner.compute(0.0, 0.0, 100.0, 0.0, no_hit_readings)
        assert "total_force" in result
        assert "state" in result
        assert "forces" in result
        assert isinstance(result["total_force"], ForceVector)
        assert isinstance(result["state"], State)
        assert isinstance(result["forces"], list)

    def test_navigate_state_when_clear(
        self, planner: APFPlanner, no_hit_readings: List[SensorReading]
    ) -> None:
        """No obstacles nearby -> NAVIGATE state."""
        result = planner.compute(0.0, 0.0, 100.0, 0.0, no_hit_readings)
        assert result["state"] == State.NAVIGATE

    def test_avoid_state_when_obstacle_detected(
        self, planner: APFPlanner, hit_reading_ahead: List[SensorReading]
    ) -> None:
        """Obstacle within influence range -> AVOID state."""
        result = planner.compute(0.0, 0.0, 100.0, 0.0, hit_reading_ahead)
        assert result["state"] == State.AVOID

    def test_stop_state_when_very_close(
        self, planner: APFPlanner, hit_reading_close: List[SensorReading]
    ) -> None:
        """Obstacle very close -> STOP state."""
        result = planner.compute(0.0, 0.0, 100.0, 0.0, hit_reading_close)
        assert result["state"] == State.STOP

    def test_total_force_is_sum(
        self, planner: APFPlanner
    ) -> None:
        """Total force should be sum of individual forces (before capping)."""
        # Use weak reading so total stays below max_speed (no capping)
        weak_reading = [SensorReading(
            angle=0.0, distance=90.0, hit=True, hit_point=(90.0, 0.0),
        )]
        result = planner.compute(0.0, 0.0, 100.0, 0.0, weak_reading)
        forces = result["forces"]
        att = [f for f in forces if f.source == "attractive"]
        rep = [f for f in forces if f.source == "repulsive"]
        total = result["total_force"]

        expected_fx = sum(f.fx for f in att) + sum(f.fx for f in rep)
        expected_fy = sum(f.fy for f in att) + sum(f.fy for f in rep)
        assert total.fx == pytest.approx(expected_fx, abs=1e-6)
        assert total.fy == pytest.approx(expected_fy, abs=1e-6)

    def test_total_force_capped_at_max_speed(
        self, planner: APFPlanner
    ) -> None:
        """Total force magnitude should not exceed max_speed."""
        # Very close obstacle -> large repulsive force exceeds max_speed
        readings = [
            SensorReading(angle=0.0, distance=1.05, hit=True, hit_point=(1.05, 0.0)),
        ]
        result = planner.compute(0.0, 0.0, 100.0, 0.0, readings)
        assert result["total_force"].magnitude == pytest.approx(
            planner.max_speed, abs=0.1
        )

    def test_goal_reached(
        self, planner: APFPlanner, no_hit_readings: List[SensorReading]
    ) -> None:
        """When at the goal, force should be zero and state NAVIGATE."""
        result = planner.compute(100.0, 100.0, 100.0, 100.0, no_hit_readings)
        assert result["total_force"].magnitude == pytest.approx(0.0)

    def test_forces_list_contains_attractive(
        self, planner: APFPlanner, no_hit_readings: List[SensorReading]
    ) -> None:
        """Forces list should always contain at least the attractive force."""
        result = planner.compute(0.0, 0.0, 100.0, 0.0, no_hit_readings)
        sources = [f.source for f in result["forces"]]
        assert "attractive" in sources

    def test_forces_list_contains_repulsive_when_hit(
        self, planner: APFPlanner, hit_reading_ahead: List[SensorReading]
    ) -> None:
        result = planner.compute(0.0, 0.0, 100.0, 0.0, hit_reading_ahead)
        sources = [f.source for f in result["forces"]]
        assert "repulsive" in sources


# --- Tests: Local Minima Escape ---


class TestLocalMinimaEscape:
    def test_no_escape_when_moving(self, planner: APFPlanner) -> None:
        """Agent that is moving should not trigger escape."""
        # Simulate movement by recording different positions
        for i in range(40):
            planner._record_position(float(i), 0.0)
        assert not planner._is_stuck()

    def test_escape_when_stuck(self, planner: APFPlanner) -> None:
        """Agent stuck in same position should trigger escape."""
        for _ in range(40):
            planner._record_position(100.0, 100.0)
        assert planner._is_stuck()

    def test_escape_force_applied_when_stuck(self, planner: APFPlanner) -> None:
        """When stuck, compute should add an escape force."""
        # Fill position history with same position
        for _ in range(40):
            planner._record_position(100.0, 100.0)

        readings = [
            SensorReading(angle=0.0, distance=50.0, hit=True, hit_point=(150.0, 100.0)),
        ]
        result = planner.compute(100.0, 100.0, 300.0, 200.0, readings)
        sources = [f.source for f in result["forces"]]
        assert "escape" in sources

    def test_escape_force_magnitude(self, planner: APFPlanner) -> None:
        """Escape force should have the configured magnitude."""
        for _ in range(40):
            planner._record_position(100.0, 100.0)

        readings = [
            SensorReading(angle=0.0, distance=50.0, hit=True, hit_point=(150.0, 100.0)),
        ]
        result = planner.compute(100.0, 100.0, 300.0, 200.0, readings)
        escape_forces = [f for f in result["forces"] if f.source == "escape"]
        assert len(escape_forces) == 1
        assert escape_forces[0].magnitude == pytest.approx(
            planner.escape_force, abs=1.0
        )

    def test_not_stuck_with_insufficient_history(self, planner: APFPlanner) -> None:
        """Should not be stuck with fewer positions than escape_window."""
        for _ in range(5):
            planner._record_position(100.0, 100.0)
        assert not planner._is_stuck()

    def test_escape_seed_reproducibility(self) -> None:
        """Escape direction should be reproducible with same seed."""
        p1 = APFPlanner(
            k_att=1.0, k_rep=100.0, influence_range=100.0,
            goal_tolerance=15.0, max_speed=60.0,
            escape_threshold=2.0, escape_window=30, escape_force=50.0,
            escape_seed=42,
        )
        p2 = APFPlanner(
            k_att=1.0, k_rep=100.0, influence_range=100.0,
            goal_tolerance=15.0, max_speed=60.0,
            escape_threshold=2.0, escape_window=30, escape_force=50.0,
            escape_seed=42,
        )
        for _ in range(40):
            p1._record_position(100.0, 100.0)
            p2._record_position(100.0, 100.0)

        readings = [SensorReading(
            angle=0.0, distance=50.0, hit=True,
            hit_point=(150.0, 100.0),
        )]
        r1 = p1.compute(100.0, 100.0, 300.0, 200.0, readings)
        r2 = p2.compute(100.0, 100.0, 300.0, 200.0, readings)
        e1 = [f for f in r1["forces"] if f.source == "escape"][0]
        e2 = [f for f in r2["forces"] if f.source == "escape"][0]
        assert e1.fx == pytest.approx(e2.fx)
        assert e1.fy == pytest.approx(e2.fy)


# --- Tests: State Classification Thresholds ---


class TestStateClassification:
    def test_stop_threshold_is_agent_radius_scale(
        self, planner: APFPlanner
    ) -> None:
        """STOP when min sensor distance <= 10% of influence range."""
        # 10% of 100 = 10
        reading = SensorReading(
            angle=0.0, distance=9.0, hit=True, hit_point=(9.0, 0.0)
        )
        result = planner.compute(0.0, 0.0, 200.0, 0.0, [reading])
        assert result["state"] == State.STOP

    def test_avoid_within_influence(self, planner: APFPlanner) -> None:
        """AVOID state when obstacle within influence range but not critical."""
        reading = SensorReading(
            angle=0.0, distance=50.0, hit=True, hit_point=(50.0, 0.0)
        )
        result = planner.compute(0.0, 0.0, 200.0, 0.0, [reading])
        assert result["state"] == State.AVOID

    def test_navigate_beyond_influence(self, planner: APFPlanner) -> None:
        """NAVIGATE state when no obstacle within influence range."""
        reading = SensorReading(
            angle=0.0, distance=150.0, hit=True, hit_point=(150.0, 0.0)
        )
        result = planner.compute(0.0, 0.0, 200.0, 0.0, [reading])
        assert result["state"] == State.NAVIGATE


# --- Tests: Compute Speed ---


class TestComputeSpeed:
    """Tests for speed decoupled from force magnitude."""

    @pytest.fixture
    def speed_planner(self) -> APFPlanner:
        """Planner with explicit speed params for testing."""
        return APFPlanner(
            k_att=20.0,
            k_rep=2000.0,
            influence_range=100.0,
            goal_tolerance=15.0,
            max_speed=80.0,
            escape_threshold=2.0,
            escape_window=30,
            escape_force=80.0,
            slow_down_distance=60.0,
            cruise_fraction=0.5,
        )

    def test_cruise_speed_clear_path(self, speed_planner: APFPlanner) -> None:
        """Clear path should yield max_speed * cruise_fraction."""
        speed = speed_planner.compute_speed(float("inf"), State.NAVIGATE)
        assert speed == pytest.approx(40.0)  # 80 * 0.5

    def test_cruise_speed_obstacle_beyond_slow_down(
        self, speed_planner: APFPlanner
    ) -> None:
        """Obstacle beyond slow_down_distance should yield full cruise speed."""
        speed = speed_planner.compute_speed(100.0, State.NAVIGATE)
        assert speed == pytest.approx(40.0)

    def test_deceleration_at_half_slow_down(
        self, speed_planner: APFPlanner
    ) -> None:
        """Obstacle at half slow_down_distance -> sqrt(0.5) * cruise."""
        speed = speed_planner.compute_speed(30.0, State.NAVIGATE)
        # 40 * sqrt(30/60) = 40 * 0.7071 ≈ 28.28
        assert speed == pytest.approx(28.28, abs=0.1)

    def test_deceleration_at_quarter_slow_down(
        self, speed_planner: APFPlanner
    ) -> None:
        """Obstacle at quarter slow_down_distance -> sqrt(0.25) * cruise."""
        speed = speed_planner.compute_speed(15.0, State.NAVIGATE)
        # 40 * sqrt(15/60) = 40 * 0.5 = 20.0
        assert speed == pytest.approx(20.0)

    def test_obstacle_at_zero_distance(
        self, speed_planner: APFPlanner
    ) -> None:
        """Obstacle at distance 0 -> minimum maneuver speed (15.0)."""
        speed = speed_planner.compute_speed(0.0, State.NAVIGATE)
        assert speed == pytest.approx(15.0)

    def test_stop_state_same_as_navigate(
        self, speed_planner: APFPlanner
    ) -> None:
        """STOP state should not reduce speed (collision response handles safety)."""
        speed_stop = speed_planner.compute_speed(float("inf"), State.STOP)
        speed_nav = speed_planner.compute_speed(float("inf"), State.NAVIGATE)
        assert speed_stop == pytest.approx(speed_nav)

    def test_stop_state_never_zero(self, speed_planner: APFPlanner) -> None:
        """STOP state near obstacle should still yield >= 15.0."""
        speed = speed_planner.compute_speed(1.0, State.STOP)
        assert speed >= 15.0

    def test_all_states_same_speed(
        self, speed_planner: APFPlanner
    ) -> None:
        """All states should yield same speed at same distance."""
        speed_nav = speed_planner.compute_speed(30.0, State.NAVIGATE)
        speed_avoid = speed_planner.compute_speed(30.0, State.AVOID)
        speed_stop = speed_planner.compute_speed(30.0, State.STOP)
        assert speed_nav == pytest.approx(speed_avoid)
        assert speed_nav == pytest.approx(speed_stop)

    def test_negative_distance_clamped(
        self, speed_planner: APFPlanner
    ) -> None:
        """Negative distance (shouldn't happen) clamps to 0."""
        speed = speed_planner.compute_speed(-5.0, State.NAVIGATE)
        assert speed >= 15.0

    def test_default_params_backward_compat(self) -> None:
        """Planner without explicit speed params uses defaults."""
        p = APFPlanner(
            k_att=1.0, k_rep=100.0, influence_range=100.0,
            goal_tolerance=15.0, max_speed=60.0,
            escape_threshold=2.0, escape_window=30, escape_force=50.0,
        )
        assert p.slow_down_distance == 60.0
        assert p.cruise_fraction == 0.5
        # compute_speed still works
        speed = p.compute_speed(float("inf"), State.NAVIGATE)
        assert speed == pytest.approx(30.0)  # 60 * 0.5

    def test_from_config_reads_new_params(self) -> None:
        """from_config reads slow_down, cruise, vortex, smoothing, symmetry."""
        config = Mock()
        config.planner.k_att = 20.0
        config.planner.k_rep = 2000.0
        config.planner.influence_range = 100.0
        config.planner.goal_tolerance = 15.0
        config.planner.max_speed = 80.0
        config.planner.escape_threshold = 2.0
        config.planner.escape_window = 30
        config.planner.escape_force = 80.0
        config.planner.slow_down_distance = 50.0
        config.planner.cruise_fraction = 0.6
        config.planner.vortex_weight = 0.4
        config.planner.force_smoothing = 0.5
        config.planner.symmetry_threshold = 0.25
        config.planner.symmetry_nudge_force = 10.0
        config.planner.forward_half_angle = 1.57
        config.planner.rear_attenuation = 0.0
        config.planner.attenuation_power = 1.0
        config.planner.commitment_duration = 0
        config.planner.commitment_force = 0.0

        p = APFPlanner.from_config(config)
        assert p.slow_down_distance == 50.0
        assert p.cruise_fraction == 0.6
        assert p.vortex_weight == pytest.approx(0.4)
        assert p.force_smoothing == pytest.approx(0.5)
        assert p.symmetry_threshold == pytest.approx(0.25)
        assert p.symmetry_nudge_force == pytest.approx(10.0)

    def test_exact_slow_down_boundary(
        self, speed_planner: APFPlanner
    ) -> None:
        """Obstacle at exactly slow_down_distance -> full cruise speed."""
        speed = speed_planner.compute_speed(60.0, State.NAVIGATE)
        assert speed == pytest.approx(40.0)  # ratio = 60/60 = 1.0


# --- Tests: Vortex Field ---


class TestVortexField:
    """Tests for the vortex field component in repulsive force."""

    @pytest.fixture
    def vortex_planner(self) -> APFPlanner:
        """Planner with vortex_weight=1.0 for pure tangential testing."""
        return APFPlanner(
            k_att=20.0,
            k_rep=2000.0,
            influence_range=100.0,
            goal_tolerance=15.0,
            max_speed=80.0,
            escape_threshold=2.0,
            escape_window=30,
            escape_force=80.0,
            vortex_weight=1.0,
        )

    @pytest.fixture
    def blended_planner(self) -> APFPlanner:
        """Planner with vortex_weight=0.5 for blended testing."""
        return APFPlanner(
            k_att=20.0,
            k_rep=2000.0,
            influence_range=100.0,
            goal_tolerance=15.0,
            max_speed=80.0,
            escape_threshold=2.0,
            escape_window=30,
            escape_force=80.0,
            vortex_weight=0.5,
        )

    def test_zero_weight_is_standard_apf(self, planner: APFPlanner) -> None:
        """vortex_weight=0 should produce identical result to standard APF."""
        reading = SensorReading(
            angle=0.0, distance=30.0, hit=True, hit_point=(30.0, 0.0)
        )
        f = planner.repulsive_force(0.0, 0.0, reading, gx=100.0, gy=0.0)
        # Standard: pushes west (away from obstacle at east)
        assert f.fx < 0
        assert f.fy == pytest.approx(0.0, abs=1e-10)

    def test_pure_vortex_perpendicular(
        self, vortex_planner: APFPlanner
    ) -> None:
        """vortex_weight=1.0 should produce purely tangential force."""
        # Obstacle east, goal east — tangential should be perpendicular to radial
        reading = SensorReading(
            angle=0.0, distance=30.0, hit=True, hit_point=(30.0, 0.0)
        )
        f = vortex_planner.repulsive_force(0.0, 0.0, reading, gx=100.0, gy=0.0)
        # Radial is west (fx<0, fy=0). Tangential is north or south.
        # With pure vortex, fx should be ~0 (no radial component)
        assert abs(f.fx) < abs(f.fy) or abs(f.fx) < 1e-6

    def test_vortex_selects_goal_side(
        self, vortex_planner: APFPlanner
    ) -> None:
        """Vortex should rotate toward the goal side."""
        reading = SensorReading(
            angle=0.0, distance=30.0, hit=True, hit_point=(30.0, 0.0)
        )
        # Goal to the north-east: tangential should push north
        f_north = vortex_planner.repulsive_force(
            0.0, 0.0, reading, gx=100.0, gy=100.0
        )
        # Goal to the south-east: tangential should push south
        f_south = vortex_planner.repulsive_force(
            0.0, 0.0, reading, gx=100.0, gy=-100.0
        )
        assert f_north.fy > 0  # pushed north
        assert f_south.fy < 0  # pushed south

    def test_blended_has_both_components(
        self, blended_planner: APFPlanner
    ) -> None:
        """vortex_weight=0.5 should have both radial and tangential."""
        reading = SensorReading(
            angle=0.0, distance=30.0, hit=True, hit_point=(30.0, 0.0)
        )
        f = blended_planner.repulsive_force(
            0.0, 0.0, reading, gx=100.0, gy=100.0
        )
        # Should have both fx (radial, pushing west) and fy (tangential)
        assert f.fx < 0  # radial component still present
        assert f.fy != pytest.approx(0.0, abs=1e-6)  # tangential present

    def test_no_hit_returns_zero(self, vortex_planner: APFPlanner) -> None:
        """No hit should return zero force regardless of vortex_weight."""
        reading = SensorReading(angle=0.0, distance=150.0, hit=False)
        f = vortex_planner.repulsive_force(
            0.0, 0.0, reading, gx=100.0, gy=0.0
        )
        assert f.magnitude == pytest.approx(0.0)

    def test_beyond_influence_returns_zero(
        self, vortex_planner: APFPlanner
    ) -> None:
        """Beyond influence range should return zero with vortex."""
        reading = SensorReading(
            angle=0.0, distance=150.0, hit=True, hit_point=(150.0, 0.0)
        )
        f = vortex_planner.repulsive_force(
            0.0, 0.0, reading, gx=100.0, gy=0.0
        )
        assert f.magnitude == pytest.approx(0.0)

    def test_vortex_magnitude_preserved(
        self, vortex_planner: APFPlanner
    ) -> None:
        """Vortex force should have same magnitude as standard repulsive."""
        reading = SensorReading(
            angle=0.0, distance=30.0, hit=True, hit_point=(30.0, 0.0)
        )
        # Standard planner (vortex=0)
        std_planner = APFPlanner(
            k_att=20.0, k_rep=2000.0, influence_range=100.0,
            goal_tolerance=15.0, max_speed=80.0,
            escape_threshold=2.0, escape_window=30, escape_force=80.0,
            vortex_weight=0.0,
        )
        f_std = std_planner.repulsive_force(0.0, 0.0, reading, gx=100.0, gy=0.0)
        f_vortex = vortex_planner.repulsive_force(
            0.0, 0.0, reading, gx=100.0, gy=0.0
        )
        # Pure tangential has same magnitude as pure radial
        assert f_vortex.magnitude == pytest.approx(f_std.magnitude, rel=0.01)

    def test_backward_compat_default_zero(self) -> None:
        """Default vortex_weight should be 0.0."""
        p = APFPlanner(
            k_att=1.0, k_rep=100.0, influence_range=100.0,
            goal_tolerance=15.0, max_speed=60.0,
            escape_threshold=2.0, escape_window=30, escape_force=50.0,
        )
        assert p.vortex_weight == 0.0

    def test_vortex_weight_clamped(self) -> None:
        """vortex_weight should be clamped to [0, 1]."""
        p = APFPlanner(
            k_att=1.0, k_rep=100.0, influence_range=100.0,
            goal_tolerance=15.0, max_speed=60.0,
            escape_threshold=2.0, escape_window=30, escape_force=50.0,
            vortex_weight=1.5,
        )
        assert p.vortex_weight == 1.0

        p2 = APFPlanner(
            k_att=1.0, k_rep=100.0, influence_range=100.0,
            goal_tolerance=15.0, max_speed=60.0,
            escape_threshold=2.0, escape_window=30, escape_force=50.0,
            vortex_weight=-0.5,
        )
        assert p2.vortex_weight == 0.0

    def test_compute_passes_goal_to_repulsive(
        self, blended_planner: APFPlanner
    ) -> None:
        """compute() should pass gx, gy through to repulsive_force."""
        readings = [
            SensorReading(
                angle=0.0, distance=30.0, hit=True, hit_point=(30.0, 0.0)
            ),
        ]
        # Goal to the north — repulsive forces should have tangential component
        result = blended_planner.compute(0.0, 0.0, 100.0, 100.0, readings)
        rep_forces = [f for f in result["forces"] if f.source == "repulsive"]
        assert len(rep_forces) == 1
        # With goal at (100, 100), tangential should push northward
        assert rep_forces[0].fy > 0

    def test_obstacle_north_goal_east_pushes_east(
        self, vortex_planner: APFPlanner
    ) -> None:
        """Obstacle to north, goal to east: tangential should push east."""
        reading = SensorReading(
            angle=math.pi / 2, distance=30.0, hit=True,
            hit_point=(0.0, 30.0),
        )
        f = vortex_planner.repulsive_force(
            0.0, 0.0, reading, gx=100.0, gy=0.0
        )
        # Radial is south (away from north obstacle). Tangential rotations:
        # CW: (0, -1) -> (-1, 0) i.e. west   CCW: (0, -1) -> (1, 0) i.e. east
        # Goal is east, so CCW should be selected
        assert f.fx > 0  # pushed east


# --- Tests: Force Smoothing (EMA) ---


class TestForceSmoothing:
    """Tests for EMA temporal smoothing of force vectors."""

    def _make_planner(self, alpha: float) -> APFPlanner:
        return APFPlanner(
            k_att=1.0,
            k_rep=100.0,
            influence_range=100.0,
            goal_tolerance=15.0,
            max_speed=200.0,  # high cap to avoid capping interference
            escape_threshold=0.0,
            escape_window=30,
            escape_force=0.0,
            force_smoothing=alpha,
            symmetry_nudge_force=0.0,  # disable symmetry to isolate EMA
        )

    def test_first_call_unsmoothed(self) -> None:
        """First compute() call returns exact force sum (no previous to blend)."""
        p = self._make_planner(alpha=0.5)
        no_hit = [SensorReading(angle=0.0, distance=150.0, hit=False)]
        result = p.compute(0.0, 0.0, 100.0, 0.0, no_hit)
        # Only attractive force: k_att=1.0 unit vector toward (100,0)
        assert result["total_force"].fx == pytest.approx(1.0, abs=1e-6)
        assert result["total_force"].fy == pytest.approx(0.0, abs=1e-6)

    def test_second_call_blended(self) -> None:
        """Second compute() blends current with previous via EMA alpha."""
        p = self._make_planner(alpha=0.5)
        no_hit = [SensorReading(angle=0.0, distance=150.0, hit=False)]

        # First call: goal east -> force (1, 0)
        p.compute(0.0, 0.0, 100.0, 0.0, no_hit)

        # Second call: goal north -> raw force (0, 1)
        # EMA: 0.5 * (0, 1) + 0.5 * (1, 0) = (0.5, 0.5)
        result = p.compute(0.0, 0.0, 0.0, 100.0, no_hit)
        assert result["total_force"].fx == pytest.approx(0.5, abs=1e-6)
        assert result["total_force"].fy == pytest.approx(0.5, abs=1e-6)

    def test_convergence_with_constant_input(self) -> None:
        """Repeated same input converges to that input."""
        p = self._make_planner(alpha=0.3)
        no_hit = [SensorReading(angle=0.0, distance=150.0, hit=False)]

        # Repeat 50 calls with same goal
        for _ in range(50):
            result = p.compute(0.0, 0.0, 100.0, 0.0, no_hit)

        assert result["total_force"].fx == pytest.approx(1.0, abs=0.01)
        assert result["total_force"].fy == pytest.approx(0.0, abs=0.01)

    def test_smoothing_disabled_when_alpha_1(self) -> None:
        """alpha=1.0 means no blending — current force passes through."""
        p = self._make_planner(alpha=1.0)
        no_hit = [SensorReading(angle=0.0, distance=150.0, hit=False)]

        # First: goal east
        p.compute(0.0, 0.0, 100.0, 0.0, no_hit)
        # Second: goal north -> should be pure (0, 1)
        result = p.compute(0.0, 0.0, 0.0, 100.0, no_hit)
        assert result["total_force"].fx == pytest.approx(0.0, abs=1e-6)
        assert result["total_force"].fy == pytest.approx(1.0, abs=1e-6)

    def test_smoothing_maximum_when_alpha_near_0(self) -> None:
        """alpha near 0 means old force persists strongly."""
        p = self._make_planner(alpha=0.01)
        no_hit = [SensorReading(angle=0.0, distance=150.0, hit=False)]

        # First: goal east -> (1, 0)
        p.compute(0.0, 0.0, 100.0, 0.0, no_hit)
        # Second: goal north -> raw (0, 1), but blended mostly with (1, 0)
        result = p.compute(0.0, 0.0, 0.0, 100.0, no_hit)
        # 0.01*(0,1) + 0.99*(1,0) = (0.99, 0.01)
        assert result["total_force"].fx == pytest.approx(0.99, abs=0.01)
        assert result["total_force"].fy == pytest.approx(0.01, abs=0.01)

    def test_adaptive_stronger_near_obstacles(self) -> None:
        """Close obstacle triggers stronger smoothing (lower effective alpha)."""
        # Two planners with same alpha=0.5, but different obstacle proximity
        p_close = self._make_planner(alpha=0.5)
        p_clear = self._make_planner(alpha=0.5)
        # Override slow_down_distance for predictability
        p_close.slow_down_distance = 60.0
        p_clear.slow_down_distance = 60.0

        # Prime both with goal east
        no_hit = [SensorReading(angle=0.0, distance=150.0, hit=False)]
        p_close.compute(0.0, 0.0, 100.0, 0.0, no_hit)
        p_clear.compute(0.0, 0.0, 100.0, 0.0, no_hit)

        # Second call: goal north, but p_close has a close obstacle (to the side)
        close_hit = [SensorReading(
            angle=math.pi / 4, distance=10.0, hit=True,
            hit_point=(7.07, 7.07),
        )]
        r_close = p_close.compute(0.0, 0.0, 0.0, 100.0, close_hit)
        r_clear = p_clear.compute(0.0, 0.0, 0.0, 100.0, no_hit)

        # Close obstacle -> lower effective alpha -> more of previous force retained
        # Previous was (1, 0), so close should retain more fx than clear
        # (repulsive force pushes away from obstacle but EMA dominates)
        # Use absolute fx to account for repulsive contribution
        assert abs(r_close["total_force"].fx) > abs(r_clear["total_force"].fx)

    def test_adaptive_floor_at_zero_distance(self) -> None:
        """Obstacle at d~0 floors alpha at 0.05, not zero."""
        p = self._make_planner(alpha=0.5)
        p.slow_down_distance = 60.0

        # Prime with goal east
        no_hit = [SensorReading(angle=0.0, distance=150.0, hit=False)]
        p.compute(0.0, 0.0, 100.0, 0.0, no_hit)

        # Second call with obstacle at ~0 distance (ahead)
        close_hit = [SensorReading(
            angle=0.0, distance=0.5, hit=True, hit_point=(0.5, 0.0),
        )]
        result = p.compute(0.0, 0.0, 0.0, 100.0, close_hit)

        # Alpha floored at 0.05: still allows 5% of new force through
        # So the result should NOT be identical to the previous force
        # Previous was ~(1, 0); new raw has fy component (goal north)
        assert result["total_force"].fy != pytest.approx(0.0, abs=1e-3)

    def test_default_smoothing_backward_compat(self) -> None:
        """Default constructor uses force_smoothing=0.3."""
        p = APFPlanner(
            k_att=1.0, k_rep=100.0, influence_range=100.0,
            goal_tolerance=15.0, max_speed=60.0,
            escape_threshold=2.0, escape_window=30, escape_force=50.0,
        )
        assert p.force_smoothing == pytest.approx(0.3)


# --- Tests: Symmetry Breaking ---


class TestSymmetryBreaking:
    """Tests for the deterministic symmetry-breaking nudge."""

    def _make_planner(
        self, nudge_force: float = 15.0, threshold: float = 0.3
    ) -> APFPlanner:
        return APFPlanner(
            k_att=1.0,
            k_rep=100.0,
            influence_range=100.0,
            goal_tolerance=15.0,
            max_speed=500.0,  # high cap to avoid capping interference
            escape_threshold=0.0,
            escape_window=30,
            escape_force=0.0,
            force_smoothing=1.0,  # disable EMA to isolate symmetry
            symmetry_threshold=threshold,
            symmetry_nudge_force=nudge_force,
        )

    def _make_trapped_planner(
        self, nudge_force: float = 15.0, threshold: float = 0.3
    ) -> APFPlanner:
        """Planner with balanced gains where a symmetric trap triggers nudge."""
        return APFPlanner(
            k_att=50.0,
            k_rep=50.0,
            influence_range=100.0,
            goal_tolerance=15.0,
            max_speed=500.0,
            escape_threshold=0.0,
            escape_window=30,
            escape_force=0.0,
            force_smoothing=1.0,
            symmetry_threshold=threshold,
            symmetry_nudge_force=nudge_force,
            slow_down_distance=60.0,
        )

    def _symmetric_readings(self) -> List[SensorReading]:
        """Two symmetric obstacles at equal distances, opposite sides."""
        return [
            SensorReading(
                angle=math.pi / 2, distance=30.0, hit=True,
                hit_point=(0.0, 30.0),
            ),
            SensorReading(
                angle=-math.pi / 2, distance=30.0, hit=True,
                hit_point=(0.0, -30.0),
            ),
        ]

    def _trapped_readings(self) -> List[SensorReading]:
        """Symmetric side obstacles + ahead obstacle creating a weak-force trap."""
        return [
            SensorReading(
                angle=math.pi / 2, distance=30.0, hit=True,
                hit_point=(0.0, 30.0),
            ),
            SensorReading(
                angle=-math.pi / 2, distance=30.0, hit=True,
                hit_point=(0.0, -30.0),
            ),
            SensorReading(
                angle=0.0, distance=42.0, hit=True,
                hit_point=(42.0, 0.0),
            ),
        ]

    def test_symmetric_readings_get_nudge(self) -> None:
        """Symmetric obstacles in a trap produce a symmetry_break force."""
        p = self._make_trapped_planner()
        result = p.compute(0.0, 0.0, 100.0, 0.0, self._trapped_readings())
        sources = [f.source for f in result["forces"]]
        assert "symmetry_break" in sources

    def test_asymmetric_readings_no_nudge(self) -> None:
        """Single-side obstacle does not trigger symmetry nudge."""
        p = self._make_planner()
        readings = [
            SensorReading(
                angle=math.pi / 4, distance=30.0, hit=True,
                hit_point=(21.2, 21.2),
            ),
        ]
        result = p.compute(0.0, 0.0, 100.0, 0.0, readings)
        sources = [f.source for f in result["forces"]]
        assert "symmetry_break" not in sources

    def test_no_repulsive_no_nudge(self) -> None:
        """Clear readings produce no symmetry nudge."""
        p = self._make_planner()
        no_hit = [SensorReading(angle=0.0, distance=150.0, hit=False)]
        result = p.compute(0.0, 0.0, 100.0, 0.0, no_hit)
        sources = [f.source for f in result["forces"]]
        assert "symmetry_break" not in sources

    def test_nudge_direction_consistent(self) -> None:
        """Same goal direction produces same nudge direction (deterministic)."""
        p1 = self._make_trapped_planner()
        p2 = self._make_trapped_planner()
        readings = self._trapped_readings()

        r1 = p1.compute(0.0, 0.0, 100.0, 0.0, readings)
        r2 = p2.compute(0.0, 0.0, 100.0, 0.0, readings)

        nudge1 = [f for f in r1["forces"] if f.source == "symmetry_break"][0]
        nudge2 = [f for f in r2["forces"] if f.source == "symmetry_break"][0]
        assert nudge1.fx == pytest.approx(nudge2.fx)
        assert nudge1.fy == pytest.approx(nudge2.fy)

    def test_nudge_perpendicular_to_goal(self) -> None:
        """Nudge is CW perpendicular to goal direction."""
        p = self._make_trapped_planner(nudge_force=10.0)
        readings = self._trapped_readings()

        # Goal east: direction (1, 0). CW perp = (0, -1) * 10
        result = p.compute(0.0, 0.0, 100.0, 0.0, readings)
        nudge = [f for f in result["forces"] if f.source == "symmetry_break"][0]
        assert nudge.fx == pytest.approx(0.0, abs=1e-6)
        assert nudge.fy == pytest.approx(-10.0, abs=1e-6)

    def test_symmetry_disabled_when_nudge_force_zero(self) -> None:
        """symmetry_nudge_force=0 disables the nudge entirely."""
        p = self._make_planner(nudge_force=0.0)
        result = p.compute(0.0, 0.0, 100.0, 0.0, self._symmetric_readings())
        sources = [f.source for f in result["forces"]]
        assert "symmetry_break" not in sources

    def test_backward_compat_defaults(self) -> None:
        """Default constructor has correct symmetry defaults."""
        p = APFPlanner(
            k_att=1.0, k_rep=100.0, influence_range=100.0,
            goal_tolerance=15.0, max_speed=60.0,
            escape_threshold=2.0, escape_window=30, escape_force=50.0,
        )
        assert p.symmetry_threshold == 0.3
        assert p.symmetry_nudge_force == 15.0

    def test_no_nudge_when_total_force_strong(self) -> None:
        """No nudge when total force is strong relative to attraction (gate A)."""
        p = self._make_planner()
        # Perpendicular symmetric obstacles: repulsive forces cancel,
        # leaving attractive force at full strength (raw_mag/att_mag = 1.0)
        result = p.compute(0.0, 0.0, 100.0, 0.0, self._symmetric_readings())
        sources = [f.source for f in result["forces"]]
        assert "symmetry_break" not in sources

    def test_no_nudge_when_obstacles_far(self) -> None:
        """No nudge when obstacles are beyond slow-down distance (gate B)."""
        p = APFPlanner(
            k_att=10.0,
            k_rep=50.0,
            influence_range=200.0,
            goal_tolerance=15.0,
            max_speed=500.0,
            escape_threshold=0.0,
            escape_window=30,
            escape_force=0.0,
            force_smoothing=1.0,
            symmetry_threshold=0.3,
            symmetry_nudge_force=15.0,
            slow_down_distance=60.0,
        )
        readings = [
            SensorReading(
                angle=math.pi / 2, distance=80.0, hit=True,
                hit_point=(0.0, 80.0),
            ),
            SensorReading(
                angle=-math.pi / 2, distance=80.0, hit=True,
                hit_point=(0.0, -80.0),
            ),
            SensorReading(
                angle=0.0, distance=150.0, hit=True,
                hit_point=(150.0, 0.0),
            ),
        ]
        result = p.compute(0.0, 0.0, 300.0, 0.0, readings)
        sources = [f.source for f in result["forces"]]
        assert "symmetry_break" not in sources

    def test_nudge_fires_when_both_conditions_met(self) -> None:
        """Nudge fires when force is weak AND obstacles are near."""
        p = self._make_trapped_planner()
        result = p.compute(0.0, 0.0, 100.0, 0.0, self._trapped_readings())
        sources = [f.source for f in result["forces"]]
        assert "symmetry_break" in sources
        nudge = [f for f in result["forces"] if f.source == "symmetry_break"][0]
        assert nudge.magnitude == pytest.approx(15.0, abs=0.1)


# --- Tests: Directional Attenuation ---


class TestDirectionalAttenuation:
    """Tests for directional attenuation of repulsive forces."""

    @pytest.fixture
    def dir_planner(self) -> APFPlanner:
        """Planner with directional attenuation enabled."""
        return APFPlanner(
            k_att=1.0,
            k_rep=100.0,
            influence_range=100.0,
            goal_tolerance=15.0,
            max_speed=60.0,
            escape_threshold=2.0,
            escape_window=30,
            escape_force=50.0,
            forward_half_angle=math.pi / 2,
            rear_attenuation=0.0,
        )

    def test_full_force_in_forward_cone(self, dir_planner: APFPlanner) -> None:
        """Obstacle within forward_half_angle gets full repulsive force."""
        reading = SensorReading(
            angle=0.3, distance=30.0, hit=True, hit_point=(30.0, 0.0)
        )
        # Agent heading = 0, reading angle = 0.3 (within pi/2)
        f = dir_planner.repulsive_force(
            0.0, 0.0, reading, agent_heading=0.0
        )
        # Compare with no-attenuation scenario: same planner, same reading
        # Forward cone -> attenuation = 1.0, so magnitude should match baseline
        baseline = dir_planner.k_rep * (1.0 - 30.0 / 100.0)
        assert f.magnitude == pytest.approx(baseline, rel=0.01)

    def test_attenuated_behind(self, dir_planner: APFPlanner) -> None:
        """Obstacle directly behind agent gets near-zero force."""
        reading = SensorReading(
            angle=math.pi, distance=30.0, hit=True, hit_point=(-30.0, 0.0)
        )
        # Agent heading = 0, reading angle = pi (directly behind)
        f = dir_planner.repulsive_force(
            0.0, 0.0, reading, agent_heading=0.0
        )
        # rear_attenuation = 0.0, so force should be ~0
        assert f.magnitude == pytest.approx(0.0, abs=1e-6)

    def test_smooth_taper_side(self, dir_planner: APFPlanner) -> None:
        """Obstacle at 90° (boundary) gets full force; beyond 90° tapers."""
        reading_at_boundary = SensorReading(
            angle=math.pi / 2, distance=30.0, hit=True,
            hit_point=(0.0, 30.0),
        )
        reading_beyond = SensorReading(
            angle=3 * math.pi / 4, distance=30.0, hit=True,
            hit_point=(-21.2, 21.2),
        )
        f_boundary = dir_planner.repulsive_force(
            0.0, 0.0, reading_at_boundary, agent_heading=0.0
        )
        f_beyond = dir_planner.repulsive_force(
            0.0, 0.0, reading_beyond, agent_heading=0.0
        )
        # At boundary (exactly forward_half_angle): full force
        baseline = dir_planner.k_rep * (1.0 - 30.0 / 100.0)
        assert f_boundary.magnitude == pytest.approx(baseline, rel=0.01)
        # Beyond 90° with rear_attenuation=0.0: force is zero
        assert f_beyond.magnitude == pytest.approx(0.0, abs=1e-6)

    def test_backward_compat_default_heading(
        self, dir_planner: APFPlanner
    ) -> None:
        """Default agent_heading=0.0 preserves backward compatibility."""
        reading = SensorReading(
            angle=0.0, distance=30.0, hit=True, hit_point=(30.0, 0.0)
        )
        # Calling without agent_heading should default to 0.0
        f = dir_planner.repulsive_force(0.0, 0.0, reading)
        assert f.magnitude > 0

    def test_forward_hemisphere_min_dist(self) -> None:
        """compute() min_obs_dist only considers forward hemisphere."""
        p = APFPlanner(
            k_att=1.0,
            k_rep=100.0,
            influence_range=100.0,
            goal_tolerance=15.0,
            max_speed=500.0,
            escape_threshold=0.0,
            escape_window=30,
            escape_force=0.0,
            force_smoothing=1.0,
            symmetry_nudge_force=0.0,
            forward_half_angle=math.pi / 2,
            rear_attenuation=0.0,
        )
        # Obstacle behind (close) and one ahead (far)
        readings = [
            SensorReading(
                angle=math.pi, distance=10.0, hit=True,
                hit_point=(-10.0, 0.0),
            ),
            SensorReading(
                angle=0.0, distance=80.0, hit=True,
                hit_point=(80.0, 0.0),
            ),
        ]
        # With agent_heading=0, the behind obstacle should be excluded
        # from min_obs_dist. The adaptive smoothing should use d=80.
        # Prime with one call, then check second call behavior
        result = p.compute(0.0, 0.0, 100.0, 0.0, readings, agent_heading=0.0)
        # The key point: rear obstacle force should be attenuated to ~0
        rep_forces = [f for f in result["forces"] if f.source == "repulsive"]
        # At least one repulsive force should have significant magnitude (ahead)
        ahead_force = [f for f in rep_forces if f.magnitude > 1.0]
        assert len(ahead_force) >= 1

    def test_from_config_reads_directional_params(self) -> None:
        """from_config reads forward_half_angle and rear_attenuation."""
        config = Mock()
        config.planner.k_att = 1.0
        config.planner.k_rep = 100.0
        config.planner.influence_range = 100.0
        config.planner.goal_tolerance = 15.0
        config.planner.max_speed = 60.0
        config.planner.escape_threshold = 2.0
        config.planner.escape_window = 30
        config.planner.escape_force = 50.0
        config.planner.slow_down_distance = 60.0
        config.planner.cruise_fraction = 0.5
        config.planner.vortex_weight = 0.0
        config.planner.force_smoothing = 0.3
        config.planner.symmetry_threshold = 0.3
        config.planner.symmetry_nudge_force = 15.0
        config.planner.forward_half_angle = 1.2
        config.planner.rear_attenuation = 0.1
        config.planner.attenuation_power = 3.0
        config.planner.commitment_duration = 10
        config.planner.commitment_force = 25.0

        p = APFPlanner.from_config(config)
        assert p.forward_half_angle == pytest.approx(1.2)
        assert p.rear_attenuation == pytest.approx(0.1)
        assert p.attenuation_power == pytest.approx(3.0)
        assert p.commitment_duration == 10
        assert p.commitment_force == pytest.approx(25.0)

    @pytest.fixture
    def steep_planner(self) -> APFPlanner:
        """Planner with steep power-cosine attenuation."""
        return APFPlanner(
            k_att=1.0,
            k_rep=100.0,
            influence_range=100.0,
            goal_tolerance=15.0,
            max_speed=60.0,
            escape_threshold=2.0,
            escape_window=30,
            escape_force=50.0,
            forward_half_angle=math.pi / 3,  # 60°
            rear_attenuation=0.1,
            attenuation_power=2.0,
        )

    def test_steep_attenuation_at_75_degrees(self, steep_planner: APFPlanner) -> None:
        """At 75° (halfway through taper zone), force is significantly reduced."""
        reading = SensorReading(
            angle=math.radians(75), distance=30.0, hit=True,
            hit_point=(
                30.0 * math.cos(math.radians(75)),
                30.0 * math.sin(math.radians(75)),
            ),
        )
        f = steep_planner.repulsive_force(0.0, 0.0, reading, agent_heading=0.0)
        baseline = steep_planner.k_rep * (1.0 - 30.0 / 100.0)
        # cos^2 at midpoint of 60°->90° should give ~0.5 attenuation
        assert f.magnitude < baseline * 0.6
        assert f.magnitude > baseline * 0.3

    def test_steep_attenuation_at_90_degrees(self, steep_planner: APFPlanner) -> None:
        """At 90° (edge of taper zone), force drops to rear_attenuation level."""
        reading = SensorReading(
            angle=math.pi / 2, distance=30.0, hit=True,
            hit_point=(0.0, 30.0),
        )
        f = steep_planner.repulsive_force(0.0, 0.0, reading, agent_heading=0.0)
        baseline = steep_planner.k_rep * (1.0 - 30.0 / 100.0)
        # At 90° with power=2, cos^2(pi/2) = 0, so attenuation = rear_attenuation = 0.1
        assert f.magnitude == pytest.approx(baseline * 0.1, rel=0.15)

    def test_steep_attenuation_at_120_degrees(self, steep_planner: APFPlanner) -> None:
        """Beyond 90°, force is capped at rear_attenuation."""
        reading = SensorReading(
            angle=math.radians(120), distance=30.0, hit=True,
            hit_point=(
                30.0 * math.cos(math.radians(120)),
                30.0 * math.sin(math.radians(120)),
            ),
        )
        f = steep_planner.repulsive_force(0.0, 0.0, reading, agent_heading=0.0)
        baseline = steep_planner.k_rep * (1.0 - 30.0 / 100.0)
        assert f.magnitude == pytest.approx(baseline * 0.1, rel=0.15)

    def test_steep_forward_cone_unchanged(self, steep_planner: APFPlanner) -> None:
        """Within forward cone, attenuation_power has no effect (still 1.0)."""
        reading = SensorReading(
            angle=0.3, distance=30.0, hit=True, hit_point=(30.0, 0.0),
        )
        f = steep_planner.repulsive_force(0.0, 0.0, reading, agent_heading=0.0)
        baseline = steep_planner.k_rep * (1.0 - 30.0 / 100.0)
        assert f.magnitude == pytest.approx(baseline, rel=0.01)


# --- Tests: Gap-Finding + Commitment ---


class TestGapFindingCommitment:
    """Tests for gap-finding and commitment bias mechanism."""

    @pytest.fixture
    def gap_planner(self) -> APFPlanner:
        """Planner with commitment enabled."""
        return APFPlanner(
            k_att=20.0,
            k_rep=80.0,
            influence_range=120.0,
            goal_tolerance=25.0,
            max_speed=80.0,
            escape_threshold=2.0,
            escape_window=30,
            escape_force=50.0,
            slow_down_distance=40.0,
            vortex_weight=0.7,
            force_smoothing=1.0,  # no EMA for test clarity
            symmetry_nudge_force=40.0,
            forward_half_angle=math.pi / 3,
            rear_attenuation=0.1,
            commitment_duration=15,
            commitment_force=30.0,
        )

    def _make_readings_with_gap(
        self, gap_center: float, gap_width: float, num_rays: int = 36
    ) -> list:
        """Create sensor readings with a clear gap at given angle."""
        readings = []
        for i in range(num_rays):
            angle = -math.pi + (2 * math.pi * i / num_rays)
            # Check if this ray falls in the gap
            angle_diff = abs(normalize_angle(angle - gap_center))
            if angle_diff < gap_width / 2:
                # Clear ray (no hit)
                readings.append(SensorReading(
                    angle=angle, distance=250.0, hit=False, hit_point=None,
                ))
            else:
                # Hit at close range (obstacle)
                hx = 30.0 * math.cos(angle)
                hy = 30.0 * math.sin(angle)
                readings.append(SensorReading(
                    angle=angle, distance=30.0, hit=True, hit_point=(hx, hy),
                ))
        return readings

    def test_find_best_gap_symmetric_obstacle(self, gap_planner: APFPlanner) -> None:
        """With gaps on both sides, should pick the one more aligned with goal."""
        # Gap at +45° and -45°, goal is up-right
        readings = self._make_readings_with_gap(math.pi / 4, math.radians(40))
        gap = gap_planner._find_best_gap(readings, agent_heading=0.0, gx=100.0, gy=50.0)
        assert gap is not None
        # Gap toward +45° should be preferred (goal is up-right)
        assert abs(normalize_angle(gap - math.pi / 4)) < math.radians(30)

    def test_find_best_gap_no_gap(self, gap_planner: APFPlanner) -> None:
        """When all rays hit close obstacles, no gap found."""
        readings = []
        for i in range(36):
            angle = -math.pi + (2 * math.pi * i / 36)
            hx = 30.0 * math.cos(angle)
            hy = 30.0 * math.sin(angle)
            readings.append(SensorReading(
                angle=angle, distance=30.0, hit=True, hit_point=(hx, hy),
            ))
        gap = gap_planner._find_best_gap(readings, agent_heading=0.0, gx=100.0, gy=0.0)
        assert gap is None

    def test_find_best_gap_all_clear(self, gap_planner: APFPlanner) -> None:
        """When all rays are clear, no commitment needed — return None."""
        readings = []
        for i in range(36):
            angle = -math.pi + (2 * math.pi * i / 36)
            readings.append(SensorReading(
                angle=angle, distance=250.0, hit=False, hit_point=None,
            ))
        gap = gap_planner._find_best_gap(readings, agent_heading=0.0, gx=100.0, gy=0.0)
        assert gap is None

    def test_commitment_activates(self, gap_planner: APFPlanner) -> None:
        """Commitment should activate when obstacle is close in forward cone."""
        readings = self._make_readings_with_gap(math.pi / 4, math.radians(40))
        result = gap_planner.compute(
            0.0, 0.0, 100.0, 0.0, readings, agent_heading=0.0
        )
        # Should have a commitment force in the forces list
        commitment_forces = [
            f for f in result["forces"] if f.source == "commitment"
        ]
        assert len(commitment_forces) == 1
        assert gap_planner._commitment_steps_remaining > 0

    def test_commitment_persists(self, gap_planner: APFPlanner) -> None:
        """Commitment force should persist across multiple compute() calls."""
        readings = self._make_readings_with_gap(math.pi / 4, math.radians(40))
        # First call activates commitment
        gap_planner.compute(0.0, 0.0, 100.0, 0.0, readings, agent_heading=0.0)
        initial_remaining = gap_planner._commitment_steps_remaining

        # Second call: even with different readings, commitment should persist
        clear_readings = [
            SensorReading(angle=0.0, distance=250.0, hit=False, hit_point=None)
        ]
        gap_planner.compute(
            5.0, 0.0, 100.0, 0.0, clear_readings, agent_heading=0.0
        )
        assert gap_planner._commitment_steps_remaining == initial_remaining - 1

    def test_commitment_expires(self, gap_planner: APFPlanner) -> None:
        """After commitment_duration steps, commitment should expire."""
        readings = self._make_readings_with_gap(math.pi / 4, math.radians(40))
        gap_planner.compute(0.0, 0.0, 100.0, 0.0, readings, agent_heading=0.0)

        # Run through remaining steps with clear readings
        clear = [SensorReading(angle=0.0, distance=250.0, hit=False, hit_point=None)]
        for i in range(20):
            gap_planner.compute(
                float(i + 1), 0.0, 100.0, 0.0, clear, agent_heading=0.0
            )

        assert gap_planner._commitment_steps_remaining == 0
        # No commitment force in last result
        result = gap_planner.compute(25.0, 0.0, 100.0, 0.0, clear, agent_heading=0.0)
        commitment_forces = [
            f for f in result["forces"] if f.source == "commitment"
        ]
        assert len(commitment_forces) == 0

    def test_commitment_suppresses_symmetry_nudge(
        self, gap_planner: APFPlanner,
    ) -> None:
        """While committed, symmetry breaker should not fire."""
        readings = self._make_readings_with_gap(math.pi / 4, math.radians(40))
        gap_planner.compute(0.0, 0.0, 100.0, 0.0, readings, agent_heading=0.0)

        # Second call with symmetric obstacle scenario
        sym_readings = [
            SensorReading(angle=0.5, distance=30.0, hit=True, hit_point=(30.0, 0.0)),
            SensorReading(angle=-0.5, distance=30.0, hit=True, hit_point=(30.0, 0.0)),
        ]
        result = gap_planner.compute(
            0.0, 0.0, 100.0, 0.0, sym_readings, agent_heading=0.0
        )
        sym_forces = [f for f in result["forces"] if f.source == "symmetry_break"]
        assert len(sym_forces) == 0
