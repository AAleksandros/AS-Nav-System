"""Unit tests for the evaluation framework."""

import math

import pytest

from src.evaluation import (
    RunData,
    Evaluator,
    MetricsPlotter,
    print_metrics_table,
)
from src.models import State


# ---------------------------------------------------------------------------
# Helpers — synthetic RunData factories
# ---------------------------------------------------------------------------


def _straight_line_run(
    start: tuple = (0.0, 0.0),
    end: tuple = (100.0, 0.0),
    steps: int = 10,
    dt: float = 0.033,
    scenario: str = "test",
    collisions: int = 0,
) -> RunData:
    """Create a RunData for a straight-line path from start to end."""
    positions = []
    headings = []
    velocities = []
    for i in range(steps):
        t = i / max(steps - 1, 1)
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        positions.append((x, y))
        headings.append(math.atan2(end[1] - start[1], end[0] - start[0]))
        velocities.append(30.0)

    return RunData(
        scenario=scenario,
        positions=positions,
        headings=headings,
        velocities=velocities,
        min_obstacle_dists=[50.0] * steps,
        states=[State.NAVIGATE] * steps,
        collisions=collisions,
        waypoints_reached=1,
        total_waypoints=1,
        steps=steps,
        dt=dt,
        waypoint_positions=[end],
        start_position=start,
        obstacles=[(50.0, 50.0, 10.0)],
    )


def _detour_run() -> RunData:
    """Create a RunData with a detour (path > optimal)."""
    # Straight optimal: (0,0) -> (100,0) = 100
    # Actual path: goes via (50, 50) — longer
    positions = [(0.0, 0.0), (50.0, 50.0), (100.0, 0.0)]
    headings = [
        math.atan2(50, 50),  # toward (50,50)
        math.atan2(-50, 50),  # toward (100,0)
        math.atan2(-50, 50),
    ]
    return RunData(
        scenario="detour",
        positions=positions,
        headings=headings,
        velocities=[30.0, 30.0, 30.0],
        min_obstacle_dists=[40.0, 20.0, 60.0],
        states=[State.NAVIGATE, State.AVOID, State.NAVIGATE],
        collisions=0,
        waypoints_reached=1,
        total_waypoints=1,
        steps=3,
        dt=0.033,
        waypoint_positions=[(100.0, 0.0)],
        start_position=(0.0, 0.0),
        obstacles=[(50.0, 30.0, 15.0)],
    )


# ---------------------------------------------------------------------------
# RunData construction tests
# ---------------------------------------------------------------------------


class TestRunData:
    """Tests for RunData dataclass."""

    def test_construction(self):
        """RunData stores all fields correctly."""
        run = _straight_line_run()
        assert run.scenario == "test"
        assert len(run.positions) == 10
        assert run.collisions == 0
        assert run.dt == 0.033

    def test_field_access(self):
        """RunData fields are accessible as attributes."""
        run = _straight_line_run(collisions=2)
        assert run.collisions == 2
        assert run.total_waypoints == 1
        assert run.start_position == (0.0, 0.0)
        assert len(run.obstacles) == 1


# ---------------------------------------------------------------------------
# Evaluator.evaluate tests
# ---------------------------------------------------------------------------


class TestEvaluatorEvaluate:
    """Tests for Evaluator.evaluate()."""

    def test_straight_line_efficiency_is_one(self):
        """Straight-line path from start to waypoint has efficiency ~1.0."""
        run = _straight_line_run()
        m = Evaluator.evaluate(run)
        assert m.path_efficiency == pytest.approx(1.0, abs=0.01)

    def test_detour_efficiency_less_than_one(self):
        """Detour path has efficiency < 1.0."""
        run = _detour_run()
        m = Evaluator.evaluate(run)
        assert m.path_efficiency < 1.0
        assert m.path_efficiency > 0.0

    def test_smoothness_constant_heading(self):
        """Constant heading gives smoothness = 0."""
        run = _straight_line_run()
        m = Evaluator.evaluate(run)
        assert m.smoothness == pytest.approx(0.0, abs=1e-10)

    def test_smoothness_varying_heading(self):
        """Varying heading gives smoothness > 0."""
        run = _detour_run()
        m = Evaluator.evaluate(run)
        assert m.smoothness > 0.0

    def test_safety_margins(self):
        """Min and avg safety margins computed correctly."""
        run = _detour_run()
        m = Evaluator.evaluate(run)
        assert m.min_safety_margin == pytest.approx(20.0)
        assert m.avg_safety_margin == pytest.approx(40.0)

    def test_success_all_waypoints_no_collisions(self):
        """Success = True when all waypoints reached and 0 collisions."""
        run = _straight_line_run(collisions=0)
        m = Evaluator.evaluate(run)
        assert m.success is True

    def test_success_false_with_collisions(self):
        """Success = False when collisions > 0."""
        run = _straight_line_run(collisions=1)
        m = Evaluator.evaluate(run)
        assert m.success is False

    def test_success_false_missing_waypoints(self):
        """Success = False when not all waypoints reached."""
        run = _straight_line_run()
        run.waypoints_reached = 0
        m = Evaluator.evaluate(run)
        assert m.success is False

    def test_state_distribution(self):
        """State fractions computed correctly."""
        run = _detour_run()  # states: NAV, AVOID, NAV
        m = Evaluator.evaluate(run)
        assert m.time_in_navigate == pytest.approx(2 / 3, abs=0.01)
        assert m.time_in_avoid == pytest.approx(1 / 3, abs=0.01)

    def test_completion_time(self):
        """Completion time = steps * dt."""
        run = _straight_line_run(steps=100, dt=0.033)
        m = Evaluator.evaluate(run)
        assert m.completion_time == pytest.approx(100 * 0.033)

    def test_avg_speed(self):
        """Average speed computed correctly."""
        run = _straight_line_run()
        m = Evaluator.evaluate(run)
        assert m.avg_speed == pytest.approx(30.0)

    def test_path_length(self):
        """Path length for straight line equals euclidean distance."""
        run = _straight_line_run(start=(0.0, 0.0), end=(100.0, 0.0), steps=11)
        m = Evaluator.evaluate(run)
        assert m.path_length == pytest.approx(100.0, abs=0.1)

    def test_optimal_length(self):
        """Optimal length is start -> waypoints chain."""
        run = _straight_line_run(start=(0.0, 0.0), end=(100.0, 0.0))
        m = Evaluator.evaluate(run)
        assert m.optimal_length == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Evaluator.summary tests
# ---------------------------------------------------------------------------


class TestEvaluatorSummary:
    """Tests for Evaluator.summary()."""

    def test_summary_aggregates_multiple(self):
        """Summary computes mean/min/max across scenarios."""
        runs = [
            _straight_line_run(scenario="a"),
            _detour_run(),
        ]
        metrics = Evaluator.evaluate_batch(runs)
        s = Evaluator.summary(metrics)

        assert s["count"] == 2
        assert 0.0 <= s["success_rate"] <= 1.0
        assert s["path_efficiency"]["min"] <= s["path_efficiency"]["max"]
        assert s["smoothness"]["min"] <= s["smoothness"]["max"]

    def test_summary_empty_returns_empty_dict(self):
        """Summary of empty list returns {}."""
        assert Evaluator.summary([]) == {}

    def test_summary_single_scenario(self):
        """Summary of one scenario has mean == min == max."""
        run = _straight_line_run()
        metrics = [Evaluator.evaluate(run)]
        s = Evaluator.summary(metrics)

        assert s["path_efficiency"]["mean"] == s["path_efficiency"]["min"]
        assert s["path_efficiency"]["mean"] == s["path_efficiency"]["max"]


# ---------------------------------------------------------------------------
# MetricsPlotter tests (mock matplotlib)
# ---------------------------------------------------------------------------


class TestMetricsPlotter:
    """Tests for MetricsPlotter — verifies figures are created and saved."""

    def test_bar_chart_creates_file(self, tmp_path):
        """Bar chart saves a PNG file."""
        plotter = MetricsPlotter(output_dir=str(tmp_path))
        runs = [_straight_line_run(scenario="corridor"), _detour_run()]
        metrics = Evaluator.evaluate_batch(runs)

        path = plotter.plot_metrics_bar_chart(metrics)
        assert path is not None
        assert path.exists()
        assert path.suffix == ".png"

    def test_trajectory_creates_file(self, tmp_path):
        """Trajectory plot saves a PNG file."""
        plotter = MetricsPlotter(output_dir=str(tmp_path))
        run = _straight_line_run(scenario="corridor")
        m = Evaluator.evaluate(run)

        path = plotter.plot_trajectory(run, m)
        assert path is not None
        assert path.exists()
        assert "corridor" in path.name

    def test_radar_chart_creates_file(self, tmp_path):
        """Radar chart saves a PNG file."""
        plotter = MetricsPlotter(output_dir=str(tmp_path))
        runs = [_straight_line_run(scenario="a"), _detour_run()]
        metrics = Evaluator.evaluate_batch(runs)

        path = plotter.plot_radar_chart(metrics)
        assert path is not None
        assert path.exists()

    def test_empty_data_returns_none(self, tmp_path):
        """Plotter returns None for empty data."""
        plotter = MetricsPlotter(output_dir=str(tmp_path))
        assert plotter.plot_metrics_bar_chart([]) is None
        assert plotter.plot_radar_chart([]) is None

    def test_trajectory_empty_positions_returns_none(self, tmp_path):
        """Trajectory plot returns None for empty positions."""
        plotter = MetricsPlotter(output_dir=str(tmp_path))
        run = _straight_line_run()
        run.positions = []
        m = Evaluator.evaluate(run)
        assert plotter.plot_trajectory(run, m) is None


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for evaluation."""

    def test_single_step(self):
        """Single step run produces valid metrics."""
        run = RunData(
            scenario="single",
            positions=[(50.0, 50.0)],
            headings=[0.0],
            velocities=[10.0],
            min_obstacle_dists=[100.0],
            states=[State.NAVIGATE],
            collisions=0,
            waypoints_reached=0,
            total_waypoints=1,
            steps=1,
            dt=0.033,
            waypoint_positions=[(200.0, 200.0)],
            start_position=(50.0, 50.0),
        )
        m = Evaluator.evaluate(run)
        assert m.path_length == 0.0
        assert m.smoothness == 0.0
        assert m.completion_time == pytest.approx(0.033)

    def test_no_waypoints(self):
        """Run with no waypoints: success since 0/0 reached."""
        run = RunData(
            scenario="empty",
            positions=[(0.0, 0.0), (10.0, 0.0)],
            headings=[0.0, 0.0],
            velocities=[10.0, 10.0],
            min_obstacle_dists=[100.0, 100.0],
            states=[State.NAVIGATE, State.NAVIGATE],
            collisions=0,
            waypoints_reached=0,
            total_waypoints=0,
            steps=2,
            dt=0.033,
            waypoint_positions=[],
            start_position=(0.0, 0.0),
        )
        m = Evaluator.evaluate(run)
        assert m.success is True
        assert m.optimal_length == 0.0

    def test_all_collisions(self):
        """Run with many collisions: success = False."""
        run = _straight_line_run(collisions=10)
        m = Evaluator.evaluate(run)
        assert m.success is False
        assert m.collisions == 10

    def test_zero_length_path(self):
        """Agent doesn't move: path_length = 0, efficiency handled."""
        run = RunData(
            scenario="stuck",
            positions=[(50.0, 50.0), (50.0, 50.0), (50.0, 50.0)],
            headings=[0.0, 0.0, 0.0],
            velocities=[0.0, 0.0, 0.0],
            min_obstacle_dists=[100.0, 100.0, 100.0],
            states=[State.NAVIGATE, State.NAVIGATE, State.NAVIGATE],
            collisions=0,
            waypoints_reached=1,
            total_waypoints=1,
            steps=3,
            dt=0.033,
            waypoint_positions=[(50.0, 50.0)],
            start_position=(50.0, 50.0),
        )
        m = Evaluator.evaluate(run)
        assert m.path_length == 0.0
        # optimal is also 0 (start == waypoint), so efficiency = 1.0
        assert m.path_efficiency == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# print_metrics_table test
# ---------------------------------------------------------------------------


class TestPrintMetricsTable:
    """Tests for terminal output formatting."""

    def test_prints_without_error(self, capsys):
        """print_metrics_table outputs formatted text."""
        runs = [_straight_line_run(scenario="corridor"), _detour_run()]
        metrics = Evaluator.evaluate_batch(runs)

        print_metrics_table(metrics)

        captured = capsys.readouterr()
        assert "corridor" in captured.out
        assert "detour" in captured.out
        assert "AGGREGATE" in captured.out

    def test_empty_metrics(self, capsys):
        """Empty metrics list prints informative message."""
        print_metrics_table([])
        captured = capsys.readouterr()
        assert "No metrics" in captured.out
