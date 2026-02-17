"""Behavioral acceptance tests for the APF navigation agent.

These tests verify that the agent achieves acceptable navigation quality
across all named scenarios and random seeds.  They use
``run_scenario(render=False)`` for fast headless execution (~0.1 s per
scenario) and assert on metrics computed by the ``Evaluator``.

Test tiers
----------
Tier 1 — Hard constraints : 0 collisions, all waypoints, safety margins
Tier 2 — Quality thresholds : per-scenario efficiency, smoothness, safety
Tier 3 — Stress testing : random seeds, aggregate success rate
Tier 4 — Behavioral properties : monotonic progress, speed utilization
"""

import math

import pytest

from src.config import load_config
from src.evaluation import Evaluator
from src.main import run_scenario


NAMED_SCENARIOS = [
    "corridor",
    "gauntlet",
    "dynamic",
    "slalom",
    "narrow_gap",
    "u_turn",
    "crossing",
    "dense",
]

# Per-scenario quality thresholds derived from baseline with ~20 % margin.
# Format: (min_efficiency, max_smoothness, min_safety_margin)
QUALITY_THRESHOLDS = {
    "corridor":   (0.90, 0.010, 30.0),
    "gauntlet":   (0.55, 0.030, 25.0),
    "dynamic":    (0.85, 0.020, 3.0),    # moving obstacles → tighter passes
    "slalom":     (0.75, 0.020, 35.0),
    "narrow_gap": (0.90, 0.010, 20.0),
    "u_turn":     (0.60, 0.030, 35.0),
    "crossing":   (0.85, 0.020, 35.0),
    "dense":      (0.65, 0.030, 20.0),
}

NUM_RANDOM_SEEDS = 20


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():
    """Load default configuration once for all behavioral tests."""
    return load_config()


@pytest.fixture(scope="module")
def scenario_runs(config):
    """Run all 8 named scenarios headless and cache (RunData, Metrics)."""
    results = {}
    for name in NAMED_SCENARIOS:
        run_data = run_scenario(config=config, scenario_name=name, render=False)
        metrics = Evaluator.evaluate(run_data)
        results[name] = (run_data, metrics)
    return results


@pytest.fixture(scope="module")
def random_runs(config):
    """Run random scenarios with seeds 0..NUM_RANDOM_SEEDS-1 and cache."""
    results = []
    for seed in range(NUM_RANDOM_SEEDS):
        run_data = run_scenario(
            config=config, scenario_name="random", seed=seed, render=False,
        )
        metrics = Evaluator.evaluate(run_data)
        results.append((seed, run_data, metrics))
    return results


# ---------------------------------------------------------------------------
# Tier 1: Hard Constraints
# ---------------------------------------------------------------------------


class TestHardConstraints:
    """Invariants that must hold for every named scenario."""

    @pytest.mark.parametrize("scenario", NAMED_SCENARIOS)
    def test_zero_collisions(self, scenario_runs, scenario):
        """Agent must complete every scenario with zero collisions."""
        _, metrics = scenario_runs[scenario]
        assert metrics.collisions == 0, (
            f"{scenario}: {metrics.collisions} collision(s)"
        )

    @pytest.mark.parametrize("scenario", NAMED_SCENARIOS)
    def test_all_waypoints_reached(self, scenario_runs, scenario):
        """Agent must reach all waypoints in every scenario."""
        _, metrics = scenario_runs[scenario]
        assert metrics.waypoints_reached == metrics.total_waypoints, (
            f"{scenario}: {metrics.waypoints_reached}/{metrics.total_waypoints}"
        )

    @pytest.mark.parametrize("scenario", NAMED_SCENARIOS)
    def test_scenario_succeeds(self, scenario_runs, scenario):
        """Every named scenario must report overall success."""
        _, metrics = scenario_runs[scenario]
        assert metrics.success is True, f"{scenario}: not successful"

    @pytest.mark.parametrize("scenario", NAMED_SCENARIOS)
    def test_positive_safety_margin(self, scenario_runs, scenario):
        """Min obstacle/wall distance must stay positive (no overlap)."""
        _, metrics = scenario_runs[scenario]
        assert metrics.min_safety_margin > 0.0, (
            f"{scenario}: min safety {metrics.min_safety_margin:.1f} <= 0"
        )

    @pytest.mark.parametrize("scenario", NAMED_SCENARIOS)
    def test_completes_within_time_limit(self, scenario_runs, scenario):
        """Scenario must finish within 60 simulated seconds."""
        _, metrics = scenario_runs[scenario]
        assert metrics.completion_time <= 60.0, (
            f"{scenario}: {metrics.completion_time:.1f}s > 60s"
        )


# ---------------------------------------------------------------------------
# Tier 2: Quality Thresholds
# ---------------------------------------------------------------------------


class TestQualityThresholds:
    """Per-scenario quality gates (baseline minus ~20 % margin)."""

    @pytest.mark.parametrize("scenario", NAMED_SCENARIOS)
    def test_path_efficiency(self, scenario_runs, scenario):
        """Path efficiency must meet per-scenario minimum."""
        _, metrics = scenario_runs[scenario]
        min_eff = QUALITY_THRESHOLDS[scenario][0]
        assert metrics.path_efficiency >= min_eff, (
            f"{scenario}: efficiency {metrics.path_efficiency:.3f} < {min_eff}"
        )

    @pytest.mark.parametrize("scenario", NAMED_SCENARIOS)
    def test_smoothness(self, scenario_runs, scenario):
        """Mean |heading change| must stay below per-scenario ceiling."""
        _, metrics = scenario_runs[scenario]
        max_smooth = QUALITY_THRESHOLDS[scenario][1]
        assert metrics.smoothness <= max_smooth, (
            f"{scenario}: smoothness {metrics.smoothness:.4f} > {max_smooth}"
        )

    @pytest.mark.parametrize("scenario", NAMED_SCENARIOS)
    def test_min_safety_margin(self, scenario_runs, scenario):
        """Min safety margin must exceed per-scenario floor."""
        _, metrics = scenario_runs[scenario]
        min_safety = QUALITY_THRESHOLDS[scenario][2]
        assert metrics.min_safety_margin >= min_safety, (
            f"{scenario}: min safety {metrics.min_safety_margin:.1f} < {min_safety}"
        )

    @pytest.mark.parametrize("scenario", NAMED_SCENARIOS)
    def test_speed_utilization(self, scenario_runs, scenario):
        """Avg speed must be at least 60 % of cruise speed (32 u/s)."""
        _, metrics = scenario_runs[scenario]
        min_avg_speed = 32.0 * 0.60  # 19.2 u/s
        assert metrics.avg_speed >= min_avg_speed, (
            f"{scenario}: avg speed {metrics.avg_speed:.1f} < {min_avg_speed:.1f}"
        )


# ---------------------------------------------------------------------------
# Tier 3: Stress Testing — Random Seeds
# ---------------------------------------------------------------------------


class TestRandomScenarioStress:
    """Robustness across randomly generated obstacle layouts.

    Random layouts are NOT guaranteed solvable — obstacles can spawn on
    the start position or form impassable barriers.  Thresholds are set
    accordingly: we test that the agent handles *most* layouts and that
    *successful* runs are clean.
    """

    def test_aggregate_success_rate(self, random_runs):
        """At least 60 % of random layouts should succeed."""
        successes = sum(1 for _, _, m in random_runs if m.success)
        rate = successes / len(random_runs)
        assert rate >= 0.60, (
            f"Random success rate {rate:.0%} < 60% "
            f"({successes}/{len(random_runs)})"
        )

    def test_successful_runs_are_clean(self, random_runs):
        """Runs that succeed must have 0 collisions (by definition)."""
        for seed, _, metrics in random_runs:
            if metrics.success:
                assert metrics.collisions == 0, (
                    f"Seed {seed}: success=True but {metrics.collisions} collisions"
                )

    def test_successful_runs_have_decent_efficiency(self, random_runs):
        """Successful runs should achieve at least 30 % path efficiency."""
        for seed, _, metrics in random_runs:
            if metrics.success:
                assert metrics.path_efficiency >= 0.30, (
                    f"Seed {seed}: efficiency {metrics.path_efficiency:.3f} < 0.30"
                )


# ---------------------------------------------------------------------------
# Tier 4: Behavioral Properties
# ---------------------------------------------------------------------------


class TestBehavioralProperties:
    """Verify emergent navigation behavior across named scenarios."""

    @pytest.mark.parametrize("scenario", NAMED_SCENARIOS)
    def test_monotonic_progress(self, scenario_runs, scenario):
        """Distance to final waypoint should decrease over the run.

        Compares mean distance in the first quarter vs last quarter.
        Allows temporary increases during obstacle avoidance.
        """
        run_data, _ = scenario_runs[scenario]
        final_wp = run_data.waypoint_positions[-1]
        distances = [
            math.sqrt((x - final_wp[0]) ** 2 + (y - final_wp[1]) ** 2)
            for x, y in run_data.positions
        ]

        n = len(distances)
        q1 = max(n // 4, 1)
        q4_start = 3 * n // 4

        first_quarter_avg = sum(distances[:q1]) / q1
        last_quarter_avg = (
            sum(distances[q4_start:]) / max(n - q4_start, 1)
        )

        assert last_quarter_avg < first_quarter_avg, (
            f"{scenario}: no net progress — Q1 avg {first_quarter_avg:.0f}, "
            f"Q4 avg {last_quarter_avg:.0f}"
        )

    @pytest.mark.parametrize("scenario", NAMED_SCENARIOS)
    def test_no_stuck_in_loop(self, scenario_runs, scenario):
        """Path length should be reasonable relative to bounding box.

        A path/bbox-diagonal ratio > 5 indicates excessive looping.
        """
        run_data, metrics = scenario_runs[scenario]
        xs = [p[0] for p in run_data.positions]
        ys = [p[1] for p in run_data.positions]
        bbox_diag = math.sqrt(
            (max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2
        )

        if bbox_diag < 1e-10:
            pytest.skip("Degenerate trajectory")

        ratio = metrics.path_length / bbox_diag
        assert ratio <= 5.0, (
            f"{scenario}: path/bbox ratio {ratio:.1f} > 5.0 "
            f"(path={metrics.path_length:.0f}, diag={bbox_diag:.0f})"
        )

    @pytest.mark.parametrize("scenario", NAMED_SCENARIOS)
    def test_no_excessive_stationary_time(self, scenario_runs, scenario):
        """Agent should not spend > 10 % of time near-stationary."""
        run_data, _ = scenario_runs[scenario]
        near_stationary = sum(
            1 for v in run_data.velocities if v < 5.0
        )
        fraction = near_stationary / max(len(run_data.velocities), 1)

        assert fraction < 0.10, (
            f"{scenario}: {fraction:.0%} of time near-stationary (< 5 u/s)"
        )
