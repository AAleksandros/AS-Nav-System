"""Evaluation framework for the APF navigation simulator.

Computes per-scenario metrics (success, path efficiency, smoothness, safety
margin) and generates publication-quality matplotlib plots for the DTU MSc
portfolio.
"""

import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.models import State

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RunData:
    """Per-step accumulator collected during a simulation run.

    Attributes
    ----------
    scenario : str
        Name of the scenario.
    positions : list of (float, float)
        Agent (x, y) each step.
    headings : list of float
        Agent heading (rad) each step.
    velocities : list of float
        Agent speed each step.
    min_obstacle_dists : list of float
        Nearest obstacle distance each step.
    states : list of State
        NAVIGATE / AVOID / STOP each step.
    collisions : int
        Total collision count.
    waypoints_reached : int
        Number of waypoints reached.
    total_waypoints : int
        Total number of waypoints.
    steps : int
        Total simulation steps taken.
    dt : float
        Time step duration (seconds).
    waypoint_positions : list of (float, float)
        Waypoint (x, y) coordinates.
    start_position : tuple of (float, float)
        Agent starting position.
    obstacles : list of (float, float, float)
        Snapshot of (x, y, radius) for each obstacle.
    """

    scenario: str
    positions: List[Tuple[float, float]]
    headings: List[float]
    velocities: List[float]
    min_obstacle_dists: List[float]
    states: List[State]
    collisions: int
    waypoints_reached: int
    total_waypoints: int
    steps: int
    dt: float
    waypoint_positions: List[Tuple[float, float]]
    start_position: Tuple[float, float]
    obstacles: List[Tuple[float, float, float]] = field(default_factory=list)


@dataclass
class ScenarioMetrics:
    """Computed metrics for a single scenario run.

    Attributes
    ----------
    scenario : str
        Name of the scenario.
    success : bool
        True if all waypoints reached and 0 collisions.
    collisions : int
        Total collision count.
    waypoints_reached : int
        Number of waypoints reached.
    total_waypoints : int
        Total number of waypoints.
    path_length : float
        Sum of euclidean step distances.
    optimal_length : float
        Sum of straight-line start -> wp1 -> wp2 -> ...
    path_efficiency : float
        optimal / actual (1.0 = perfect, <1.0 = detour).
    smoothness : float
        Mean |heading_change| per step (rad/step).
    min_safety_margin : float
        Minimum nearest-obstacle distance across run.
    avg_safety_margin : float
        Mean nearest-obstacle distance across run.
    avg_speed : float
        Mean velocity.
    completion_time : float
        steps * dt (simulated seconds).
    time_in_avoid : float
        Fraction of steps in AVOID state.
    time_in_navigate : float
        Fraction of steps in NAVIGATE state.
    """

    scenario: str
    success: bool
    collisions: int
    waypoints_reached: int
    total_waypoints: int
    path_length: float
    optimal_length: float
    path_efficiency: float
    smoothness: float
    min_safety_margin: float
    avg_safety_margin: float
    avg_speed: float
    completion_time: float
    time_in_avoid: float
    time_in_navigate: float


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Pure-function metric computation from RunData."""

    @staticmethod
    def evaluate(run: RunData) -> ScenarioMetrics:
        """Compute metrics for a single simulation run.

        Parameters
        ----------
        run : RunData
            Collected per-step data from a simulation run.

        Returns
        -------
        ScenarioMetrics
            Computed metrics.
        """
        # Path length — sum of euclidean distances between consecutive positions
        path_length = 0.0
        for i in range(1, len(run.positions)):
            x0, y0 = run.positions[i - 1]
            x1, y1 = run.positions[i]
            path_length += math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # Optimal length — straight-line start -> wp1 -> wp2 -> ...
        chain = [run.start_position] + run.waypoint_positions
        optimal_length = 0.0
        for i in range(1, len(chain)):
            x0, y0 = chain[i - 1]
            x1, y1 = chain[i]
            optimal_length += math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # Path efficiency
        if path_length > 1e-10:
            path_efficiency = min(optimal_length / path_length, 1.0)
        else:
            path_efficiency = 1.0 if optimal_length < 1e-10 else 0.0

        # Smoothness — mean |heading_change| per step
        heading_changes: List[float] = []
        for i in range(1, len(run.headings)):
            delta = run.headings[i] - run.headings[i - 1]
            # Normalize to [-pi, pi]
            delta = math.atan2(math.sin(delta), math.cos(delta))
            heading_changes.append(abs(delta))
        smoothness = (
            sum(heading_changes) / len(heading_changes)
            if heading_changes
            else 0.0
        )

        # Safety margins
        dists = run.min_obstacle_dists
        min_safety = min(dists) if dists else float("inf")
        avg_safety = sum(dists) / len(dists) if dists else float("inf")

        # Speed
        avg_speed = (
            sum(run.velocities) / len(run.velocities)
            if run.velocities
            else 0.0
        )

        # Completion time
        completion_time = run.steps * run.dt

        # State distribution
        n = len(run.states) if run.states else 1
        time_in_avoid = sum(1 for s in run.states if s == State.AVOID) / n
        time_in_navigate = sum(
            1 for s in run.states if s == State.NAVIGATE
        ) / n

        # Success
        success = (
            run.waypoints_reached >= run.total_waypoints
            and run.collisions == 0
        )

        return ScenarioMetrics(
            scenario=run.scenario,
            success=success,
            collisions=run.collisions,
            waypoints_reached=run.waypoints_reached,
            total_waypoints=run.total_waypoints,
            path_length=path_length,
            optimal_length=optimal_length,
            path_efficiency=path_efficiency,
            smoothness=smoothness,
            min_safety_margin=min_safety,
            avg_safety_margin=avg_safety,
            avg_speed=avg_speed,
            completion_time=completion_time,
            time_in_avoid=time_in_avoid,
            time_in_navigate=time_in_navigate,
        )

    @staticmethod
    def evaluate_batch(runs: List[RunData]) -> List[ScenarioMetrics]:
        """Compute metrics for multiple runs.

        Parameters
        ----------
        runs : list of RunData
            Collected data from multiple simulation runs.

        Returns
        -------
        list of ScenarioMetrics
        """
        return [Evaluator.evaluate(r) for r in runs]

    @staticmethod
    def summary(metrics: List[ScenarioMetrics]) -> Dict:
        """Aggregate mean/min/max across scenarios.

        Parameters
        ----------
        metrics : list of ScenarioMetrics
            Metrics from multiple scenarios.

        Returns
        -------
        dict
            Keys: success_rate, mean/min/max for path_efficiency,
            smoothness, min_safety_margin, avg_speed, collisions.
        """
        if not metrics:
            return {}

        n = len(metrics)
        effs = [m.path_efficiency for m in metrics]
        smooths = [m.smoothness for m in metrics]
        safeties = [m.min_safety_margin for m in metrics]
        speeds = [m.avg_speed for m in metrics]
        colls = [m.collisions for m in metrics]

        return {
            "count": n,
            "success_rate": sum(1 for m in metrics if m.success) / n,
            "total_collisions": sum(colls),
            "path_efficiency": {
                "mean": sum(effs) / n,
                "min": min(effs),
                "max": max(effs),
            },
            "smoothness": {
                "mean": sum(smooths) / n,
                "min": min(smooths),
                "max": max(smooths),
            },
            "min_safety_margin": {
                "mean": sum(safeties) / n,
                "min": min(safeties),
                "max": max(safeties),
            },
            "avg_speed": {
                "mean": sum(speeds) / n,
                "min": min(speeds),
                "max": max(speeds),
            },
        }


# ---------------------------------------------------------------------------
# MetricsPlotter
# ---------------------------------------------------------------------------


class MetricsPlotter:
    """Matplotlib figure generation for evaluation results."""

    def __init__(self, output_dir: str = "output/eval") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_metrics_bar_chart(
        self,
        metrics: List[ScenarioMetrics],
        show: bool = False,
    ) -> Optional[Path]:
        """Grouped bar chart: path efficiency, smoothness (inverted), avg safety.

        Parameters
        ----------
        metrics : list of ScenarioMetrics
            One entry per scenario.
        show : bool
            If True, display interactively.

        Returns
        -------
        Path or None
            Path to saved PNG, or None if no data.
        """
        import matplotlib.pyplot as plt  # type: ignore

        if not metrics:
            return None

        scenarios = [m.scenario for m in metrics]
        efficiency = [m.path_efficiency for m in metrics]
        # Invert smoothness: lower is better → show as (1 - normalized)
        max_smooth = max(m.smoothness for m in metrics) or 1.0
        smoothness_inv = [1.0 - m.smoothness / max_smooth for m in metrics]
        safety = [m.avg_safety_margin for m in metrics]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        axes[0].bar(scenarios, efficiency, color="#2196F3")
        axes[0].set_title("Path Efficiency")
        axes[0].set_ylabel("Optimal / Actual")
        axes[0].set_ylim(0, 1.05)
        axes[0].tick_params(axis="x", rotation=45)

        axes[1].bar(scenarios, smoothness_inv, color="#4CAF50")
        axes[1].set_title("Smoothness (higher = smoother)")
        axes[1].set_ylabel("1 - normalized smoothness")
        axes[1].set_ylim(0, 1.05)
        axes[1].tick_params(axis="x", rotation=45)

        axes[2].bar(scenarios, safety, color="#FF9800")
        axes[2].set_title("Avg Safety Margin")
        axes[2].set_ylabel("Distance (units)")
        axes[2].tick_params(axis="x", rotation=45)

        fig.suptitle("APF Navigation — Scenario Metrics", fontsize=14)
        fig.tight_layout()

        path = self.output_dir / "metrics_bar_chart.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        logger.info("Saved bar chart: %s", path)

        if show:
            plt.show()
        plt.close(fig)

        return path

    def plot_trajectory(
        self,
        run: RunData,
        metrics: ScenarioMetrics,
        show: bool = False,
    ) -> Optional[Path]:
        """Agent path overlaid on obstacle circles for a single scenario.

        Parameters
        ----------
        run : RunData
            Collected simulation data.
        metrics : ScenarioMetrics
            Computed metrics for title annotation.
        show : bool
            If True, display interactively.

        Returns
        -------
        Path or None
            Path to saved PNG.
        """
        import matplotlib.pyplot as plt  # type: ignore

        if not run.positions:
            return None

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        # Draw obstacles
        for ox, oy, r in run.obstacles:
            circle = plt.Circle(  # type: ignore
                (ox, oy), r, color="#B0BEC5", alpha=0.6
            )
            ax.add_patch(circle)

        # Draw trajectory
        xs = [p[0] for p in run.positions]
        ys = [p[1] for p in run.positions]
        ax.plot(xs, ys, "-", color="#1976D2", linewidth=1.5, label="Path")

        # Start and end
        ax.plot(xs[0], ys[0], "go", markersize=10, label="Start")
        ax.plot(xs[-1], ys[-1], "rs", markersize=10, label="End")

        # Waypoints
        for i, (wx, wy) in enumerate(run.waypoint_positions):
            ax.plot(wx, wy, "m^", markersize=10)
            ax.annotate(
                f"WP{i + 1}",
                (wx, wy),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        ax.set_aspect("equal")
        ax.set_xlabel("X (world units)")
        ax.set_ylabel("Y (world units)")
        ax.set_title(
            f"{run.scenario} — Eff={metrics.path_efficiency:.2f}, "
            f"Safety={metrics.min_safety_margin:.0f}, "
            f"{'OK' if metrics.success else 'FAIL'}"
        )
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = self.output_dir / f"trajectory_{run.scenario}.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        logger.info("Saved trajectory: %s", path)

        if show:
            plt.show()
        plt.close(fig)

        return path

    def plot_radar_chart(
        self,
        metrics: List[ScenarioMetrics],
        show: bool = False,
    ) -> Optional[Path]:
        """Radar/spider chart with normalized metrics per scenario.

        Axes: path_efficiency, smoothness (inverted), min_safety (normalized),
        avg_speed (normalized), success (0 or 1).

        Parameters
        ----------
        metrics : list of ScenarioMetrics
            One entry per scenario.
        show : bool
            If True, display interactively.

        Returns
        -------
        Path or None
            Path to saved PNG.
        """
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore

        if not metrics:
            return None

        categories = [
            "Path Efficiency",
            "Smoothness",
            "Min Safety",
            "Avg Speed",
            "Success",
        ]
        n_cats = len(categories)
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        # Normalization bounds
        max_smooth = max(m.smoothness for m in metrics) or 1.0
        max_safety = max(m.min_safety_margin for m in metrics) or 1.0
        max_speed = max(m.avg_speed for m in metrics) or 1.0

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={"polar": True})

        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))  # type: ignore

        for i, m in enumerate(metrics):
            values = [
                m.path_efficiency,
                1.0 - m.smoothness / max_smooth,
                m.min_safety_margin / max_safety,
                m.avg_speed / max_speed,
                1.0 if m.success else 0.0,
            ]
            values += values[:1]
            ax.plot(
                angles, values, "o-", linewidth=1.5,
                label=m.scenario, color=colors[i],
            )
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        ax.set_thetagrids(  # type: ignore[attr-defined]
            [a * 180 / np.pi for a in angles[:-1]], categories
        )
        ax.set_ylim(0, 1.05)
        ax.set_title("APF Navigation — Scenario Comparison", fontsize=14, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        fig.tight_layout()

        path = self.output_dir / "radar_chart.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        logger.info("Saved radar chart: %s", path)

        if show:
            plt.show()
        plt.close(fig)

        return path


# ---------------------------------------------------------------------------
# Terminal table formatting
# ---------------------------------------------------------------------------


def print_metrics_table(metrics: List[ScenarioMetrics]) -> None:
    """Print a formatted terminal table of scenario metrics.

    Parameters
    ----------
    metrics : list of ScenarioMetrics
        One entry per scenario.
    """
    if not metrics:
        print("No metrics to display.")
        return

    header = (
        f"{'Scenario':<14} {'OK':>3} {'Col':>4} {'WP':>6} "
        f"{'Eff':>6} {'Smooth':>7} {'MinSaf':>7} {'AvgSaf':>7} "
        f"{'AvgSpd':>7} {'Time':>6} {'Avoid%':>7}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for m in metrics:
        wp_str = f"{m.waypoints_reached}/{m.total_waypoints}"
        print(
            f"{m.scenario:<14} {'Y' if m.success else 'N':>3} "
            f"{m.collisions:>4} {wp_str:>6} "
            f"{m.path_efficiency:>6.3f} {m.smoothness:>7.4f} "
            f"{m.min_safety_margin:>7.1f} {m.avg_safety_margin:>7.1f} "
            f"{m.avg_speed:>7.1f} {m.completion_time:>6.1f} "
            f"{m.time_in_avoid * 100:>6.1f}%"
        )

    print("-" * len(header))

    # Aggregate row
    s = Evaluator.summary(metrics)
    pe = s["path_efficiency"]
    sm = s["smoothness"]
    sf = s["min_safety_margin"]
    sp = s["avg_speed"]
    print(
        f"{'AGGREGATE':<14} "
        f"{s['success_rate'] * 100:>3.0f}% "
        f"{s['total_collisions']:>3} "
        f"{'':>6} "
        f"{pe['mean']:>6.3f} {sm['mean']:>7.4f} "
        f"{sf['mean']:>7.1f} {'':>7} "
        f"{sp['mean']:>7.1f} {'':>6} {'':>7}"
    )
    print("=" * len(header))
