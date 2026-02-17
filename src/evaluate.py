"""CLI batch evaluation entry point for the APF navigation simulator.

Usage:
    python -m src.evaluate                          # all 8 named, terminal table
    python -m src.evaluate --plots                  # + matplotlib figures
    python -m src.evaluate --random 20              # + 20 random seeds
    python -m src.evaluate --scenario corridor      # single scenario
    python -m src.evaluate --plots --show           # interactive display
    python -m src.evaluate -o output/eval           # custom output dir
"""

import argparse
import sys
import time

from src.config import load_config
from src.evaluation import Evaluator, MetricsPlotter, print_metrics_table
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


def parse_arguments() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(
        description="APF Navigation — Batch Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Run a single scenario (default: all 8 named)",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=0,
        metavar="N",
        help="Also run N random scenarios with sequential seeds",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate matplotlib plots (bar chart, trajectories, radar)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (requires --plots)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="output/eval",
        help="Output directory for plots (default: output/eval)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Configuration file path",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Override simulation duration in seconds",
    )
    return parser.parse_args()


def main() -> int:
    """Run batch evaluation and print results.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    args = parse_arguments()

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1

    # Determine which scenarios to run
    if args.scenario:
        scenarios = [args.scenario]
    else:
        scenarios = list(NAMED_SCENARIOS)

    # Add random scenarios
    for i in range(args.random):
        scenarios.append(f"random:{i}")

    print(f"Evaluating {len(scenarios)} scenario(s)...\n")

    runs: list = []
    start_time = time.time()

    for scenario_spec in scenarios:
        # Parse "random:N" format
        if scenario_spec.startswith("random:"):
            seed = int(scenario_spec.split(":")[1])
            name = f"random_{seed}"
            scenario_name = "random"
        else:
            seed = None
            name = scenario_spec
            scenario_name = scenario_spec

        print(f"  Running {name}...", end=" ", flush=True)
        t0 = time.time()

        try:
            run_data = run_scenario(
                config=config,
                scenario_name=scenario_name,
                seed=seed,
                duration=args.duration,
                render=False,
            )
            # Override scenario name for display
            run_data.scenario = name
            runs.append(run_data)
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")

    total_time = time.time() - start_time
    print(f"\nAll scenarios complete in {total_time:.1f}s\n")

    # Compute metrics
    metrics = Evaluator.evaluate_batch(runs)

    # Print terminal table
    print_metrics_table(metrics)

    # Summary
    summary = Evaluator.summary(metrics)
    if summary:
        print(f"\nSuccess rate: {summary['success_rate'] * 100:.0f}%")
        print(f"Total collisions: {summary['total_collisions']}")

    # Generate plots
    if args.plots:
        print(f"\nGenerating plots in {args.output_dir}/...")
        plotter = MetricsPlotter(output_dir=args.output_dir)

        path = plotter.plot_metrics_bar_chart(metrics, show=args.show)
        if path:
            print(f"  Bar chart: {path}")

        for run_data, m in zip(runs, metrics):
            path = plotter.plot_trajectory(run_data, m, show=args.show)
            if path:
                print(f"  Trajectory: {path}")

        path = plotter.plot_radar_chart(metrics, show=args.show)
        if path:
            print(f"  Radar chart: {path}")

        print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
