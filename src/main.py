"""Main entry point for the APF navigation simulator.

This module provides the CLI interface and orchestrates the simulation loop:
environment setup, sensor scanning, APF planning, PID control, and visualization.
"""

import argparse
import math
import sys
import time
from pathlib import Path

from src.config import load_config, print_config
from src.utils.logger import setup_logger
from src.utils.video_utils import VideoWriter
from src.environment import Environment
from src.sensors import RangeSensor
from src.apf_planner import APFPlanner
from src.pid_controller import PIDController
from src.control import AgentController
from src.visualization import VisualizationRenderer
from src.waypoint_navigator import Waypoint, WaypointNavigator
from src.models import ControlCommand, State


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Autonomous Navigation Agent — APF Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Corridor scenario
  python -m src.main --scenario corridor -o output/corridor.mp4

  # Random scenario with seed
  python -m src.main --scenario random --seed 42 -o output/random.mp4

  # Dynamic scenario, short run
  python -m src.main --scenario dynamic --duration 5 -o output/dynamic.mp4

  # Debug mode
  python -m src.main --scenario corridor -o output/debug.mp4 -v
        """
    )

    parser.add_argument(
        '--scenario',
        type=str,
        required=True,
        help='Scenario name from config (e.g., corridor, gauntlet, dynamic) or "random"'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output video file path'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='RNG seed for random scenario'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Override simulation duration in seconds'
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        default=None,
        help='Configuration file path (default: config/default_config.yaml)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable DEBUG logging'
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the APF simulation.

    Returns:
        Exit code (0 = success, 1 = error)
    """
    args = parse_arguments()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1

    # Determine log level
    log_level = 'DEBUG' if args.verbose else config.logging.level  # type: ignore

    # Setup logging
    logger = setup_logger(
        name='navigation',
        level=log_level,
        log_file=config.logging.log_file,  # type: ignore
        log_to_console=config.logging.log_to_console,  # type: ignore
    )

    # Validate output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Create environment ---
    try:
        if args.scenario == "random":
            env_width = float(config.environment.width)  # type: ignore
            env_height = float(config.environment.height)  # type: ignore
            env = Environment.random_scenario(
                env_width, env_height, seed=args.seed
            )
            # Default waypoint for random: center-right
            scenario_waypoints = [(env_width * 0.85, env_height * 0.5)]
            scenario_start = (env_width * 0.1, env_height * 0.5)
            scenario_heading = 0.0  # east
        else:
            env = Environment.from_scenario(config, args.scenario)
            # Extract scenario data from config
            scenarios = config.environment.scenarios  # type: ignore
            scenario_data = getattr(scenarios, args.scenario, None)
            if scenario_data is None:
                logger.error(f"Scenario '{args.scenario}' not found in config")
                return 1

            scenario_waypoints = [
                (float(wp[0]), float(wp[1]))
                for wp in scenario_data.waypoints  # type: ignore
            ]
            scenario_start = (
                float(scenario_data.start[0]),  # type: ignore
                float(scenario_data.start[1]),  # type: ignore
            )
            scenario_heading = float(scenario_data.start_heading)  # type: ignore
    except (ValueError, AttributeError) as e:
        logger.error(f"Failed to create environment: {e}")
        return 1

    # --- Simulation parameters ---
    sim = config.simulation  # type: ignore
    dt = float(sim.dt)  # type: ignore
    max_steps = int(sim.max_steps)  # type: ignore
    fps = int(sim.fps)  # type: ignore
    video_width = int(sim.video_width)  # type: ignore
    video_height = int(sim.video_height)  # type: ignore

    if args.duration is not None:
        max_steps = int(args.duration / dt)

    # --- Create components ---
    sensor = RangeSensor.from_config(config)
    planner = APFPlanner.from_config(config)
    pid = PIDController.from_config(config)

    # Override control config with scenario start position
    # AgentController.__init__ calls math.radians(start_heading), so convert
    config.control.start_x = scenario_start[0]  # type: ignore
    config.control.start_y = scenario_start[1]  # type: ignore
    config.control.start_heading = math.degrees(scenario_heading)  # type: ignore

    controller = AgentController(config, pid=pid)
    renderer = VisualizationRenderer(config)

    # Build waypoints
    goal_tolerance = float(config.planner.goal_tolerance)  # type: ignore
    waypoints = [
        Waypoint(x=wx, y=wy, tolerance=goal_tolerance)
        for wx, wy in scenario_waypoints
    ]
    navigator = WaypointNavigator(waypoints)

    agent_radius = float(config.environment.agent_radius)  # type: ignore

    # --- Log summary ---
    logger.info("=" * 60)
    logger.info("Autonomous Navigation Agent — APF Simulator")
    logger.info("=" * 60)
    logger.info(f"Scenario: {args.scenario}")
    logger.info(f"Output: {args.output}")
    logger.info(
        "Start: (%.0f, %.0f), heading=%.2f rad",
        scenario_start[0], scenario_start[1], scenario_heading,
    )
    logger.info(f"Waypoints: {len(waypoints)}")
    logger.info(f"Max steps: {max_steps}, dt={dt:.3f}s, FPS={fps}")
    logger.info(f"Resolution: {video_width}x{video_height}")
    logger.info("=" * 60)

    if args.verbose:
        logger.debug("Full configuration:")
        print_config(config)

    # --- Simulation loop ---
    try:
        with VideoWriter(
            args.output, fps=fps, width=video_width, height=video_height
        ) as writer:
            start_time = time.time()
            collisions = 0

            for step in range(max_steps):
                # 1. Advance dynamic obstacles
                env.step(dt)

                # 2. Check and advance waypoints
                agent = controller.get_agent_state()
                navigator.check_and_advance(agent.x, agent.y)

                # 3. Get current goal
                goal = navigator.current_goal
                if goal is None:
                    # All waypoints reached — render completion banner
                    banner_frames = int(2.0 * fps)
                    for _ in range(banner_frames):
                        frame = env.render_background(config)
                        obstacles = env.get_obstacles()
                        renderer.render_simulation(
                            frame,
                            agent=agent,
                            state=State.NAVIGATE,
                            obstacles=obstacles,
                            sensor_readings=[],
                            forces=[],
                            waypoints=[(wp.x, wp.y) for wp in waypoints],
                            current_waypoint_idx=len(waypoints),
                            goal_status="GOAL REACHED",
                            is_goal_complete=True,
                        )
                        writer.write_frame(frame)
                    logger.info("All waypoints reached!")
                    break

                gx, gy = goal

                # 4. Sensor scan
                readings = sensor.scan(
                    agent.x, agent.y,
                    controller._heading_rad,
                    env.get_obstacles(),
                    env.get_boundary_segments(),
                )

                # 5. APF planning
                result = planner.compute(
                    agent.x, agent.y, gx, gy, readings
                )
                total_force = result["total_force"]
                state = result["state"]
                forces = result["forces"]

                # 6. Create control command
                if total_force.magnitude < 1e-10:
                    desired_heading = controller._heading_rad
                else:
                    desired_heading = total_force.heading

                hit_distances = [r.distance for r in readings if r.hit]
                min_obstacle_dist = (
                    min(hit_distances) if hit_distances else float("inf")
                )
                desired_speed = planner.compute_speed(min_obstacle_dist, state)

                command = ControlCommand(
                    desired_heading=desired_heading,
                    desired_speed=desired_speed,
                )

                # 7. PID heading control + kinematics
                prev_x, prev_y = agent.x, agent.y
                controller.update(command, dt)
                agent = controller.get_agent_state()

                # 8. Collision response — push agent away from obstacle
                if env.check_collision(agent.x, agent.y, agent_radius):
                    collisions += 1
                    safe_x, safe_y = prev_x, prev_y
                    # Find nearest obstacle and push away from it
                    for obs in env.get_obstacles():
                        dx = agent.x - obs.x
                        dy = agent.y - obs.y
                        dist = math.sqrt(dx * dx + dy * dy)
                        min_sep = agent_radius + obs.radius + 1.0
                        if dist < min_sep and dist > 1e-10:
                            # Push to just outside collision
                            scale = min_sep / dist
                            safe_x = obs.x + dx * scale
                            safe_y = obs.y + dy * scale
                            break
                    # Clamp to world boundaries
                    safe_x = max(agent_radius, min(
                        safe_x, env.width - agent_radius))
                    safe_y = max(agent_radius, min(
                        safe_y, env.height - agent_radius))
                    controller.agent = controller.agent.__class__(
                        x=safe_x,
                        y=safe_y,
                        heading=agent.heading,
                        velocity=agent.velocity,
                        trajectory=agent.trajectory,
                    )
                    agent = controller.get_agent_state()
                    if collisions <= 5:
                        logger.warning(
                            "Collision at (%.1f, %.1f) step %d",
                            safe_x, safe_y, step,
                        )

                # 9. Render frame
                frame = env.render_background(config)

                wp_positions = [(wp.x, wp.y) for wp in waypoints]
                status_text = (
                    f"WP {navigator.current_waypoint_idx + 1}/{len(waypoints)} | "
                    f"Progress: {navigator.progress * 100:.0f}%"
                )

                renderer.render_simulation(
                    frame,
                    agent=agent,
                    state=state,
                    obstacles=env.get_obstacles(),
                    sensor_readings=readings,
                    forces=forces,
                    waypoints=wp_positions,
                    current_waypoint_idx=navigator.current_waypoint_idx,
                    goal_status=status_text,
                    is_goal_complete=False,
                )

                # 10. Write frame
                writer.write_frame(frame)

                # Progress logging
                if step > 0 and step % 100 == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        "Step %d/%d | State=%s | Speed=%.1f"
                        " | Collisions=%d | %.1f steps/s",
                        step, max_steps, state.value,
                        agent.velocity, collisions,
                        step / elapsed,
                    )

            # Final summary
            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info("Simulation complete!")
            logger.info(f"Steps: {min(step + 1, max_steps)}")
            logger.info(f"Collisions: {collisions}")
            logger.info(
                "Waypoints reached: %d/%d",
                navigator.current_waypoint_idx, len(waypoints),
            )
            logger.info(f"Time: {elapsed:.1f}s")
            logger.info(f"Output: {args.output}")
            logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Simulation error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
