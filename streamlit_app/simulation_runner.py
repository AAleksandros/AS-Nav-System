"""Headless simulation runner for Streamlit interactive demo.

Mirrors the simulation loop in src/main.py:run_scenario() but collects
rendered frames into an in-memory MP4 via a temp file, returning
(video_bytes, RunData) for display in the Streamlit UI.
"""

import logging
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np

from src.apf_planner import APFPlanner
from src.config import Config
from src.control import AgentController
from src.environment import Environment
from src.evaluation import RunData
from src.models import ControlCommand, Obstacle, State
from src.pid_controller import PIDController
from src.sensors import RangeSensor
from src.utils.math_utils import normalize_angle
from src.visualization import VisualizationRenderer
from src.waypoint_navigator import Waypoint, WaypointNavigator

logger = logging.getLogger(__name__)


def frames_to_video_bytes(frames: List[np.ndarray], fps: int = 30) -> bytes:
    """Encode a list of BGR frames to an MP4 byte string via a temp file.

    Parameters
    ----------
    frames : list of np.ndarray
        BGR frames (H, W, 3).
    fps : int
        Frames per second.

    Returns
    -------
    bytes
        MP4 file contents.
    """
    if not frames:
        return b""

    h, w = frames[0].shape[:2]

    # Write frames with mp4v first (universally supported by OpenCV)
    raw_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_path, fourcc, fps, (w, h))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    # Re-encode to H.264 with ffmpeg for browser compatibility
    h264_path = tempfile.mktemp(suffix=".mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", raw_path,
                "-c:v", "libx264", "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                h264_path,
            ],
            capture_output=True, check=True,
        )
        video_bytes = Path(h264_path).read_bytes()
    except (FileNotFoundError, subprocess.CalledProcessError):
        # ffmpeg not available — try avc1 codec directly (macOS)
        logger.warning("ffmpeg not found, trying avc1 codec directly")
        avc1_path = tempfile.mktemp(suffix=".mp4")
        fourcc_h264 = cv2.VideoWriter_fourcc(*"avc1")
        writer2 = cv2.VideoWriter(avc1_path, fourcc_h264, fps, (w, h))
        try:
            for frame in frames:
                writer2.write(frame)
        finally:
            writer2.release()
        if Path(avc1_path).exists() and Path(avc1_path).stat().st_size > 0:
            video_bytes = Path(avc1_path).read_bytes()
        else:
            # Last resort: return mp4v (may not play in browser)
            video_bytes = Path(raw_path).read_bytes()
        Path(avc1_path).unlink(missing_ok=True)
    finally:
        Path(raw_path).unlink(missing_ok=True)
        Path(h264_path).unlink(missing_ok=True)

    return video_bytes


def run_scenario_headless(
    config: Config,
    *,
    scenario_name: Optional[str] = None,
    obstacles: Optional[List[Obstacle]] = None,
    waypoints_xy: Optional[List[Tuple[float, float]]] = None,
    start_pos: Optional[Tuple[float, float]] = None,
    start_heading: float = 0.0,
    duration: Optional[int] = None,
    frame_skip: int = 2,
    speed_multiplier: float = 1.0,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[bytes, RunData]:
    """Run a simulation and return video bytes + RunData.

    For preset scenarios, pass ``scenario_name``.
    For custom scenarios, pass ``obstacles``, ``waypoints_xy``, and ``start_pos``.

    Parameters
    ----------
    config : Config
        Full configuration object.
    scenario_name : str or None
        Named scenario from config. If provided, overrides custom params.
    obstacles : list of Obstacle or None
        Custom obstacles (ignored if scenario_name is set).
    waypoints_xy : list of (float, float) or None
        Custom waypoint positions (ignored if scenario_name is set).
    start_pos : tuple of (float, float) or None
        Custom start position (ignored if scenario_name is set).
    start_heading : float
        Start heading in radians (default east).
    duration : int or None
        Override simulation duration in seconds.
    frame_skip : int
        Render every N-th frame to save memory.
    speed_multiplier : float
        Video playback speed multiplier (e.g. 2.0 = 2x faster).
    progress_callback : callable or None
        Called with progress fraction [0, 1] each step.

    Returns
    -------
    (bytes, RunData)
        MP4 video bytes and collected simulation data.
    """
    # --- Resolve scenario ---
    if scenario_name is not None:
        env = Environment.from_scenario(config, scenario_name)
        scenarios = config.environment.scenarios  # type: ignore
        scenario_data = getattr(scenarios, scenario_name, None)
        if scenario_data is None:
            raise ValueError(f"Scenario '{scenario_name}' not found in config")

        scenario_waypoints = [
            (float(wp[0]), float(wp[1]))
            for wp in scenario_data.waypoints  # type: ignore
        ]
        scenario_start = (
            float(scenario_data.start[0]),  # type: ignore
            float(scenario_data.start[1]),  # type: ignore
        )
        scenario_heading = float(scenario_data.start_heading)  # type: ignore
        label = scenario_name
    else:
        # Custom scenario
        env_width = float(config.environment.width)  # type: ignore
        env_height = float(config.environment.height)  # type: ignore
        env = Environment(env_width, env_height, obstacles or [])
        scenario_waypoints = waypoints_xy or [(env_width * 0.85, env_height * 0.5)]
        scenario_start = start_pos or (env_width * 0.1, env_height * 0.5)
        scenario_heading = start_heading
        label = "custom"

    # --- Simulation parameters ---
    sim = config.simulation  # type: ignore
    dt = float(sim.dt)  # type: ignore
    max_steps = int(sim.max_steps)  # type: ignore
    fps = int(sim.fps)  # type: ignore

    if duration is not None:
        max_steps = int(duration / dt)

    # --- Create components ---
    sensor = RangeSensor.from_config(config)
    planner = APFPlanner.from_config(config)
    pid = PIDController.from_config(config)

    config.control.start_x = scenario_start[0]  # type: ignore
    config.control.start_y = scenario_start[1]  # type: ignore
    config.control.start_heading = math.degrees(scenario_heading)  # type: ignore

    controller = AgentController(config, pid=pid)

    goal_tolerance = float(config.planner.goal_tolerance)  # type: ignore
    nav_waypoints = [
        Waypoint(x=wx, y=wy, tolerance=goal_tolerance)
        for wx, wy in scenario_waypoints
    ]
    navigator = WaypointNavigator(nav_waypoints)

    agent_radius = float(config.environment.agent_radius)  # type: ignore

    # --- Rendering ---
    renderer = VisualizationRenderer(config)

    # --- Data collectors ---
    positions: List[Tuple[float, float]] = []
    headings_list: List[float] = []
    velocities_list: List[float] = []
    min_obstacle_dists: List[float] = []
    states_list: List[State] = []
    collisions = 0
    frames: List[np.ndarray] = []

    obstacle_snapshot = [(o.x, o.y, o.radius) for o in env.get_obstacles()]

    # --- Simulation loop ---
    for step in range(max_steps):
        # 1. Advance dynamic obstacles
        env.step(dt)

        # 2. Check and advance waypoints
        agent = controller.get_agent_state()
        navigator.check_and_advance(agent.x, agent.y)

        # 3. Get current goal
        goal = navigator.current_goal
        if goal is None:
            # Render a few "GOAL REACHED" frames
            for _ in range(min(int(1.0 * fps), 30)):
                frame = env.render_background(config)
                renderer.render_simulation(
                    frame,
                    agent=agent,
                    state=State.NAVIGATE,
                    obstacles=env.get_obstacles(),
                    sensor_readings=[],
                    forces=[],
                    waypoints=[(wp.x, wp.y) for wp in nav_waypoints],
                    current_waypoint_idx=len(nav_waypoints),
                    goal_status="GOAL REACHED",
                    is_goal_complete=True,
                    agent_start=scenario_start,
                )
                frames.append(frame.copy())
            break

        gx, gy = goal

        # 4. Sensor scan
        heading = controller._heading_rad
        readings = sensor.scan(
            agent.x, agent.y, heading,
            env.get_obstacles(), env.get_boundary_segments(),
        )

        # 5. APF planning
        result = planner.compute(
            agent.x, agent.y, gx, gy, readings, agent_heading=heading,
        )
        total_force = result["total_force"]
        state = result["state"]
        forces = result["forces"]

        # 6. Control command
        if total_force.magnitude < 1e-10:
            desired_heading = heading
        else:
            desired_heading = total_force.heading

        fwd_cone = planner.forward_half_angle
        hit_distances = [
            r.distance for r in readings
            if r.hit and abs(normalize_angle(r.angle - heading)) <= fwd_cone
        ]
        min_obstacle_dist = min(hit_distances) if hit_distances else float("inf")
        desired_speed = planner.compute_speed(min_obstacle_dist, state)

        command = ControlCommand(
            desired_heading=desired_heading, desired_speed=desired_speed,
        )

        # 7. PID control + kinematics
        prev_x, prev_y = agent.x, agent.y
        controller.update(command, dt)
        agent = controller.get_agent_state()

        # 8. Collision response
        if env.check_collision(agent.x, agent.y, agent_radius):
            collisions += 1
            safe_x, safe_y = prev_x, prev_y
            for obs in env.get_obstacles():
                dx = agent.x - obs.x
                dy = agent.y - obs.y
                dist = math.sqrt(dx * dx + dy * dy)
                min_sep = agent_radius + obs.radius + 1.0
                if dist < min_sep and dist > 1e-10:
                    scale = min_sep / dist
                    safe_x = obs.x + dx * scale
                    safe_y = obs.y + dy * scale
                    break
            safe_x = max(agent_radius, min(safe_x, env.width - agent_radius))
            safe_y = max(agent_radius, min(safe_y, env.height - agent_radius))
            controller.agent = controller.agent.__class__(
                x=safe_x, y=safe_y,
                heading=agent.heading, velocity=agent.velocity,
                trajectory=agent.trajectory,
            )
            agent = controller.get_agent_state()

        # --- Collect data ---
        positions.append((agent.x, agent.y))
        headings_list.append(controller._heading_rad)
        velocities_list.append(agent.velocity)
        min_obstacle_dists.append(min_obstacle_dist)
        states_list.append(state)

        # --- Render frame (with skip) ---
        if step % frame_skip == 0:
            frame = env.render_background(config)
            wp_positions = [(wp.x, wp.y) for wp in nav_waypoints]
            status_text = (
                f"WP {navigator.current_waypoint_idx + 1}/{len(nav_waypoints)} | "
                f"Progress: {navigator.progress * 100:.0f}%"
            )
            renderer.render_simulation(
                frame, agent=agent, state=state,
                obstacles=env.get_obstacles(), sensor_readings=readings,
                forces=forces, waypoints=wp_positions,
                current_waypoint_idx=navigator.current_waypoint_idx,
                goal_status=status_text, is_goal_complete=False,
                agent_start=scenario_start,
            )
            frames.append(frame.copy())

        # Progress callback
        if progress_callback is not None:
            progress_callback((step + 1) / max_steps)

    # --- Encode video ---
    effective_fps = int(max(1, fps // frame_skip) * speed_multiplier)
    video_bytes = frames_to_video_bytes(frames, fps=effective_fps)

    run_data = RunData(
        scenario=label,
        positions=positions,
        headings=headings_list,
        velocities=velocities_list,
        min_obstacle_dists=min_obstacle_dists,
        states=states_list,
        collisions=collisions,
        waypoints_reached=navigator.current_waypoint_idx,
        total_waypoints=len(nav_waypoints),
        steps=len(positions),
        dt=dt,
        waypoint_positions=scenario_waypoints,
        start_position=scenario_start,
        obstacles=obstacle_snapshot,
    )

    return video_bytes, run_data
