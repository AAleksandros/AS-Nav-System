"""Synthetic demo pipeline runner.

This module orchestrates the synthetic demo mode, combining scene generation,
waypoint navigation, blended planning, agent control, and visualization
into a complete pipeline that generates compelling navigation demos.
"""

import logging
from typing import List

from src.blended_planner import BlendedNavigationPlanner
from src.config import Config
from src.control import AgentController
from src.synthetic_scene import (
    SyntheticScene,
    create_gauntlet_scenario,
    create_crossing_scenario,
    create_converging_scenario,
)
from src.utils.video_utils import VideoWriter
from src.visualization import VisualizationRenderer
from src.waypoint_navigator import Waypoint, WaypointNavigator


logger = logging.getLogger(__name__)


def run_synthetic_pipeline(config: Config, output_path: str) -> int:
    """Run synthetic demo pipeline.

    Orchestrates all components to generate a synthetic navigation demo video:
    - Creates synthetic scene with moving obstacles
    - Sets up waypoint-based navigation with blended avoidance
    - Simulates agent movement with boundary clamping
    - Renders visualization with waypoints and goal status
    - Writes output video

    Args:
        config: System configuration with synthetic settings
        output_path: Path for output video file

    Returns:
        0 on success, non-zero on error
    """
    logger.info("Starting synthetic pipeline")

    try:
        # Extract synthetic config
        width: int = config.synthetic.width  # type: ignore[attr-defined]
        height: int = config.synthetic.height  # type: ignore[attr-defined]
        fps: int = config.synthetic.fps  # type: ignore[attr-defined]
        duration: int = (
            config.synthetic.duration_seconds  # type: ignore[attr-defined]
        )
        scenario: str = config.synthetic.scenario  # type: ignore[attr-defined]
        waypoint_coords: List[List[float]] = (
            config.synthetic.waypoints  # type: ignore[attr-defined]
        )
        waypoint_tolerance: float = (
            config.synthetic.waypoint_tolerance  # type: ignore[attr-defined]
        )

        logger.info(
            f"Synthetic config: {width}x{height} @ {fps}fps, "
            f"{duration}s, scenario={scenario}, "
            f"{len(waypoint_coords)} waypoints"
        )

        # Create synthetic scene with chosen scenario
        if scenario == "gauntlet":
            obstacles = create_gauntlet_scenario(width, height)
        elif scenario == "crossing":
            obstacles = create_crossing_scenario(width, height)
        elif scenario == "converging":
            obstacles = create_converging_scenario(width, height)
        else:
            logger.error(f"Unknown scenario: {scenario}")
            return 1

        scene = SyntheticScene(
            width=width, height=height, obstacles=obstacles, waypoints=waypoint_coords
        )

        # Create waypoint navigator
        waypoints = [
            Waypoint(x=coord[0], y=coord[1], tolerance=waypoint_tolerance)
            for coord in waypoint_coords
        ]
        navigator = WaypointNavigator(
            waypoints,
            turn_rate=config.control.turn_rate,  # type: ignore[attr-defined]
            speed=config.control.agent_speed,  # type: ignore[attr-defined]
        )

        # Create blended navigation planner
        planner = BlendedNavigationPlanner(config, navigator)

        # Create agent controller with boundary clamping
        controller = AgentController(
            config, frame_width=width, frame_height=height
        )

        # Create visualization renderer
        renderer = VisualizationRenderer(config)

        # Create video writer
        writer = VideoWriter(
            output_path=output_path, fps=float(fps), width=width, height=height
        )

        logger.info(f"Writing output to: {output_path}")

        # Frame loop
        total_frames = duration * fps
        goal_reached_hold_frames = fps * 2  # Hold for 2 seconds after goal
        frames_after_goal = 0

        for frame_idx in range(total_frames):
            # Render scene frame with obstacles
            frame, detections = scene.render_frame()

            # Get agent state
            agent_state = controller.get_agent_state()

            # Compute navigation state and action
            state, action = planner.update(detections, agent_state)

            # Execute action
            controller.execute_action(action)

            # Get updated agent state for visualization
            agent_state = controller.get_agent_state()

            # Render visualization with waypoints and goal status
            is_goal_complete = planner.is_goal_reached
            goal_status = (
                "GOAL REACHED"
                if is_goal_complete
                else planner.goal_status(agent_state)
            )

            annotated_frame = renderer.render(
                frame=frame,
                detections=detections,
                agent=agent_state,
                state=state,
                waypoints=waypoint_coords,
                waypoint_idx=navigator.current_waypoint_idx,
                goal_status=goal_status,
                is_goal_complete=is_goal_complete,
            )

            # Write frame
            writer.write_frame(annotated_frame)

            # Check if goal reached and should stop early
            if is_goal_complete:
                frames_after_goal += 1
                if frames_after_goal >= goal_reached_hold_frames:
                    logger.info(
                        f"Goal reached at frame {frame_idx}, "
                        f"held for {frames_after_goal} frames, ending"
                    )
                    break

            # Log progress periodically
            if (frame_idx + 1) % (fps * 5) == 0 or frame_idx == total_frames - 1:
                progress = (frame_idx + 1) / total_frames * 100
                logger.info(
                    f"Progress: {frame_idx + 1}/{total_frames} "
                    f"({progress:.1f}%), State: {state.value}"
                )

        # Release resources
        writer.release()

        logger.info(
            f"Synthetic pipeline complete: {frame_idx + 1} frames written"
        )
        return 0

    except Exception as e:
        logger.exception(f"Synthetic pipeline failed: {e}")
        return 1
