"""Visualization module for rendering APF simulation overlays.

This module provides the VisualizationRenderer class that renders:
circular obstacles, sensor rays, force vectors, agent with heading arrow,
trajectory, waypoints, state indicators, and metrics using world_to_screen
coordinate conversion (y-up to y-down).
"""

import logging
import math
from typing import List, Dict, Optional, Tuple

import cv2  # type: ignore
import numpy as np

from src.config import Config
from src.models import State, AgentState, Obstacle, SensorReading, ForceVector

logger = logging.getLogger(__name__)

# Default colors (BGR format for OpenCV)
_DEFAULT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "agent": (0, 200, 255),
    "trajectory": (100, 100, 255),
    "obstacle_fill": (80, 80, 80),
    "obstacle_outline": (200, 200, 200),
    "sensor_ray_miss": (60, 60, 60),
    "sensor_ray_hit": (255, 80, 80),
    "force_attractive": (0, 255, 100),
    "force_repulsive": (255, 80, 80),
    "force_total": (0, 255, 255),
    "state_navigate": (0, 255, 0),
    "state_avoid": (255, 165, 0),
    "state_stop": (255, 0, 0),
    "waypoint": (255, 255, 0),
    "waypoint_reached": (100, 100, 100),
    "background": (40, 40, 40),
    "text": (255, 255, 255),
}

# Map State enum to color keys
_STATE_COLOR_KEY = {
    State.NAVIGATE: "state_navigate",
    State.AVOID: "state_avoid",
    State.STOP: "state_stop",
}


class VisualizationRenderer:
    """Renderer for APF simulation visualization overlays.

    All drawing uses world coordinates (y-up) converted to screen coordinates
    (y-down) via ``world_to_screen()``.

    Parameters
    ----------
    config : Config
        System configuration with visualization settings.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        vis = config.visualization  # type: ignore[attr-defined]

        # --- APF keys (with defaults) ---
        self.show_obstacles: bool = getattr(vis, "show_obstacles", True)
        self.show_sensor_rays: bool = getattr(vis, "show_sensor_rays", True)
        self.show_force_vectors: bool = getattr(vis, "show_force_vectors", True)
        self.show_waypoints: bool = getattr(vis, "show_waypoints", True)
        self.show_goal_status: bool = getattr(vis, "show_goal_status", True)
        try:
            self.force_vector_scale: float = float(
                vis.force_vector_scale  # type: ignore[attr-defined]
            )
        except (AttributeError, TypeError):
            self.force_vector_scale = 3.0

        # --- Shared keys ---
        self.show_agent: bool = getattr(vis, "show_agent", True)
        self.show_trajectory: bool = getattr(vis, "show_trajectory", True)
        self.show_state: bool = getattr(vis, "show_state", True)
        self.show_metrics: bool = getattr(vis, "show_metrics", True)

        # --- Colors from config or defaults ---
        self.colors: Dict[str, Tuple[int, int, int]] = dict(_DEFAULT_COLORS)
        try:
            cfg_colors = vis.colors  # type: ignore[attr-defined]
            for key in list(_DEFAULT_COLORS.keys()):
                val = getattr(cfg_colors, key, None)
                if val is not None:
                    self.colors[key] = tuple(val)  # type: ignore[assignment]
        except (AttributeError, TypeError):
            pass

        # --- World height for coordinate conversion ---
        try:
            self.world_height: float = float(
                config.environment.height  # type: ignore[attr-defined]
            )
        except (AttributeError, TypeError):
            self.world_height = 600.0

        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2

        # Agent visualization settings
        self.agent_radius = 10
        self.arrow_length = 30

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world (y-up) to screen (y-down) coordinates.

        Parameters
        ----------
        wx : float
            World x coordinate.
        wy : float
            World y coordinate.

        Returns
        -------
        Tuple[int, int]
            Screen (x, y) in pixel coordinates.
        """
        return (int(wx), int(self.world_height - wy))

    # ------------------------------------------------------------------
    # Drawing methods
    # ------------------------------------------------------------------

    def draw_obstacles(
        self, frame: np.ndarray, obstacles: List[Obstacle]
    ) -> None:
        """Draw circular obstacles with fill and outline.

        Parameters
        ----------
        frame : np.ndarray
            The frame to draw on (modified in-place).
        obstacles : List[Obstacle]
            Obstacles in world coordinates.
        """
        if not self.show_obstacles:
            return

        for obs in obstacles:
            center = self.world_to_screen(obs.x, obs.y)
            radius = int(obs.radius)
            cv2.circle(frame, center, radius, self.colors["obstacle_fill"], -1)
            cv2.circle(
                frame, center, radius, self.colors["obstacle_outline"], 2
            )

    def draw_sensor_rays(
        self,
        frame: np.ndarray,
        agent_x: float,
        agent_y: float,
        readings: List[SensorReading],
    ) -> None:
        """Draw sensor ray lines from agent to hit/max-range points.

        Parameters
        ----------
        frame : np.ndarray
            The frame to draw on (modified in-place).
        agent_x : float
            Agent x in world coordinates.
        agent_y : float
            Agent y in world coordinates.
        readings : List[SensorReading]
            Sensor readings with angle, distance, hit info.
        """
        if not self.show_sensor_rays:
            return

        start = self.world_to_screen(agent_x, agent_y)

        for reading in readings:
            if reading.hit and reading.hit_point is not None:
                end = self.world_to_screen(*reading.hit_point)
                color = self.colors["sensor_ray_hit"]
            else:
                end_wx = agent_x + reading.distance * math.cos(reading.angle)
                end_wy = agent_y + reading.distance * math.sin(reading.angle)
                end = self.world_to_screen(end_wx, end_wy)
                color = self.colors["sensor_ray_miss"]

            cv2.line(frame, start, end, color, 1)

    def draw_force_vectors(
        self,
        frame: np.ndarray,
        agent_x: float,
        agent_y: float,
        forces: List[ForceVector],
    ) -> None:
        """Draw force arrows from agent position, colored by source.

        Parameters
        ----------
        frame : np.ndarray
            The frame to draw on (modified in-place).
        agent_x : float
            Agent x in world coordinates.
        agent_y : float
            Agent y in world coordinates.
        forces : List[ForceVector]
            Force vectors with fx, fy, source label.
        """
        if not self.show_force_vectors:
            return

        start = self.world_to_screen(agent_x, agent_y)

        for force in forces:
            end_wx = agent_x + force.fx * self.force_vector_scale
            end_wy = agent_y + force.fy * self.force_vector_scale
            end = self.world_to_screen(end_wx, end_wy)
            color_key = f"force_{force.source}"
            color = self.colors.get(color_key, self.colors["force_total"])
            cv2.arrowedLine(frame, start, end, color, 2, tipLength=0.3)

    def draw_agent(self, frame: np.ndarray, agent: AgentState) -> None:
        """Draw agent circle and heading arrow (world coords, radians).

        Parameters
        ----------
        frame : np.ndarray
            The frame to draw on (modified in-place).
        agent : AgentState
            Agent state with position in world coords and heading in radians.
        """
        if not self.show_agent:
            return

        center = self.world_to_screen(agent.x, agent.y)

        # Draw agent circle
        cv2.circle(
            frame, center, self.agent_radius, self.colors["agent"], -1
        )

        # Heading arrow (heading in radians, y-up convention)
        end_wx = agent.x + self.arrow_length * math.cos(agent.heading)
        end_wy = agent.y + self.arrow_length * math.sin(agent.heading)
        end = self.world_to_screen(end_wx, end_wy)

        cv2.arrowedLine(
            frame, center, end, self.colors["agent"], 2, tipLength=0.3
        )

    def draw_trajectory(
        self, frame: np.ndarray, trajectory: List[Tuple[float, float]]
    ) -> None:
        """Draw trajectory polyline with world-to-screen conversion.

        Parameters
        ----------
        frame : np.ndarray
            The frame to draw on (modified in-place).
        trajectory : List[Tuple[float, float]]
            List of (x, y) positions in world coordinates.
        """
        if not self.show_trajectory or len(trajectory) < 2:
            return

        screen_points = [self.world_to_screen(x, y) for x, y in trajectory]
        points = np.array(screen_points, dtype=np.int32).reshape((-1, 1, 2))

        cv2.polylines(
            frame,
            [points],
            isClosed=False,
            color=self.colors["trajectory"],
            thickness=2,
        )

    def draw_state_indicator(self, frame: np.ndarray, state: State) -> None:
        """Draw state indicator in top-left corner.

        Parameters
        ----------
        frame : np.ndarray
            The frame to draw on (modified in-place).
        state : State
            Current navigation state.
        """
        if not self.show_state:
            return

        text = f"STATE: {state.value.upper()}"
        position = (20, 40)
        color = self.colors.get(
            _STATE_COLOR_KEY.get(state, ""), self.colors["text"]
        )

        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness
        )
        cv2.rectangle(
            frame,
            (10, 10),
            (30 + text_width, 50 + baseline),
            self.colors["background"],
            -1,
        )
        cv2.putText(
            frame,
            text,
            position,
            self.font,
            self.font_scale,
            color,
            self.font_thickness,
        )

    def draw_metrics(self, frame: np.ndarray, metrics: Dict[str, float]) -> None:
        """Draw metrics panel in bottom-left corner.

        Parameters
        ----------
        frame : np.ndarray
            The frame to draw on (modified in-place).
        metrics : Dict[str, float]
            Dictionary with keys: speed, heading, obstacles.
        """
        if not self.show_metrics:
            return

        speed = metrics.get("speed", 0.0)
        heading = metrics.get("heading", 0.0)
        obstacles = int(metrics.get("obstacles", 0))

        text = (
            f"Speed: {speed:.1f} | "
            f"Heading: {heading:.2f} rad | "
            f"Obstacles: {obstacles}"
        )

        frame_height = frame.shape[0]
        position = (20, frame_height - 30)

        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness
        )
        cv2.rectangle(
            frame,
            (10, frame_height - text_height - baseline - 50),
            (30 + text_width, frame_height - 10),
            self.colors["background"],
            -1,
        )
        cv2.putText(
            frame,
            text,
            position,
            self.font,
            self.font_scale,
            self.colors["text"],
            self.font_thickness,
        )

    def draw_waypoints(
        self,
        frame: np.ndarray,
        waypoints: List[Tuple[float, float]],
        current_idx: int,
        agent_start: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Draw waypoints with color-coded status (world coords).

        Parameters
        ----------
        frame : np.ndarray
            The frame to draw on (modified in-place).
        waypoints : List[Tuple[float, float]]
            List of (x, y) waypoint positions in world coordinates.
        current_idx : int
            Index of the current target waypoint.
        agent_start : Optional[Tuple[float, float]]
            Agent starting position (x, y) in world coordinates.
            If provided, a line is drawn from start to first waypoint.
        """
        if not self.show_waypoints or not waypoints:
            return

        # Draw line from agent start to first waypoint
        if agent_start is not None:
            start_screen = self.world_to_screen(*agent_start)
            first_wp_screen = self.world_to_screen(*waypoints[0])
            cv2.line(frame, start_screen, first_wp_screen, (180, 180, 180), 2)

        # Draw connecting lines between waypoints
        for i in range(len(waypoints) - 1):
            start = self.world_to_screen(*waypoints[i])
            end = self.world_to_screen(*waypoints[i + 1])
            cv2.line(frame, start, end, (180, 180, 180), 2)

        # Draw waypoint markers with color based on status
        for i, wp in enumerate(waypoints):
            center = self.world_to_screen(*wp)

            if i < current_idx:
                # Reached waypoints
                color = self.colors["waypoint_reached"]
            elif i == current_idx:
                # Current target waypoint
                color = self.colors["waypoint"]
            else:
                # Future waypoints
                color = (100, 100, 100)

            cv2.circle(frame, center, 8, color, -1)
            cv2.circle(frame, center, 10, (255, 255, 255), 2)

    def draw_goal_status(
        self,
        frame: np.ndarray,
        status_text: str,
        is_complete: bool,
    ) -> None:
        """Draw goal status text or completion banner.

        Parameters
        ----------
        frame : np.ndarray
            The frame to draw on (modified in-place).
        status_text : str
            Status text to display.
        is_complete : bool
            Whether goal is complete (shows large banner if True).
        """
        if is_complete:
            # Draw large centered "GOAL REACHED" banner
            banner_text = "GOAL REACHED"
            banner_font_scale = 2.0
            banner_thickness = 3

            (text_width, text_height), baseline = cv2.getTextSize(
                banner_text, self.font, banner_font_scale, banner_thickness
            )

            frame_height, frame_width = frame.shape[:2]
            center_x = frame_width // 2
            center_y = frame_height // 2

            rect_padding = 30
            cv2.rectangle(
                frame,
                (
                    center_x - text_width // 2 - rect_padding,
                    center_y - text_height // 2 - rect_padding,
                ),
                (
                    center_x + text_width // 2 + rect_padding,
                    center_y + text_height // 2 + rect_padding + baseline,
                ),
                (0, 200, 0),
                -1,
            )
            cv2.putText(
                frame,
                banner_text,
                (center_x - text_width // 2, center_y + text_height // 2),
                self.font,
                banner_font_scale,
                self.colors["text"],
                banner_thickness,
            )
        else:
            # Draw small status text in top-right corner
            (text_width, text_height), baseline = cv2.getTextSize(
                status_text, self.font, self.font_scale, self.font_thickness
            )

            frame_width = frame.shape[1]
            position = (frame_width - text_width - 20, 40)

            cv2.rectangle(
                frame,
                (frame_width - text_width - 30, 10),
                (frame_width - 10, 50 + baseline),
                self.colors["background"],
                -1,
            )
            cv2.putText(
                frame,
                status_text,
                position,
                self.font,
                self.font_scale,
                self.colors["text"],
                self.font_thickness,
            )

    # ------------------------------------------------------------------
    # Top-level orchestrator
    # ------------------------------------------------------------------

    def render_simulation(
        self,
        frame: np.ndarray,
        agent: AgentState,
        state: State,
        obstacles: List[Obstacle],
        sensor_readings: List[SensorReading],
        forces: List[ForceVector],
        waypoints: Optional[List[Tuple[float, float]]] = None,
        current_waypoint_idx: int = 0,
        goal_status: Optional[str] = None,
        is_goal_complete: bool = False,
        agent_start: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """Render APF simulation visualization (world coords, radians).

        Calls all drawing methods and returns the annotated frame.

        Parameters
        ----------
        frame : np.ndarray
            The input frame to annotate.
        agent : AgentState
            Agent state (world coords, heading in radians).
        state : State
            Current navigation state.
        obstacles : List[Obstacle]
            Obstacles in world coordinates.
        sensor_readings : List[SensorReading]
            Sensor readings to visualize.
        forces : List[ForceVector]
            Force vectors to visualize.
        waypoints : Optional[List[Tuple[float, float]]]
            Optional waypoints in world coordinates.
        current_waypoint_idx : int
            Current target waypoint index.
        goal_status : Optional[str]
            Optional goal status text.
        is_goal_complete : bool
            Whether goal is complete.
        agent_start : Optional[Tuple[float, float]]
            Agent starting position for waypoint line drawing.

        Returns
        -------
        np.ndarray
            The annotated frame.
        """
        self.draw_obstacles(frame, obstacles)
        self.draw_sensor_rays(frame, agent.x, agent.y, sensor_readings)
        self.draw_force_vectors(frame, agent.x, agent.y, forces)
        self.draw_trajectory(frame, agent.trajectory)
        self.draw_agent(frame, agent)
        self.draw_state_indicator(frame, state)

        metrics: Dict[str, float] = {
            "speed": agent.velocity,
            "heading": agent.heading,
            "obstacles": float(len(obstacles)),
        }
        self.draw_metrics(frame, metrics)

        if waypoints:
            self.draw_waypoints(frame, waypoints, current_waypoint_idx,
                                agent_start=agent_start)

        if goal_status is not None:
            self.draw_goal_status(frame, goal_status, is_goal_complete)

        return frame
