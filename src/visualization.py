"""Visualization module for rendering frame overlays.

This module provides the VisualizationRenderer class that renders bounding boxes,
agent position/heading, trajectory history, state indicators, metrics panels,
waypoints, and goal status on video frames.
"""

import logging
import math
from typing import List, Dict, Optional, Tuple

import cv2  # type: ignore
import numpy as np

from src.config import Config
from src.models import Detection, State, AgentState


class VisualizationRenderer:
    """Renderer for navigation visualization overlays.

    Renders detection bounding boxes, agent visualization, trajectory lines,
    state indicators, and metrics panels on video frames.

    Parameters
    ----------
    config : Config
        System configuration with visualization settings.
    """

    # Color scheme (BGR format for OpenCV)
    COLORS = {
        State.NAVIGATE: (0, 255, 0),      # Green
        State.AVOID: (0, 165, 255),       # Orange
        State.STOP: (0, 0, 255),          # Red
        "agent": (255, 255, 0),           # Cyan
        "trajectory": (0, 255, 255),      # Yellow
        "text": (255, 255, 255),          # White
        "background": (0, 0, 0),          # Black
    }

    def __init__(self, config: Config) -> None:
        """Initialize renderer with configuration.

        Parameters
        ----------
        config : Config
            System configuration.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Extract visualization settings
        self.show_boxes: bool = (
            config.visualization.show_boxes  # type: ignore[attr-defined]
        )
        self.show_labels: bool = (
            config.visualization.show_labels  # type: ignore[attr-defined]
        )
        self.show_confidence: bool = (
            config.visualization.show_confidence  # type: ignore[attr-defined]
        )
        self.show_agent: bool = (
            config.visualization.show_agent  # type: ignore[attr-defined]
        )
        self.show_trajectory: bool = (
            config.visualization.show_trajectory  # type: ignore[attr-defined]
        )
        self.show_state: bool = (
            config.visualization.show_state  # type: ignore[attr-defined]
        )
        self.show_metrics: bool = (
            config.visualization.show_metrics  # type: ignore[attr-defined]
        )

        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2

        # Agent visualization settings
        self.agent_radius = 10
        self.arrow_length = 30

    def draw_detections(
        self, frame: np.ndarray, detections: List[Detection], state: State
    ) -> None:
        """Draw bounding boxes and labels for detections.

        Parameters
        ----------
        frame : np.ndarray
            The frame to draw on (modified in-place).
        detections : List[Detection]
            List of detected obstacles.
        state : State
            Current navigation state (determines box color).
        """
        if not self.show_boxes:
            return

        color = self.COLORS[state]

        for detection in detections:
            x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2

            # Draw bounding box
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2,
            )

            # Draw label if enabled
            if self.show_labels:
                label = detection.class_name
                if self.show_confidence:
                    label += f" {detection.confidence:.2f}"

                # Draw label with background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, self.font, self.font_scale, self.font_thickness
                )
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1) - text_height - baseline - 5),
                    (int(x1) + text_width, int(y1)),
                    color,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 5),
                    self.font,
                    self.font_scale,
                    self.COLORS["text"],
                    self.font_thickness,
                )

    def draw_agent(self, frame: np.ndarray, agent: AgentState) -> None:
        """Draw agent circle and heading arrow.

        Parameters
        ----------
        frame : np.ndarray
            The frame to draw on (modified in-place).
        agent : AgentState
            Current agent state with position and heading.
        """
        if not self.show_agent:
            return

        center = (int(agent.x), int(agent.y))

        # Draw agent circle
        cv2.circle(
            frame,
            center,
            self.agent_radius,
            self.COLORS["agent"],
            -1,
        )

        # Calculate heading arrow endpoint
        heading_rad = math.radians(agent.heading)
        end_x = int(agent.x + self.arrow_length * math.cos(heading_rad))
        end_y = int(agent.y - self.arrow_length * math.sin(heading_rad))

        # Draw heading arrow
        cv2.arrowedLine(
            frame,
            center,
            (end_x, end_y),
            self.COLORS["agent"],
            2,
            tipLength=0.3,
        )

    def draw_trajectory(
        self, frame: np.ndarray, trajectory: List[Tuple[float, float]]
    ) -> None:
        """Draw trajectory polyline.

        Parameters
        ----------
        frame : np.ndarray
            The frame to draw on (modified in-place).
        trajectory : List[Tuple[float, float]]
            List of (x, y) positions in the agent's trajectory.
        """
        if not self.show_trajectory or len(trajectory) < 2:
            return

        # Convert trajectory to numpy array of integer points
        points = np.array(trajectory, dtype=np.int32)
        points = points.reshape((-1, 1, 2))

        # Draw polyline
        cv2.polylines(
            frame,
            [points],
            isClosed=False,
            color=self.COLORS["trajectory"],
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
        color = self.COLORS[state]

        # Draw background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness
        )
        cv2.rectangle(
            frame,
            (10, 10),
            (30 + text_width, 50 + baseline),
            self.COLORS["background"],
            -1,
        )

        # Draw text
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
            Dictionary with keys: detections, velocity, heading.
        """
        if not self.show_metrics:
            return

        # Format metrics text
        detections = int(metrics.get("detections", 0))
        velocity = metrics.get("velocity", 0.0)
        heading = metrics.get("heading", 0.0)

        text = (
            f"Detections: {detections} | "
            f"Velocity: {velocity:.1f} | "
            f"Heading: {heading:.0f}°"
        )

        # Position at bottom-left
        frame_height = frame.shape[0]
        position = (20, frame_height - 30)

        # Draw background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness
        )
        cv2.rectangle(
            frame,
            (10, frame_height - text_height - baseline - 50),
            (30 + text_width, frame_height - 10),
            self.COLORS["background"],
            -1,
        )

        # Draw text
        cv2.putText(
            frame,
            text,
            position,
            self.font,
            self.font_scale,
            self.COLORS["text"],
            self.font_thickness,
        )

    def draw_waypoints(
        self,
        frame: np.ndarray,
        waypoints: List[List[float]],
        current_idx: int
    ) -> None:
        """Draw waypoints with color-coded status.

        Parameters
        ----------
        frame : np.ndarray
            The frame to draw on (modified in-place).
        waypoints : List[List[float]]
            List of [x, y] waypoint coordinates.
        current_idx : int
            Index of the current target waypoint.
        """
        if not waypoints:
            return

        # Draw connecting lines between waypoints
        for i in range(len(waypoints) - 1):
            start = (int(waypoints[i][0]), int(waypoints[i][1]))
            end = (int(waypoints[i + 1][0]), int(waypoints[i + 1][1]))
            cv2.line(frame, start, end, (180, 180, 180), 2)

        # Draw waypoint markers with color based on status
        for i, waypoint in enumerate(waypoints):
            center = (int(waypoint[0]), int(waypoint[1]))

            if i < current_idx:
                # Reached waypoints: green
                color = (0, 255, 0)
            elif i == current_idx:
                # Current target waypoint: yellow
                color = (0, 255, 255)
            else:
                # Future waypoints: gray
                color = (100, 100, 100)

            # Draw filled circle
            cv2.circle(frame, center, 8, color, -1)
            # Draw outline
            cv2.circle(frame, center, 10, (255, 255, 255), 2)

    def draw_goal_status(
        self,
        frame: np.ndarray,
        status_text: str,
        is_complete: bool
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
                banner_text,
                self.font,
                banner_font_scale,
                banner_thickness
            )

            frame_height, frame_width = frame.shape[:2]
            center_x = frame_width // 2
            center_y = frame_height // 2

            # Draw semi-transparent background
            rect_padding = 30
            cv2.rectangle(
                frame,
                (center_x - text_width // 2 - rect_padding,
                 center_y - text_height // 2 - rect_padding),
                (center_x + text_width // 2 + rect_padding,
                 center_y + text_height // 2 + rect_padding + baseline),
                (0, 200, 0),  # Green background
                -1
            )

            # Draw banner text
            cv2.putText(
                frame,
                banner_text,
                (center_x - text_width // 2, center_y + text_height // 2),
                self.font,
                banner_font_scale,
                self.COLORS["text"],
                banner_thickness
            )
        else:
            # Draw small status text in top-right corner
            (text_width, text_height), baseline = cv2.getTextSize(
                status_text,
                self.font,
                self.font_scale,
                self.font_thickness
            )

            frame_width = frame.shape[1]
            position = (frame_width - text_width - 20, 40)

            # Draw background
            cv2.rectangle(
                frame,
                (frame_width - text_width - 30, 10),
                (frame_width - 10, 50 + baseline),
                self.COLORS["background"],
                -1
            )

            # Draw text
            cv2.putText(
                frame,
                status_text,
                position,
                self.font,
                self.font_scale,
                self.COLORS["text"],
                self.font_thickness
            )

    def render(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        agent: AgentState,
        state: State,
        waypoints: Optional[List[List[float]]] = None,
        waypoint_idx: Optional[int] = None,
        goal_status: Optional[str] = None,
        is_goal_complete: bool = False,
    ) -> np.ndarray:
        """Render all visualization elements on the frame.

        Parameters
        ----------
        frame : np.ndarray
            The input frame to annotate.
        detections : List[Detection]
            Detected obstacles.
        agent : AgentState
            Current agent state.
        state : State
            Current navigation state.
        waypoints : Optional[List[List[float]]]
            Optional list of [x, y] waypoint coordinates to draw.
        waypoint_idx : Optional[int]
            Optional current waypoint index (for color-coding).
        goal_status : Optional[str]
            Optional goal status text to display.
        is_goal_complete : bool
            Whether the goal has been reached (shows banner if True).

        Returns
        -------
        np.ndarray
            The annotated frame.
        """
        # Draw all standard elements (modifies frame in-place)
        self.draw_detections(frame, detections, state)
        self.draw_trajectory(frame, agent.trajectory)
        self.draw_agent(frame, agent)
        self.draw_state_indicator(frame, state)

        # Prepare and draw metrics
        metrics = {
            "detections": len(detections),
            "velocity": agent.velocity,
            "heading": agent.heading,
        }
        self.draw_metrics(frame, metrics)

        # Draw waypoints if provided
        if waypoints is not None and waypoint_idx is not None:
            self.draw_waypoints(frame, waypoints, waypoint_idx)

        # Draw goal status if provided
        if goal_status is not None:
            self.draw_goal_status(frame, goal_status, is_goal_complete)

        return frame
