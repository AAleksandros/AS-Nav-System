"""Synthetic scene generation for demo mode.

This module provides synthetic obstacle and scene generation for creating
compelling navigation demos without relying on real video footage.
"""

import logging
from typing import List, Tuple

import cv2  # type: ignore
import numpy as np

from src.models import Detection


logger = logging.getLogger(__name__)


class SyntheticObstacle:
    """Moving obstacle for synthetic scenes.

    An obstacle with position, velocity, size, and appearance that moves
    within scene boundaries and bounces off walls.

    Attributes:
        x: X position in pixels
        y: Y position in pixels
        vx: X velocity in pixels per frame
        vy: Y velocity in pixels per frame
        width: Obstacle width in pixels
        height: Obstacle height in pixels
        color: BGR color tuple for rendering
        label: Class label (e.g., "car", "person")
        class_id: COCO class ID
    """

    def __init__(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        width: int,
        height: int,
        color: Tuple[int, int, int],
        label: str,
        class_id: int = 0,
    ):
        """Initialize synthetic obstacle.

        Args:
            x: Initial x position
            y: Initial y position
            vx: X velocity (pixels per frame)
            vy: Y velocity (pixels per frame)
            width: Obstacle width
            height: Obstacle height
            color: BGR color for rendering
            label: Class name
            class_id: COCO class ID
        """
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.width = width
        self.height = height
        self.color = color
        self.label = label
        self.class_id = class_id

    def update(self, frame_width: int, frame_height: int) -> None:
        """Update obstacle position and handle boundary bouncing.

        Args:
            frame_width: Scene width in pixels
            frame_height: Scene height in pixels
        """
        # Update position
        self.x += self.vx
        self.y += self.vy

        # Bounce off horizontal boundaries
        if self.x < 0:
            self.x = 0
            self.vx = abs(self.vx)
        elif self.x + self.width > frame_width:
            self.x = frame_width - self.width
            self.vx = -abs(self.vx)

        # Bounce off vertical boundaries
        if self.y < 0:
            self.y = 0
            self.vy = abs(self.vy)
        elif self.y + self.height > frame_height:
            self.y = frame_height - self.height
            self.vy = -abs(self.vy)

    def draw(self, frame: np.ndarray) -> None:
        """Draw obstacle on frame.

        Args:
            frame: Frame to draw on (modified in-place)
        """
        x1, y1 = int(self.x), int(self.y)
        x2, y2 = int(self.x + self.width), int(self.y + self.height)

        # Draw filled rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, -1)

        # Draw white border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Draw label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(
            self.label, font, font_scale, font_thickness
        )

        # Center text in obstacle
        text_x = x1 + (self.width - text_width) // 2
        text_y = y1 + (self.height + text_height) // 2

        cv2.putText(
            frame,
            self.label,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )

    def to_detection(self) -> Detection:
        """Convert obstacle to Detection dataclass.

        Returns:
            Detection with obstacle bounding box and class info
        """
        return Detection(
            x1=self.x,
            y1=self.y,
            x2=self.x + self.width,
            y2=self.y + self.height,
            class_id=self.class_id,
            class_name=self.label,
            confidence=1.0,
        )


class SyntheticScene:
    """Manages full synthetic scene with background, obstacles, and waypoints.

    Attributes:
        width: Scene width in pixels
        height: Scene height in pixels
        obstacles: List of moving obstacles
        waypoints: List of [x, y] waypoint positions
    """

    def __init__(
        self,
        width: int,
        height: int,
        obstacles: List[SyntheticObstacle],
        waypoints: List[List[float]],
    ):
        """Initialize synthetic scene.

        Args:
            width: Scene width in pixels
            height: Scene height in pixels
            obstacles: List of obstacles to simulate
            waypoints: List of [x, y] waypoint coordinates
        """
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.waypoints = waypoints

        logger.info(
            f"SyntheticScene initialized: {width}x{height}, "
            f"{len(obstacles)} obstacles, {len(waypoints)} waypoints"
        )

    def update(self) -> None:
        """Advance all obstacle positions."""
        for obstacle in self.obstacles:
            obstacle.update(self.width, self.height)

    def render_frame(self) -> Tuple[np.ndarray, List[Detection]]:
        """Render complete scene frame with obstacles and waypoints.

        Returns:
            Tuple of (frame, detections) where frame is RGB numpy array
            and detections is list of Detection objects
        """
        # Update obstacle positions
        self.update()

        # Create light gray background
        frame = np.full((self.height, self.width, 3), 200, dtype=np.uint8)

        # Draw grid pattern
        grid_spacing = 50
        grid_color = (220, 220, 220)

        for x in range(0, self.width, grid_spacing):
            cv2.line(frame, (x, 0), (x, self.height), grid_color, 1)

        for y in range(0, self.height, grid_spacing):
            cv2.line(frame, (0, y), (self.width, y), grid_color, 1)

        # Draw waypoint path as dashed line
        if len(self.waypoints) >= 2:
            self._draw_waypoint_path(frame)

        # Draw obstacles
        for obstacle in self.obstacles:
            obstacle.draw(frame)

        # Convert obstacles to detections
        detections = [obs.to_detection() for obs in self.obstacles]

        return frame, detections

    def _draw_waypoint_path(self, frame: np.ndarray) -> None:
        """Draw dashed line connecting waypoints.

        Args:
            frame: Frame to draw on (modified in-place)
        """
        path_color = (150, 150, 150)
        dash_length = 10
        gap_length = 5

        for i in range(len(self.waypoints) - 1):
            start = (int(self.waypoints[i][0]), int(self.waypoints[i][1]))
            end = (int(self.waypoints[i + 1][0]), int(self.waypoints[i + 1][1]))

            # Calculate line segments for dashed effect
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            distance = np.sqrt(dx * dx + dy * dy)

            if distance < 1:
                continue

            num_dashes = int(distance / (dash_length + gap_length))

            for j in range(num_dashes + 1):
                t1 = j * (dash_length + gap_length) / distance
                t2 = min(1.0, (j * (dash_length + gap_length) + dash_length) / distance)

                p1 = (
                    int(start[0] + t1 * dx),
                    int(start[1] + t1 * dy)
                )
                p2 = (
                    int(start[0] + t2 * dx),
                    int(start[1] + t2 * dy)
                )

                cv2.line(frame, p1, p2, path_color, 2)

        # Draw waypoint markers as circles
        for waypoint in self.waypoints:
            center = (int(waypoint[0]), int(waypoint[1]))
            cv2.circle(frame, center, 5, (100, 100, 100), -1)
            cv2.circle(frame, center, 8, (150, 150, 150), 2)


def create_gauntlet_scenario(width: int, height: int) -> List[SyntheticObstacle]:
    """Create gauntlet scenario with corridor of obstacles.

    Obstacles form a corridor with gaps that the agent must navigate through.

    Args:
        width: Scene width in pixels
        height: Scene height in pixels

    Returns:
        List of obstacles forming a gauntlet
    """
    obstacles = []

    # Left side obstacles
    for i in range(4):
        y = 150 + i * 150
        obstacles.append(
            SyntheticObstacle(
                x=width * 0.25,
                y=y,
                vx=0.0,
                vy=1.0 if i % 2 == 0 else -1.0,
                width=80,
                height=60,
                color=(200, 100, 50),
                label="car",
                class_id=2,
            )
        )

    # Right side obstacles
    for i in range(4):
        y = 200 + i * 150
        obstacles.append(
            SyntheticObstacle(
                x=width * 0.65,
                y=y,
                vx=0.0,
                vy=-1.0 if i % 2 == 0 else 1.0,
                width=80,
                height=60,
                color=(50, 100, 200),
                label="truck",
                class_id=7,
            )
        )

    logger.info(f"Created gauntlet scenario with {len(obstacles)} obstacles")
    return obstacles


def create_crossing_scenario(width: int, height: int) -> List[SyntheticObstacle]:
    """Create crossing scenario with obstacles crossing agent's path.

    Obstacles move horizontally across the scene at different heights.

    Args:
        width: Scene width in pixels
        height: Scene height in pixels

    Returns:
        List of crossing obstacles
    """
    obstacles = []

    # Horizontal crossing obstacles at different heights
    heights = [200, 350, 500]
    for i, y in enumerate(heights):
        # Left-to-right obstacle
        obstacles.append(
            SyntheticObstacle(
                x=50.0 + i * 100,
                y=y,
                vx=2.5,
                vy=0.0,
                width=70,
                height=50,
                color=(180, 50, 50),
                label="car",
                class_id=2,
            )
        )

        # Right-to-left obstacle
        obstacles.append(
            SyntheticObstacle(
                x=width - 150.0 - i * 100,
                y=y + 50,
                vx=-2.0,
                vy=0.0,
                width=60,
                height=45,
                color=(50, 180, 50),
                label="person",
                class_id=0,
            )
        )

    logger.info(f"Created crossing scenario with {len(obstacles)} obstacles")
    return obstacles


def create_converging_scenario(width: int, height: int) -> List[SyntheticObstacle]:
    """Create converging scenario with obstacles moving toward center.

    Obstacles start from edges and move toward the center of the scene.

    Args:
        width: Scene width in pixels
        height: Scene height in pixels

    Returns:
        List of converging obstacles
    """
    obstacles = []
    center_x, center_y = width / 2, height / 2

    # Create obstacles from four corners
    corner_configs = [
        # (start_x, start_y, vx, vy, color, label)
        (100.0, 100.0, 1.5, 1.2, (200, 50, 50), "car"),
        (width - 100.0, 100.0, -1.5, 1.2, (50, 200, 50), "bicycle"),
        (100.0, height - 100.0, 1.5, -1.2, (50, 50, 200), "person"),
        (width - 100.0, height - 100.0, -1.5, -1.2, (200, 200, 50), "motorcycle"),
    ]

    for x, y, vx, vy, color, label in corner_configs:
        obstacles.append(
            SyntheticObstacle(
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                width=65,
                height=50,
                color=color,
                label=label,
                class_id=0,
            )
        )

    # Add some from sides
    side_configs = [
        (50.0, center_y - 50, 2.0, 0.5, (100, 150, 200), "bus"),
        (width - 100.0, center_y + 50, -2.0, -0.5, (200, 150, 100), "truck"),
    ]

    for x, y, vx, vy, color, label in side_configs:
        obstacles.append(
            SyntheticObstacle(
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                width=80,
                height=60,
                color=color,
                label=label,
                class_id=0,
            )
        )

    logger.info(f"Created converging scenario with {len(obstacles)} obstacles")
    return obstacles
