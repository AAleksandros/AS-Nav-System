"""2D simulation environment with circular obstacles and boundary walls.

Provides the world for the autonomous navigation agent, including static and
dynamic obstacles, collision detection, and background rendering.
"""

import logging
import random
from typing import List, Optional, Tuple

import cv2  # type: ignore
import numpy as np

from src.config import Config
from src.models import Obstacle

logger = logging.getLogger(__name__)


class Environment:
    """2D world with circular obstacles, collision detection, and boundary walls.

    The environment uses a y-up coordinate system internally. Rendering converts
    to y-down for OpenCV via world_to_screen().

    Parameters
    ----------
    width : float
        World width in units.
    height : float
        World height in units.
    obstacles : List[Obstacle]
        Initial list of obstacles (static and dynamic).
    """

    def __init__(
        self,
        width: float,
        height: float,
        obstacles: Optional[List[Obstacle]] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.obstacles: List[Obstacle] = obstacles if obstacles is not None else []

        logger.info(
            "Environment initialized: %.0fx%.0f, %d obstacles (%d dynamic)",
            width,
            height,
            len(self.obstacles),
            sum(1 for o in self.obstacles if o.is_dynamic),
        )

    def step(self, dt: float) -> None:
        """Advance dynamic obstacles by one time step.

        Dynamic obstacles move according to velocity and bounce off boundaries.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        """
        for obs in self.obstacles:
            if not obs.is_dynamic:
                continue

            # Update position
            obs.x += obs.vx * dt
            obs.y += obs.vy * dt

            # Bounce off horizontal boundaries (keep center + radius inside)
            if obs.x - obs.radius < 0:
                obs.x = obs.radius
                obs.vx = abs(obs.vx)
            elif obs.x + obs.radius > self.width:
                obs.x = self.width - obs.radius
                obs.vx = -abs(obs.vx)

            # Bounce off vertical boundaries
            if obs.y - obs.radius < 0:
                obs.y = obs.radius
                obs.vy = abs(obs.vy)
            elif obs.y + obs.radius > self.height:
                obs.y = self.height - obs.radius
                obs.vy = -abs(obs.vy)

    def get_obstacles(self) -> List[Obstacle]:
        """Get all current obstacles.

        Returns
        -------
        List[Obstacle]
            All obstacles in the environment.
        """
        return list(self.obstacles)

    def check_collision(self, x: float, y: float, radius: float) -> bool:
        """Check if a circle collides with any obstacle or boundary.

        Parameters
        ----------
        x : float
            Circle center x.
        y : float
            Circle center y.
        radius : float
            Circle radius.

        Returns
        -------
        bool
            True if collision detected.
        """
        # Check boundary collision
        if x - radius < 0 or x + radius > self.width:
            return True
        if y - radius < 0 or y + radius > self.height:
            return True

        # Check obstacle collisions (circle-circle)
        for obs in self.obstacles:
            dx = x - obs.x
            dy = y - obs.y
            dist_sq = dx * dx + dy * dy
            min_dist = radius + obs.radius
            if dist_sq < min_dist * min_dist:
                return True

        return False

    def get_boundary_segments(self) -> List[Tuple[float, float, float, float]]:
        """Get the four wall segments as (x1, y1, x2, y2) tuples.

        Returns
        -------
        List[Tuple[float, float, float, float]]
            Four wall segments: bottom, top, left, right.
        """
        return [
            (0, 0, self.width, 0),           # bottom wall
            (0, self.height, self.width, self.height),  # top wall
            (0, 0, 0, self.height),           # left wall
            (self.width, 0, self.width, self.height),   # right wall
        ]

    def render_background(self, config: Optional[Config] = None) -> np.ndarray:
        """Render the environment background with grid and boundaries.

        Parameters
        ----------
        config : Optional[Config]
            Configuration for colors. Uses defaults if None.

        Returns
        -------
        np.ndarray
            BGR image of the background.
        """
        w = int(self.width)
        h = int(self.height)

        # Default colors
        bg_color = (40, 40, 40)
        grid_color = (60, 60, 60)
        boundary_color = (200, 200, 200)

        if config is not None:
            try:
                colors = config.visualization.colors  # type: ignore[attr-defined]
                bg_color = tuple(colors.background)  # type: ignore[attr-defined]
                grid_color = tuple(colors.grid)  # type: ignore[attr-defined]
                boundary_color = tuple(colors.boundary)  # type: ignore[attr-defined]
            except AttributeError:
                pass

        # Create background
        frame = np.full((h, w, 3), bg_color, dtype=np.uint8)

        # Draw grid (every 50 units)
        grid_spacing = 50
        for gx in range(0, w, grid_spacing):
            cv2.line(frame, (gx, 0), (gx, h), grid_color, 1)
        for gy in range(0, h, grid_spacing):
            cv2.line(frame, (0, gy), (w, gy), grid_color, 1)

        # Draw boundary rectangle
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), boundary_color, 2)

        return frame

    @classmethod
    def from_scenario(cls, config: Config, scenario_name: str) -> "Environment":
        """Create environment from a named scenario in config.

        Parameters
        ----------
        config : Config
            System configuration containing scenario definitions.
        scenario_name : str
            Name of the scenario (e.g., "corridor", "gauntlet", "dynamic").

        Returns
        -------
        Environment
            Configured environment.

        Raises
        ------
        ValueError
            If scenario name is not found in config.
        """
        env_config = config.environment  # type: ignore[attr-defined]
        width = float(env_config.width)  # type: ignore[attr-defined]
        height = float(env_config.height)  # type: ignore[attr-defined]
        scenarios = env_config.scenarios  # type: ignore[attr-defined]

        # scenarios is a Config object with dynamic attributes
        scenario = getattr(scenarios, scenario_name, None)
        if scenario is None:
            available = [k for k in vars(scenarios) if not k.startswith("_")]
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. "
                f"Available: {available}"
            )

        obstacles: List[Obstacle] = []

        # Parse static obstacles
        for obs_data in scenario.static_obstacles:  # type: ignore[attr-defined]
            if isinstance(obs_data, dict):
                obstacles.append(Obstacle(
                    x=float(obs_data["x"]),
                    y=float(obs_data["y"]),
                    radius=float(obs_data["radius"]),
                ))
            else:
                obstacles.append(Obstacle(
                    x=float(obs_data.x),  # type: ignore[attr-defined]
                    y=float(obs_data.y),  # type: ignore[attr-defined]
                    radius=float(obs_data.radius),  # type: ignore[attr-defined]
                ))

        # Parse dynamic obstacles
        for obs_data in scenario.dynamic_obstacles:  # type: ignore[attr-defined]
            if isinstance(obs_data, dict):
                obstacles.append(Obstacle(
                    x=float(obs_data["x"]),
                    y=float(obs_data["y"]),
                    radius=float(obs_data["radius"]),
                    vx=float(obs_data.get("vx", 0)),
                    vy=float(obs_data.get("vy", 0)),
                    label="dynamic",
                ))
            else:
                obstacles.append(Obstacle(
                    x=float(obs_data.x),  # type: ignore[attr-defined]
                    y=float(obs_data.y),  # type: ignore[attr-defined]
                    radius=float(obs_data.radius),  # type: ignore[attr-defined]
                    vx=float(getattr(obs_data, "vx", 0)),
                    vy=float(getattr(obs_data, "vy", 0)),
                    label="dynamic",
                ))

        logger.info(
            "Created '%s' scenario: %d obstacles", scenario_name, len(obstacles)
        )
        return cls(width, height, obstacles)

    @classmethod
    def random_scenario(
        cls,
        width: float,
        height: float,
        num_static: int = 5,
        num_dynamic: int = 2,
        seed: Optional[int] = None,
    ) -> "Environment":
        """Create a randomized environment.

        Parameters
        ----------
        width : float
            World width.
        height : float
            World height.
        num_static : int
            Number of static obstacles.
        num_dynamic : int
            Number of dynamic obstacles.
        seed : Optional[int]
            Random seed for reproducibility.

        Returns
        -------
        Environment
            Randomly generated environment.
        """
        rng = random.Random(seed)
        obstacles: List[Obstacle] = []

        margin = 50.0  # keep obstacles away from edges

        for i in range(num_static):
            radius = rng.uniform(15.0, 45.0)
            x = rng.uniform(margin + radius, width - margin - radius)
            y = rng.uniform(margin + radius, height - margin - radius)
            obstacles.append(Obstacle(
                x=x, y=y, radius=radius, label=f"static_{i}"
            ))

        for i in range(num_dynamic):
            radius = rng.uniform(10.0, 25.0)
            x = rng.uniform(margin + radius, width - margin - radius)
            y = rng.uniform(margin + radius, height - margin - radius)
            vx = rng.uniform(-40.0, 40.0)
            vy = rng.uniform(-40.0, 40.0)
            obstacles.append(Obstacle(
                x=x, y=y, radius=radius, vx=vx, vy=vy, label=f"dynamic_{i}"
            ))

        logger.info(
            "Created random scenario: %d static + %d dynamic obstacles (seed=%s)",
            num_static,
            num_dynamic,
            seed,
        )
        return cls(width, height, obstacles)
