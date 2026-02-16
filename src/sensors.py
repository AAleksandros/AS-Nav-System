"""Simulated range sensor with ray casting.

Provides a configurable ray-cast range sensor that detects obstacles and
boundary walls in the 2D environment. Supports Gaussian measurement noise
and arbitrary field-of-view.
"""

import logging
import math
import random
from typing import List, Optional, Tuple

from src.config import Config
from src.models import Obstacle, SensorReading

logger = logging.getLogger(__name__)


class RangeSensor:
    """Ray-casting range sensor for 2D obstacle detection.

    Casts evenly-spaced rays across the field of view and returns the
    distance to the nearest obstacle or wall for each ray.

    Parameters
    ----------
    num_rays : int
        Number of rays to cast per scan.
    max_range : float
        Maximum detection range (world units).
    fov : float
        Field of view in radians (2*pi for 360 degrees).
    noise_stddev : float
        Standard deviation of Gaussian measurement noise. 0 for no noise.
    noise_seed : Optional[int]
        Random seed for noise reproducibility.
    """

    def __init__(
        self,
        num_rays: int,
        max_range: float,
        fov: float,
        noise_stddev: float = 0.0,
        noise_seed: Optional[int] = None,
    ) -> None:
        self.num_rays = num_rays
        self.max_range = max_range
        self.fov = fov
        self.noise_stddev = noise_stddev
        self._rng = random.Random(noise_seed)

        logger.info(
            "RangeSensor: %d rays, range=%.1f, fov=%.2f rad, noise=%.2f",
            num_rays,
            max_range,
            fov,
            noise_stddev,
        )

    @classmethod
    def from_config(cls, config: Config) -> "RangeSensor":
        """Create sensor from configuration.

        Parameters
        ----------
        config : Config
            System configuration with sensor section.

        Returns
        -------
        RangeSensor
            Configured sensor instance.
        """
        sensor_cfg = config.sensor  # type: ignore[attr-defined]
        return cls(
            num_rays=int(sensor_cfg.num_rays),  # type: ignore[attr-defined]
            max_range=float(sensor_cfg.max_range),  # type: ignore[attr-defined]
            fov=float(sensor_cfg.fov),  # type: ignore[attr-defined]
            noise_stddev=float(sensor_cfg.noise_stddev),  # type: ignore[attr-defined]
            noise_seed=sensor_cfg.noise_seed,  # type: ignore[attr-defined]
        )

    def scan(
        self,
        x: float,
        y: float,
        heading: float,
        obstacles: List[Obstacle],
        boundary_segments: List[Tuple[float, float, float, float]],
    ) -> List[SensorReading]:
        """Cast all rays and return sensor readings.

        Parameters
        ----------
        x : float
            Agent x position (world units).
        y : float
            Agent y position (world units).
        heading : float
            Agent heading in radians (0 = east).
        obstacles : List[Obstacle]
            Obstacles to detect.
        boundary_segments : List[Tuple[float, float, float, float]]
            Wall segments as (x1, y1, x2, y2).

        Returns
        -------
        List[SensorReading]
            One reading per ray.
        """
        readings: List[SensorReading] = []

        for i in range(self.num_rays):
            # Compute ray angle: evenly spaced across FOV, centered on heading
            if self.num_rays == 1:
                ray_angle = heading
            else:
                ray_angle = heading - self.fov / 2 + i * self.fov / (self.num_rays - 1)

            dx = math.cos(ray_angle)
            dy = math.sin(ray_angle)

            # Find nearest intersection
            nearest_dist = self.max_range
            hit = False
            hit_point: Optional[Tuple[float, float]] = None

            # Check obstacles
            for obs in obstacles:
                d = self._ray_circle_intersection(
                    x, y, dx, dy, obs.x, obs.y, obs.radius
                )
                if d is not None and d < nearest_dist:
                    nearest_dist = d
                    hit = True
                    hit_point = (x + dx * d, y + dy * d)

            # Check boundary walls
            for seg in boundary_segments:
                d = self._ray_segment_intersection(
                    x, y, dx, dy, seg[0], seg[1], seg[2], seg[3]
                )
                if d is not None and d < nearest_dist:
                    nearest_dist = d
                    hit = True
                    hit_point = (x + dx * d, y + dy * d)

            # Apply noise
            if self.noise_stddev > 0.0 and hit:
                noise = self._rng.gauss(0.0, self.noise_stddev)
                nearest_dist = max(0.0, min(self.max_range, nearest_dist + noise))

            readings.append(
                SensorReading(
                    angle=ray_angle,
                    distance=nearest_dist,
                    hit=hit,
                    hit_point=hit_point,
                )
            )

        return readings

    def _ray_circle_intersection(
        self,
        ox: float,
        oy: float,
        dx: float,
        dy: float,
        cx: float,
        cy: float,
        r: float,
    ) -> Optional[float]:
        """Find distance from ray origin to nearest circle intersection.

        Parameters
        ----------
        ox, oy : float
            Ray origin.
        dx, dy : float
            Ray direction (unit vector).
        cx, cy : float
            Circle center.
        r : float
            Circle radius.

        Returns
        -------
        Optional[float]
            Distance to nearest intersection, or None if no hit.
        """
        # Vector from ray origin to circle center
        fx = ox - cx
        fy = oy - cy

        a = dx * dx + dy * dy
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - r * r

        discriminant = b * b - 4.0 * a * c

        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        # Return nearest positive t
        if t1 >= 0:
            return t1
        if t2 >= 0:
            return t2
        return None

    def _ray_segment_intersection(
        self,
        ox: float,
        oy: float,
        dx: float,
        dy: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> Optional[float]:
        """Find distance from ray origin to line segment intersection.

        Parameters
        ----------
        ox, oy : float
            Ray origin.
        dx, dy : float
            Ray direction (unit vector).
        x1, y1, x2, y2 : float
            Segment endpoints.

        Returns
        -------
        Optional[float]
            Distance to intersection, or None if no hit.
        """
        # Segment direction
        sx = x2 - x1
        sy = y2 - y1

        denom = dx * sy - dy * sx

        if abs(denom) < 1e-10:
            return None  # Parallel

        # Vector from segment start to ray origin
        qx = ox - x1
        qy = oy - y1

        t = -(qx * sy - qy * sx) / denom  # Ray parameter
        u = -(qx * dy - qy * dx) / denom  # Segment parameter

        if t >= 0 and 0 <= u <= 1:
            return t

        return None
