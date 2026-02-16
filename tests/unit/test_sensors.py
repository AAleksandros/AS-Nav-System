"""Tests for the simulated range sensor with ray casting."""

import math
from typing import List, Tuple
from unittest.mock import Mock

import pytest

from src.models import Obstacle, SensorReading
from src.sensors import RangeSensor


# --- Fixtures ---


@pytest.fixture
def sensor() -> RangeSensor:
    """Default sensor: 36 rays, 150 range, 360 deg FOV, no noise."""
    return RangeSensor(
        num_rays=36,
        max_range=150.0,
        fov=2 * math.pi,
        noise_stddev=0.0,
    )


@pytest.fixture
def narrow_sensor() -> RangeSensor:
    """Narrow FOV sensor: 5 rays, 90 deg FOV, no noise."""
    return RangeSensor(
        num_rays=5,
        max_range=100.0,
        fov=math.pi / 2,
        noise_stddev=0.0,
    )


@pytest.fixture
def single_ray_sensor() -> RangeSensor:
    """Single ray sensor pointing forward."""
    return RangeSensor(
        num_rays=1,
        max_range=200.0,
        fov=0.0,
        noise_stddev=0.0,
    )


@pytest.fixture
def obstacle_ahead() -> List[Obstacle]:
    """Single obstacle directly east of origin."""
    return [Obstacle(x=80.0, y=0.0, radius=10.0)]


@pytest.fixture
def walls() -> List[Tuple[float, float, float, float]]:
    """Boundary walls for a 400x300 world."""
    return [
        (0, 0, 400, 0),      # bottom
        (0, 300, 400, 300),   # top
        (0, 0, 0, 300),       # left
        (400, 0, 400, 300),   # right
    ]


# --- Tests: Init ---


class TestRangeSensorInit:
    def test_basic_attributes(self, sensor: RangeSensor) -> None:
        assert sensor.num_rays == 36
        assert sensor.max_range == 150.0
        assert sensor.fov == pytest.approx(2 * math.pi)
        assert sensor.noise_stddev == 0.0

    def test_single_ray(self, single_ray_sensor: RangeSensor) -> None:
        assert single_ray_sensor.num_rays == 1

    def test_from_config(self) -> None:
        config = Mock()
        config.sensor.num_rays = 18
        config.sensor.max_range = 200.0
        config.sensor.fov = math.pi
        config.sensor.noise_stddev = 2.0
        config.sensor.noise_seed = 42

        s = RangeSensor.from_config(config)
        assert s.num_rays == 18
        assert s.max_range == 200.0
        assert s.fov == pytest.approx(math.pi)
        assert s.noise_stddev == 2.0

    def test_from_config_null_seed(self) -> None:
        config = Mock()
        config.sensor.num_rays = 10
        config.sensor.max_range = 100.0
        config.sensor.fov = 2 * math.pi
        config.sensor.noise_stddev = 1.0
        config.sensor.noise_seed = None

        s = RangeSensor.from_config(config)
        assert s.num_rays == 10


# --- Tests: Ray-Circle Intersection ---


class TestRayCircleIntersection:
    def test_hit_circle_ahead(self, single_ray_sensor: RangeSensor) -> None:
        """Ray pointing east hits a circle directly ahead."""
        dist = single_ray_sensor._ray_circle_intersection(
            ox=0.0, oy=0.0, dx=1.0, dy=0.0,
            cx=50.0, cy=0.0, r=10.0,
        )
        # Nearest intersection at 50 - 10 = 40
        assert dist == pytest.approx(40.0)

    def test_miss_circle_behind(self, single_ray_sensor: RangeSensor) -> None:
        """Ray pointing east misses a circle behind it."""
        dist = single_ray_sensor._ray_circle_intersection(
            ox=0.0, oy=0.0, dx=1.0, dy=0.0,
            cx=-50.0, cy=0.0, r=10.0,
        )
        assert dist is None

    def test_miss_circle_to_side(self, single_ray_sensor: RangeSensor) -> None:
        """Ray pointing east misses a circle far to the side."""
        dist = single_ray_sensor._ray_circle_intersection(
            ox=0.0, oy=0.0, dx=1.0, dy=0.0,
            cx=50.0, cy=100.0, r=10.0,
        )
        assert dist is None

    def test_ray_tangent_to_circle(self, single_ray_sensor: RangeSensor) -> None:
        """Ray tangent to circle (discriminant ~ 0)."""
        dist = single_ray_sensor._ray_circle_intersection(
            ox=0.0, oy=10.0, dx=1.0, dy=0.0,
            cx=50.0, cy=0.0, r=10.0,
        )
        # Tangent: discriminant = 0, single intersection at cx = 50
        assert dist == pytest.approx(50.0)

    def test_ray_origin_inside_circle(self, single_ray_sensor: RangeSensor) -> None:
        """Ray starting inside circle should return distance to exit point."""
        dist = single_ray_sensor._ray_circle_intersection(
            ox=50.0, oy=0.0, dx=1.0, dy=0.0,
            cx=50.0, cy=0.0, r=10.0,
        )
        # Origin at center, exit at 50 + 10 = 60, so dist = 10
        assert dist is not None
        assert dist == pytest.approx(10.0)

    def test_diagonal_ray_hits_circle(self, single_ray_sensor: RangeSensor) -> None:
        """Diagonal ray hits offset circle."""
        # Ray at 45 degrees toward obstacle at (50, 50) radius 10
        dx = math.cos(math.pi / 4)
        dy = math.sin(math.pi / 4)
        dist = single_ray_sensor._ray_circle_intersection(
            ox=0.0, oy=0.0, dx=dx, dy=dy,
            cx=50.0, cy=50.0, r=10.0,
        )
        # Distance to center is sqrt(50^2 + 50^2) ~ 70.7, minus radius
        assert dist is not None
        assert dist == pytest.approx(math.sqrt(50**2 + 50**2) - 10.0, abs=0.1)


# --- Tests: Ray-Segment Intersection ---


class TestRaySegmentIntersection:
    def test_hit_vertical_wall(self, single_ray_sensor: RangeSensor) -> None:
        """Ray pointing east hits a vertical wall."""
        dist = single_ray_sensor._ray_segment_intersection(
            ox=0.0, oy=50.0, dx=1.0, dy=0.0,
            x1=100.0, y1=0.0, x2=100.0, y2=100.0,
        )
        assert dist == pytest.approx(100.0)

    def test_hit_horizontal_wall(self, single_ray_sensor: RangeSensor) -> None:
        """Ray pointing north hits a horizontal wall."""
        dist = single_ray_sensor._ray_segment_intersection(
            ox=50.0, oy=0.0, dx=0.0, dy=1.0,
            x1=0.0, y1=100.0, x2=100.0, y2=100.0,
        )
        assert dist == pytest.approx(100.0)

    def test_miss_wall_behind(self, single_ray_sensor: RangeSensor) -> None:
        """Ray pointing east misses a wall behind it."""
        dist = single_ray_sensor._ray_segment_intersection(
            ox=200.0, oy=50.0, dx=1.0, dy=0.0,
            x1=100.0, y1=0.0, x2=100.0, y2=100.0,
        )
        assert dist is None

    def test_miss_wall_outside_segment(self, single_ray_sensor: RangeSensor) -> None:
        """Ray misses because intersection is outside segment endpoints."""
        dist = single_ray_sensor._ray_segment_intersection(
            ox=0.0, oy=150.0, dx=1.0, dy=0.0,
            x1=100.0, y1=0.0, x2=100.0, y2=100.0,
        )
        assert dist is None

    def test_parallel_ray_no_hit(self, single_ray_sensor: RangeSensor) -> None:
        """Ray parallel to wall segment never hits."""
        dist = single_ray_sensor._ray_segment_intersection(
            ox=0.0, oy=50.0, dx=1.0, dy=0.0,
            x1=0.0, y1=100.0, x2=200.0, y2=100.0,
        )
        assert dist is None


# --- Tests: Scan ---


class TestScan:
    def test_scan_returns_correct_count(self, sensor: RangeSensor) -> None:
        readings = sensor.scan(200.0, 150.0, 0.0, [], [])
        assert len(readings) == 36

    def test_scan_single_ray_returns_one(self, single_ray_sensor: RangeSensor) -> None:
        readings = single_ray_sensor.scan(0.0, 0.0, 0.0, [], [])
        assert len(readings) == 1

    def test_scan_returns_sensor_readings(self, sensor: RangeSensor) -> None:
        readings = sensor.scan(200.0, 150.0, 0.0, [], [])
        assert all(isinstance(r, SensorReading) for r in readings)

    def test_scan_no_obstacles_all_max_range(self, sensor: RangeSensor) -> None:
        """In empty world (no walls, no obstacles) all readings should be max_range."""
        readings = sensor.scan(200.0, 150.0, 0.0, [], [])
        for r in readings:
            assert r.distance == pytest.approx(sensor.max_range)
            assert r.hit is False
            assert r.hit_point is None

    def test_scan_hits_obstacle(
        self, single_ray_sensor: RangeSensor, obstacle_ahead: List[Obstacle]
    ) -> None:
        """Single ray pointing east hits obstacle at x=80, r=10."""
        readings = single_ray_sensor.scan(0.0, 0.0, 0.0, obstacle_ahead, [])
        assert len(readings) == 1
        assert readings[0].hit is True
        assert readings[0].distance == pytest.approx(70.0)  # 80 - 10
        assert readings[0].hit_point is not None
        assert readings[0].hit_point[0] == pytest.approx(70.0)
        assert readings[0].hit_point[1] == pytest.approx(0.0)

    def test_scan_hits_wall(self, single_ray_sensor: RangeSensor) -> None:
        """Single ray pointing east hits right wall."""
        walls = [(100.0, 0.0, 100.0, 200.0)]
        readings = single_ray_sensor.scan(0.0, 100.0, 0.0, [], walls)
        assert readings[0].hit is True
        assert readings[0].distance == pytest.approx(100.0)

    def test_scan_returns_nearest_hit(self, single_ray_sensor: RangeSensor) -> None:
        """When multiple obstacles along ray, return nearest hit."""
        obstacles = [
            Obstacle(x=100.0, y=0.0, radius=10.0),  # nearer: hit at 90
            Obstacle(x=150.0, y=0.0, radius=10.0),   # farther: hit at 140
        ]
        readings = single_ray_sensor.scan(0.0, 0.0, 0.0, obstacles, [])
        assert readings[0].distance == pytest.approx(90.0)

    def test_scan_obstacle_beyond_max_range(
        self, single_ray_sensor: RangeSensor
    ) -> None:
        """Obstacle beyond max_range should not register as hit."""
        far_obstacle = [Obstacle(x=500.0, y=0.0, radius=10.0)]
        readings = single_ray_sensor.scan(0.0, 0.0, 0.0, far_obstacle, [])
        assert readings[0].hit is False
        assert readings[0].distance == pytest.approx(200.0)  # max_range

    def test_scan_ray_angles_360_fov(self, sensor: RangeSensor) -> None:
        """Ray angles should span full 360 degrees evenly."""
        readings = sensor.scan(0.0, 0.0, 0.0, [], [])
        angles = [r.angle for r in readings]
        # First ray should be at heading - fov/2
        expected_start = 0.0 - math.pi  # heading(0) - fov/2(pi)
        assert angles[0] == pytest.approx(expected_start, abs=1e-6)

    def test_scan_respects_heading(
        self, single_ray_sensor: RangeSensor, obstacle_ahead: List[Obstacle]
    ) -> None:
        """When heading is north (pi/2), forward ray should point north, not east."""
        readings = single_ray_sensor.scan(
            0.0, 0.0, math.pi / 2, obstacle_ahead, []
        )
        # Obstacle is east, but agent faces north -> no hit
        assert readings[0].hit is False

    def test_scan_narrow_fov_coverage(self, narrow_sensor: RangeSensor) -> None:
        """Narrow FOV sensor should only cover 90 degrees."""
        readings = narrow_sensor.scan(0.0, 0.0, 0.0, [], [])
        angles = [r.angle for r in readings]
        # FOV = pi/2, so angles should span from -pi/4 to +pi/4
        assert angles[0] == pytest.approx(-math.pi / 4, abs=1e-6)
        assert angles[-1] == pytest.approx(math.pi / 4, abs=1e-6)

    def test_scan_wall_closer_than_obstacle(
        self, single_ray_sensor: RangeSensor
    ) -> None:
        """Wall closer than obstacle should be the reported hit."""
        obstacles = [Obstacle(x=150.0, y=0.0, radius=10.0)]
        walls = [(100.0, -50.0, 100.0, 50.0)]
        readings = single_ray_sensor.scan(0.0, 0.0, 0.0, obstacles, walls)
        assert readings[0].distance == pytest.approx(100.0)  # wall, not obstacle


# --- Tests: Noise ---


class TestSensorNoise:
    def test_zero_noise_exact_distances(self) -> None:
        """With zero noise, distances should be exact."""
        s = RangeSensor(num_rays=1, max_range=200.0, fov=0.0, noise_stddev=0.0)
        obstacles = [Obstacle(x=100.0, y=0.0, radius=10.0)]
        readings = s.scan(0.0, 0.0, 0.0, obstacles, [])
        assert readings[0].distance == pytest.approx(90.0)

    def test_noise_changes_distance(self) -> None:
        """With noise, distances should differ from exact values."""
        s = RangeSensor(
            num_rays=1, max_range=200.0, fov=0.0, noise_stddev=5.0, noise_seed=42
        )
        obstacles = [Obstacle(x=100.0, y=0.0, radius=10.0)]
        distances = []
        for _ in range(10):
            readings = s.scan(0.0, 0.0, 0.0, obstacles, [])
            distances.append(readings[0].distance)
        # With noise, not all distances should be exactly 90
        assert not all(d == pytest.approx(90.0) for d in distances)

    def test_noise_seed_reproducibility(self) -> None:
        """Same seed should produce same noise sequence."""
        s1 = RangeSensor(
            num_rays=1, max_range=200.0, fov=0.0, noise_stddev=5.0, noise_seed=42
        )
        s2 = RangeSensor(
            num_rays=1, max_range=200.0, fov=0.0, noise_stddev=5.0, noise_seed=42
        )
        obstacles = [Obstacle(x=100.0, y=0.0, radius=10.0)]
        r1 = s1.scan(0.0, 0.0, 0.0, obstacles, [])
        r2 = s2.scan(0.0, 0.0, 0.0, obstacles, [])
        assert r1[0].distance == pytest.approx(r2[0].distance)

    def test_noisy_distance_clamped_to_zero(self) -> None:
        """Noise should not produce negative distances."""
        s = RangeSensor(
            num_rays=1, max_range=200.0, fov=0.0, noise_stddev=100.0, noise_seed=1
        )
        # Very close obstacle + huge noise
        obstacles = [Obstacle(x=5.0, y=0.0, radius=4.0)]
        for _ in range(20):
            readings = s.scan(0.0, 0.0, 0.0, obstacles, [])
            assert readings[0].distance >= 0.0

    def test_noisy_distance_clamped_to_max_range(self) -> None:
        """Noise should not produce distances above max_range."""
        s = RangeSensor(
            num_rays=1, max_range=200.0, fov=0.0, noise_stddev=100.0, noise_seed=7
        )
        obstacles = [Obstacle(x=195.0, y=0.0, radius=5.0)]
        for _ in range(20):
            readings = s.scan(0.0, 0.0, 0.0, obstacles, [])
            assert readings[0].distance <= 200.0
