"""Tests for the 2D simulation environment."""

from unittest.mock import Mock

import numpy as np
import pytest

from src.environment import Environment
from src.models import Obstacle


# --- Fixtures ---


@pytest.fixture
def empty_env() -> Environment:
    """Environment with no obstacles."""
    return Environment(width=800.0, height=600.0)


@pytest.fixture
def static_env() -> Environment:
    """Environment with static obstacles."""
    obstacles = [
        Obstacle(x=200.0, y=300.0, radius=30.0, label="a"),
        Obstacle(x=500.0, y=200.0, radius=40.0, label="b"),
    ]
    return Environment(width=800.0, height=600.0, obstacles=obstacles)


@pytest.fixture
def dynamic_env() -> Environment:
    """Environment with dynamic obstacles."""
    obstacles = [
        Obstacle(x=100.0, y=100.0, radius=20.0, vx=50.0, vy=30.0, label="dyn"),
        Obstacle(x=400.0, y=300.0, radius=25.0, label="static"),
    ]
    return Environment(width=800.0, height=600.0, obstacles=obstacles)


# --- Tests: Init ---


class TestEnvironmentInit:
    def test_dimensions(self, empty_env: Environment) -> None:
        assert empty_env.width == 800.0
        assert empty_env.height == 600.0

    def test_empty_obstacles(self, empty_env: Environment) -> None:
        assert len(empty_env.obstacles) == 0

    def test_with_obstacles(self, static_env: Environment) -> None:
        assert len(static_env.obstacles) == 2

    def test_none_obstacles_default(self) -> None:
        env = Environment(100.0, 100.0)
        assert env.obstacles == []


# --- Tests: Step ---


class TestEnvironmentStep:
    def test_static_obstacles_unchanged(self, static_env: Environment) -> None:
        before_x = static_env.obstacles[0].x
        before_y = static_env.obstacles[0].y
        static_env.step(dt=0.1)
        assert static_env.obstacles[0].x == before_x
        assert static_env.obstacles[0].y == before_y

    def test_dynamic_obstacle_moves(self, dynamic_env: Environment) -> None:
        obs = dynamic_env.obstacles[0]
        orig_x, orig_y = obs.x, obs.y
        dynamic_env.step(dt=0.1)
        assert obs.x == pytest.approx(orig_x + 50.0 * 0.1)
        assert obs.y == pytest.approx(orig_y + 30.0 * 0.1)

    def test_bounce_left_wall(self) -> None:
        obs = Obstacle(x=5.0, y=50.0, radius=10.0, vx=-100.0, vy=0.0)
        env = Environment(100.0, 100.0, [obs])
        env.step(dt=0.1)
        # After step: x = 5 - 10 = -5, clamped to radius=10, vx flipped
        assert obs.x == 10.0
        assert obs.vx > 0

    def test_bounce_right_wall(self) -> None:
        obs = Obstacle(x=95.0, y=50.0, radius=10.0, vx=100.0, vy=0.0)
        env = Environment(100.0, 100.0, [obs])
        env.step(dt=0.1)
        assert obs.x == 90.0  # width - radius
        assert obs.vx < 0

    def test_bounce_bottom_wall(self) -> None:
        obs = Obstacle(x=50.0, y=5.0, radius=10.0, vx=0.0, vy=-100.0)
        env = Environment(100.0, 100.0, [obs])
        env.step(dt=0.1)
        assert obs.y == 10.0  # radius
        assert obs.vy > 0

    def test_bounce_top_wall(self) -> None:
        obs = Obstacle(x=50.0, y=95.0, radius=10.0, vx=0.0, vy=100.0)
        env = Environment(100.0, 100.0, [obs])
        env.step(dt=0.1)
        assert obs.y == 90.0  # height - radius
        assert obs.vy < 0


# --- Tests: Collision Detection ---


class TestCollisionDetection:
    def test_no_collision_empty_env(self, empty_env: Environment) -> None:
        assert not empty_env.check_collision(400.0, 300.0, 10.0)

    def test_collision_with_obstacle(self, static_env: Environment) -> None:
        # static_env has obstacle at (200, 300) radius 30
        # check point at (220, 300) radius 15 -> dist=20 < 30+15=45 -> collision
        assert static_env.check_collision(220.0, 300.0, 15.0)

    def test_no_collision_with_obstacle(self, static_env: Environment) -> None:
        # check far away point
        assert not static_env.check_collision(700.0, 500.0, 10.0)

    def test_boundary_collision_left(self, empty_env: Environment) -> None:
        assert empty_env.check_collision(5.0, 300.0, 10.0)

    def test_boundary_collision_right(self, empty_env: Environment) -> None:
        assert empty_env.check_collision(795.0, 300.0, 10.0)

    def test_boundary_collision_bottom(self, empty_env: Environment) -> None:
        assert empty_env.check_collision(400.0, 5.0, 10.0)

    def test_boundary_collision_top(self, empty_env: Environment) -> None:
        assert empty_env.check_collision(400.0, 595.0, 10.0)

    def test_touching_boundary_exactly(self, empty_env: Environment) -> None:
        # x - radius = 0 -> not inside, no collision by < check
        assert not empty_env.check_collision(10.0, 300.0, 10.0)

    def test_touching_obstacle_exactly(self) -> None:
        obs = Obstacle(x=50.0, y=50.0, radius=10.0)
        env = Environment(100.0, 100.0, [obs])
        # Distance = 30, sum of radii = 10 + 20 = 30, not < 30
        assert not env.check_collision(80.0, 50.0, 20.0)


# --- Tests: Boundary Segments ---


class TestBoundarySegments:
    def test_returns_four_segments(self, empty_env: Environment) -> None:
        segments = empty_env.get_boundary_segments()
        assert len(segments) == 4

    def test_segment_format(self, empty_env: Environment) -> None:
        segments = empty_env.get_boundary_segments()
        for seg in segments:
            assert len(seg) == 4  # (x1, y1, x2, y2)

    def test_bottom_wall(self, empty_env: Environment) -> None:
        segments = empty_env.get_boundary_segments()
        assert segments[0] == (0, 0, 800.0, 0)

    def test_top_wall(self, empty_env: Environment) -> None:
        segments = empty_env.get_boundary_segments()
        assert segments[1] == (0, 600.0, 800.0, 600.0)


# --- Tests: Get Obstacles ---


class TestGetObstacles:
    def test_returns_copy(self, static_env: Environment) -> None:
        obs = static_env.get_obstacles()
        obs.pop()  # modify returned list
        assert len(static_env.obstacles) == 2  # original unchanged

    def test_obstacle_data(self, static_env: Environment) -> None:
        obs = static_env.get_obstacles()
        assert obs[0].x == 200.0
        assert obs[0].radius == 30.0


# --- Tests: Render Background ---


class TestRenderBackground:
    def test_returns_numpy_array(self, empty_env: Environment) -> None:
        frame = empty_env.render_background()
        assert isinstance(frame, np.ndarray)

    def test_correct_dimensions(self, empty_env: Environment) -> None:
        frame = empty_env.render_background()
        assert frame.shape == (600, 800, 3)

    def test_default_background_color(self, empty_env: Environment) -> None:
        frame = empty_env.render_background()
        # Check a center pixel (avoid grid lines and boundary)
        assert frame[301, 401, 0] == 40  # dark gray B
        assert frame[301, 401, 1] == 40  # dark gray G
        assert frame[301, 401, 2] == 40  # dark gray R

    def test_custom_colors_from_config(self, empty_env: Environment) -> None:
        config = Mock()
        config.visualization.colors.background = [10, 20, 30]
        config.visualization.colors.grid = [40, 50, 60]
        config.visualization.colors.boundary = [100, 110, 120]
        frame = empty_env.render_background(config=config)
        # Center pixel should have custom bg color
        assert frame[301, 401, 0] == 10
        assert frame[301, 401, 1] == 20
        assert frame[301, 401, 2] == 30

    def test_config_without_colors_uses_defaults(self, empty_env: Environment) -> None:
        config = Mock()
        config.visualization.colors = Mock(side_effect=AttributeError)
        del config.visualization.colors
        frame = empty_env.render_background(config=config)
        # Should fall back to default dark gray
        assert frame[301, 401, 0] == 40


# --- Tests: Random Scenario ---


class TestRandomScenario:
    def test_creates_correct_count(self) -> None:
        env = Environment.random_scenario(
            800.0, 600.0, num_static=3, num_dynamic=2, seed=42
        )
        static_count = sum(1 for o in env.obstacles if not o.is_dynamic)
        dynamic_count = sum(1 for o in env.obstacles if o.is_dynamic)
        assert static_count == 3
        assert dynamic_count == 2

    def test_obstacles_within_bounds(self) -> None:
        env = Environment.random_scenario(
            800.0, 600.0, num_static=10, num_dynamic=5, seed=42
        )
        for obs in env.obstacles:
            assert obs.x - obs.radius >= 0
            assert obs.x + obs.radius <= 800.0
            assert obs.y - obs.radius >= 0
            assert obs.y + obs.radius <= 600.0

    def test_reproducible_with_seed(self) -> None:
        env1 = Environment.random_scenario(800.0, 600.0, seed=42)
        env2 = Environment.random_scenario(800.0, 600.0, seed=42)
        assert len(env1.obstacles) == len(env2.obstacles)
        for o1, o2 in zip(env1.obstacles, env2.obstacles):
            assert o1.x == o2.x
            assert o1.y == o2.y

    def test_different_seeds_differ(self) -> None:
        env1 = Environment.random_scenario(800.0, 600.0, seed=42)
        env2 = Environment.random_scenario(800.0, 600.0, seed=99)
        # Very unlikely to have same positions
        positions_differ = any(
            o1.x != o2.x for o1, o2 in zip(env1.obstacles, env2.obstacles)
        )
        assert positions_differ

    def test_dimensions_set(self) -> None:
        env = Environment.random_scenario(1000.0, 500.0, seed=1)
        assert env.width == 1000.0
        assert env.height == 500.0


# --- Tests: From Scenario ---


class TestFromScenario:
    @pytest.fixture
    def mock_config(self) -> Mock:
        """Config with corridor scenario."""
        config = Mock()
        config.environment.width = 800.0
        config.environment.height = 600.0

        corridor = Mock()
        corridor.static_obstacles = [
            {"x": 200, "y": 300, "radius": 40},
            {"x": 400, "y": 200, "radius": 35},
        ]
        corridor.dynamic_obstacles = [
            {"x": 100, "y": 150, "radius": 20, "vx": 30, "vy": 20},
        ]
        config.environment.scenarios = Mock(spec=[])
        config.environment.scenarios.corridor = corridor

        return config

    def test_creates_environment(self, mock_config: Mock) -> None:
        env = Environment.from_scenario(mock_config, "corridor")
        assert env.width == 800.0
        assert env.height == 600.0

    def test_loads_obstacles(self, mock_config: Mock) -> None:
        env = Environment.from_scenario(mock_config, "corridor")
        assert len(env.obstacles) == 3  # 2 static + 1 dynamic

    def test_static_obstacle_data(self, mock_config: Mock) -> None:
        env = Environment.from_scenario(mock_config, "corridor")
        assert env.obstacles[0].x == 200.0
        assert env.obstacles[0].radius == 40.0
        assert not env.obstacles[0].is_dynamic

    def test_dynamic_obstacle_data(self, mock_config: Mock) -> None:
        env = Environment.from_scenario(mock_config, "corridor")
        dyn = env.obstacles[2]
        assert dyn.vx == 30.0
        assert dyn.vy == 20.0
        assert dyn.is_dynamic

    def test_unknown_scenario_raises(self, mock_config: Mock) -> None:
        with pytest.raises(ValueError, match="Unknown scenario"):
            Environment.from_scenario(mock_config, "nonexistent")

    def test_config_object_style_obstacles(self) -> None:
        """Test obstacle parsing when data is Config objects (not dicts)."""
        config = Mock()
        config.environment.width = 800.0
        config.environment.height = 600.0

        obs1 = Mock()
        obs1.x = 100.0
        obs1.y = 200.0
        obs1.radius = 25.0

        dyn1 = Mock()
        dyn1.x = 300.0
        dyn1.y = 400.0
        dyn1.radius = 15.0
        dyn1.vx = 10.0
        dyn1.vy = 20.0

        scenario = Mock()
        # Use non-dict objects (not isinstance(x, dict))
        scenario.static_obstacles = [obs1]
        scenario.dynamic_obstacles = [dyn1]

        config.environment.scenarios = Mock(spec=[])
        config.environment.scenarios.test = scenario

        env = Environment.from_scenario(config, "test")
        assert len(env.obstacles) == 2
        assert env.obstacles[0].x == 100.0
        assert env.obstacles[1].vx == 10.0
