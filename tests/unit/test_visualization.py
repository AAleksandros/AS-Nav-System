"""Unit tests for visualization module (APF simulator rendering)."""

import math

import numpy as np
import pytest
from unittest.mock import Mock, patch

from src.models import State, AgentState, Obstacle, SensorReading, ForceVector
from src.visualization import VisualizationRenderer


@pytest.fixture
def mock_config():
    """Create mock configuration with all new APF keys."""
    config = Mock()

    # Visualization settings
    vis = Mock()
    vis.show_obstacles = True
    vis.show_sensor_rays = True
    vis.show_force_vectors = True
    vis.show_agent = True
    vis.show_trajectory = True
    vis.show_state = True
    vis.show_metrics = True
    vis.show_waypoints = True
    vis.show_goal_status = True
    vis.force_vector_scale = 3.0

    # Legacy keys (still read defensively)
    vis.show_boxes = True
    vis.show_labels = True
    vis.show_confidence = True

    # Colors
    colors = Mock()
    colors.agent = [0, 200, 255]
    colors.trajectory = [100, 100, 255]
    colors.obstacle_fill = [80, 80, 80]
    colors.obstacle_outline = [200, 200, 200]
    colors.sensor_ray_miss = [60, 60, 60]
    colors.sensor_ray_hit = [255, 80, 80]
    colors.force_attractive = [0, 255, 100]
    colors.force_repulsive = [255, 80, 80]
    colors.force_total = [0, 255, 255]
    colors.state_navigate = [0, 255, 0]
    colors.state_avoid = [255, 165, 0]
    colors.state_stop = [255, 0, 0]
    colors.waypoint = [255, 255, 0]
    colors.waypoint_reached = [100, 100, 100]
    colors.background = [40, 40, 40]
    vis.colors = colors

    config.visualization = vis

    # Environment settings (for world_height)
    config.environment = Mock()
    config.environment.height = 600.0

    return config


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing (800x600)."""
    return np.zeros((600, 800, 3), dtype=np.uint8)


@pytest.fixture
def sample_obstacles():
    """Create sample obstacles."""
    return [
        Obstacle(x=200, y=300, radius=40),
        Obstacle(x=500, y=200, radius=30),
    ]


@pytest.fixture
def sample_readings():
    """Create sample sensor readings (one hit, one miss)."""
    return [
        SensorReading(angle=0.0, distance=100.0, hit=True, hit_point=(200.0, 300.0)),
        SensorReading(angle=math.pi / 2, distance=150.0, hit=False, hit_point=None),
    ]


@pytest.fixture
def sample_forces():
    """Create sample force vectors."""
    return [
        ForceVector(fx=10.0, fy=5.0, source="attractive"),
        ForceVector(fx=-3.0, fy=-2.0, source="repulsive"),
        ForceVector(fx=7.0, fy=3.0, source="total"),
    ]


@pytest.fixture
def sample_agent():
    """Create sample agent state (world coords, radians heading)."""
    return AgentState(
        x=100.0,
        y=200.0,
        heading=0.0,  # radians: facing right
        velocity=5.0,
        trajectory=[(80.0, 200.0), (90.0, 200.0), (100.0, 200.0)],
    )


class TestWorldToScreen:
    """Tests for world_to_screen coordinate conversion."""

    def test_origin_maps_to_bottom_left(self, mock_config):
        """(0, 0) in world maps to (0, world_height) in screen."""
        renderer = VisualizationRenderer(mock_config)
        sx, sy = renderer.world_to_screen(0.0, 0.0)
        assert sx == 0
        assert sy == 600  # world_height

    def test_top_maps_to_screen_top(self, mock_config):
        """(x, world_height) in world maps to (x, 0) in screen."""
        renderer = VisualizationRenderer(mock_config)
        sx, sy = renderer.world_to_screen(400.0, 600.0)
        assert sx == 400
        assert sy == 0

    def test_general_conversion(self, mock_config):
        """(x, y) maps to (x, world_height - y)."""
        renderer = VisualizationRenderer(mock_config)
        sx, sy = renderer.world_to_screen(200.0, 150.0)
        assert sx == 200
        assert sy == 450  # 600 - 150


class TestInitConfig:
    """Tests for __init__ config reading."""

    def test_reads_new_config_keys(self, mock_config):
        """Renderer reads all new APF config keys."""
        renderer = VisualizationRenderer(mock_config)
        assert renderer.show_obstacles is True
        assert renderer.show_sensor_rays is True
        assert renderer.show_force_vectors is True
        assert renderer.show_waypoints is True
        assert renderer.show_goal_status is True
        assert renderer.world_height == 600.0
        assert renderer.force_vector_scale == 3.0

    def test_defaults_when_keys_missing(self):
        """Renderer falls back to defaults when new keys are missing."""
        config = Mock()
        config.visualization = Mock(spec=[])
        config.environment = Mock(spec=[])

        renderer = VisualizationRenderer(config)
        assert renderer.show_obstacles is True
        assert renderer.show_sensor_rays is True
        assert renderer.show_force_vectors is True
        assert renderer.world_height == 600.0


class TestDrawObstacles:
    """Tests for draw_obstacles method."""

    @patch('src.visualization.cv2.circle')
    def test_draws_circles_per_obstacle(
        self, mock_circle, mock_config, sample_frame, sample_obstacles
    ):
        """Draws filled circle + outline per obstacle."""
        renderer = VisualizationRenderer(mock_config)
        renderer.draw_obstacles(sample_frame, sample_obstacles)

        # 2 obstacles * 2 calls each (fill + outline) = 4
        assert mock_circle.call_count == 4

    @patch('src.visualization.cv2.circle')
    def test_respects_toggle(
        self, mock_circle, mock_config, sample_frame, sample_obstacles
    ):
        """Does not draw when show_obstacles is False."""
        mock_config.visualization.show_obstacles = False
        renderer = VisualizationRenderer(mock_config)
        renderer.draw_obstacles(sample_frame, sample_obstacles)

        assert mock_circle.call_count == 0

    @patch('src.visualization.cv2.circle')
    def test_empty_list(self, mock_circle, mock_config, sample_frame):
        """Empty obstacle list draws nothing."""
        renderer = VisualizationRenderer(mock_config)
        renderer.draw_obstacles(sample_frame, [])

        assert mock_circle.call_count == 0


class TestDrawSensorRays:
    """Tests for draw_sensor_rays method."""

    @patch('src.visualization.cv2.line')
    def test_draws_lines(
        self, mock_line, mock_config, sample_frame, sample_readings
    ):
        """Draws line per sensor reading."""
        renderer = VisualizationRenderer(mock_config)
        renderer.draw_sensor_rays(sample_frame, 100.0, 300.0, sample_readings)

        # One line per reading
        assert mock_line.call_count == 2

    @patch('src.visualization.cv2.line')
    def test_respects_toggle(
        self, mock_line, mock_config, sample_frame, sample_readings
    ):
        """Does not draw when show_sensor_rays is False."""
        mock_config.visualization.show_sensor_rays = False
        renderer = VisualizationRenderer(mock_config)
        renderer.draw_sensor_rays(sample_frame, 100.0, 300.0, sample_readings)

        assert mock_line.call_count == 0

    @patch('src.visualization.cv2.line')
    def test_empty_readings(self, mock_line, mock_config, sample_frame):
        """Empty readings draws nothing."""
        renderer = VisualizationRenderer(mock_config)
        renderer.draw_sensor_rays(sample_frame, 100.0, 300.0, [])

        assert mock_line.call_count == 0


class TestDrawForceVectors:
    """Tests for draw_force_vectors method."""

    @patch('src.visualization.cv2.arrowedLine')
    def test_draws_arrows(
        self, mock_arrow, mock_config, sample_frame, sample_forces
    ):
        """Draws arrowed line per force vector."""
        renderer = VisualizationRenderer(mock_config)
        renderer.draw_force_vectors(sample_frame, 100.0, 300.0, sample_forces)

        # One arrow per force
        assert mock_arrow.call_count == 3

    @patch('src.visualization.cv2.arrowedLine')
    def test_respects_toggle(
        self, mock_arrow, mock_config, sample_frame, sample_forces
    ):
        """Does not draw when show_force_vectors is False."""
        mock_config.visualization.show_force_vectors = False
        renderer = VisualizationRenderer(mock_config)
        renderer.draw_force_vectors(sample_frame, 100.0, 300.0, sample_forces)

        assert mock_arrow.call_count == 0

    @patch('src.visualization.cv2.arrowedLine')
    def test_empty_list(self, mock_arrow, mock_config, sample_frame):
        """Empty force list draws nothing."""
        renderer = VisualizationRenderer(mock_config)
        renderer.draw_force_vectors(sample_frame, 100.0, 300.0, [])

        assert mock_arrow.call_count == 0


class TestDrawAgent:
    """Tests for draw_agent method (world coords, radians heading)."""

    @patch('src.visualization.cv2.arrowedLine')
    @patch('src.visualization.cv2.circle')
    def test_draws_circle_and_arrow(
        self, mock_circle, mock_arrow, mock_config, sample_frame, sample_agent
    ):
        """Draws agent circle and heading arrow."""
        renderer = VisualizationRenderer(mock_config)
        renderer.draw_agent(sample_frame, sample_agent)

        assert mock_circle.call_count == 1
        assert mock_arrow.call_count == 1

    @patch('src.visualization.cv2.arrowedLine')
    @patch('src.visualization.cv2.circle')
    def test_uses_world_to_screen(
        self, mock_circle, mock_arrow, mock_config, sample_frame
    ):
        """Agent position is converted via world_to_screen."""
        renderer = VisualizationRenderer(mock_config)
        # Agent at world (100, 200), world_height=600 -> screen (100, 400)
        agent = AgentState(x=100.0, y=200.0, heading=0.0, velocity=5.0, trajectory=[])
        renderer.draw_agent(sample_frame, agent)

        # Circle center should be screen coords
        circle_center = mock_circle.call_args[0][1]
        assert circle_center == (100, 400)  # 600 - 200 = 400

    @patch('src.visualization.cv2.circle')
    def test_respects_toggle(
        self, mock_circle, mock_config, sample_frame, sample_agent
    ):
        """Does not draw when show_agent is False."""
        mock_config.visualization.show_agent = False
        renderer = VisualizationRenderer(mock_config)
        renderer.draw_agent(sample_frame, sample_agent)

        assert mock_circle.call_count == 0


class TestDrawTrajectory:
    """Tests for draw_trajectory method (world coords)."""

    @patch('src.visualization.cv2.polylines')
    def test_draws_polyline_with_world_coords(
        self, mock_polylines, mock_config, sample_frame
    ):
        """Draws trajectory polyline with world-to-screen converted points."""
        renderer = VisualizationRenderer(mock_config)
        trajectory = [(100.0, 200.0), (200.0, 300.0), (300.0, 400.0)]
        renderer.draw_trajectory(sample_frame, trajectory)

        assert mock_polylines.call_count == 1
        # Verify points are screen-converted
        # world (100, 200) -> screen (100, 400)
        # world (300, 400) -> screen (300, 200)
        points_arg = mock_polylines.call_args[0][1]
        points = points_arg[0]
        assert tuple(points[0][0]) == (100, 400)
        assert tuple(points[2][0]) == (300, 200)

    @patch('src.visualization.cv2.polylines')
    def test_handles_short_list(self, mock_polylines, mock_config, sample_frame):
        """Less than 2 points draws nothing."""
        renderer = VisualizationRenderer(mock_config)

        renderer.draw_trajectory(sample_frame, [(100.0, 200.0)])
        assert mock_polylines.call_count == 0

        renderer.draw_trajectory(sample_frame, [])
        assert mock_polylines.call_count == 0


class TestDrawStateIndicator:
    """Tests for draw_state_indicator method."""

    @patch('src.visualization.cv2.putText')
    @patch('src.visualization.cv2.rectangle')
    def test_draws_text_with_state(
        self, mock_rect, mock_text, mock_config, sample_frame
    ):
        """Draws state text with correct state name."""
        renderer = VisualizationRenderer(mock_config)
        renderer.draw_state_indicator(sample_frame, State.NAVIGATE)

        assert mock_text.call_count >= 1
        text = mock_text.call_args[0][1]
        assert "NAVIGATE" in text

    @patch('src.visualization.cv2.rectangle')
    def test_respects_toggle(self, mock_rect, mock_config, sample_frame):
        """Does not draw when show_state is False."""
        mock_config.visualization.show_state = False
        renderer = VisualizationRenderer(mock_config)
        renderer.draw_state_indicator(sample_frame, State.NAVIGATE)

        assert mock_rect.call_count == 0


class TestDrawMetrics:
    """Tests for draw_metrics method."""

    @patch('src.visualization.cv2.putText')
    @patch('src.visualization.cv2.rectangle')
    def test_shows_speed_heading_obstacles(
        self, mock_rect, mock_text, mock_config, sample_frame
    ):
        """Draws metrics panel with speed, heading, and obstacle count."""
        renderer = VisualizationRenderer(mock_config)
        metrics = {"speed": 5.0, "heading": 1.57, "obstacles": 3}
        renderer.draw_metrics(sample_frame, metrics)

        assert mock_text.call_count >= 1
        text = mock_text.call_args[0][1]
        assert "5.0" in text or "Speed" in text

    @patch('src.visualization.cv2.rectangle')
    def test_respects_toggle(self, mock_rect, mock_config, sample_frame):
        """Does not draw when show_metrics is False."""
        mock_config.visualization.show_metrics = False
        renderer = VisualizationRenderer(mock_config)
        renderer.draw_metrics(sample_frame, {"speed": 5.0})

        assert mock_rect.call_count == 0


class TestDrawWaypoints:
    """Tests for draw_waypoints method (world coords)."""

    @patch('src.visualization.cv2.circle')
    @patch('src.visualization.cv2.line')
    def test_draws_markers(
        self, mock_line, mock_circle, mock_config, sample_frame
    ):
        """Draws circle markers and connecting lines for waypoints."""
        renderer = VisualizationRenderer(mock_config)
        waypoints = [(100.0, 200.0), (300.0, 400.0), (500.0, 300.0)]
        renderer.draw_waypoints(sample_frame, waypoints, 1)

        # Should draw circles for each waypoint (fill + outline = 6)
        assert mock_circle.call_count >= 3
        # Should draw connecting lines between waypoints
        assert mock_line.call_count >= 2

    @patch('src.visualization.cv2.circle')
    def test_respects_toggle(self, mock_circle, mock_config, sample_frame):
        """Does not draw when show_waypoints is False."""
        mock_config.visualization.show_waypoints = False
        renderer = VisualizationRenderer(mock_config)
        renderer.draw_waypoints(sample_frame, [(100.0, 200.0)], 0)

        assert mock_circle.call_count == 0


class TestDrawGoalStatus:
    """Tests for draw_goal_status method."""

    @patch('src.visualization.cv2.putText')
    @patch('src.visualization.cv2.rectangle')
    def test_draws_text_or_banner(
        self, mock_rect, mock_text, mock_config, sample_frame
    ):
        """Draws status text when incomplete, banner when complete."""
        renderer = VisualizationRenderer(mock_config)

        # Incomplete
        renderer.draw_goal_status(sample_frame, "Waypoint 1/3", False)
        assert mock_text.call_count >= 1

        mock_text.reset_mock()
        mock_rect.reset_mock()

        # Complete
        renderer.draw_goal_status(sample_frame, "GOAL REACHED", True)
        assert mock_text.call_count >= 1
        text = mock_text.call_args[0][1]
        assert "GOAL REACHED" in text


class TestRenderSimulation:
    """Tests for render_simulation method."""

    @patch('src.visualization.cv2.arrowedLine')
    @patch('src.visualization.cv2.circle')
    @patch('src.visualization.cv2.line')
    @patch('src.visualization.cv2.polylines')
    @patch('src.visualization.cv2.putText')
    @patch('src.visualization.cv2.rectangle')
    def test_calls_all_draw_methods(
        self, mock_rect, mock_text, mock_polylines, mock_line,
        mock_circle, mock_arrow, mock_config, sample_frame,
        sample_agent, sample_obstacles, sample_readings, sample_forces
    ):
        """render_simulation calls all drawing methods and returns frame."""
        renderer = VisualizationRenderer(mock_config)
        result = renderer.render_simulation(
            frame=sample_frame,
            agent=sample_agent,
            state=State.NAVIGATE,
            obstacles=sample_obstacles,
            sensor_readings=sample_readings,
            forces=sample_forces,
        )

        assert result is not None
        assert result.shape == sample_frame.shape
        # Should have drawn obstacles (circles)
        assert mock_circle.call_count > 0
        # Should have drawn sensor rays (lines)
        assert mock_line.call_count > 0
        # Should have drawn force vectors (arrows)
        assert mock_arrow.call_count > 0

    @patch('src.visualization.cv2.arrowedLine')
    @patch('src.visualization.cv2.circle')
    @patch('src.visualization.cv2.putText')
    @patch('src.visualization.cv2.rectangle')
    def test_minimal_args(
        self, mock_rect, mock_text, mock_circle, mock_arrow,
        mock_config, sample_frame, sample_agent
    ):
        """render_simulation works with only required parameters."""
        renderer = VisualizationRenderer(mock_config)
        result = renderer.render_simulation(
            frame=sample_frame,
            agent=sample_agent,
            state=State.NAVIGATE,
            obstacles=[],
            sensor_readings=[],
            forces=[],
        )

        assert result is not None
        assert result.shape == sample_frame.shape
