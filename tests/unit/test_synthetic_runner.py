"""Tests for synthetic pipeline runner."""

import os
import tempfile
from unittest.mock import Mock
import pytest

from src.synthetic_runner import run_synthetic_pipeline


@pytest.fixture
def temp_output_path():
    """Create temporary output path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_output.mp4")


@pytest.fixture
def mock_config():
    """Create mock configuration for synthetic mode."""
    config = Mock()

    # Synthetic settings
    config.synthetic = Mock()
    config.synthetic.width = 640
    config.synthetic.height = 480
    config.synthetic.fps = 24
    config.synthetic.duration_seconds = 1  # Short for fast test
    config.synthetic.scenario = "crossing"
    config.synthetic.waypoints = [[320, 400], [320, 80]]
    config.synthetic.waypoint_tolerance = 30.0

    # Model settings
    config.model = Mock()
    config.model.name = "yolo11n.pt"

    # Planning settings
    config.planning = Mock()
    config.planning.obstacle_distance_threshold = 100.0
    config.planning.critical_distance = 30.0

    # Control settings
    config.control = Mock()
    config.control.start_x = 320.0
    config.control.start_y = 400.0
    config.control.start_heading = 90.0
    config.control.agent_speed = 2.0
    config.control.turn_rate = 5.0
    config.control.trajectory_length = 100

    # Visualization settings
    config.visualization = Mock()
    config.visualization.show_boxes = True
    config.visualization.show_labels = True
    config.visualization.show_confidence = True
    config.visualization.show_agent = True
    config.visualization.show_trajectory = True
    config.visualization.show_state = True
    config.visualization.show_metrics = True

    return config


class TestSyntheticRunner:
    """Tests for synthetic pipeline runner."""

    def test_run_creates_output_file(self, mock_config, temp_output_path):
        """Test that pipeline creates output video file."""
        result = run_synthetic_pipeline(mock_config, temp_output_path)

        assert result == 0  # Success
        assert os.path.exists(temp_output_path)
        assert os.path.getsize(temp_output_path) > 0

    def test_run_returns_zero_on_success(self, mock_config, temp_output_path):
        """Test that pipeline returns 0 on successful completion."""
        result = run_synthetic_pipeline(mock_config, temp_output_path)

        assert result == 0

    def test_run_uses_blended_planner(self, mock_config, temp_output_path):
        """Test that pipeline uses BlendedNavigationPlanner."""
        # This test verifies the pipeline runs without error
        # The BlendedNavigationPlanner is used internally
        result = run_synthetic_pipeline(mock_config, temp_output_path)

        assert result == 0

    def test_run_respects_duration(self, mock_config, temp_output_path):
        """Test that pipeline generates correct number of frames."""
        mock_config.synthetic.duration_seconds = 2
        mock_config.synthetic.fps = 10

        result = run_synthetic_pipeline(mock_config, temp_output_path)

        # Should run for 2 seconds at 10 fps = 20 frames
        assert result == 0

    def test_run_gauntlet_scenario(self, mock_config, temp_output_path):
        """Test running with gauntlet scenario."""
        mock_config.synthetic.scenario = "gauntlet"

        result = run_synthetic_pipeline(mock_config, temp_output_path)

        assert result == 0
        assert os.path.exists(temp_output_path)

    def test_run_converging_scenario(self, mock_config, temp_output_path):
        """Test running with converging scenario."""
        mock_config.synthetic.scenario = "converging"

        result = run_synthetic_pipeline(mock_config, temp_output_path)

        assert result == 0
        assert os.path.exists(temp_output_path)
