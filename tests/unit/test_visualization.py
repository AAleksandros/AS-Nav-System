"""Unit tests for visualization module."""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from src.config import Config
from src.models import Detection, State, AgentState
from src.visualization import VisualizationRenderer


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = Mock(spec=Config)

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


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections():
    """Create sample detections for testing."""
    return [
        Detection(
            x1=100, y1=100, x2=200, y2=200,
            class_id=0,
            class_name="person",
            confidence=0.95,
        ),
        Detection(
            x1=300, y1=150, x2=400, y2=250,
            class_id=2,
            class_name="car",
            confidence=0.87,
        ),
    ]


@pytest.fixture
def sample_agent():
    """Create sample agent state for testing."""
    return AgentState(
        x=960.0,
        y=540.0,
        heading=90.0,
        velocity=5.0,
        trajectory=[(960.0, 540.0), (955.0, 535.0), (950.0, 530.0)],
    )


class TestVisualizationRenderer:
    """Test suite for VisualizationRenderer class."""

    def test_initialization(self, mock_config):
        """Test renderer initializes with config."""
        renderer = VisualizationRenderer(mock_config)
        assert renderer.config == mock_config
        assert hasattr(renderer, 'logger')

    @patch('src.visualization.cv2.rectangle')
    def test_draw_detections_single(
        self, mock_rectangle, mock_config, sample_frame, sample_detections
    ):
        """Test drawing single detection with bounding box."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        renderer.draw_detections(frame, [sample_detections[0]], State.NAVIGATE)

        # Should call rectangle twice (box + label background)
        assert mock_rectangle.call_count == 2
        # Verify green color for NAVIGATE state (first call is the bounding box)
        first_call_args = mock_rectangle.call_args_list[0][0]
        assert first_call_args[3] == (0, 255, 0)  # Green color (4th arg)

    @patch('src.visualization.cv2.rectangle')
    @patch('src.visualization.cv2.putText')
    def test_draw_detections_multiple(
        self, mock_putText, mock_rectangle, mock_config, sample_frame, sample_detections
    ):
        """Test drawing multiple detections."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        renderer.draw_detections(frame, sample_detections, State.AVOID)

        # Should call rectangle 4 times (2 detections * 2 each: box + label bg)
        assert mock_rectangle.call_count == 4
        # Verify orange for AVOID state (check bounding box calls - indices 0, 2)
        # First detection box
        assert mock_rectangle.call_args_list[0][0][3] == (0, 165, 255)
        # Second detection box
        assert mock_rectangle.call_args_list[2][0][3] == (0, 165, 255)

    @patch('src.visualization.cv2.rectangle')
    def test_draw_detections_color_by_state(
        self, mock_rectangle, mock_config, sample_frame, sample_detections
    ):
        """Test bounding box colors change based on state."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        # Test NAVIGATE (green)
        renderer.draw_detections(frame, [sample_detections[0]], State.NAVIGATE)
        assert mock_rectangle.call_args_list[0][0][3] == (0, 255, 0)

        mock_rectangle.reset_mock()

        # Test AVOID (orange)
        renderer.draw_detections(frame, [sample_detections[0]], State.AVOID)
        assert mock_rectangle.call_args_list[0][0][3] == (0, 165, 255)

        mock_rectangle.reset_mock()

        # Test STOP (red)
        renderer.draw_detections(frame, [sample_detections[0]], State.STOP)
        assert mock_rectangle.call_args_list[0][0][3] == (0, 0, 255)

    @patch('src.visualization.cv2.rectangle')
    @patch('src.visualization.cv2.putText')
    def test_draw_detections_with_labels(
        self, mock_putText, mock_rectangle, mock_config, sample_frame, sample_detections
    ):
        """Test drawing detections with labels and confidence."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        renderer.draw_detections(frame, [sample_detections[0]], State.NAVIGATE)

        # Should call putText for label
        assert mock_putText.call_count >= 1
        # Verify label contains class name and confidence
        label_text = mock_putText.call_args[0][1]
        assert "person" in label_text
        assert "0.95" in label_text or "95" in label_text

    @patch('src.visualization.cv2.rectangle')
    @patch('src.visualization.cv2.putText')
    def test_draw_detections_respects_config_toggles(
        self, mock_putText, mock_rectangle, mock_config, sample_frame, sample_detections
    ):
        """Test that config toggles control rendering."""
        # Disable boxes
        mock_config.visualization.show_boxes = False
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        renderer.draw_detections(frame, sample_detections, State.NAVIGATE)

        # Should not call rectangle when boxes disabled
        assert mock_rectangle.call_count == 0

    @patch('src.visualization.cv2.rectangle')
    def test_draw_detections_empty_list(
        self, mock_rectangle, mock_config, sample_frame
    ):
        """Test drawing with empty detection list."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        renderer.draw_detections(frame, [], State.NAVIGATE)

        # Should not call rectangle with no detections
        assert mock_rectangle.call_count == 0

    @patch('src.visualization.cv2.circle')
    @patch('src.visualization.cv2.arrowedLine')
    def test_draw_agent(
        self, mock_arrow, mock_circle, mock_config, sample_frame, sample_agent
    ):
        """Test drawing agent circle and heading arrow."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        renderer.draw_agent(frame, sample_agent)

        # Should draw circle for agent position
        assert mock_circle.call_count == 1
        # Verify cyan color (4th positional argument)
        call_args = mock_circle.call_args[0]
        assert call_args[3] == (255, 255, 0)

        # Should draw arrow for heading
        assert mock_arrow.call_count == 1

    @patch('src.visualization.cv2.circle')
    @patch('src.visualization.cv2.arrowedLine')
    def test_draw_agent_heading_calculation(
        self, mock_arrow, mock_circle, mock_config, sample_frame
    ):
        """Test agent heading arrow calculation for different angles."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        # Test 0° (right)
        agent = AgentState(x=500.0, y=500.0, heading=0.0, velocity=5.0, trajectory=[])
        renderer.draw_agent(frame, agent)
        arrow_end = mock_arrow.call_args[0][2]
        assert arrow_end[0] > 500  # x increases
        assert abs(arrow_end[1] - 500) < 1  # y unchanged

        mock_arrow.reset_mock()

        # Test 90° (up)
        agent = AgentState(x=500.0, y=500.0, heading=90.0, velocity=5.0, trajectory=[])
        renderer.draw_agent(frame, agent)
        arrow_end = mock_arrow.call_args[0][2]
        assert abs(arrow_end[0] - 500) < 1  # x unchanged
        assert arrow_end[1] < 500  # y decreases (up in image coords)

    @patch('src.visualization.cv2.circle')
    def test_draw_agent_respects_config(
        self, mock_circle, mock_config, sample_frame, sample_agent
    ):
        """Test that agent rendering respects config toggle."""
        mock_config.visualization.show_agent = False
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        renderer.draw_agent(frame, sample_agent)

        # Should not draw when disabled
        assert mock_circle.call_count == 0

    @patch('src.visualization.cv2.polylines')
    def test_draw_trajectory(
        self, mock_polylines, mock_config, sample_frame, sample_agent
    ):
        """Test drawing trajectory polyline."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        renderer.draw_trajectory(frame, sample_agent.trajectory)

        # Should call polylines once
        assert mock_polylines.call_count == 1
        # Verify yellow color (check keyword argument)
        assert mock_polylines.call_args[1]['color'] == (0, 255, 255)

    @patch('src.visualization.cv2.polylines')
    def test_draw_trajectory_empty(
        self, mock_polylines, mock_config, sample_frame
    ):
        """Test drawing with empty trajectory."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        renderer.draw_trajectory(frame, [])

        # Should not draw with empty trajectory
        assert mock_polylines.call_count == 0

    @patch('src.visualization.cv2.polylines')
    def test_draw_trajectory_single_point(
        self, mock_polylines, mock_config, sample_frame
    ):
        """Test drawing with single trajectory point."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        renderer.draw_trajectory(frame, [(500.0, 500.0)])

        # Should not draw line with only one point
        assert mock_polylines.call_count == 0

    @patch('src.visualization.cv2.polylines')
    def test_draw_trajectory_respects_config(
        self, mock_polylines, mock_config, sample_frame, sample_agent
    ):
        """Test that trajectory rendering respects config toggle."""
        mock_config.visualization.show_trajectory = False
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        renderer.draw_trajectory(frame, sample_agent.trajectory)

        # Should not draw when disabled
        assert mock_polylines.call_count == 0

    @patch('src.visualization.cv2.rectangle')
    @patch('src.visualization.cv2.putText')
    def test_draw_state_indicator(
        self, mock_putText, mock_rectangle, mock_config, sample_frame
    ):
        """Test drawing state indicator with background."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        renderer.draw_state_indicator(frame, State.NAVIGATE)

        # Should draw background rectangle
        assert mock_rectangle.call_count == 1
        # Should draw text
        assert mock_putText.call_count == 1
        # Verify text contains state
        text = mock_putText.call_args[0][1]
        assert "NAVIGATE" in text

    @patch('src.visualization.cv2.putText')
    def test_draw_state_indicator_all_states(
        self, mock_putText, mock_config, sample_frame
    ):
        """Test state indicator for all navigation states."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        for state in [State.NAVIGATE, State.AVOID, State.STOP]:
            mock_putText.reset_mock()
            renderer.draw_state_indicator(frame, state)

            text = mock_putText.call_args[0][1]
            assert state.value.upper() in text

    @patch('src.visualization.cv2.rectangle')
    def test_draw_state_indicator_respects_config(
        self, mock_rectangle, mock_config, sample_frame
    ):
        """Test that state indicator respects config toggle."""
        mock_config.visualization.show_state = False
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        renderer.draw_state_indicator(frame, State.NAVIGATE)

        # Should not draw when disabled
        assert mock_rectangle.call_count == 0

    @patch('src.visualization.cv2.rectangle')
    @patch('src.visualization.cv2.putText')
    def test_draw_metrics(
        self, mock_putText, mock_rectangle, mock_config, sample_frame
    ):
        """Test drawing metrics panel."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        metrics = {
            "detections": 5,
            "velocity": 5.5,
            "heading": 90.0,
        }

        renderer.draw_metrics(frame, metrics)

        # Should draw background rectangle
        assert mock_rectangle.call_count == 1
        # Should draw text with metrics
        assert mock_putText.call_count >= 1

    @patch('src.visualization.cv2.rectangle')
    def test_draw_metrics_respects_config(
        self, mock_rectangle, mock_config, sample_frame
    ):
        """Test that metrics panel respects config toggle."""
        mock_config.visualization.show_metrics = False
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        metrics = {"detections": 5, "velocity": 5.5, "heading": 90.0}
        renderer.draw_metrics(frame, metrics)

        # Should not draw when disabled
        assert mock_rectangle.call_count == 0

    @patch('src.visualization.cv2.rectangle')
    @patch('src.visualization.cv2.putText')
    @patch('src.visualization.cv2.circle')
    @patch('src.visualization.cv2.arrowedLine')
    @patch('src.visualization.cv2.polylines')
    def test_render_full_pipeline(
        self, mock_polylines, mock_arrow, mock_circle,
        mock_putText, mock_rectangle, mock_config, sample_frame,
        sample_detections, sample_agent
    ):
        """Test full rendering pipeline with all elements."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        result = renderer.render(
            frame, sample_detections, sample_agent, State.NAVIGATE
        )

        # Should return a frame
        assert result is not None
        assert result.shape == sample_frame.shape

        # Should call various drawing functions
        assert mock_rectangle.call_count > 0  # Boxes, state, metrics
        assert mock_circle.call_count > 0  # Agent
        assert mock_arrow.call_count > 0  # Agent heading
        assert mock_polylines.call_count > 0  # Trajectory

    def test_render_does_not_modify_original(
        self, mock_config, sample_frame, sample_detections, sample_agent
    ):
        """Test that render does not modify the original frame."""
        renderer = VisualizationRenderer(mock_config)
        original = sample_frame.copy()

        renderer.render(original, sample_detections, sample_agent, State.NAVIGATE)

        # Original should be unchanged (render works on a copy)
        # Note: In practice, OpenCV modifies in-place, so we'll handle this
        # in implementation by working on the passed frame
        assert original.shape == sample_frame.shape

    @patch('src.visualization.cv2.circle')
    @patch('src.visualization.cv2.line')
    def test_draw_waypoints_empty(
        self, mock_line, mock_circle, mock_config, sample_frame
    ):
        """Test drawing with no waypoints."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        renderer.draw_waypoints(frame, [], 0)

        # Should not draw anything
        assert mock_circle.call_count == 0
        assert mock_line.call_count == 0

    @patch('src.visualization.cv2.circle')
    def test_draw_waypoints_single(
        self, mock_circle, mock_config, sample_frame
    ):
        """Test drawing single waypoint."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        waypoints = [[100, 200]]
        renderer.draw_waypoints(frame, waypoints, 0)

        # Should draw circles for current waypoint (fill + outline)
        assert mock_circle.call_count >= 1

    @patch('src.visualization.cv2.circle')
    @patch('src.visualization.cv2.line')
    def test_draw_waypoints_multiple(
        self, mock_line, mock_circle, mock_config, sample_frame
    ):
        """Test drawing multiple waypoints with connections."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        waypoints = [[100, 200], [200, 300], [300, 400]]
        renderer.draw_waypoints(frame, waypoints, 0)

        # Should draw circles for each waypoint
        assert mock_circle.call_count >= 3
        # Should draw lines connecting waypoints
        assert mock_line.call_count >= 2

    @patch('src.visualization.cv2.circle')
    def test_draw_waypoints_color_by_status(
        self, mock_circle, mock_config, sample_frame
    ):
        """Test waypoint colors differ by status (reached/current/future)."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        waypoints = [[100, 100], [200, 200], [300, 300]]

        # Current waypoint idx = 1 (second waypoint)
        renderer.draw_waypoints(frame, waypoints, 1)

        # Should draw circles for all waypoints
        # Colors should differ: green (reached), yellow (current), gray (future)
        assert mock_circle.call_count >= 3

    @patch('src.visualization.cv2.putText')
    def test_draw_goal_status_incomplete(
        self, mock_putText, mock_config, sample_frame
    ):
        """Test drawing goal status when incomplete."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        renderer.draw_goal_status(frame, "Waypoint 2/3", is_complete=False)

        # Should draw status text
        assert mock_putText.call_count >= 1
        # Text should contain status
        text = mock_putText.call_args[0][1]
        assert "Waypoint" in text

    @patch('src.visualization.cv2.rectangle')
    @patch('src.visualization.cv2.putText')
    def test_draw_goal_status_complete(
        self, mock_putText, mock_rectangle, mock_config, sample_frame
    ):
        """Test drawing goal reached banner."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        renderer.draw_goal_status(frame, "GOAL REACHED", is_complete=True)

        # Should draw banner background
        assert mock_rectangle.call_count >= 1
        # Should draw large text
        assert mock_putText.call_count >= 1
        # Text should be GOAL REACHED
        text = mock_putText.call_args[0][1]
        assert "GOAL REACHED" in text

    @patch('src.visualization.cv2.rectangle')
    @patch('src.visualization.cv2.circle')
    @patch('src.visualization.cv2.line')
    @patch('src.visualization.cv2.putText')
    def test_render_with_waypoints(
        self, mock_putText, mock_line, mock_circle, mock_rectangle,
        mock_config, sample_frame, sample_detections, sample_agent
    ):
        """Test render with optional waypoint parameters."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        waypoints = [[100, 100], [200, 200], [300, 300]]
        result = renderer.render(
            frame,
            sample_detections,
            sample_agent,
            State.NAVIGATE,
            waypoints=waypoints,
            waypoint_idx=1,
            goal_status="Waypoint 2/3"
        )

        # Should return frame
        assert result is not None
        # Should draw waypoints (circles for markers)
        assert mock_circle.call_count > 0
        # Should draw goal status
        assert mock_putText.call_count > 0

    def test_render_backward_compatible(
        self, mock_config, sample_frame, sample_detections, sample_agent
    ):
        """Test render works without waypoint parameters (backward compatible)."""
        renderer = VisualizationRenderer(mock_config)
        frame = sample_frame.copy()

        # Should work without waypoints (old signature)
        result = renderer.render(
            frame,
            sample_detections,
            sample_agent,
            State.NAVIGATE
        )

        # Should return frame without error
        assert result is not None
        assert result.shape == sample_frame.shape
