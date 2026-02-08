"""Integration tests for full video processing pipeline."""

import cv2  # type: ignore
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import sys

from src.main import main


def create_test_video(output_path: Path, num_frames: int = 10) -> None:
    """Create a simple test video for integration testing.

    Parameters
    ----------
    output_path : Path
        Where to save the test video
    num_frames : int, optional
        Number of frames to generate (default: 10)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create video writer (640x480, 30fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        30.0,
        (640, 480)
    )

    # Generate simple frames with gradient background
    for i in range(num_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add a moving gradient
        frame[:, :, 0] = (i * 25) % 256  # Blue channel changes
        frame[:, :, 1] = 128  # Green constant
        frame[:, :, 2] = 200  # Red constant
        writer.write(frame)

    writer.release()


@pytest.fixture
def test_video(tmp_path):
    """Create a test video fixture."""
    video_path = tmp_path / "test_input.mp4"
    create_test_video(video_path, num_frames=10)
    return video_path


@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model that returns fixed detections."""
    with patch('src.perception.YOLO') as mock_yolo:
        # Create mock model
        mock_model = Mock()

        # Mock detection results - simulate detecting one obstacle
        mock_boxes = Mock()
        # Bounding box at (300, 200) to (350, 250) - far from agent
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
            [300.0, 200.0, 350.0, 250.0]
        ])
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.85])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0.0])
        mock_boxes.__len__ = Mock(return_value=1)

        mock_result = Mock()
        mock_result.boxes = mock_boxes

        mock_model.predict.return_value = [mock_result]
        mock_model.names = {0: 'person'}

        mock_yolo.return_value = mock_model
        yield mock_yolo


class TestPipelineIntegration:
    """Integration tests for the full video processing pipeline."""

    def test_pipeline_processes_video_end_to_end(
        self,
        test_video,
        tmp_path,
        mock_yolo_model,
        monkeypatch
    ):
        """Test full pipeline processes video and creates output."""
        output_path = tmp_path / "test_output.mp4"

        # Mock command-line arguments
        test_args = [
            'src.main',
            '-i', str(test_video),
            '-o', str(output_path),
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        # Run main pipeline
        exit_code = main()

        # Verify successful execution
        assert exit_code == 0

        # Verify output video was created
        assert output_path.exists()

        # Verify output video is valid
        cap = cv2.VideoCapture(str(output_path))
        assert cap.isOpened()

        # Verify frame count matches input
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count == 10

        # Verify dimensions match
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert width == 640
        assert height == 480

        cap.release()

    def test_pipeline_with_frame_skip(
        self,
        test_video,
        tmp_path,
        mock_yolo_model,
        monkeypatch
    ):
        """Test pipeline with frame skip option."""
        output_path = tmp_path / "test_output_skip.mp4"

        test_args = [
            'src.main',
            '-i', str(test_video),
            '-o', str(output_path),
            '--frame-skip', '2'
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        exit_code = main()

        assert exit_code == 0
        assert output_path.exists()

    def test_pipeline_with_max_frames(
        self,
        test_video,
        tmp_path,
        mock_yolo_model,
        monkeypatch
    ):
        """Test pipeline with max frames limit."""
        output_path = tmp_path / "test_output_max.mp4"

        test_args = [
            'src.main',
            '-i', str(test_video),
            '-o', str(output_path),
            '--max-frames', '5'
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        exit_code = main()

        assert exit_code == 0
        assert output_path.exists()

        # Verify only 5 frames were written
        cap = cv2.VideoCapture(str(output_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Note: actual frame count may vary slightly due to codec,
        # but should be close to 5
        assert 4 <= frame_count <= 6

    def test_pipeline_with_invalid_input(self, tmp_path, monkeypatch):
        """Test pipeline handles invalid input gracefully."""
        output_path = tmp_path / "test_output.mp4"

        test_args = [
            'src.main',
            '-i', 'nonexistent_video.mp4',
            '-o', str(output_path),
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        exit_code = main()

        # Should fail with error code 1
        assert exit_code == 1
