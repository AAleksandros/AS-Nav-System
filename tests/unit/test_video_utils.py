"""Unit tests for video I/O utilities."""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from src.utils.video_utils import VideoReader, VideoWriter


class TestVideoReader:
    """Test VideoReader class."""

    @patch('cv2.VideoCapture')
    def test_init_valid_path(self, mock_capture):
        """Test VideoReader initialization with valid path."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        # fps, width, height, frame_count
        mock_cap.get.side_effect = [30.0, 640.0, 480.0, 100.0]
        mock_capture.return_value = mock_cap

        reader = VideoReader("test_video.mp4")

        assert reader.fps == 30.0
        assert reader.width == 640
        assert reader.height == 480
        assert reader.frame_count == 100
        mock_capture.assert_called_once_with("test_video.mp4")

    @patch('cv2.VideoCapture')
    def test_init_invalid_path(self, mock_capture):
        """Test VideoReader initialization with invalid path."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap

        with pytest.raises(ValueError, match="Failed to open video file"):
            VideoReader("nonexistent.mp4")

    @patch('cv2.VideoCapture')
    def test_frame_iteration(self, mock_capture):
        """Test frame iteration protocol."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [30.0, 640.0, 480.0, 3.0]  # 3 frames

        # Simulate reading 3 frames then EOF
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.ones((480, 640, 3), dtype=np.uint8)),
            (True, np.full((480, 640, 3), 255, dtype=np.uint8)),
            (False, None),  # EOF
        ]
        mock_capture.return_value = mock_cap

        reader = VideoReader("test.mp4")
        frames = list(reader)

        assert len(frames) == 3
        assert frames[0].shape == (480, 640, 3)
        assert np.all(frames[0] == 0)
        assert np.all(frames[1] == 1)
        assert np.all(frames[2] == 255)

    @patch('cv2.VideoCapture')
    def test_context_manager(self, mock_capture):
        """Test VideoReader as context manager."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [30.0, 640.0, 480.0, 10.0]
        mock_capture.return_value = mock_cap

        with VideoReader("test.mp4") as reader:
            assert reader.fps == 30.0

        mock_cap.release.assert_called_once()

    @patch('cv2.VideoCapture')
    def test_release(self, mock_capture):
        """Test explicit release."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [30.0, 640.0, 480.0, 10.0]
        mock_capture.return_value = mock_cap

        reader = VideoReader("test.mp4")
        reader.release()

        mock_cap.release.assert_called_once()


class TestVideoWriter:
    """Test VideoWriter class."""

    @patch('cv2.VideoWriter')
    def test_init_valid_params(self, mock_writer):
        """Test VideoWriter initialization."""
        mock_vid = Mock()
        mock_vid.isOpened.return_value = True
        mock_writer.return_value = mock_vid

        writer = VideoWriter("output.mp4", fps=30.0, width=640, height=480)

        assert writer.fps == 30.0
        assert writer.width == 640
        assert writer.height == 480
        # Check that fourcc was created and VideoWriter called
        mock_writer.assert_called_once()
        call_args = mock_writer.call_args[0]
        assert call_args[0] == "output.mp4"
        assert call_args[2] == 30.0
        assert call_args[3] == (640, 480)

    @patch('cv2.VideoWriter')
    def test_init_creates_parent_dir(self, mock_writer, tmp_path):
        """Test VideoWriter creates parent directory."""
        mock_vid = Mock()
        mock_vid.isOpened.return_value = True
        mock_writer.return_value = mock_vid

        output_path = tmp_path / "subdir" / "output.mp4"
        VideoWriter(str(output_path), fps=30.0, width=640, height=480)

        assert output_path.parent.exists()

    @patch('cv2.VideoWriter')
    def test_init_invalid_writer(self, mock_writer):
        """Test VideoWriter initialization failure."""
        mock_vid = Mock()
        mock_vid.isOpened.return_value = False
        mock_writer.return_value = mock_vid

        with pytest.raises(ValueError, match="Failed to create video writer"):
            VideoWriter("output.mp4", fps=30.0, width=640, height=480)

    @patch('cv2.VideoWriter')
    def test_write_frame(self, mock_writer):
        """Test writing a frame."""
        mock_vid = Mock()
        mock_vid.isOpened.return_value = True
        mock_writer.return_value = mock_vid

        writer = VideoWriter("output.mp4", fps=30.0, width=640, height=480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        writer.write_frame(frame)

        mock_vid.write.assert_called_once()
        written_frame = mock_vid.write.call_args[0][0]
        assert np.array_equal(written_frame, frame)

    @patch('cv2.VideoWriter')
    def test_write_frame_wrong_shape(self, mock_writer):
        """Test writing frame with wrong dimensions."""
        mock_vid = Mock()
        mock_vid.isOpened.return_value = True
        mock_writer.return_value = mock_vid

        writer = VideoWriter("output.mp4", fps=30.0, width=640, height=480)
        wrong_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Frame dimensions"):
            writer.write_frame(wrong_frame)

    @patch('cv2.VideoWriter')
    def test_context_manager(self, mock_writer):
        """Test VideoWriter as context manager."""
        mock_vid = Mock()
        mock_vid.isOpened.return_value = True
        mock_writer.return_value = mock_vid

        with VideoWriter("output.mp4", fps=30.0, width=640, height=480) as writer:
            assert writer.fps == 30.0

        mock_vid.release.assert_called_once()

    @patch('cv2.VideoWriter')
    def test_release(self, mock_writer):
        """Test explicit release."""
        mock_vid = Mock()
        mock_vid.isOpened.return_value = True
        mock_writer.return_value = mock_vid

        writer = VideoWriter("output.mp4", fps=30.0, width=640, height=480)
        writer.release()

        mock_vid.release.assert_called_once()
