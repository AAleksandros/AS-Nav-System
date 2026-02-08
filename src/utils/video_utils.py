"""Video I/O utilities for reading and writing video files.

This module provides VideoReader and VideoWriter classes that wrap OpenCV's
video capture and writer functionality with a clean, Pythonic interface.
"""

import cv2  # type: ignore
import numpy as np
from pathlib import Path
from typing import Iterator


class VideoReader:
    """Read video frames with iterator protocol.

    Provides a convenient interface for reading video files frame-by-frame
    using OpenCV VideoCapture. Supports context manager protocol for
    automatic resource cleanup.

    Parameters
    ----------
    video_path : str
        Path to the input video file

    Raises
    ------
    ValueError
        If the video file cannot be opened

    Examples
    --------
    >>> with VideoReader("input.mp4") as reader:
    ...     print(f"FPS: {reader.fps}, Frames: {reader.frame_count}")
    ...     for frame in reader:
    ...         process(frame)
    """

    def __init__(self, video_path: str) -> None:
        """Initialize video reader.

        Parameters
        ----------
        video_path : str
            Path to the input video file
        """
        self.video_path = video_path
        self._cap = cv2.VideoCapture(video_path)

        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        # Cache video properties
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float:
        """Get video frame rate (frames per second)."""
        return self._fps

    @property
    def width(self) -> int:
        """Get video frame width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Get video frame height in pixels."""
        return self._height

    @property
    def frame_count(self) -> int:
        """Get total number of frames in video."""
        return self._frame_count

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over video frames.

        Yields
        ------
        np.ndarray
            Video frame as numpy array (height, width, channels)
        """
        return self

    def __next__(self) -> np.ndarray:
        """Get next video frame.

        Returns
        -------
        np.ndarray
            Video frame as numpy array

        Raises
        ------
        StopIteration
            When all frames have been read
        """
        ret, frame = self._cap.read()
        if not ret:
            raise StopIteration
        return frame

    def release(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "VideoReader":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and release resources."""
        self.release()


class VideoWriter:
    """Write video frames with context manager support.

    Provides a convenient interface for writing video files frame-by-frame
    using OpenCV VideoWriter. Automatically creates parent directories and
    supports context manager protocol.

    Parameters
    ----------
    output_path : str
        Path to the output video file
    fps : float
        Frame rate for output video
    width : int
        Frame width in pixels
    height : int
        Frame height in pixels
    codec : str, optional
        FourCC codec code (default: 'mp4v')

    Raises
    ------
    ValueError
        If the video writer cannot be created

    Examples
    --------
    >>> with VideoWriter("output.mp4", fps=30, width=640, height=480) as writer:
    ...     for frame in frames:
    ...         writer.write_frame(frame)
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v"
    ) -> None:
        """Initialize video writer.

        Parameters
        ----------
        output_path : str
            Path to the output video file
        fps : float
            Frame rate for output video
        width : int
            Frame width in pixels
        height : int
            Frame height in pixels
        codec : str, optional
            FourCC codec code (default: 'mp4v')
        """
        self.output_path = output_path
        self._fps = fps
        self._width = width
        self._height = height

        # Create parent directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore
        self._writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        if not self._writer.isOpened():
            raise ValueError(f"Failed to create video writer for: {output_path}")

    @property
    def fps(self) -> float:
        """Get output video frame rate."""
        return self._fps

    @property
    def width(self) -> int:
        """Get output video frame width."""
        return self._width

    @property
    def height(self) -> int:
        """Get output video frame height."""
        return self._height

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single frame to the video.

        Parameters
        ----------
        frame : np.ndarray
            Video frame as numpy array (height, width, channels)

        Raises
        ------
        ValueError
            If frame dimensions don't match expected width/height
        """
        # Validate frame dimensions
        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            raise ValueError(
                f"Frame dimensions {frame.shape[:2]} don't match "
                f"expected ({self._height}, {self._width})"
            )

        self._writer.write(frame)

    def release(self) -> None:
        """Release video writer resources."""
        if self._writer is not None:
            self._writer.release()

    def __enter__(self) -> "VideoWriter":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and release resources."""
        self.release()
