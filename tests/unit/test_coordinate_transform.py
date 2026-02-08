"""Unit tests for the coordinate transform module."""

import math
from unittest.mock import Mock

import pytest

from src.config import Config
from src.coordinate_transform import CoordinateTransform
from src.models import AgentState, Detection


@pytest.fixture
def sample_config():
    """Create a mock config for testing."""
    config = Mock(spec=Config)
    return config


@pytest.fixture
def sample_agent():
    """Create an agent at a known position."""
    return AgentState(x=320.0, y=400.0)


@pytest.fixture
def sample_detections():
    """Create detections at known positions for distance testing."""
    return [
        Detection(
            x1=100.0, y1=100.0, x2=200.0, y2=200.0,
            class_id=0, class_name="person", confidence=0.9,
        ),  # center = (150, 150)
        Detection(
            x1=300.0, y1=380.0, x2=340.0, y2=420.0,
            class_id=2, class_name="car", confidence=0.85,
        ),  # center = (320, 400) -- same as agent
        Detection(
            x1=500.0, y1=300.0, x2=600.0, y2=400.0,
            class_id=1, class_name="bicycle", confidence=0.7,
        ),  # center = (550, 350)
    ]


class TestCalculateDistance:
    """Tests for calculate_distance method."""

    def test_distance_known_values(self, sample_config, sample_agent):
        """Test distance calculation with known Euclidean distance."""
        ct = CoordinateTransform(sample_config)

        # Detection centered at (150, 150), agent at (320, 400)
        detection = Detection(
            x1=100.0, y1=100.0, x2=200.0, y2=200.0,
            class_id=0, class_name="person", confidence=0.9,
        )

        distance = ct.calculate_distance(detection, sample_agent)

        # dx = 150 - 320 = -170, dy = 150 - 400 = -250
        expected = math.sqrt(170**2 + 250**2)
        assert distance == pytest.approx(expected)

    def test_distance_zero(self, sample_config, sample_agent):
        """Test distance is zero when detection center matches agent position."""
        ct = CoordinateTransform(sample_config)

        # Detection centered at agent position (320, 400)
        detection = Detection(
            x1=300.0, y1=380.0, x2=340.0, y2=420.0,
            class_id=0, class_name="person", confidence=0.9,
        )

        distance = ct.calculate_distance(detection, sample_agent)
        assert distance == 0.0

    def test_distance_diagonal_345(self, sample_config):
        """Test with 3-4-5 triangle for easy verification."""
        ct = CoordinateTransform(sample_config)

        # Agent at origin (0, 0), detection centered at (3, 4)
        agent = AgentState(x=0.0, y=0.0)
        detection = Detection(
            x1=1.0, y1=2.0, x2=5.0, y2=6.0,
            class_id=0, class_name="person", confidence=0.9,
        )  # center = (3.0, 4.0)

        distance = ct.calculate_distance(detection, agent)
        assert distance == pytest.approx(5.0)


class TestFindNearestObstacle:
    """Tests for find_nearest_obstacle method."""

    def test_finds_nearest_among_multiple(
        self, sample_config, sample_agent, sample_detections
    ):
        """Test that nearest detection is returned from multiple."""
        ct = CoordinateTransform(sample_config)

        result = ct.find_nearest_obstacle(sample_detections, sample_agent)

        assert result is not None
        distance, detection = result
        # The second detection (center 320,400) is at agent position -> distance 0
        assert detection.class_name == "car"
        assert distance == 0.0

    def test_returns_none_for_empty_list(self, sample_config, sample_agent):
        """Test that empty detection list returns None."""
        ct = CoordinateTransform(sample_config)

        result = ct.find_nearest_obstacle([], sample_agent)

        assert result is None

    def test_single_detection(self, sample_config, sample_agent):
        """Test with a single detection returns that detection."""
        ct = CoordinateTransform(sample_config)

        detection = Detection(
            x1=400.0, y1=300.0, x2=500.0, y2=400.0,
            class_id=0, class_name="person", confidence=0.9,
        )  # center = (450, 350)

        result = ct.find_nearest_obstacle([detection], sample_agent)

        assert result is not None
        distance, returned_detection = result
        assert returned_detection is detection

    def test_nearest_returns_correct_distance(self, sample_config):
        """Test that the returned distance value is correct."""
        ct = CoordinateTransform(sample_config)

        agent = AgentState(x=0.0, y=0.0)
        detections = [
            Detection(
                x1=5.0, y1=5.0, x2=15.0, y2=15.0,
                class_id=0, class_name="person", confidence=0.9,
            ),  # center = (10, 10), distance = sqrt(200)
            Detection(
                x1=1.0, y1=1.0, x2=5.0, y2=5.0,
                class_id=1, class_name="car", confidence=0.85,
            ),  # center = (3, 3), distance = sqrt(18)
        ]

        result = ct.find_nearest_obstacle(detections, agent)

        assert result is not None
        distance, detection = result
        expected_distance = math.sqrt(3**2 + 3**2)
        assert distance == pytest.approx(expected_distance)
        assert detection.class_name == "car"


class TestImageToWorld:
    """Tests for image_to_world stub method."""

    def test_identity_transform(self, sample_config):
        """Test that identity transform returns input unchanged."""
        ct = CoordinateTransform(sample_config)

        wx, wy = ct.image_to_world(100.0, 200.0)

        assert wx == 100.0
        assert wy == 200.0

    def test_identity_preserves_values(self, sample_config):
        """Test identity transform with multiple coordinate pairs."""
        ct = CoordinateTransform(sample_config)

        test_coords = [(0.0, 0.0), (320.0, 240.0), (639.0, 479.0)]
        for x, y in test_coords:
            wx, wy = ct.image_to_world(x, y)
            assert wx == x
            assert wy == y
