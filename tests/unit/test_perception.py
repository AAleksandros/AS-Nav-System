"""Unit tests for the perception module (YOLO-based object detection)."""

import logging
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest

from src.config import Config
from src.models import Detection


@pytest.fixture
def sample_config():
    """Create a mock config for testing."""
    config = Mock(spec=Config)
    config.model = Mock()
    config.model.name = "yolo11n.pt"
    config.model.device = "cpu"
    config.model.confidence_threshold = 0.5
    config.planning = Mock()
    config.planning.obstacle_classes = ["person", "car", "bicycle"]
    return config


@pytest.fixture
def sample_config_cuda():
    """Create a mock config requesting CUDA."""
    config = Mock(spec=Config)
    config.model = Mock()
    config.model.name = "yolo11n.pt"
    config.model.device = "cuda"
    config.model.confidence_threshold = 0.5
    config.planning = Mock()
    config.planning.obstacle_classes = ["person", "car"]
    return config


class TestObjectDetectorInitialization:
    """Tests for ObjectDetector initialization."""

    def test_initialization_cpu_device(self, sample_config):
        """Test that ObjectDetector initializes correctly with CPU device."""
        from src.perception import ObjectDetector

        with patch('src.perception.YOLO') as mock_yolo_class:
            mock_model = Mock()
            mock_yolo_class.return_value = mock_model

            detector = ObjectDetector(sample_config)

            # Verify YOLO was initialized with correct model
            mock_yolo_class.assert_called_once_with("yolo11n.pt")

            # Verify model was moved to CPU
            mock_model.to.assert_called_once_with("cpu")

            # Verify config and model stored
            assert detector.config == sample_config
            assert detector.model == mock_model

    def test_initialization_cuda_fallback_to_cpu(self, sample_config_cuda):
        """Test that CUDA request falls back to CPU when CUDA unavailable."""
        from src.perception import ObjectDetector

        with patch('src.perception.YOLO') as mock_yolo_class, \
             patch('src.perception.torch') as mock_torch:
            mock_model = Mock()
            mock_yolo_class.return_value = mock_model
            mock_torch.cuda.is_available.return_value = False  # No CUDA available

            detector = ObjectDetector(sample_config_cuda)

            # Verify model was moved to CPU (fallback)
            mock_model.to.assert_called_once_with("cpu")

    def test_initialization_model_load_failure(self, sample_config):
        """Test that initialization raises RuntimeError when model fails to load."""
        from src.perception import ObjectDetector

        with patch('src.perception.YOLO') as mock_yolo_class:
            mock_yolo_class.side_effect = Exception("Model file not found")

            with pytest.raises(RuntimeError, match="Failed to load YOLO model"):
                ObjectDetector(sample_config)


class TestObjectDetectorDetection:
    """Tests for ObjectDetector detection methods."""

    @pytest.fixture
    def mock_yolo_results(self):
        """Create mock YOLO results with detections."""
        # Mock the results structure
        mock_result = Mock()
        mock_boxes = Mock()

        # Create numpy arrays for detection data
        # 3 detections: person, car, dog
        xyxy = np.array([
            [100.0, 150.0, 200.0, 250.0],  # person
            [300.0, 100.0, 450.0, 300.0],  # car
            [500.0, 200.0, 600.0, 350.0],  # dog
        ])
        conf = np.array([0.85, 0.92, 0.75])
        cls = np.array([0, 2, 16], dtype=int)  # COCO IDs: person=0, car=2, dog=16

        # Mock boxes object with __len__ support
        mock_boxes.xyxy = Mock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = xyxy
        mock_boxes.conf = Mock()
        mock_boxes.conf.cpu.return_value.numpy.return_value = conf
        mock_boxes.cls = Mock()
        mock_boxes.cls.cpu.return_value.numpy.return_value = cls
        mock_boxes.__len__ = Mock(return_value=3)  # 3 detections

        mock_result.boxes = mock_boxes

        return [mock_result]

    def test_detect_frame_returns_detections(self, sample_config, mock_yolo_results):
        """Test that detect_frame returns correct Detection objects."""
        from src.perception import ObjectDetector

        with patch('src.perception.YOLO') as mock_yolo_class:
            mock_model = Mock()
            mock_model.names = {0: "person", 2: "car", 16: "dog"}
            mock_model.predict.return_value = mock_yolo_results
            mock_yolo_class.return_value = mock_model

            detector = ObjectDetector(sample_config)

            # Create dummy frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Run detection
            detections = detector.detect_frame(frame)

            # Should return 2 detections (person and car are obstacles, dog is not)
            assert len(detections) == 2

            # Verify first detection (person)
            assert detections[0].class_name == "person"
            assert detections[0].x1 == 100.0
            assert detections[0].y1 == 150.0
            assert detections[0].x2 == 200.0
            assert detections[0].y2 == 250.0
            assert detections[0].confidence == 0.85

            # Verify second detection (car)
            assert detections[1].class_name == "car"
            assert detections[1].x1 == 300.0
            assert detections[1].confidence == 0.92

    def test_detect_frame_empty_results(self, sample_config):
        """Test that detect_frame handles frames with no detections."""
        from src.perception import ObjectDetector

        with patch('src.perception.YOLO') as mock_yolo_class:
            mock_model = Mock()
            mock_model.names = {0: "person"}

            # Mock empty results
            mock_result = Mock()
            mock_result.boxes = None  # No detections
            mock_model.predict.return_value = [mock_result]

            mock_yolo_class.return_value = mock_model

            detector = ObjectDetector(sample_config)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            detections = detector.detect_frame(frame)

            # Should return empty list
            assert detections == []

    def test_detect_frame_invalid_bounding_boxes(self, sample_config):
        """Test that invalid bounding boxes are filtered out."""
        from src.perception import ObjectDetector

        with patch('src.perception.YOLO') as mock_yolo_class:
            mock_model = Mock()
            mock_model.names = {0: "person"}

            # Create mock results with invalid boxes
            mock_result = Mock()
            mock_boxes = Mock()

            # Invalid: x1 >= x2, y1 >= y2, negative coords
            xyxy = np.array([
                [200.0, 150.0, 100.0, 250.0],  # x1 > x2 (invalid)
                [100.0, 250.0, 200.0, 150.0],  # y1 > y2 (invalid)
                [-10.0, 150.0, 200.0, 250.0],  # negative x1 (invalid)
                [100.0, 150.0, 200.0, 250.0],  # valid
            ])
            conf = np.array([0.85, 0.90, 0.80, 0.95])
            cls = np.array([0, 0, 0, 0], dtype=int)

            mock_boxes.xyxy = Mock()
            mock_boxes.xyxy.cpu.return_value.numpy.return_value = xyxy
            mock_boxes.conf = Mock()
            mock_boxes.conf.cpu.return_value.numpy.return_value = conf
            mock_boxes.cls = Mock()
            mock_boxes.cls.cpu.return_value.numpy.return_value = cls
            mock_boxes.__len__ = Mock(return_value=4)

            mock_result.boxes = mock_boxes
            mock_model.predict.return_value = [mock_result]
            mock_yolo_class.return_value = mock_model

            detector = ObjectDetector(sample_config)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            detections = detector.detect_frame(frame)

            # Should only return the valid detection
            assert len(detections) == 1
            assert detections[0].x1 == 100.0
            assert detections[0].confidence == 0.95

    def test_detect_frame_error_handling(self, sample_config):
        """Test that detect_frame returns empty list on errors."""
        from src.perception import ObjectDetector

        with patch('src.perception.YOLO') as mock_yolo_class:
            mock_model = Mock()
            mock_model.predict.side_effect = Exception("CUDA out of memory")
            mock_yolo_class.return_value = mock_model

            detector = ObjectDetector(sample_config)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            detections = detector.detect_frame(frame)

            # Should return empty list, not raise exception
            assert detections == []

    def test_filter_obstacle_classes(self, sample_config):
        """Test that filtering keeps only obstacle classes."""
        from src.perception import ObjectDetector

        with patch('src.perception.YOLO') as mock_yolo_class:
            mock_model = Mock()
            # person and car are obstacles (in config), dog is not
            mock_model.names = {0: "person", 2: "car", 16: "dog", 1: "bicycle"}

            # Create mock results with mixed classes
            mock_result = Mock()
            mock_boxes = Mock()

            xyxy = np.array([
                [100.0, 150.0, 200.0, 250.0],  # person (obstacle)
                [300.0, 100.0, 450.0, 300.0],  # car (obstacle)
                [500.0, 200.0, 600.0, 350.0],  # dog (not obstacle)
                [50.0, 50.0, 150.0, 150.0],    # bicycle (obstacle)
            ])
            conf = np.array([0.85, 0.92, 0.75, 0.88])
            cls = np.array([0, 2, 16, 1], dtype=int)

            mock_boxes.xyxy = Mock()
            mock_boxes.xyxy.cpu.return_value.numpy.return_value = xyxy
            mock_boxes.conf = Mock()
            mock_boxes.conf.cpu.return_value.numpy.return_value = conf
            mock_boxes.cls = Mock()
            mock_boxes.cls.cpu.return_value.numpy.return_value = cls
            mock_boxes.__len__ = Mock(return_value=4)

            mock_result.boxes = mock_boxes
            mock_model.predict.return_value = [mock_result]
            mock_yolo_class.return_value = mock_model

            detector = ObjectDetector(sample_config)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            detections = detector.detect_frame(frame)

            # Should return 3 detections (person, car, bicycle), not dog
            assert len(detections) == 3
            class_names = [d.class_name for d in detections]
            assert "person" in class_names
            assert "car" in class_names
            assert "bicycle" in class_names
            assert "dog" not in class_names
