"""Tests for synthetic scene generation module."""

import numpy as np
import pytest
from src.synthetic_scene import (
    SyntheticObstacle,
    SyntheticScene,
    create_gauntlet_scenario,
    create_crossing_scenario,
    create_converging_scenario,
)
from src.models import Detection


class TestSyntheticObstacle:
    """Tests for SyntheticObstacle class."""

    def test_init(self):
        """Test obstacle initialization."""
        obs = SyntheticObstacle(
            x=100.0, y=100.0, vx=2.0, vy=1.0,
            width=50, height=30,
            color=(255, 0, 0),
            label="test",
            class_id=5
        )

        assert obs.x == 100.0
        assert obs.y == 100.0
        assert obs.vx == 2.0
        assert obs.vy == 1.0
        assert obs.width == 50
        assert obs.height == 30
        assert obs.color == (255, 0, 0)
        assert obs.label == "test"
        assert obs.class_id == 5

    def test_update_moves_obstacle(self):
        """Test that update moves the obstacle by velocity."""
        obs = SyntheticObstacle(
            x=100.0, y=100.0, vx=2.0, vy=3.0,
            width=50, height=30,
            color=(255, 0, 0),
            label="test"
        )

        obs.update(frame_width=1280, frame_height=720)

        assert obs.x == 102.0
        assert obs.y == 103.0

    def test_update_bounces_right_boundary(self):
        """Test obstacle bounces off right boundary."""
        obs = SyntheticObstacle(
            x=1275.0, y=100.0, vx=10.0, vy=0.0,
            width=50, height=30,
            color=(255, 0, 0),
            label="test"
        )

        obs.update(frame_width=1280, frame_height=720)

        # Should bounce and reverse x velocity
        assert obs.x <= 1280
        assert obs.vx < 0

    def test_update_bounces_left_boundary(self):
        """Test obstacle bounces off left boundary."""
        obs = SyntheticObstacle(
            x=5.0, y=100.0, vx=-10.0, vy=0.0,
            width=50, height=30,
            color=(255, 0, 0),
            label="test"
        )

        obs.update(frame_width=1280, frame_height=720)

        # Should bounce and reverse x velocity
        assert obs.x >= 0
        assert obs.vx > 0

    def test_update_bounces_top_boundary(self):
        """Test obstacle bounces off top boundary."""
        obs = SyntheticObstacle(
            x=100.0, y=5.0, vx=0.0, vy=-10.0,
            width=50, height=30,
            color=(255, 0, 0),
            label="test"
        )

        obs.update(frame_width=1280, frame_height=720)

        # Should bounce and reverse y velocity
        assert obs.y >= 0
        assert obs.vy > 0

    def test_update_bounces_bottom_boundary(self):
        """Test obstacle bounces off bottom boundary."""
        obs = SyntheticObstacle(
            x=100.0, y=715.0, vx=0.0, vy=10.0,
            width=50, height=30,
            color=(255, 0, 0),
            label="test"
        )

        obs.update(frame_width=1280, frame_height=720)

        # Should bounce and reverse y velocity
        assert obs.y <= 720
        assert obs.vy < 0

    def test_draw_modifies_frame(self):
        """Test that draw modifies the frame."""
        obs = SyntheticObstacle(
            x=100.0, y=100.0, vx=0.0, vy=0.0,
            width=50, height=30,
            color=(255, 0, 0),
            label="test"
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original_sum = frame.sum()

        obs.draw(frame)

        # Frame should have changed (pixels added)
        assert frame.sum() > original_sum

    def test_to_detection_correct_values(self):
        """Test conversion to Detection dataclass."""
        obs = SyntheticObstacle(
            x=100.0, y=100.0, vx=0.0, vy=0.0,
            width=50, height=30,
            color=(255, 0, 0),
            label="car",
            class_id=2
        )

        detection = obs.to_detection()

        assert isinstance(detection, Detection)
        assert detection.x1 == 100.0
        assert detection.y1 == 100.0
        assert detection.x2 == 150.0
        assert detection.y2 == 130.0
        assert detection.class_id == 2
        assert detection.class_name == "car"
        assert detection.confidence == 1.0

    def test_to_detection_center_property(self):
        """Test that Detection center property is correct."""
        obs = SyntheticObstacle(
            x=100.0, y=100.0, vx=0.0, vy=0.0,
            width=50, height=30,
            color=(255, 0, 0),
            label="car"
        )

        detection = obs.to_detection()
        center_x, center_y = detection.center

        assert center_x == 125.0  # (100 + 150) / 2
        assert center_y == 115.0  # (100 + 130) / 2


class TestSyntheticScene:
    """Tests for SyntheticScene class."""

    def test_init_empty_obstacles(self):
        """Test scene initialization with no obstacles."""
        waypoints = [[100, 100], [200, 200]]
        scene = SyntheticScene(
            width=640,
            height=480,
            obstacles=[],
            waypoints=waypoints
        )

        assert scene.width == 640
        assert scene.height == 480
        assert len(scene.obstacles) == 0
        assert len(scene.waypoints) == 2

    def test_init_with_obstacles(self):
        """Test scene initialization with obstacles."""
        obstacles = [
            SyntheticObstacle(
                x=100.0, y=100.0, vx=1.0, vy=1.0,
                width=50, height=30,
                color=(255, 0, 0),
                label="car"
            )
        ]
        waypoints = [[100, 100]]

        scene = SyntheticScene(
            width=640,
            height=480,
            obstacles=obstacles,
            waypoints=waypoints
        )

        assert len(scene.obstacles) == 1
        assert scene.obstacles[0].x == 100.0

    def test_update_advances_all_obstacles(self):
        """Test that update advances all obstacles."""
        obstacles = [
            SyntheticObstacle(
                x=100.0, y=100.0, vx=2.0, vy=0.0,
                width=50, height=30,
                color=(255, 0, 0),
                label="car1"
            ),
            SyntheticObstacle(
                x=200.0, y=200.0, vx=0.0, vy=3.0,
                width=50, height=30,
                color=(0, 255, 0),
                label="car2"
            ),
        ]
        waypoints = [[100, 100]]

        scene = SyntheticScene(
            width=1280,
            height=720,
            obstacles=obstacles,
            waypoints=waypoints
        )

        scene.update()

        # Check both obstacles moved
        assert scene.obstacles[0].x == 102.0
        assert scene.obstacles[0].y == 100.0
        assert scene.obstacles[1].x == 200.0
        assert scene.obstacles[1].y == 203.0

    def test_render_frame_returns_correct_shape(self):
        """Test that render_frame returns correctly sized frame."""
        waypoints = [[100, 100]]
        scene = SyntheticScene(
            width=640,
            height=480,
            obstacles=[],
            waypoints=waypoints
        )

        frame, detections = scene.render_frame()

        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8

    def test_render_frame_returns_detections(self):
        """Test that render_frame returns detection list."""
        obstacles = [
            SyntheticObstacle(
                x=100.0, y=100.0, vx=0.0, vy=0.0,
                width=50, height=30,
                color=(255, 0, 0),
                label="car"
            ),
            SyntheticObstacle(
                x=200.0, y=200.0, vx=0.0, vy=0.0,
                width=40, height=40,
                color=(0, 255, 0),
                label="person"
            ),
        ]
        waypoints = [[100, 100]]

        scene = SyntheticScene(
            width=640,
            height=480,
            obstacles=obstacles,
            waypoints=waypoints
        )

        frame, detections = scene.render_frame()

        assert len(detections) == 2
        assert all(isinstance(d, Detection) for d in detections)

    def test_render_frame_updates_obstacles(self):
        """Test that render_frame calls update on obstacles."""
        obstacles = [
            SyntheticObstacle(
                x=100.0, y=100.0, vx=2.0, vy=0.0,
                width=50, height=30,
                color=(255, 0, 0),
                label="car"
            )
        ]
        waypoints = [[100, 100]]

        scene = SyntheticScene(
            width=1280,
            height=720,
            obstacles=obstacles,
            waypoints=waypoints
        )

        initial_x = scene.obstacles[0].x
        frame, detections = scene.render_frame()

        # Obstacle should have moved
        assert scene.obstacles[0].x > initial_x


class TestScenarioFactories:
    """Tests for scenario factory functions."""

    def test_create_gauntlet_scenario_returns_obstacles(self):
        """Test gauntlet scenario creates obstacles."""
        obstacles = create_gauntlet_scenario(width=1280, height=720)

        assert len(obstacles) > 0
        assert all(isinstance(obs, SyntheticObstacle) for obs in obstacles)

    def test_create_crossing_scenario_returns_obstacles(self):
        """Test crossing scenario creates obstacles."""
        obstacles = create_crossing_scenario(width=1280, height=720)

        assert len(obstacles) > 0
        assert all(isinstance(obs, SyntheticObstacle) for obs in obstacles)

    def test_create_converging_scenario_returns_obstacles(self):
        """Test converging scenario creates obstacles."""
        obstacles = create_converging_scenario(width=1280, height=720)

        assert len(obstacles) > 0
        assert all(isinstance(obs, SyntheticObstacle) for obs in obstacles)

    def test_gauntlet_obstacles_form_corridor(self):
        """Test gauntlet obstacles are positioned to form a corridor."""
        obstacles = create_gauntlet_scenario(width=1280, height=720)

        # Should have left and right obstacles
        left_obs = [o for o in obstacles if o.x < 640]
        right_obs = [o for o in obstacles if o.x >= 640]

        assert len(left_obs) > 0
        assert len(right_obs) > 0

    def test_crossing_obstacles_have_movement(self):
        """Test crossing obstacles have non-zero velocity."""
        obstacles = create_crossing_scenario(width=1280, height=720)

        # At least some obstacles should be moving
        moving_obs = [o for o in obstacles if o.vx != 0 or o.vy != 0]
        assert len(moving_obs) > 0

    def test_converging_obstacles_move_toward_center(self):
        """Test converging obstacles have velocity toward center."""
        obstacles = create_converging_scenario(width=1280, height=720)

        center_x, center_y = 640, 360

        # Check that at least some obstacles are moving toward center
        converging_count = 0
        for obs in obstacles:
            dx_to_center = center_x - obs.x
            dy_to_center = center_y - obs.y

            # Check if velocity has component toward center
            if (dx_to_center * obs.vx > 0) or (dy_to_center * obs.vy > 0):
                converging_count += 1

        assert converging_count > 0
