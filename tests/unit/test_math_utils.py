"""Tests for mathematical utility functions."""

import math

import pytest

from src.utils.math_utils import (
    normalize_angle,
    angle_difference,
    heading_to_vector,
    vector_to_heading,
    distance,
    clamp,
    lerp_angle,
)


class TestNormalizeAngle:
    """Tests for normalize_angle function."""

    def test_zero(self) -> None:
        assert normalize_angle(0.0) == pytest.approx(0.0)

    def test_pi(self) -> None:
        # pi is the boundary — should map to pi (or -pi, both valid)
        result = normalize_angle(math.pi)
        assert result == pytest.approx(math.pi) or result == pytest.approx(-math.pi)

    def test_negative_pi(self) -> None:
        result = normalize_angle(-math.pi)
        assert result == pytest.approx(math.pi) or result == pytest.approx(-math.pi)

    def test_two_pi(self) -> None:
        assert normalize_angle(2 * math.pi) == pytest.approx(0.0, abs=1e-10)

    def test_negative_two_pi(self) -> None:
        assert normalize_angle(-2 * math.pi) == pytest.approx(0.0, abs=1e-10)

    def test_three_pi(self) -> None:
        result = normalize_angle(3 * math.pi)
        assert result == pytest.approx(math.pi) or result == pytest.approx(-math.pi)

    def test_positive_quarter_pi(self) -> None:
        assert normalize_angle(math.pi / 4) == pytest.approx(math.pi / 4)

    def test_large_positive(self) -> None:
        # 5pi should normalize to pi
        result = normalize_angle(5 * math.pi)
        assert result == pytest.approx(math.pi) or result == pytest.approx(-math.pi)

    def test_large_negative(self) -> None:
        # -3pi/2 = pi/2
        assert normalize_angle(-3 * math.pi / 2) == pytest.approx(math.pi / 2)

    def test_small_positive(self) -> None:
        assert normalize_angle(0.1) == pytest.approx(0.1)

    def test_small_negative(self) -> None:
        assert normalize_angle(-0.1) == pytest.approx(-0.1)


class TestAngleDifference:
    """Tests for angle_difference function."""

    def test_same_angle(self) -> None:
        assert angle_difference(1.0, 1.0) == pytest.approx(0.0)

    def test_positive_difference(self) -> None:
        # pi/2 - 0 = pi/2 (counter-clockwise)
        assert angle_difference(math.pi / 2, 0.0) == pytest.approx(math.pi / 2)

    def test_negative_difference(self) -> None:
        # 0 - pi/2 = -pi/2 (clockwise)
        assert angle_difference(0.0, math.pi / 2) == pytest.approx(-math.pi / 2)

    def test_wraparound_positive(self) -> None:
        # From -3pi/4 to 3pi/4 — shortest path is pi/2 backward (negative)
        result = angle_difference(3 * math.pi / 4, -3 * math.pi / 4)
        # 3pi/4 - (-3pi/4) = 6pi/4 = 3pi/2, normalized = -pi/2
        assert result == pytest.approx(-math.pi / 2)

    def test_wraparound_negative(self) -> None:
        # From 3pi/4 to -3pi/4 — shortest path is pi/2 forward (positive)
        result = angle_difference(-3 * math.pi / 4, 3 * math.pi / 4)
        assert result == pytest.approx(math.pi / 2)

    def test_opposite_directions(self) -> None:
        # pi difference — could be either direction
        result = angle_difference(math.pi, 0.0)
        assert abs(result) == pytest.approx(math.pi)


class TestHeadingToVector:
    """Tests for heading_to_vector function."""

    def test_east(self) -> None:
        dx, dy = heading_to_vector(0.0)
        assert dx == pytest.approx(1.0)
        assert dy == pytest.approx(0.0, abs=1e-10)

    def test_north(self) -> None:
        dx, dy = heading_to_vector(math.pi / 2)
        assert dx == pytest.approx(0.0, abs=1e-10)
        assert dy == pytest.approx(1.0)

    def test_west(self) -> None:
        dx, dy = heading_to_vector(math.pi)
        assert dx == pytest.approx(-1.0)
        assert dy == pytest.approx(0.0, abs=1e-10)

    def test_south(self) -> None:
        dx, dy = heading_to_vector(-math.pi / 2)
        assert dx == pytest.approx(0.0, abs=1e-10)
        assert dy == pytest.approx(-1.0)

    def test_unit_magnitude(self) -> None:
        dx, dy = heading_to_vector(0.7)
        mag = math.sqrt(dx * dx + dy * dy)
        assert mag == pytest.approx(1.0)


class TestVectorToHeading:
    """Tests for vector_to_heading function."""

    def test_east(self) -> None:
        assert vector_to_heading(1.0, 0.0) == pytest.approx(0.0)

    def test_north(self) -> None:
        assert vector_to_heading(0.0, 1.0) == pytest.approx(math.pi / 2)

    def test_west(self) -> None:
        assert vector_to_heading(-1.0, 0.0) == pytest.approx(math.pi)

    def test_south(self) -> None:
        assert vector_to_heading(0.0, -1.0) == pytest.approx(-math.pi / 2)

    def test_northeast(self) -> None:
        assert vector_to_heading(1.0, 1.0) == pytest.approx(math.pi / 4)

    def test_roundtrip(self) -> None:
        heading = 1.23
        dx, dy = heading_to_vector(heading)
        assert vector_to_heading(dx, dy) == pytest.approx(heading)


class TestDistance:
    """Tests for distance function."""

    def test_same_point(self) -> None:
        assert distance((0.0, 0.0), (0.0, 0.0)) == pytest.approx(0.0)

    def test_horizontal(self) -> None:
        assert distance((0.0, 0.0), (3.0, 0.0)) == pytest.approx(3.0)

    def test_vertical(self) -> None:
        assert distance((0.0, 0.0), (0.0, 4.0)) == pytest.approx(4.0)

    def test_diagonal(self) -> None:
        assert distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)

    def test_negative_coords(self) -> None:
        assert distance((-1.0, -1.0), (2.0, 3.0)) == pytest.approx(5.0)

    def test_symmetry(self) -> None:
        p1 = (1.0, 2.0)
        p2 = (4.0, 6.0)
        assert distance(p1, p2) == pytest.approx(distance(p2, p1))


class TestClamp:
    """Tests for clamp function."""

    def test_within_range(self) -> None:
        assert clamp(5.0, 0.0, 10.0) == pytest.approx(5.0)

    def test_below_min(self) -> None:
        assert clamp(-5.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_above_max(self) -> None:
        assert clamp(15.0, 0.0, 10.0) == pytest.approx(10.0)

    def test_at_min(self) -> None:
        assert clamp(0.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_at_max(self) -> None:
        assert clamp(10.0, 0.0, 10.0) == pytest.approx(10.0)

    def test_negative_range(self) -> None:
        assert clamp(0.0, -5.0, -1.0) == pytest.approx(-1.0)


class TestLerpAngle:
    """Tests for lerp_angle function."""

    def test_t_zero(self) -> None:
        assert lerp_angle(0.0, math.pi / 2, 0.0) == pytest.approx(0.0)

    def test_t_one(self) -> None:
        assert lerp_angle(0.0, math.pi / 2, 1.0) == pytest.approx(math.pi / 2)

    def test_t_half(self) -> None:
        assert lerp_angle(0.0, math.pi / 2, 0.5) == pytest.approx(math.pi / 4)

    def test_wraparound_shortest_arc(self) -> None:
        # From 3pi/4 to -3pi/4 (should go through pi, not through 0)
        result = lerp_angle(3 * math.pi / 4, -3 * math.pi / 4, 0.5)
        assert result == pytest.approx(math.pi) or result == pytest.approx(-math.pi)

    def test_wraparound_other_direction(self) -> None:
        # From -pi/4 to pi/4 — should go through 0
        result = lerp_angle(-math.pi / 4, math.pi / 4, 0.5)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_same_angle(self) -> None:
        assert lerp_angle(1.0, 1.0, 0.5) == pytest.approx(1.0)
