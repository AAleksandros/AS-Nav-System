"""Mathematical utility functions for the navigation system.

Provides core geometric and trigonometric functions used throughout the
autonomous navigation pipeline. All angles are in radians, using standard
math convention (y-up, counter-clockwise positive).
"""

import math
from typing import Tuple


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi].

    Parameters
    ----------
    angle : float
        Angle in radians.

    Returns
    -------
    float
        Normalized angle in [-pi, pi].
    """
    result = math.fmod(angle + math.pi, 2 * math.pi)
    if result < 0:
        result += 2 * math.pi
    return result - math.pi


def angle_difference(target: float, current: float) -> float:
    """Compute shortest signed angular difference from current to target.

    Parameters
    ----------
    target : float
        Target angle in radians.
    current : float
        Current angle in radians.

    Returns
    -------
    float
        Signed difference in [-pi, pi]. Positive means counter-clockwise.
    """
    return normalize_angle(target - current)


def heading_to_vector(heading: float) -> Tuple[float, float]:
    """Convert heading angle to unit direction vector.

    Parameters
    ----------
    heading : float
        Heading in radians (0 = east, pi/2 = north in y-up).

    Returns
    -------
    Tuple[float, float]
        Unit vector (dx, dy).
    """
    return (math.cos(heading), math.sin(heading))


def vector_to_heading(dx: float, dy: float) -> float:
    """Convert direction vector to heading angle.

    Parameters
    ----------
    dx : float
        X component of direction.
    dy : float
        Y component of direction.

    Returns
    -------
    float
        Heading in radians [-pi, pi].
    """
    return math.atan2(dy, dx)


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two points.

    Parameters
    ----------
    p1 : Tuple[float, float]
        First point (x, y).
    p2 : Tuple[float, float]
        Second point (x, y).

    Returns
    -------
    float
        Distance between points.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to [min_val, max_val].

    Parameters
    ----------
    value : float
        Value to clamp.
    min_val : float
        Minimum bound.
    max_val : float
        Maximum bound.

    Returns
    -------
    float
        Clamped value.
    """
    return max(min_val, min(max_val, value))


def lerp_angle(a1: float, a2: float, t: float) -> float:
    """Linearly interpolate between two angles along the shortest arc.

    Parameters
    ----------
    a1 : float
        Start angle in radians.
    a2 : float
        End angle in radians.
    t : float
        Interpolation parameter [0, 1]. 0 returns a1, 1 returns a2.

    Returns
    -------
    float
        Interpolated angle in [-pi, pi].
    """
    diff = angle_difference(a2, a1)
    return normalize_angle(a1 + t * diff)
