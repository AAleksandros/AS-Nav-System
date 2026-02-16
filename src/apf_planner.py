"""Artificial Potential Fields planner for 2D navigation.

Implements the Khatib (1986) potential field approach with attractive forces
toward the goal and repulsive forces away from obstacles. Includes a local
minima escape heuristic based on position history.
"""

import logging
import math
import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from src.config import Config
from src.models import ForceVector, SensorReading, State

logger = logging.getLogger(__name__)

# Minimum distance to avoid division by zero in repulsive force
_MIN_DIST = 1.0


class APFPlanner:
    """Artificial Potential Fields navigation planner.

    Computes attractive force toward the goal and repulsive forces away from
    detected obstacles. Classifies the navigation state and handles local
    minima via random escape perturbation.

    Parameters
    ----------
    k_att : float
        Attractive force gain.
    k_rep : float
        Repulsive force gain.
    influence_range : float
        Maximum distance at which obstacles exert repulsive force.
    goal_tolerance : float
        Distance within which the goal is considered reached.
    max_speed : float
        Maximum magnitude of the total force vector.
    escape_threshold : float
        Position change threshold below which the agent is considered stuck.
    escape_window : int
        Number of recent positions to track for stuck detection.
    escape_force : float
        Magnitude of the random escape perturbation.
    escape_seed : Optional[int]
        Random seed for escape direction reproducibility.
    force_smoothing : float
        EMA alpha for temporal force smoothing (0 = full previous, 1 = no smoothing).
    symmetry_threshold : float
        Net/sum magnitude ratio below which symmetry-breaking nudge activates.
    symmetry_nudge_force : float
        Magnitude of the deterministic symmetry-breaking force.
    """

    def __init__(
        self,
        k_att: float,
        k_rep: float,
        influence_range: float,
        goal_tolerance: float,
        max_speed: float,
        escape_threshold: float,
        escape_window: int,
        escape_force: float,
        escape_seed: Optional[int] = None,
        slow_down_distance: float = 60.0,
        cruise_fraction: float = 0.5,
        vortex_weight: float = 0.0,
        force_smoothing: float = 0.3,
        symmetry_threshold: float = 0.3,
        symmetry_nudge_force: float = 15.0,
    ) -> None:
        self.k_att = k_att
        self.k_rep = k_rep
        self.influence_range = influence_range
        self.goal_tolerance = goal_tolerance
        self.max_speed = max_speed
        self.escape_threshold = escape_threshold
        self.escape_window = escape_window
        self.escape_force = escape_force
        self.slow_down_distance = slow_down_distance
        self.cruise_fraction = cruise_fraction
        self.vortex_weight = max(0.0, min(1.0, vortex_weight))
        self.force_smoothing = max(0.0, min(1.0, force_smoothing))
        self.symmetry_threshold = symmetry_threshold
        self.symmetry_nudge_force = symmetry_nudge_force
        self._position_history: Deque[tuple] = deque(maxlen=escape_window)
        self._rng = random.Random(escape_seed)
        self._prev_force_x: float = 0.0
        self._prev_force_y: float = 0.0
        self._has_prev_force: bool = False

        logger.info(
            "APFPlanner: k_att=%.1f, k_rep=%.1f, influence=%.1f, max_speed=%.1f",
            k_att,
            k_rep,
            influence_range,
            max_speed,
        )

    @classmethod
    def from_config(cls, config: Config) -> "APFPlanner":
        """Create planner from configuration.

        Parameters
        ----------
        config : Config
            System configuration with planner section.

        Returns
        -------
        APFPlanner
            Configured planner instance.
        """
        p = config.planner  # type: ignore[attr-defined]
        return cls(
            k_att=float(p.k_att),  # type: ignore[attr-defined]
            k_rep=float(p.k_rep),  # type: ignore[attr-defined]
            influence_range=float(p.influence_range),  # type: ignore[attr-defined]
            goal_tolerance=float(p.goal_tolerance),  # type: ignore[attr-defined]
            max_speed=float(p.max_speed),  # type: ignore[attr-defined]
            escape_threshold=float(p.escape_threshold),  # type: ignore[attr-defined]
            escape_window=int(p.escape_window),  # type: ignore[attr-defined]
            escape_force=float(p.escape_force),  # type: ignore[attr-defined]
            slow_down_distance=float(  # type: ignore
                getattr(p, 'slow_down_distance', 60.0)),
            cruise_fraction=float(  # type: ignore
                getattr(p, 'cruise_fraction', 0.5)),
            vortex_weight=float(  # type: ignore
                getattr(p, 'vortex_weight', 0.0)),
            force_smoothing=float(  # type: ignore
                getattr(p, 'force_smoothing', 0.3)),
            symmetry_threshold=float(  # type: ignore
                getattr(p, 'symmetry_threshold', 0.3)),
            symmetry_nudge_force=float(  # type: ignore
                getattr(p, 'symmetry_nudge_force', 15.0)),
        )

    def attractive_force(
        self, ax: float, ay: float, gx: float, gy: float
    ) -> ForceVector:
        """Compute attractive force toward the goal.

        Returns a unit vector from agent to goal scaled by k_att.
        Returns zero force when agent is at the goal.

        Parameters
        ----------
        ax, ay : float
            Agent position.
        gx, gy : float
            Goal position.

        Returns
        -------
        ForceVector
            Attractive force vector.
        """
        dx = gx - ax
        dy = gy - ay
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1e-10:
            return ForceVector(fx=0.0, fy=0.0, source="attractive")

        return ForceVector(
            fx=self.k_att * dx / dist,
            fy=self.k_att * dy / dist,
            source="attractive",
        )

    def repulsive_force(
        self,
        ax: float,
        ay: float,
        reading: SensorReading,
        gx: float = 0.0,
        gy: float = 0.0,
    ) -> ForceVector:
        """Compute repulsive force from a single sensor reading.

        Uses linear repulsive field with optional vortex component:
        F_rep = k_rep * (1 - d/d0) * direction
        When vortex_weight > 0, blends radial with tangential component
        rotated toward the goal side.

        Parameters
        ----------
        ax, ay : float
            Agent position.
        reading : SensorReading
            Sensor measurement.
        gx, gy : float
            Goal position (used for vortex direction selection).

        Returns
        -------
        ForceVector
            Repulsive force vector (zero if no hit or beyond influence range).
        """
        if not reading.hit or reading.hit_point is None:
            return ForceVector(fx=0.0, fy=0.0, source="repulsive")

        d = reading.distance
        d0 = self.influence_range

        if d >= d0:
            return ForceVector(fx=0.0, fy=0.0, source="repulsive")

        # Clamp distance to avoid division by zero
        d = max(d, _MIN_DIST)

        # Direction away from obstacle (from hit_point toward agent)
        hx, hy = reading.hit_point
        away_x = ax - hx
        away_y = ay - hy
        away_dist = math.sqrt(away_x * away_x + away_y * away_y)

        if away_dist < 1e-10:
            # Agent is at the hit point; push in opposite direction of ray
            away_x = -math.cos(reading.angle)
            away_y = -math.sin(reading.angle)
            away_dist = 1.0

        # Normalize direction
        away_x /= away_dist
        away_y /= away_dist

        # Linear repulsive force magnitude
        magnitude = self.k_rep * (1.0 - d / d0)

        # Apply vortex field if enabled
        if self.vortex_weight > 0.0:
            # Two tangential candidates: CW and CCW 90-degree rotations
            tan_cw_x, tan_cw_y = away_y, -away_x
            tan_ccw_x, tan_ccw_y = -away_y, away_x

            # Goal direction from agent
            goal_dx = gx - ax
            goal_dy = gy - ay

            # Pick rotation more aligned with goal direction (dot product)
            dot_cw = tan_cw_x * goal_dx + tan_cw_y * goal_dy
            dot_ccw = tan_ccw_x * goal_dx + tan_ccw_y * goal_dy

            if dot_cw >= dot_ccw:
                tan_x, tan_y = tan_cw_x, tan_cw_y
            else:
                tan_x, tan_y = tan_ccw_x, tan_ccw_y

            # Blend radial and tangential
            vw = self.vortex_weight
            fx = magnitude * ((1.0 - vw) * away_x + vw * tan_x)
            fy = magnitude * ((1.0 - vw) * away_y + vw * tan_y)
        else:
            fx = magnitude * away_x
            fy = magnitude * away_y

        return ForceVector(fx=fx, fy=fy, source="repulsive")

    def compute(
        self,
        ax: float,
        ay: float,
        gx: float,
        gy: float,
        sensor_readings: List[SensorReading],
    ) -> Dict[str, Any]:
        """Compute total force, navigation state, and individual forces.

        Parameters
        ----------
        ax, ay : float
            Agent position.
        gx, gy : float
            Goal position.
        sensor_readings : List[SensorReading]
            Current sensor scan results.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - ``total_force``: Combined ForceVector
            - ``state``: Navigation State
            - ``forces``: List of individual ForceVector components
        """
        forces: List[ForceVector] = []

        # Attractive force
        f_att = self.attractive_force(ax, ay, gx, gy)
        forces.append(f_att)

        # Repulsive forces from each sensor reading
        for reading in sensor_readings:
            f_rep = self.repulsive_force(ax, ay, reading, gx, gy)
            if f_rep.magnitude > 1e-10:
                forces.append(f_rep)

        # Compute min obstacle distance (used by symmetry gate and adaptive smoothing)
        min_obs_dist = float("inf")
        for r in sensor_readings:
            if r.hit and r.distance < min_obs_dist:
                min_obs_dist = r.distance

        # Symmetry breaking: detect when opposing repulsive forces cancel
        rep_forces = [f for f in forces if f.source == "repulsive"]
        if rep_forces and self.symmetry_nudge_force > 0.0:
            sum_individual = sum(f.magnitude for f in rep_forces)
            net_rx = sum(f.fx for f in rep_forces)
            net_ry = sum(f.fy for f in rep_forces)
            net_mag = math.sqrt(net_rx * net_rx + net_ry * net_ry)
            ratio = net_mag / sum_individual
            if ratio < self.symmetry_threshold:
                # Gate A: total force is weak relative to attraction
                raw_fx = sum(f.fx for f in forces)
                raw_fy = sum(f.fy for f in forces)
                raw_mag = math.sqrt(raw_fx * raw_fx + raw_fy * raw_fy)
                att_mag = f_att.magnitude
                force_is_weak = att_mag < 1e-10 or (raw_mag / att_mag) < 0.5

                # Gate B: nearest obstacle is within slow-down zone
                obstacle_is_near = min_obs_dist <= self.slow_down_distance

                if force_is_weak and obstacle_is_near:
                    # CW perpendicular to goal direction
                    goal_dx = gx - ax
                    goal_dy = gy - ay
                    goal_dist = math.sqrt(goal_dx * goal_dx + goal_dy * goal_dy)
                    if goal_dist > 1e-10:
                        nudge_x = (goal_dy / goal_dist) * self.symmetry_nudge_force
                        nudge_y = (-goal_dx / goal_dist) * self.symmetry_nudge_force
                        forces.append(ForceVector(
                            fx=nudge_x, fy=nudge_y, source="symmetry_break",
                        ))

        # Record position and check for local minima
        self._record_position(ax, ay)
        if self._is_stuck():
            f_escape = self._escape_perturbation()
            forces.append(f_escape)

        # Sum forces
        total_fx = sum(f.fx for f in forces)
        total_fy = sum(f.fy for f in forces)

        # Adaptive EMA: stronger smoothing when closer to obstacles
        alpha = self.force_smoothing
        if min_obs_dist < self.slow_down_distance:
            proximity = max(min_obs_dist, 0.0) / self.slow_down_distance
            alpha = self.force_smoothing * proximity
            alpha = max(alpha, 0.05)  # floor: never fully frozen

        if self._has_prev_force:
            total_fx = alpha * total_fx + (1.0 - alpha) * self._prev_force_x
            total_fy = alpha * total_fy + (1.0 - alpha) * self._prev_force_y
        self._prev_force_x = total_fx
        self._prev_force_y = total_fy
        self._has_prev_force = True

        # Cap magnitude
        magnitude = math.sqrt(total_fx * total_fx + total_fy * total_fy)
        if magnitude > self.max_speed:
            scale = self.max_speed / magnitude
            total_fx *= scale
            total_fy *= scale

        total = ForceVector(fx=total_fx, fy=total_fy, source="total")

        # Classify state
        state = self._classify_state(sensor_readings)

        return {
            "total_force": total,
            "state": state,
            "forces": forces,
        }

    def compute_speed(
        self, min_obstacle_distance: float, state: "State"
    ) -> float:
        """Compute desired speed based on obstacle proximity.

        Decouples speed from force magnitude. Uses cruise_fraction of
        max_speed as baseline, linearly reducing when obstacles are
        within slow_down_distance. State is accepted for interface
        compatibility but does not affect speed — collision response
        prevents actual overlap.

        Parameters
        ----------
        min_obstacle_distance : float
            Distance to nearest detected obstacle (inf if none).
        state : State
            Current navigation state (unused, kept for interface).

        Returns
        -------
        float
            Desired speed in units/s (always > 0).
        """
        base_speed = self.max_speed * self.cruise_fraction

        if min_obstacle_distance < self.slow_down_distance:
            ratio = max(min_obstacle_distance, 0.0) / self.slow_down_distance
            base_speed = base_speed * math.sqrt(ratio)

        # Never fully zero — ensure enough speed to maneuver
        return max(base_speed, 15.0)

    def _classify_state(self, readings: List[SensorReading]) -> State:
        """Classify navigation state based on sensor readings.

        Parameters
        ----------
        readings : List[SensorReading]
            Current sensor scan results.

        Returns
        -------
        State
            STOP if any obstacle within critical distance (10% of influence range),
            AVOID if any obstacle within influence range,
            NAVIGATE otherwise.
        """
        critical_dist = self.influence_range * 0.1
        min_dist = float("inf")

        for r in readings:
            if r.hit and r.distance < min_dist:
                min_dist = r.distance

        if min_dist <= critical_dist:
            return State.STOP
        if min_dist < self.influence_range:
            return State.AVOID
        return State.NAVIGATE

    def _record_position(self, x: float, y: float) -> None:
        """Record current position for stuck detection.

        Parameters
        ----------
        x, y : float
            Current agent position.
        """
        self._position_history.append((x, y))

    def _is_stuck(self) -> bool:
        """Check if the agent is stuck in a local minimum.

        Returns True if the total position displacement over the escape window
        is below the escape threshold.

        Returns
        -------
        bool
            True if agent appears stuck.
        """
        if len(self._position_history) < self.escape_window:
            return False

        oldest = self._position_history[0]
        newest = self._position_history[-1]
        dx = newest[0] - oldest[0]
        dy = newest[1] - oldest[1]
        displacement = math.sqrt(dx * dx + dy * dy)

        return displacement < self.escape_threshold

    def _escape_perturbation(self) -> ForceVector:
        """Generate a random escape force to break out of local minima.

        Returns
        -------
        ForceVector
            Random-direction force with magnitude ``escape_force``.
        """
        angle = self._rng.uniform(-math.pi, math.pi)
        return ForceVector(
            fx=self.escape_force * math.cos(angle),
            fy=self.escape_force * math.sin(angle),
            source="escape",
        )
