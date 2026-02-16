"""PID heading controller for continuous angular velocity control.

Implements a standard PID controller with anti-windup (integral clamping)
and output saturation. Used to track a desired heading by computing the
angular velocity needed to reduce heading error.
"""

import logging

from src.config import Config

logger = logging.getLogger(__name__)


class PIDController:
    """PID controller for heading tracking.

    Computes angular velocity from heading error using proportional,
    integral, and derivative terms with anti-windup and output clamping.

    Parameters
    ----------
    kp : float
        Proportional gain.
    ki : float
        Integral gain.
    kd : float
        Derivative gain.
    max_omega : float
        Maximum angular velocity magnitude (rad/s).
    integral_limit : float
        Anti-windup clamp for the integral accumulator.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        max_omega: float,
        integral_limit: float,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_omega = max_omega
        self.integral_limit = integral_limit

        self._integral: float = 0.0
        self._prev_error: float = 0.0

        logger.info(
            "PIDController: kp=%.2f, ki=%.2f, kd=%.2f, max_omega=%.2f",
            kp, ki, kd, max_omega,
        )

    @classmethod
    def from_config(cls, config: Config) -> "PIDController":
        """Create controller from configuration.

        Parameters
        ----------
        config : Config
            System configuration with controller section.

        Returns
        -------
        PIDController
            Configured controller instance.
        """
        c = config.controller  # type: ignore[attr-defined]
        return cls(
            kp=float(c.kp),  # type: ignore[attr-defined]
            ki=float(c.ki),  # type: ignore[attr-defined]
            kd=float(c.kd),  # type: ignore[attr-defined]
            max_omega=float(c.max_omega),  # type: ignore[attr-defined]
            integral_limit=float(c.integral_limit),  # type: ignore[attr-defined]
        )

    def compute(self, error: float, dt: float) -> float:
        """Compute angular velocity from heading error.

        Parameters
        ----------
        error : float
            Heading error in radians (desired - current), in [-pi, pi].
        dt : float
            Time step in seconds.

        Returns
        -------
        float
            Angular velocity in rad/s, clamped to [-max_omega, max_omega].
        """
        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self._integral += error * dt
        self._integral = max(
            -self.integral_limit, min(self.integral_limit, self._integral)
        )
        i_term = self.ki * self._integral

        # Derivative term (guard against zero dt)
        if dt > 0.0:
            d_term = self.kd * (error - self._prev_error) / dt
        else:
            d_term = 0.0

        self._prev_error = error

        # Sum and clamp output
        output = p_term + i_term + d_term
        return max(-self.max_omega, min(self.max_omega, output))

    def reset(self) -> None:
        """Reset controller state.

        Zeros the integral accumulator and previous error.
        """
        self._integral = 0.0
        self._prev_error = 0.0
