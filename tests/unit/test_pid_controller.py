"""Tests for the PID heading controller."""

import math
from unittest.mock import Mock

import pytest

from src.pid_controller import PIDController


# --- Fixtures ---


@pytest.fixture
def pid() -> PIDController:
    """Default PID controller with standard gains."""
    return PIDController(
        kp=4.0,
        ki=0.1,
        kd=0.5,
        max_omega=4.0,
        integral_limit=2.0,
    )


@pytest.fixture
def p_only() -> PIDController:
    """Proportional-only controller."""
    return PIDController(
        kp=2.0,
        ki=0.0,
        kd=0.0,
        max_omega=10.0,
        integral_limit=1.0,
    )


@pytest.fixture
def pi_controller() -> PIDController:
    """PI controller (no derivative)."""
    return PIDController(
        kp=1.0,
        ki=0.5,
        kd=0.0,
        max_omega=10.0,
        integral_limit=2.0,
    )


@pytest.fixture
def pd_controller() -> PIDController:
    """PD controller (no integral)."""
    return PIDController(
        kp=1.0,
        ki=0.0,
        kd=2.0,
        max_omega=10.0,
        integral_limit=1.0,
    )


# --- Tests: Init ---


class TestPIDControllerInit:
    def test_basic_attributes(self, pid: PIDController) -> None:
        assert pid.kp == 4.0
        assert pid.ki == 0.1
        assert pid.kd == 0.5
        assert pid.max_omega == 4.0
        assert pid.integral_limit == 2.0

    def test_initial_state_zeroed(self, pid: PIDController) -> None:
        """Internal state (integral, prev_error) should start at zero."""
        assert pid._integral == 0.0
        assert pid._prev_error == 0.0

    def test_from_config(self) -> None:
        config = Mock()
        config.controller.kp = 3.0
        config.controller.ki = 0.2
        config.controller.kd = 0.8
        config.controller.max_omega = 5.0
        config.controller.integral_limit = 3.0

        p = PIDController.from_config(config)
        assert p.kp == 3.0
        assert p.ki == 0.2
        assert p.kd == 0.8
        assert p.max_omega == 5.0
        assert p.integral_limit == 3.0


# --- Tests: Proportional ---


class TestProportional:
    def test_positive_error_gives_positive_output(
        self, p_only: PIDController
    ) -> None:
        """Positive heading error -> positive angular velocity."""
        omega = p_only.compute(1.0, dt=0.033)
        assert omega > 0

    def test_negative_error_gives_negative_output(
        self, p_only: PIDController
    ) -> None:
        """Negative heading error -> negative angular velocity."""
        omega = p_only.compute(-1.0, dt=0.033)
        assert omega < 0

    def test_zero_error_gives_zero_output(self, p_only: PIDController) -> None:
        """Zero error -> zero output."""
        omega = p_only.compute(0.0, dt=0.033)
        assert omega == pytest.approx(0.0)

    def test_proportional_scales_with_kp(self, p_only: PIDController) -> None:
        """Output should be kp * error for P-only controller on first call."""
        error = 0.5
        omega = p_only.compute(error, dt=0.033)
        # P term = kp * error = 2.0 * 0.5 = 1.0
        # D term on first call: kd=0 so 0
        # I term: ki=0 so 0
        assert omega == pytest.approx(2.0 * 0.5)


# --- Tests: Integral ---


class TestIntegral:
    def test_integral_accumulates(self, pi_controller: PIDController) -> None:
        """Integral term should accumulate error over time."""
        dt = 0.1
        # First call: I = ki * error * dt = 0.5 * 1.0 * 0.1 = 0.05
        omega1 = pi_controller.compute(1.0, dt=dt)
        # Second call: I = 0.5 * (0.05 + 0.1) = 0.5 * 0.15
        omega2 = pi_controller.compute(1.0, dt=dt)
        # Second output should be larger due to integral accumulation
        assert omega2 > omega1

    def test_anti_windup_clamps_integral(
        self, pi_controller: PIDController
    ) -> None:
        """Integral should be clamped to [-integral_limit, integral_limit]."""
        dt = 1.0
        # Accumulate large error: integral = error * dt = 1.0 * 1.0 = 1.0
        pi_controller.compute(1.0, dt=dt)
        # Again: integral would be 2.0 but limit is 2.0 so just at limit
        pi_controller.compute(1.0, dt=dt)
        # Once more: integral would be 3.0 but clamped to 2.0
        pi_controller.compute(1.0, dt=dt)
        assert pi_controller._integral == pytest.approx(2.0)

    def test_negative_integral_clamped(
        self, pi_controller: PIDController
    ) -> None:
        """Negative integral should also be clamped."""
        dt = 1.0
        for _ in range(5):
            pi_controller.compute(-1.0, dt=dt)
        assert pi_controller._integral == pytest.approx(-2.0)


# --- Tests: Derivative ---


class TestDerivative:
    def test_derivative_responds_to_error_change(
        self, pd_controller: PIDController
    ) -> None:
        """Derivative term should respond to change in error."""
        dt = 0.1
        # First call with error=0: D contribution based on (0-0)/dt = 0
        pd_controller.compute(0.0, dt=dt)
        # Second call with error=1.0: D = kd * (1.0-0.0)/0.1 = 2.0*10 = 20
        # But clamped to max_omega=10
        omega = pd_controller.compute(1.0, dt=dt)
        # Should be P + D = 1.0*1.0 + 2.0*(1.0-0.0)/0.1 = 1 + 20 = 21 -> clamped to 10
        assert omega == pytest.approx(10.0)

    def test_constant_error_zero_derivative(
        self, pd_controller: PIDController
    ) -> None:
        """Constant error should produce zero derivative contribution."""
        dt = 0.1
        pd_controller.compute(1.0, dt=dt)
        # Second call with same error: D = kd * (1.0-1.0)/dt = 0
        omega = pd_controller.compute(1.0, dt=dt)
        # Should be P only: kp * 1.0 = 1.0
        assert omega == pytest.approx(1.0)

    def test_decreasing_error_reduces_output(
        self, pd_controller: PIDController
    ) -> None:
        """When error decreases, derivative opposes (dampens)."""
        dt = 0.1
        pd_controller.compute(1.0, dt=dt)
        # Error drops from 1.0 to 0.5: D = kd * (0.5-1.0)/0.1 = 2.0*(-5) = -10
        # P = 1.0 * 0.5 = 0.5, total = 0.5 + (-10) = -9.5
        omega = pd_controller.compute(0.5, dt=dt)
        assert omega < 0  # derivative dominates, pulling back


# --- Tests: Full PID ---


class TestFullPID:
    def test_combined_output(self, pid: PIDController) -> None:
        """Full PID output should combine P, I, and D terms."""
        dt = 0.1
        error = 0.5
        omega = pid.compute(error, dt=dt)
        # P = 4.0 * 0.5 = 2.0
        # I = 0.1 * (0.5 * 0.1) = 0.1 * 0.05 = 0.005
        # D = 0.5 * (0.5 - 0.0) / 0.1 = 0.5 * 5.0 = 2.5
        # Total = 2.0 + 0.005 + 2.5 = 4.505 -> clamped to 4.0
        assert omega == pytest.approx(4.0)

    def test_settling_behavior(self, pid: PIDController) -> None:
        """PID should produce decreasing output as error decreases."""
        dt = 0.033
        outputs = []
        error = 1.0
        for _ in range(10):
            omega = pid.compute(error, dt=dt)
            outputs.append(omega)
            # Simulate the error reducing over time
            error *= 0.8

        # Later outputs should be smaller in magnitude
        assert abs(outputs[-1]) < abs(outputs[0])


# --- Tests: Output Clamping ---


class TestOutputClamping:
    def test_positive_clamped_to_max_omega(self, pid: PIDController) -> None:
        """Large positive error should clamp output to max_omega."""
        omega = pid.compute(10.0, dt=0.033)
        assert omega == pytest.approx(pid.max_omega)

    def test_negative_clamped_to_negative_max_omega(
        self, pid: PIDController
    ) -> None:
        """Large negative error should clamp output to -max_omega."""
        omega = pid.compute(-10.0, dt=0.033)
        assert omega == pytest.approx(-pid.max_omega)


# --- Tests: Reset ---


class TestReset:
    def test_reset_clears_integral(self, pid: PIDController) -> None:
        """Reset should zero out the integral accumulator."""
        pid.compute(1.0, dt=0.1)
        pid.compute(1.0, dt=0.1)
        pid.reset()
        assert pid._integral == pytest.approx(0.0)

    def test_reset_clears_prev_error(self, pid: PIDController) -> None:
        """Reset should zero out previous error."""
        pid.compute(1.0, dt=0.1)
        pid.reset()
        assert pid._prev_error == pytest.approx(0.0)

    def test_output_after_reset_matches_fresh(
        self, pid: PIDController
    ) -> None:
        """After reset, output should match a fresh controller."""
        # Build up state
        pid.compute(1.0, dt=0.1)
        pid.compute(0.5, dt=0.1)
        pid.reset()

        fresh = PIDController(
            kp=pid.kp,
            ki=pid.ki,
            kd=pid.kd,
            max_omega=pid.max_omega,
            integral_limit=pid.integral_limit,
        )

        error = 0.3
        dt = 0.033
        assert pid.compute(error, dt=dt) == pytest.approx(
            fresh.compute(error, dt=dt)
        )


# --- Tests: Edge Cases ---


class TestEdgeCases:
    def test_zero_dt_no_crash(self, pid: PIDController) -> None:
        """Zero dt should not crash (derivative term handled gracefully)."""
        omega = pid.compute(1.0, dt=0.0)
        assert math.isfinite(omega)

    def test_very_small_dt(self, pid: PIDController) -> None:
        """Very small dt should produce finite output."""
        omega = pid.compute(0.1, dt=1e-6)
        assert math.isfinite(omega)
        # Should be clamped to max_omega due to large derivative
        assert abs(omega) <= pid.max_omega + 1e-10

    def test_very_large_error(self, pid: PIDController) -> None:
        """Very large error should still clamp to max_omega."""
        omega = pid.compute(1000.0, dt=0.033)
        assert omega == pytest.approx(pid.max_omega)

    def test_alternating_errors(self, pid: PIDController) -> None:
        """Alternating positive/negative errors should work correctly."""
        dt = 0.1
        o1 = pid.compute(1.0, dt=dt)
        o2 = pid.compute(-1.0, dt=dt)
        assert o1 > 0
        assert o2 < 0
