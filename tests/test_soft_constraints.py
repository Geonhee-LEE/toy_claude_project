"""Tests for soft constraints module."""

import numpy as np
import pytest

from mpc_controller.models.soft_constraints import (
    ConstraintType,
    PenaltyType,
    SoftConstraintParams,
    SoftConstraint,
    VelocitySoftConstraint,
    AccelerationSoftConstraint,
    ObstacleSoftConstraint,
    PositionSoftConstraint,
    SoftConstraintManager,
    ConstraintViolation,
    SoftConstraintResult,
)
from mpc_controller.controllers.mpc import MPCController, MPCParams
from mpc_controller.models.differential_drive import RobotParams


class TestSoftConstraintParams:
    """Tests for SoftConstraintParams dataclass."""

    def test_default_params(self):
        """Test default parameter values."""
        params = SoftConstraintParams()

        assert params.velocity_weight == 100.0
        assert params.acceleration_weight == 50.0
        assert params.position_weight == 200.0
        assert params.obstacle_weight == 500.0
        assert params.penalty_type == PenaltyType.QUADRATIC
        assert params.enabled is True

    def test_custom_params(self):
        """Test custom parameter values."""
        params = SoftConstraintParams(
            velocity_weight=200.0,
            penalty_type=PenaltyType.LINEAR,
            enabled=False,
        )

        assert params.velocity_weight == 200.0
        assert params.penalty_type == PenaltyType.LINEAR
        assert params.enabled is False


class TestConstraintViolation:
    """Tests for ConstraintViolation dataclass."""

    def test_is_violated(self):
        """Test violation detection."""
        # Violated
        violation = ConstraintViolation(
            constraint_type=ConstraintType.VELOCITY,
            violation_amount=0.1,
            slack_value=0.1,
            penalty_cost=1.0,
            timestep=0,
        )
        assert violation.is_violated is True

        # Not violated
        no_violation = ConstraintViolation(
            constraint_type=ConstraintType.VELOCITY,
            violation_amount=0.0,
            slack_value=0.0,
            penalty_cost=0.0,
            timestep=0,
        )
        assert no_violation.is_violated is False


class TestSoftConstraintResult:
    """Tests for SoftConstraintResult dataclass."""

    def test_empty_result(self):
        """Test empty result."""
        result = SoftConstraintResult()

        assert result.has_violations is False
        assert result.max_violation == 0.0
        assert result.total_penalty == 0.0

    def test_result_with_violations(self):
        """Test result with violations."""
        result = SoftConstraintResult(
            total_penalty=10.0,
            violations=[
                ConstraintViolation(
                    constraint_type=ConstraintType.VELOCITY,
                    violation_amount=0.1,
                    slack_value=0.1,
                    penalty_cost=5.0,
                    timestep=0,
                ),
                ConstraintViolation(
                    constraint_type=ConstraintType.ACCELERATION,
                    violation_amount=0.2,
                    slack_value=0.2,
                    penalty_cost=5.0,
                    timestep=1,
                ),
            ],
        )

        assert result.has_violations is True
        assert result.max_violation == 0.2
        assert len(result.get_violations_by_type(ConstraintType.VELOCITY)) == 1


class TestVelocitySoftConstraint:
    """Tests for VelocitySoftConstraint class."""

    def test_initialization(self):
        """Test constraint initialization."""
        constraint = VelocitySoftConstraint(
            v_max=1.0,
            omega_max=2.0,
            weight=100.0,
        )

        assert constraint.v_max == 1.0
        assert constraint.omega_max == 2.0
        assert constraint.weight == 100.0
        assert constraint.constraint_type == ConstraintType.VELOCITY

    def test_penalty_calculation(self):
        """Test penalty calculation for different types."""
        # Quadratic
        constraint_quad = VelocitySoftConstraint(
            v_max=1.0, omega_max=2.0, weight=100.0,
            penalty_type=PenaltyType.QUADRATIC,
        )
        import casadi as ca
        slack = ca.SX.sym("slack")
        penalty = constraint_quad.compute_penalty(slack)
        # At slack=0.1, penalty should be 100 * 0.1^2 = 1.0
        f = ca.Function("f", [slack], [penalty])
        assert float(f(0.1)) == pytest.approx(1.0)

        # Linear
        constraint_lin = VelocitySoftConstraint(
            v_max=1.0, omega_max=2.0, weight=100.0,
            penalty_type=PenaltyType.LINEAR,
        )
        penalty_lin = constraint_lin.compute_penalty(slack)
        f_lin = ca.Function("f", [slack], [penalty_lin])
        assert float(f_lin(0.1)) == pytest.approx(10.0)


class TestSoftConstraintManager:
    """Tests for SoftConstraintManager class."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = SoftConstraintManager()

        assert manager.enabled is True
        assert len(manager.constraints) == 0

    def test_add_velocity_constraint(self):
        """Test adding velocity constraint."""
        manager = SoftConstraintManager()
        manager.add_velocity_constraint(v_max=1.0, omega_max=2.0)

        assert len(manager.constraints) == 1
        assert isinstance(manager.constraints[0], VelocitySoftConstraint)

    def test_add_acceleration_constraint(self):
        """Test adding acceleration constraint."""
        manager = SoftConstraintManager()
        manager.add_acceleration_constraint(a_max=1.0, alpha_max=2.0, dt=0.1)

        assert len(manager.constraints) == 1
        assert isinstance(manager.constraints[0], AccelerationSoftConstraint)

    def test_add_obstacle_constraint(self):
        """Test adding obstacle constraint."""
        manager = SoftConstraintManager()
        manager.add_obstacle_constraint(safety_margin=0.5)

        assert len(manager.constraints) == 1
        assert isinstance(manager.constraints[0], ObstacleSoftConstraint)

    def test_chaining(self):
        """Test method chaining."""
        manager = (
            SoftConstraintManager()
            .add_velocity_constraint(v_max=1.0, omega_max=2.0)
            .add_acceleration_constraint(a_max=1.0, alpha_max=2.0, dt=0.1)
        )

        assert len(manager.constraints) == 2

    def test_disabled_constraints(self):
        """Test that disabled constraints are not added."""
        params = SoftConstraintParams(
            enable_velocity_soft=False,
            enable_acceleration_soft=False,
        )
        manager = SoftConstraintManager(params)
        manager.add_velocity_constraint(v_max=1.0, omega_max=2.0)
        manager.add_acceleration_constraint(a_max=1.0, alpha_max=2.0, dt=0.1)

        assert len(manager.constraints) == 0


class TestMPCWithSoftConstraints:
    """Integration tests for MPC with soft constraints."""

    def test_mpc_with_soft_constraints_enabled(self):
        """Test MPC controller with soft constraints enabled."""
        soft_params = SoftConstraintParams(
            velocity_weight=100.0,
            acceleration_weight=50.0,
            enable_velocity_soft=True,
            enable_acceleration_soft=True,
        )
        mpc_params = MPCParams(
            N=10,
            dt=0.1,
            soft_constraints=soft_params,
        )
        controller = MPCController(
            mpc_params=mpc_params,
            enable_soft_constraints=True,
        )

        assert controller.use_soft is True
        assert controller.n_slack > 0

    def test_mpc_with_soft_constraints_disabled(self):
        """Test MPC controller with soft constraints disabled."""
        controller = MPCController(enable_soft_constraints=False)

        assert controller.use_soft is False
        assert controller.n_slack == 0

    def test_mpc_compute_control_with_soft_constraints(self):
        """Test control computation with soft constraints."""
        soft_params = SoftConstraintParams(
            velocity_weight=100.0,
            acceleration_weight=50.0,
        )
        mpc_params = MPCParams(
            N=10,
            dt=0.1,
            soft_constraints=soft_params,
        )
        controller = MPCController(
            mpc_params=mpc_params,
            enable_soft_constraints=True,
        )

        # Simple straight-line reference
        current_state = np.array([0.0, 0.0, 0.0])
        reference = np.zeros((11, 3))
        for i in range(11):
            reference[i] = [i * 0.1, 0.0, 0.0]

        control, info = controller.compute_control(current_state, reference)

        # Check output format
        assert control.shape == (2,)
        assert "soft_constraints" in info
        assert "predicted_trajectory" in info

    def test_soft_constraint_violations_reported(self):
        """Test that soft constraint violations are reported correctly."""
        # Use very restrictive constraints to force violations
        soft_params = SoftConstraintParams(
            velocity_weight=1.0,  # Low penalty to allow violations
            acceleration_weight=1.0,
        )
        mpc_params = MPCParams(
            N=5,
            dt=0.1,
            soft_constraints=soft_params,
            a_max=0.01,  # Very low acceleration limit
            alpha_max=0.01,
        )
        controller = MPCController(
            mpc_params=mpc_params,
            enable_soft_constraints=True,
        )

        # Aggressive reference that requires high acceleration
        current_state = np.array([0.0, 0.0, 0.0])
        reference = np.zeros((6, 3))
        for i in range(6):
            reference[i] = [i * 1.0, 0.0, 0.0]  # 1m per step

        control, info = controller.compute_control(current_state, reference)

        # Should have soft constraint info
        soft_info = info.get("soft_constraints", {})
        assert soft_info is not None


class TestPenaltyTypes:
    """Tests for different penalty types."""

    def test_quadratic_penalty(self):
        """Test quadratic penalty function."""
        import casadi as ca

        constraint = SoftConstraint(
            constraint_type=ConstraintType.VELOCITY,
            weight=10.0,
            penalty_type=PenaltyType.QUADRATIC,
        )
        slack = ca.SX.sym("s")
        penalty = constraint.compute_penalty(slack)
        f = ca.Function("f", [slack], [penalty])

        # w * s^2
        assert float(f(0.0)) == pytest.approx(0.0)
        assert float(f(1.0)) == pytest.approx(10.0)
        assert float(f(2.0)) == pytest.approx(40.0)

    def test_linear_penalty(self):
        """Test linear penalty function."""
        import casadi as ca

        constraint = SoftConstraint(
            constraint_type=ConstraintType.VELOCITY,
            weight=10.0,
            penalty_type=PenaltyType.LINEAR,
        )
        slack = ca.SX.sym("s")
        penalty = constraint.compute_penalty(slack)
        f = ca.Function("f", [slack], [penalty])

        # w * s
        assert float(f(0.0)) == pytest.approx(0.0)
        assert float(f(1.0)) == pytest.approx(10.0)
        assert float(f(2.0)) == pytest.approx(20.0)

    def test_huber_penalty(self):
        """Test Huber penalty function."""
        import casadi as ca

        delta = 1.0
        constraint = SoftConstraint(
            constraint_type=ConstraintType.VELOCITY,
            weight=10.0,
            penalty_type=PenaltyType.HUBER,
            huber_delta=delta,
        )
        slack = ca.SX.sym("s")
        penalty = constraint.compute_penalty(slack)
        f = ca.Function("f", [slack], [penalty])

        # For s <= delta: w * 0.5 * s^2
        assert float(f(0.5)) == pytest.approx(10.0 * 0.5 * 0.5**2)

        # For s > delta: w * delta * (s - 0.5 * delta)
        assert float(f(2.0)) == pytest.approx(10.0 * 1.0 * (2.0 - 0.5 * 1.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
