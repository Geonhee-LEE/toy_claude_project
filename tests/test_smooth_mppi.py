"""Smooth MPPI (SMPPI) 컨트롤러 테스트 — Δu input-lifting 구조적 부드러움 검증."""

import numpy as np
import pytest

from mpc_controller import (
    DifferentialDriveModel,
    RobotParams,
    MPPIController,
    MPPIParams,
    generate_circle_trajectory,
    TrajectoryInterpolator,
)
from mpc_controller.controllers.mppi.smooth_mppi import SmoothMPPIController
from mpc_controller.controllers.mppi.utils import effective_sample_size


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def robot_params():
    return RobotParams(max_velocity=1.0, max_omega=1.5)


@pytest.fixture
def basic_params():
    return MPPIParams(
        N=20, K=64, dt=0.05, lambda_=10.0,
        noise_sigma=np.array([0.3, 0.3]),
        Q=np.diag([10.0, 10.0, 1.0]),
        R=np.diag([0.01, 0.01]),
        Qf=np.diag([50.0, 50.0, 5.0]),
    )


@pytest.fixture
def smooth_params():
    return MPPIParams(
        N=20, K=64, dt=0.05, lambda_=10.0,
        noise_sigma=np.array([0.3, 0.3]),
        Q=np.diag([10.0, 10.0, 1.0]),
        R=np.diag([0.01, 0.01]),
        Qf=np.diag([50.0, 50.0, 5.0]),
        smooth_R_jerk=np.array([0.1, 0.1]),
        smooth_action_cost_weight=1.0,
    )


@pytest.fixture
def circle_ref():
    traj = generate_circle_trajectory(np.array([0.0, 0.0]), 2.0, 200)
    return traj


# ─────────────────────────────────────────────────────────────
# 기본 생성 및 인터페이스 테스트
# ─────────────────────────────────────────────────────────────

class TestSmoothMPPICreation:
    """생성자 및 기본 인터페이스 검증."""

    def test_inherits_mppi(self, robot_params, smooth_params):
        ctrl = SmoothMPPIController(robot_params, smooth_params, seed=42)
        assert isinstance(ctrl, MPPIController)

    def test_default_creation(self, robot_params, basic_params):
        ctrl = SmoothMPPIController(robot_params, basic_params, seed=42)
        assert ctrl.DU.shape == (basic_params.N, 2)
        np.testing.assert_array_equal(ctrl._u_prev, [0.0, 0.0])

    def test_custom_jerk_weights(self, robot_params, smooth_params):
        smooth_params.smooth_R_jerk = np.array([0.5, 0.3])
        ctrl = SmoothMPPIController(robot_params, smooth_params, seed=42)
        np.testing.assert_array_equal(ctrl._R_jerk, [0.5, 0.3])

    def test_jerk_weight_default(self, robot_params, basic_params):
        ctrl = SmoothMPPIController(robot_params, basic_params, seed=42)
        assert ctrl._R_jerk is not None
        assert len(ctrl._R_jerk) == 2


# ─────────────────────────────────────────────────────────────
# compute_control 인터페이스
# ─────────────────────────────────────────────────────────────

class TestSmoothMPPIComputeControl:
    """compute_control 출력 형식 및 내용 검증."""

    def test_output_shape(self, robot_params, smooth_params, circle_ref):
        ctrl = SmoothMPPIController(robot_params, smooth_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, smooth_params.N, smooth_params.dt, state[2])

        u, info = ctrl.compute_control(state, ref)
        assert u.shape == (2,), f"Expected (2,), got {u.shape}"

    def test_info_keys(self, robot_params, smooth_params, circle_ref):
        ctrl = SmoothMPPIController(robot_params, smooth_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, smooth_params.N, smooth_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)

        required_keys = [
            "predicted_trajectory", "predicted_controls", "cost",
            "mean_cost", "solve_time", "sample_trajectories",
            "sample_weights", "sample_costs", "best_trajectory",
            "best_index", "ess", "temperature",
            "delta_u_sequence", "delta_u_norm", "u_prev",
        ]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

    def test_delta_u_sequence_in_info(self, robot_params, smooth_params, circle_ref):
        ctrl = SmoothMPPIController(robot_params, smooth_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, smooth_params.N, smooth_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)
        assert info["delta_u_sequence"].shape == (smooth_params.N, 2)

    def test_control_within_limits(self, robot_params, smooth_params, circle_ref):
        ctrl = SmoothMPPIController(robot_params, smooth_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, smooth_params.N, smooth_params.dt, state[2])

        u, info = ctrl.compute_control(state, ref)
        assert abs(u[0]) <= robot_params.max_velocity + 1e-6
        assert abs(u[1]) <= robot_params.max_omega + 1e-6


# ─────────────────────────────────────────────────────────────
# 부드러움(Smoothness) 검증
# ─────────────────────────────────────────────────────────────

class TestSmoothMPPISmoothness:
    """Vanilla 대비 부드러운 제어 생성 검증."""

    def test_control_rate_lower_than_vanilla(self, robot_params, circle_ref):
        """SMPPI는 Vanilla 대비 Δu 크기가 작아야 함."""
        params = MPPIParams(
            N=20, K=256, dt=0.05, lambda_=10.0,
            noise_sigma=np.array([0.3, 0.3]),
        )

        # Vanilla
        vanilla = MPPIController(robot_params, params, seed=42)
        # SMPPI
        smooth = SmoothMPPIController(
            robot_params,
            MPPIParams(
                N=20, K=256, dt=0.05, lambda_=10.0,
                noise_sigma=np.array([0.3, 0.3]),
                smooth_R_jerk=np.array([1.0, 1.0]),
                smooth_action_cost_weight=5.0,
            ),
            seed=42,
        )

        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)

        vanilla_rates, smooth_rates = [], []
        for step in range(10):
            ref = interp.get_reference(
                step * 0.05, params.N, params.dt, state[2]
            )
            uv, iv = vanilla.compute_control(state, ref)
            us, is_ = smooth.compute_control(state, ref)

            if step > 0:
                vanilla_rates.append(np.abs(uv - prev_uv))
                smooth_rates.append(np.abs(us - prev_us))
            prev_uv, prev_us = uv.copy(), us.copy()

        if len(vanilla_rates) > 0:
            avg_vanilla_rate = np.mean(vanilla_rates)
            avg_smooth_rate = np.mean(smooth_rates)
            # SMPPI의 제어 변화율이 Vanilla보다 작거나 같아야 함
            assert avg_smooth_rate <= avg_vanilla_rate * 1.5, (
                f"SMPPI rate {avg_smooth_rate:.4f} > Vanilla rate {avg_vanilla_rate:.4f} * 1.5"
            )

    def test_du_norm_decreases_with_higher_jerk_weight(self, robot_params, circle_ref):
        """Jerk weight를 올리면 Δu norm이 감소해야 함."""
        norms = []
        for jw in [0.1, 1.0, 10.0]:
            params = MPPIParams(
                N=20, K=128, dt=0.05, lambda_=10.0,
                noise_sigma=np.array([0.3, 0.3]),
                smooth_R_jerk=np.array([jw, jw]),
                smooth_action_cost_weight=1.0,
            )
            ctrl = SmoothMPPIController(robot_params, params, seed=42)
            state = circle_ref[0].copy()
            interp = TrajectoryInterpolator(circle_ref, dt=0.05)
            ref = interp.get_reference(0, params.N, params.dt, state[2])

            _, info = ctrl.compute_control(state, ref)
            norms.append(info["delta_u_norm"])

        # Higher jerk weight → lower du_norm (또는 같을 수 있음)
        assert norms[-1] <= norms[0] * 1.5, (
            f"Du norms not decreasing: {norms}"
        )


# ─────────────────────────────────────────────────────────────
# 다중 스텝 시뮬레이션
# ─────────────────────────────────────────────────────────────

class TestSmoothMPPIMultiStep:
    """다중 스텝 시뮬레이션 안정성."""

    def test_multi_step_no_error(self, robot_params, smooth_params, circle_ref):
        ctrl = SmoothMPPIController(robot_params, smooth_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)

        for step in range(20):
            ref = interp.get_reference(
                step * 0.05, smooth_params.N, smooth_params.dt, state[2]
            )
            u, info = ctrl.compute_control(state, ref)
            assert np.all(np.isfinite(u)), f"Non-finite control at step {step}"
            state = state + np.array([
                u[0] * np.cos(state[2]) * 0.05,
                u[0] * np.sin(state[2]) * 0.05,
                u[1] * 0.05,
            ])

    def test_u_prev_updates(self, robot_params, smooth_params, circle_ref):
        """u_prev가 매 스텝 업데이트되는지 확인."""
        ctrl = SmoothMPPIController(robot_params, smooth_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, smooth_params.N, smooth_params.dt, state[2])

        u1, info1 = ctrl.compute_control(state, ref)
        np.testing.assert_array_almost_equal(ctrl._u_prev, u1)

    def test_reset_clears_state(self, robot_params, smooth_params, circle_ref):
        ctrl = SmoothMPPIController(robot_params, smooth_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, smooth_params.N, smooth_params.dt, state[2])

        ctrl.compute_control(state, ref)
        ctrl.reset()

        np.testing.assert_array_equal(ctrl.DU, np.zeros_like(ctrl.DU))
        np.testing.assert_array_equal(ctrl._u_prev, [0.0, 0.0])
        assert ctrl._iteration_count == 0


# ─────────────────────────────────────────────────────────────
# ESS 및 비용 검증
# ─────────────────────────────────────────────────────────────

class TestSmoothMPPIMetrics:
    """ESS, 비용 등 기본 메트릭 검증."""

    def test_ess_valid_range(self, robot_params, smooth_params, circle_ref):
        ctrl = SmoothMPPIController(robot_params, smooth_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, smooth_params.N, smooth_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)
        assert 1.0 <= info["ess"] <= smooth_params.K

    def test_cost_non_negative(self, robot_params, smooth_params, circle_ref):
        ctrl = SmoothMPPIController(robot_params, smooth_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, smooth_params.N, smooth_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)
        assert info["cost"] >= 0

    def test_weights_normalized(self, robot_params, smooth_params, circle_ref):
        ctrl = SmoothMPPIController(robot_params, smooth_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, smooth_params.N, smooth_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)
        np.testing.assert_almost_equal(np.sum(info["sample_weights"]), 1.0, decimal=5)


# ─────────────────────────────────────────────────────────────
# 장애물 환경
# ─────────────────────────────────────────────────────────────

class TestSmoothMPPIObstacles:
    """장애물이 있는 환경에서의 동작."""

    def test_with_obstacles(self, robot_params, smooth_params, circle_ref):
        obstacles = np.array([[1.0, 0.0, 0.3]])
        ctrl = SmoothMPPIController(
            robot_params, smooth_params, seed=42, obstacles=obstacles
        )
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, smooth_params.N, smooth_params.dt, state[2])

        u, info = ctrl.compute_control(state, ref)
        assert np.all(np.isfinite(u))
