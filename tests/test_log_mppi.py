"""Log-MPPI 컨트롤러 테스트 — 수치 안정성 및 Vanilla 호환성 검증."""

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
from mpc_controller.controllers.mppi.log_mppi import LogMPPIController
from mpc_controller.controllers.mppi.utils import log_sum_exp


# ─────────────────────────────────────────────────────────────
# 기본 초기화 및 인터페이스
# ─────────────────────────────────────────────────────────────

class TestLogMPPIInit:
    """LogMPPIController 초기화 테스트."""

    def test_creation(self):
        ctrl = LogMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1),
            seed=42,
        )
        assert ctrl is not None
        assert isinstance(ctrl, MPPIController)

    def test_inherits_mppi(self):
        ctrl = LogMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1),
            seed=42,
        )
        assert hasattr(ctrl, "compute_control")
        assert hasattr(ctrl, "_compute_weights")
        assert hasattr(ctrl, "_get_current_lambda")

    def test_interface_compatible(self):
        """compute_control 출력 형태가 Vanilla와 동일."""
        ctrl = LogMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1),
            seed=42,
        )
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))

        control, info = ctrl.compute_control(state, ref)

        assert control.shape == (2,)
        assert "sample_trajectories" in info
        assert "sample_weights" in info
        assert "ess" in info
        assert "temperature" in info


# ─────────────────────────────────────────────────────────────
# 수치 안정성
# ─────────────────────────────────────────────────────────────

class TestLogMPPINumericalStability:
    """극단적 비용 범위에서 NaN/Inf 방지 테스트."""

    def test_extreme_large_costs_no_nan(self):
        """비용이 1e15 수준이어도 가중치가 유한."""
        ctrl = LogMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1),
            seed=42,
        )
        costs = np.random.default_rng(0).uniform(1e12, 1e15, size=64)
        weights = ctrl._compute_weights(costs)

        assert np.all(np.isfinite(weights))
        np.testing.assert_almost_equal(np.sum(weights), 1.0)

    def test_extreme_small_costs(self):
        """비용이 매우 작아도(1e-15) 정상 동작."""
        ctrl = LogMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1),
            seed=42,
        )
        costs = np.random.default_rng(0).uniform(1e-15, 1e-12, size=64)
        weights = ctrl._compute_weights(costs)

        assert np.all(np.isfinite(weights))
        np.testing.assert_almost_equal(np.sum(weights), 1.0)

    def test_mixed_extreme_costs(self):
        """비용 범위가 극단적으로 넓어도(1e-10 ~ 1e10) 안전."""
        ctrl = LogMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, lambda_=1.0),
            seed=42,
        )
        rng = np.random.default_rng(0)
        costs = np.concatenate([
            rng.uniform(1e-10, 1e-5, size=32),
            rng.uniform(1e5, 1e10, size=32),
        ])
        weights = ctrl._compute_weights(costs)

        assert np.all(np.isfinite(weights))
        np.testing.assert_almost_equal(np.sum(weights), 1.0)

    def test_identical_costs(self):
        """모든 비용이 동일하면 균등 가중치."""
        ctrl = LogMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1),
            seed=42,
        )
        costs = np.ones(64) * 100.0
        weights = ctrl._compute_weights(costs)

        expected = np.ones(64) / 64
        np.testing.assert_allclose(weights, expected, atol=1e-10)

    def test_compute_control_extreme_scenario(self):
        """극단적 참조 궤적에서도 compute_control이 NaN 없이 동작."""
        ctrl = LogMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1),
            seed=42,
        )
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ref[:, 0] = np.linspace(0, 1000, 11)  # 매우 먼 목표

        control, info = ctrl.compute_control(state, ref)

        assert np.all(np.isfinite(control))
        assert np.all(np.isfinite(info["sample_weights"]))


# ─────────────────────────────────────────────────────────────
# Vanilla MPPI와의 동등성
# ─────────────────────────────────────────────────────────────

class TestLogMPPIWeights:
    """가중치 속성 테스트."""

    def test_weights_sum_to_one(self):
        """가중치 합이 정확히 1.0."""
        ctrl = LogMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1),
            seed=42,
        )
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ref[:, 0] = np.linspace(0, 2, 11)

        _, info = ctrl.compute_control(state, ref)
        np.testing.assert_almost_equal(np.sum(info["sample_weights"]), 1.0)


class TestLogMPPIVanillaEquivalence:
    """일반 비용 범위에서 Vanilla MPPI와 동일 결과 검증."""

    def test_weights_match_vanilla_small_range(self):
        """비용 범위가 작으면 Vanilla와 차이 < 1e-6."""
        params = MPPIParams(N=10, K=64, dt=0.1, lambda_=10.0)
        vanilla = MPPIController(mppi_params=params, seed=42)
        log_ctrl = LogMPPIController(mppi_params=params, seed=42)

        costs = np.random.default_rng(0).uniform(1.0, 10.0, size=64)

        w_vanilla = vanilla._compute_weights(costs)
        w_log = log_ctrl._compute_weights(costs)

        np.testing.assert_allclose(w_vanilla, w_log, atol=1e-6)

    def test_control_output_close_to_vanilla(self):
        """동일 시드 + 동일 상태에서 Vanilla와 거의 동일한 제어."""
        params = MPPIParams(N=10, K=128, dt=0.1, lambda_=10.0)
        vanilla = MPPIController(mppi_params=params, seed=42)
        log_ctrl = LogMPPIController(mppi_params=params, seed=42)

        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ref[:, 0] = np.linspace(0, 2, 11)

        u_v, _ = vanilla.compute_control(state, ref)
        u_l, _ = log_ctrl.compute_control(state, ref)

        np.testing.assert_allclose(u_v, u_l, atol=1e-6)


# ─────────────────────────────────────────────────────────────
# 궤적 추적 성능
# ─────────────────────────────────────────────────────────────

class TestLogMPPITracking:
    """Log-MPPI 궤적 추적 성능."""

    def test_circle_tracking_rmse(self):
        """원형 궤적 추적 RMSE < 0.2m."""
        params = MPPIParams(N=20, K=512, dt=0.05, lambda_=10.0)
        ctrl = LogMPPIController(mppi_params=params, seed=42)

        trajectory = generate_circle_trajectory(
            center=np.array([0.0, 0.0]),
            radius=2.0,
            num_points=200,
        )
        interpolator = TrajectoryInterpolator(trajectory, dt=0.05)

        state = np.array([2.0, 0.0, np.pi / 2])
        model = DifferentialDriveModel()

        errors = []
        for i in range(100):
            t = i * 0.05
            ref = interpolator.get_reference(
                t, params.N, params.dt, current_theta=state[2]
            )
            control, _ = ctrl.compute_control(state, ref)
            state = model.forward_simulate(state, control, 0.05)
            dist = np.sqrt(state[0] ** 2 + state[1] ** 2)
            errors.append(abs(dist - 2.0))

        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        assert rmse < 0.2, f"Log-MPPI circle RMSE = {rmse:.4f} (> 0.2m)"


# ─────────────────────────────────────────────────────────────
# Adaptive Temperature 연동
# ─────────────────────────────────────────────────────────────

class TestLogMPPIAdaptiveTemp:
    """Log-MPPI + Adaptive Temperature."""

    def test_adaptive_temp_integration(self):
        """Adaptive temperature와 함께 정상 동작."""
        params = MPPIParams(
            N=10, K=64, dt=0.1,
            adaptive_temperature=True,
            adaptive_temp_config={"target_ess_ratio": 0.5, "adaptation_rate": 0.1},
        )
        ctrl = LogMPPIController(mppi_params=params, seed=42)

        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ref[:, 0] = np.linspace(0, 2, 11)

        temps = []
        for _ in range(5):
            _, info = ctrl.compute_control(state, ref)
            temps.append(info["temperature"])

        assert all(t > 0 for t in temps)
        assert all(np.isfinite(t) for t in temps)


# ─────────────────────────────────────────────────────────────
# 장애물 회피
# ─────────────────────────────────────────────────────────────

class TestLogMPPIObstacle:
    """Log-MPPI 장애물 회피."""

    def test_obstacle_avoidance(self):
        """장애물이 있을 때 정상 동작."""
        obstacles = np.array([[1.5, 0.0, 0.3]])
        params = MPPIParams(N=15, K=256, dt=0.05)
        ctrl = LogMPPIController(
            mppi_params=params, seed=42, obstacles=obstacles,
        )

        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((16, 3))
        ref[:, 0] = np.linspace(0, 3, 16)

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert info["cost"] >= 0


# ─────────────────────────────────────────────────────────────
# Reset
# ─────────────────────────────────────────────────────────────

class TestLogMPPIReset:
    """Log-MPPI 리셋."""

    def test_reset(self):
        ctrl = LogMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1),
            seed=42,
        )
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ctrl.compute_control(state, ref)

        ctrl.reset()
        assert ctrl._iteration_count == 0
        np.testing.assert_array_equal(ctrl.U, np.zeros((10, 2)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
