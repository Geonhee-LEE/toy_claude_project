"""Risk-Aware MPPI 컨트롤러 테스트 — CVaR 가중치 절단 검증."""

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
from mpc_controller.controllers.mppi.risk_aware_mppi import RiskAwareMPPIController
from mpc_controller.controllers.mppi.utils import (
    softmax_weights,
    effective_sample_size,
)


# ─────────────────────────────────────────────────────────────
# 기본 초기화 및 인터페이스
# ─────────────────────────────────────────────────────────────

class TestRiskAwareMPPIInit:
    """RiskAwareMPPIController 초기화 및 상속 검증."""

    def test_creation(self):
        """기본 생성 확인."""
        ctrl = RiskAwareMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, cvar_alpha=0.5),
            seed=42,
        )
        assert ctrl is not None
        assert ctrl.params.cvar_alpha == 0.5

    def test_inherits_mppi(self):
        """MPPIController 상속 확인."""
        ctrl = RiskAwareMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, cvar_alpha=0.5),
            seed=42,
        )
        assert isinstance(ctrl, MPPIController)

    def test_interface_compatible(self):
        """compute_control 인터페이스 호환."""
        ctrl = RiskAwareMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, cvar_alpha=0.5),
            seed=42,
        )
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        control, info = ctrl.compute_control(state, ref)

        assert control.shape == (2,)
        assert "sample_weights" in info
        assert "ess" in info


# ─────────────────────────────────────────────────────────────
# 가중치 기본 속성
# ─────────────────────────────────────────────────────────────

class TestRiskAwareWeights:
    """가중치 기본 속성 검증."""

    def test_weights_sum_to_one(self):
        """가중치 합 = 1."""
        ctrl = RiskAwareMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, cvar_alpha=0.5),
            seed=42,
        )
        costs = np.random.default_rng(0).uniform(1, 100, size=64)
        weights = ctrl._compute_weights(costs)
        np.testing.assert_almost_equal(np.sum(weights), 1.0)

    def test_weights_non_negative(self):
        """가중치 >= 0."""
        ctrl = RiskAwareMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, cvar_alpha=0.5),
            seed=42,
        )
        costs = np.random.default_rng(0).uniform(1, 100, size=64)
        weights = ctrl._compute_weights(costs)
        assert np.all(weights >= 0)

    def test_lower_cost_higher_weight(self):
        """비용 낮을수록 가중치 높음."""
        ctrl = RiskAwareMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, cvar_alpha=0.5),
            seed=42,
        )
        costs = np.array([1.0, 100.0])
        weights = ctrl._compute_weights(costs)
        assert weights[0] > weights[1]

    def test_zero_weight_for_truncated(self):
        """alpha=0.5 → 절반의 비용 높은 샘플 가중치 = 0."""
        ctrl = RiskAwareMPPIController(
            mppi_params=MPPIParams(N=10, K=100, dt=0.1, cvar_alpha=0.5),
            seed=42,
        )
        costs = np.arange(100, dtype=float)  # 0~99
        weights = ctrl._compute_weights(costs)

        n_keep = 50  # ceil(0.5 * 100) = 50
        n_zero = np.sum(weights == 0.0)
        assert n_zero == 100 - n_keep, f"Expected {100 - n_keep} zeros, got {n_zero}"


# ─────────────────────────────────────────────────────────────
# alpha=1.0 Vanilla 동등성
# ─────────────────────────────────────────────────────────────

class TestRiskAwareVanillaEquiv:
    """alpha=1.0일 때 Vanilla MPPI와 동일."""

    def test_weights_alpha1_match_vanilla(self):
        """alpha=1.0 가중치가 Vanilla와 동일."""
        params = MPPIParams(N=10, K=64, dt=0.1, lambda_=10.0, cvar_alpha=1.0)
        vanilla = MPPIController(mppi_params=params, seed=42)
        ra = RiskAwareMPPIController(mppi_params=params, seed=42)

        costs = np.random.default_rng(0).uniform(1.0, 100.0, size=64)
        w_v = vanilla._compute_weights(costs)
        w_ra = ra._compute_weights(costs)

        np.testing.assert_allclose(w_v, w_ra, atol=1e-10)

    def test_control_alpha1_match_vanilla(self):
        """alpha=1.0 제어 출력이 Vanilla와 동일."""
        params = MPPIParams(N=10, K=128, dt=0.1, lambda_=10.0, cvar_alpha=1.0)
        vanilla = MPPIController(mppi_params=params, seed=42)
        ra = RiskAwareMPPIController(mppi_params=params, seed=42)

        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ref[:, 0] = np.linspace(0, 2, 11)

        u_v, _ = vanilla.compute_control(state, ref)
        u_ra, _ = ra.compute_control(state, ref)

        np.testing.assert_allclose(u_v, u_ra, atol=1e-6)


# ─────────────────────────────────────────────────────────────
# CVaR 절단 동작
# ─────────────────────────────────────────────────────────────

class TestRiskAwareTruncation:
    """CVaR 절단 메커니즘 검증."""

    def test_alpha_50_keeps_half(self):
        """alpha=0.5 → 50%만 유지."""
        ctrl = RiskAwareMPPIController(
            mppi_params=MPPIParams(N=10, K=100, dt=0.1, cvar_alpha=0.5),
            seed=42,
        )
        costs = np.random.default_rng(0).uniform(0, 100, size=100)
        weights = ctrl._compute_weights(costs)

        n_nonzero = np.count_nonzero(weights)
        assert n_nonzero == 50, f"Expected 50 nonzero, got {n_nonzero}"

    def test_alpha_30_keeps_30pct(self):
        """alpha=0.3 → 30%만 유지."""
        ctrl = RiskAwareMPPIController(
            mppi_params=MPPIParams(N=10, K=100, dt=0.1, cvar_alpha=0.3),
            seed=42,
        )
        costs = np.random.default_rng(0).uniform(0, 100, size=100)
        weights = ctrl._compute_weights(costs)

        n_nonzero = np.count_nonzero(weights)
        assert n_nonzero == 30, f"Expected 30 nonzero, got {n_nonzero}"

    def test_min_one_sample(self):
        """alpha가 매우 작아도 최소 1개 샘플 유지."""
        ctrl = RiskAwareMPPIController(
            mppi_params=MPPIParams(N=10, K=100, dt=0.1, cvar_alpha=0.001),
            seed=42,
        )
        costs = np.random.default_rng(0).uniform(0, 100, size=100)
        weights = ctrl._compute_weights(costs)

        n_nonzero = np.count_nonzero(weights)
        assert n_nonzero >= 1

    def test_selects_lowest_costs(self):
        """절단 후 남은 샘플이 최저 비용 샘플임을 확인."""
        ctrl = RiskAwareMPPIController(
            mppi_params=MPPIParams(N=10, K=10, dt=0.1, cvar_alpha=0.5),
            seed=42,
        )
        costs = np.array([9, 3, 7, 1, 5, 8, 2, 6, 4, 10], dtype=float)
        weights = ctrl._compute_weights(costs)

        # n_keep = ceil(0.5 * 10) = 5
        # 최저 5개 비용: 1, 2, 3, 4, 5 → 인덱스 3, 6, 1, 8, 4
        nonzero_idx = set(np.where(weights > 0)[0])
        lowest_idx = set(np.argpartition(costs, 5)[:5])
        assert nonzero_idx == lowest_idx


# ─────────────────────────────────────────────────────────────
# ESS 행동
# ─────────────────────────────────────────────────────────────

class TestRiskAwareESSBehavior:
    """alpha에 따른 ESS 변화 검증."""

    def test_lower_alpha_lower_ess(self):
        """alpha가 작을수록 ESS가 작음."""
        costs = np.random.default_rng(0).uniform(1.0, 100.0, size=256)

        params_a10 = MPPIParams(N=10, K=256, dt=0.1, lambda_=10.0, cvar_alpha=1.0)
        params_a05 = MPPIParams(N=10, K=256, dt=0.1, lambda_=10.0, cvar_alpha=0.5)

        ctrl_a10 = RiskAwareMPPIController(mppi_params=params_a10, seed=42)
        ctrl_a05 = RiskAwareMPPIController(mppi_params=params_a05, seed=42)

        w_a10 = ctrl_a10._compute_weights(costs)
        w_a05 = ctrl_a05._compute_weights(costs)

        ess_a10 = effective_sample_size(w_a10)
        ess_a05 = effective_sample_size(w_a05)

        assert ess_a05 < ess_a10, (
            f"ESS(alpha=0.5)={ess_a05:.1f} should < ESS(alpha=1.0)={ess_a10:.1f}"
        )

    def test_ess_leq_n_keep(self):
        """ESS <= n_keep (활성 샘플 수 이하)."""
        K = 256
        alpha = 0.3
        n_keep = int(np.ceil(alpha * K))  # 77

        ctrl = RiskAwareMPPIController(
            mppi_params=MPPIParams(N=10, K=K, dt=0.1, lambda_=10.0, cvar_alpha=alpha),
            seed=42,
        )
        costs = np.random.default_rng(0).uniform(1.0, 100.0, size=K)
        weights = ctrl._compute_weights(costs)
        ess = effective_sample_size(weights)

        assert ess <= n_keep + 1, (
            f"ESS={ess:.1f} should <= n_keep={n_keep}"
        )


# ─────────────────────────────────────────────────────────────
# 궤적 추적 성능
# ─────────────────────────────────────────────────────────────

class TestRiskAwareTracking:
    """Risk-Aware MPPI 궤적 추적 성능."""

    def test_circle_tracking(self):
        """alpha=0.5로 원형 궤적 추적 RMSE < 0.3m."""
        params = MPPIParams(N=20, K=512, dt=0.05, lambda_=10.0, cvar_alpha=0.5)
        ctrl = RiskAwareMPPIController(mppi_params=params, seed=42)

        trajectory = generate_circle_trajectory(
            center=np.array([0.0, 0.0]), radius=2.0, num_points=200,
        )
        interpolator = TrajectoryInterpolator(trajectory, dt=0.05)
        state = np.array([2.0, 0.0, np.pi / 2])
        model = DifferentialDriveModel()

        errors = []
        for i in range(100):
            ref = interpolator.get_reference(
                i * 0.05, params.N, params.dt, current_theta=state[2]
            )
            control, _ = ctrl.compute_control(state, ref)
            state = model.forward_simulate(state, control, 0.05)
            dist = np.sqrt(state[0] ** 2 + state[1] ** 2)
            errors.append(abs(dist - 2.0))

        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        assert rmse < 0.3, f"Risk-Aware alpha=0.5 circle RMSE = {rmse:.4f} (> 0.3m)"


# ─────────────────────────────────────────────────────────────
# Adaptive Temperature 연동
# ─────────────────────────────────────────────────────────────

class TestRiskAwareAdaptiveTemp:
    def test_adaptive_temp_integration(self):
        """adaptive temperature와 CVaR 동시 사용."""
        params = MPPIParams(
            N=10, K=64, dt=0.1, cvar_alpha=0.5,
            adaptive_temperature=True,
            adaptive_temp_config={"target_ess_ratio": 0.5, "adaptation_rate": 0.1},
        )
        ctrl = RiskAwareMPPIController(mppi_params=params, seed=42)

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

class TestRiskAwareObstacle:
    def test_obstacle_avoidance(self):
        """장애물 존재 시 정상 동작."""
        obstacles = np.array([[1.5, 0.0, 0.3]])
        params = MPPIParams(N=15, K=256, dt=0.05, cvar_alpha=0.5)
        ctrl = RiskAwareMPPIController(
            mppi_params=params, seed=42, obstacles=obstacles,
        )

        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((16, 3))
        ref[:, 0] = np.linspace(0, 3, 16)

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert info["cost"] >= 0

    def test_risk_averse_more_conservative(self):
        """risk-averse(alpha=0.3)가 neutral(alpha=1.0)보다 장애물 회피에 보수적.

        장애물 쪽으로 가는 고비용 샘플을 잘라내므로
        risk-averse 컨트롤러의 가중 평균 궤적이 장애물에서 더 멀어야 한다.
        """
        obstacles = np.array([[1.5, 0.0, 0.3]])
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((16, 3))
        ref[:, 0] = np.linspace(0, 3, 16)

        # risk-neutral (alpha=1.0)
        params_neutral = MPPIParams(N=15, K=512, dt=0.05, lambda_=10.0, cvar_alpha=1.0)
        ctrl_neutral = RiskAwareMPPIController(
            mppi_params=params_neutral, seed=42, obstacles=obstacles,
        )
        _, info_neutral = ctrl_neutral.compute_control(state, ref)

        # risk-averse (alpha=0.3)
        params_averse = MPPIParams(N=15, K=512, dt=0.05, lambda_=10.0, cvar_alpha=0.3)
        ctrl_averse = RiskAwareMPPIController(
            mppi_params=params_averse, seed=42, obstacles=obstacles,
        )
        _, info_averse = ctrl_averse.compute_control(state, ref)

        # 가중 평균 궤적에서 장애물까지 최소 거리 비교
        traj_neutral = info_neutral["predicted_trajectory"]
        traj_averse = info_averse["predicted_trajectory"]

        obs_pos = obstacles[0, :2]
        min_dist_neutral = np.min(np.linalg.norm(traj_neutral[:, :2] - obs_pos, axis=1))
        min_dist_averse = np.min(np.linalg.norm(traj_averse[:, :2] - obs_pos, axis=1))

        assert min_dist_averse >= min_dist_neutral * 0.9, (
            f"Risk-averse min_dist={min_dist_averse:.3f} should >= "
            f"neutral min_dist={min_dist_neutral:.3f} * 0.9"
        )


# ─────────────────────────────────────────────────────────────
# 엣지 케이스
# ─────────────────────────────────────────────────────────────

class TestRiskAwareEdgeCases:
    """경계값 및 기본값 테스트."""

    def test_default_alpha_is_one(self):
        """cvar_alpha 기본값이 1.0."""
        params = MPPIParams()
        assert params.cvar_alpha == 1.0

    def test_boundary_alpha_values(self):
        """극단 alpha 값에서도 안정 동작."""
        for alpha in [0.01, 0.1, 0.5, 0.99, 1.0]:
            ctrl = RiskAwareMPPIController(
                mppi_params=MPPIParams(N=10, K=64, dt=0.1, cvar_alpha=alpha),
                seed=42,
            )
            costs = np.random.default_rng(0).uniform(1, 100, 64)
            weights = ctrl._compute_weights(costs)
            assert np.all(np.isfinite(weights))
            np.testing.assert_almost_equal(np.sum(weights), 1.0)


# ─────────────────────────────────────────────────────────────
# Reset
# ─────────────────────────────────────────────────────────────

class TestRiskAwareReset:
    def test_reset(self):
        """reset 후 제어열 초기화."""
        ctrl = RiskAwareMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, cvar_alpha=0.5),
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
