"""Tsallis-MPPI 컨트롤러 테스트 — q-exponential 가중치 및 하위 호환성 검증."""

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
from mpc_controller.controllers.mppi.tsallis_mppi import TsallisMPPIController
from mpc_controller.controllers.mppi.utils import (
    q_exponential,
    q_logarithm,
    effective_sample_size,
)


# ─────────────────────────────────────────────────────────────
# q-exponential / q-logarithm 수학 함수
# ─────────────────────────────────────────────────────────────

class TestQExponential:
    """q_exponential 수학적 정확성."""

    def test_q1_equals_exp(self):
        """q=1.0일 때 표준 exp와 동일."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = q_exponential(x, q=1.0)
        expected = np.exp(x)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_q2_known_values(self):
        """q=2.0일 때 알려진 값: exp_2(x) = [1-x]_+^{-1} = 1/(1-x) if x<1."""
        x = np.array([0.0, 0.5, -1.0])
        result = q_exponential(x, q=2.0)
        # q=2: exp_q(x) = [1 + (1-2)*x]^{1/(1-2)} = [1-x]^{-1}
        expected = np.array([1.0, 1.0 / 0.5, 1.0 / 2.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_q05_known_values(self):
        """q=0.5일 때 알려진 값."""
        x = np.array([0.0, 1.0, 2.0])
        result = q_exponential(x, q=0.5)
        # q=0.5: exp_q(x) = [1 + 0.5*x]^{2}
        expected = np.array([1.0, (1.5) ** 2, (2.0) ** 2])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_negative_base_clipped(self):
        """base가 음수이면 0으로 클리핑."""
        # q=2: base = 1 + (1-2)*x = 1-x; x=2 → base=-1 → 0
        result = q_exponential(np.array([2.0]), q=2.0)
        assert result[0] == 0.0

    def test_q0_exp(self):
        """q=0일 때: exp_0(x) = [1+x]_+."""
        x = np.array([-0.5, 0.0, 1.0, 2.0])
        result = q_exponential(x, q=0.0)
        # q=0: exp_q(x) = [1+(1-0)*x]^{1/(1-0)} = (1+x)
        expected = np.maximum(1.0 + x, 0.0)
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestQLogarithm:
    """q_logarithm 수학적 정확성."""

    def test_q1_equals_log(self):
        """q=1.0일 때 표준 log와 동일."""
        x = np.array([0.5, 1.0, 2.0, 10.0])
        result = q_logarithm(x, q=1.0)
        expected = np.log(x)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_inverse_of_q_exp(self):
        """ln_q(exp_q(x)) ≈ x (역함수 관계)."""
        x = np.array([-1.0, 0.0, 0.5, 1.0])
        for q in [0.5, 1.0, 1.5]:
            y = q_exponential(x, q)
            # 양수 값만 ln_q 적용
            mask = y > 0
            recovered = q_logarithm(y[mask], q)
            np.testing.assert_allclose(recovered, x[mask], atol=1e-8)


# ─────────────────────────────────────────────────────────────
# 기본 초기화 및 인터페이스
# ─────────────────────────────────────────────────────────────

class TestTsallisMPPIInit:
    """TsallisMPPIController 초기화."""

    def test_creation(self):
        ctrl = TsallisMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, tsallis_q=1.5),
            seed=42,
        )
        assert ctrl is not None
        assert isinstance(ctrl, MPPIController)
        assert ctrl.params.tsallis_q == 1.5

    def test_interface_compatible(self):
        ctrl = TsallisMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, tsallis_q=1.5),
            seed=42,
        )
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        control, info = ctrl.compute_control(state, ref)

        assert control.shape == (2,)
        assert "sample_weights" in info
        assert "ess" in info


# ─────────────────────────────────────────────────────────────
# q=1.0 하위 호환성 (Vanilla 동등성)
# ─────────────────────────────────────────────────────────────

class TestTsallisVanillaEquivalence:
    """q=1.0일 때 Vanilla MPPI와 동일 결과."""

    def test_weights_q1_match_vanilla(self):
        """q=1.0 가중치가 Vanilla와 차이 < 1e-8."""
        params = MPPIParams(N=10, K=64, dt=0.1, lambda_=10.0, tsallis_q=1.0)
        vanilla = MPPIController(mppi_params=params, seed=42)
        tsallis = TsallisMPPIController(mppi_params=params, seed=42)

        costs = np.random.default_rng(0).uniform(1.0, 100.0, size=64)
        w_v = vanilla._compute_weights(costs)
        w_t = tsallis._compute_weights(costs)

        np.testing.assert_allclose(w_v, w_t, atol=1e-8)

    def test_control_q1_match_vanilla(self):
        """q=1.0 제어 출력이 Vanilla와 거의 동일."""
        params = MPPIParams(N=10, K=128, dt=0.1, lambda_=10.0, tsallis_q=1.0)
        vanilla = MPPIController(mppi_params=params, seed=42)
        tsallis = TsallisMPPIController(mppi_params=params, seed=42)

        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ref[:, 0] = np.linspace(0, 2, 11)

        u_v, _ = vanilla.compute_control(state, ref)
        u_t, _ = tsallis.compute_control(state, ref)

        np.testing.assert_allclose(u_v, u_t, atol=1e-6)


# ─────────────────────────────────────────────────────────────
# ESS 분포 특성 (q>1 탐색↑, q<1 집중↑)
# ─────────────────────────────────────────────────────────────

class TestTsallisESSBehavior:
    """q 파라미터에 따른 ESS 변화 검증."""

    def test_q_gt1_increases_ess(self):
        """q>1 (heavy-tail)이면 ESS가 q=1보다 높음 (탐색 확대)."""
        costs = np.random.default_rng(0).uniform(1.0, 100.0, size=256)

        params_q1 = MPPIParams(N=10, K=256, dt=0.1, lambda_=10.0, tsallis_q=1.0)
        params_q15 = MPPIParams(N=10, K=256, dt=0.1, lambda_=10.0, tsallis_q=1.5)

        ctrl_q1 = TsallisMPPIController(mppi_params=params_q1, seed=42)
        ctrl_q15 = TsallisMPPIController(mppi_params=params_q15, seed=42)

        w_q1 = ctrl_q1._compute_weights(costs)
        w_q15 = ctrl_q15._compute_weights(costs)

        ess_q1 = effective_sample_size(w_q1)
        ess_q15 = effective_sample_size(w_q15)

        assert ess_q15 > ess_q1, (
            f"ESS(q=1.5)={ess_q15:.1f} should > ESS(q=1.0)={ess_q1:.1f}"
        )

    def test_q_lt1_decreases_ess(self):
        """q<1 (light-tail)이면 ESS가 q=1보다 낮음 (집중)."""
        costs = np.random.default_rng(0).uniform(1.0, 100.0, size=256)

        params_q1 = MPPIParams(N=10, K=256, dt=0.1, lambda_=10.0, tsallis_q=1.0)
        params_q05 = MPPIParams(N=10, K=256, dt=0.1, lambda_=10.0, tsallis_q=0.5)

        ctrl_q1 = TsallisMPPIController(mppi_params=params_q1, seed=42)
        ctrl_q05 = TsallisMPPIController(mppi_params=params_q05, seed=42)

        w_q1 = ctrl_q1._compute_weights(costs)
        w_q05 = ctrl_q05._compute_weights(costs)

        ess_q1 = effective_sample_size(w_q1)
        ess_q05 = effective_sample_size(w_q05)

        assert ess_q05 < ess_q1, (
            f"ESS(q=0.5)={ess_q05:.1f} should < ESS(q=1.0)={ess_q1:.1f}"
        )


# ─────────────────────────────────────────────────────────────
# 가중치 기본 속성
# ─────────────────────────────────────────────────────────────

class TestTsallisWeights:
    """가중치 기본 속성 검증."""

    def test_weights_sum_to_one(self):
        ctrl = TsallisMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, tsallis_q=1.5),
            seed=42,
        )
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ref[:, 0] = np.linspace(0, 2, 11)

        _, info = ctrl.compute_control(state, ref)
        np.testing.assert_almost_equal(np.sum(info["sample_weights"]), 1.0)

    def test_weights_non_negative(self):
        ctrl = TsallisMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, tsallis_q=1.5),
            seed=42,
        )
        costs = np.random.default_rng(0).uniform(0, 100, size=64)
        weights = ctrl._compute_weights(costs)
        assert np.all(weights >= 0)

    def test_lower_cost_higher_weight(self):
        ctrl = TsallisMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, tsallis_q=1.5),
            seed=42,
        )
        costs = np.array([1.0, 100.0])
        weights = ctrl._compute_weights(costs)
        assert weights[0] > weights[1]


# ─────────────────────────────────────────────────────────────
# 궤적 추적 성능
# ─────────────────────────────────────────────────────────────

class TestTsallisTracking:
    """Tsallis-MPPI 궤적 추적 성능."""

    def test_circle_tracking_q15(self):
        """q=1.5로 원형 궤적 RMSE < 0.2m."""
        params = MPPIParams(N=20, K=512, dt=0.05, lambda_=10.0, tsallis_q=1.5)
        ctrl = TsallisMPPIController(mppi_params=params, seed=42)

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
        assert rmse < 0.2, f"Tsallis q=1.5 circle RMSE = {rmse:.4f} (> 0.2m)"

    def test_circle_tracking_q05(self):
        """q=0.5로 원형 궤적 RMSE < 0.2m."""
        params = MPPIParams(N=20, K=512, dt=0.05, lambda_=10.0, tsallis_q=0.5)
        ctrl = TsallisMPPIController(mppi_params=params, seed=42)

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
        assert rmse < 0.2, f"Tsallis q=0.5 circle RMSE = {rmse:.4f} (> 0.2m)"


# ─────────────────────────────────────────────────────────────
# Adaptive Temperature 연동
# ─────────────────────────────────────────────────────────────

class TestTsallisAdaptiveTemp:
    def test_adaptive_temp_integration(self):
        params = MPPIParams(
            N=10, K=64, dt=0.1, tsallis_q=1.5,
            adaptive_temperature=True,
            adaptive_temp_config={"target_ess_ratio": 0.5, "adaptation_rate": 0.1},
        )
        ctrl = TsallisMPPIController(mppi_params=params, seed=42)

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

class TestTsallisObstacle:
    def test_obstacle_avoidance(self):
        obstacles = np.array([[1.5, 0.0, 0.3]])
        params = MPPIParams(N=15, K=256, dt=0.05, tsallis_q=1.5)
        ctrl = TsallisMPPIController(
            mppi_params=params, seed=42, obstacles=obstacles,
        )

        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((16, 3))
        ref[:, 0] = np.linspace(0, 3, 16)

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert info["cost"] >= 0


# ─────────────────────────────────────────────────────────────
# 파라미터 경계값
# ─────────────────────────────────────────────────────────────

class TestTsallisEdgeCases:
    """경계값 q 파라미터 테스트."""

    def test_q0(self):
        """q=0 (극단 light-tail)에서도 동작."""
        params = MPPIParams(N=10, K=64, dt=0.1, tsallis_q=0.0)
        ctrl = TsallisMPPIController(mppi_params=params, seed=42)
        costs = np.random.default_rng(0).uniform(1, 10, 64)
        weights = ctrl._compute_weights(costs)
        assert np.all(np.isfinite(weights))
        np.testing.assert_almost_equal(np.sum(weights), 1.0)

    def test_q2(self):
        """q=2 (strong heavy-tail)에서도 동작."""
        params = MPPIParams(N=10, K=64, dt=0.1, tsallis_q=2.0)
        ctrl = TsallisMPPIController(mppi_params=params, seed=42)
        costs = np.random.default_rng(0).uniform(1, 10, 64)
        weights = ctrl._compute_weights(costs)
        assert np.all(np.isfinite(weights))
        np.testing.assert_almost_equal(np.sum(weights), 1.0)

    def test_default_q1(self):
        """tsallis_q 기본값이 1.0."""
        params = MPPIParams()
        assert params.tsallis_q == 1.0


# ─────────────────────────────────────────────────────────────
# Reset
# ─────────────────────────────────────────────────────────────

class TestTsallisReset:
    def test_reset(self):
        ctrl = TsallisMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, tsallis_q=1.5),
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
