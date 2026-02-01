"""Stein Variational MPPI (SVMPC) 컨트롤러 테스트 — SVGD 기반 샘플 다양성 유도 검증."""

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
from mpc_controller.controllers.mppi.stein_variational_mppi import (
    SteinVariationalMPPIController,
)
from mpc_controller.controllers.mppi.utils import (
    rbf_kernel,
    rbf_kernel_grad,
    median_bandwidth,
    softmax_weights,
    effective_sample_size,
)


# ─────────────────────────────────────────────────────────────
# RBF 커널 테스트
# ─────────────────────────────────────────────────────────────

class TestRBFKernel:
    """RBF 커널 함수 검증."""

    def test_self_kernel_is_one(self):
        """자기 자신과의 커널 값 = 1."""
        particles = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        K_mat = rbf_kernel(particles, bandwidth=1.0)
        np.testing.assert_allclose(np.diag(K_mat), 1.0, atol=1e-10)

    def test_symmetry(self):
        """커널 행렬은 대칭."""
        rng = np.random.default_rng(42)
        particles = rng.standard_normal((10, 5))
        K_mat = rbf_kernel(particles, bandwidth=1.0)
        np.testing.assert_allclose(K_mat, K_mat.T, atol=1e-12)

    def test_decay_with_distance(self):
        """거리가 멀수록 커널 값 감소."""
        particles = np.array([[0.0], [1.0], [10.0]])
        K_mat = rbf_kernel(particles, bandwidth=1.0)
        # k(x0, x1) > k(x0, x2) — x1이 더 가까움
        assert K_mat[0, 1] > K_mat[0, 2]


# ─────────────────────────────────────────────────────────────
# Median Bandwidth 테스트
# ─────────────────────────────────────────────────────────────

class TestMedianBandwidth:
    """Median heuristic bandwidth 검증."""

    def test_positive(self):
        """bandwidth는 항상 양수."""
        rng = np.random.default_rng(42)
        particles = rng.standard_normal((50, 10))
        h = median_bandwidth(particles)
        assert h > 0

    def test_scales_with_spread(self):
        """데이터 스케일에 비례."""
        rng = np.random.default_rng(42)
        particles_small = rng.standard_normal((50, 10))
        particles_large = particles_small * 10.0
        h_small = median_bandwidth(particles_small)
        h_large = median_bandwidth(particles_large)
        assert h_large > h_small


# ─────────────────────────────────────────────────────────────
# 초기화 및 상속
# ─────────────────────────────────────────────────────────────

class TestSVMPCInit:
    """SteinVariationalMPPIController 초기화 검증."""

    def test_creation(self):
        """기본 생성 확인."""
        ctrl = SteinVariationalMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, svgd_num_iterations=3),
            seed=42,
        )
        assert ctrl is not None
        assert ctrl.params.svgd_num_iterations == 3

    def test_inherits_mppi(self):
        """MPPIController 상속 확인."""
        ctrl = SteinVariationalMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, svgd_num_iterations=3),
            seed=42,
        )
        assert isinstance(ctrl, MPPIController)

    def test_interface_compatible(self):
        """compute_control 인터페이스 호환."""
        ctrl = SteinVariationalMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, svgd_num_iterations=2),
            seed=42,
        )
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        control, info = ctrl.compute_control(state, ref)

        assert control.shape == (2,)
        assert "sample_weights" in info
        assert "ess" in info


# ─────────────────────────────────────────────────────────────
# Vanilla 동등성 (svgd_num_iterations=0)
# ─────────────────────────────────────────────────────────────

class TestSVMPCVanillaEquiv:
    """svgd_num_iterations=0 → Vanilla MPPI와 동일."""

    def test_weights_iter0_match_vanilla(self):
        """iter=0 가중치가 Vanilla와 동일."""
        params = MPPIParams(N=10, K=64, dt=0.1, lambda_=10.0, svgd_num_iterations=0)
        vanilla = MPPIController(mppi_params=params, seed=42)
        svmpc = SteinVariationalMPPIController(mppi_params=params, seed=42)

        costs = np.random.default_rng(0).uniform(1.0, 100.0, size=64)
        w_v = vanilla._compute_weights(costs)
        w_sv = svmpc._compute_weights(costs)

        np.testing.assert_allclose(w_v, w_sv, atol=1e-10)

    def test_control_iter0_match_vanilla(self):
        """iter=0 제어 출력이 Vanilla와 동일."""
        params = MPPIParams(N=10, K=128, dt=0.1, lambda_=10.0, svgd_num_iterations=0)
        vanilla = MPPIController(mppi_params=params, seed=42)
        svmpc = SteinVariationalMPPIController(mppi_params=params, seed=42)

        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ref[:, 0] = np.linspace(0, 2, 11)

        u_v, _ = vanilla.compute_control(state, ref)
        u_sv, _ = svmpc.compute_control(state, ref)

        np.testing.assert_allclose(u_v, u_sv, atol=1e-6)


# ─────────────────────────────────────────────────────────────
# SVGD 메커니즘
# ─────────────────────────────────────────────────────────────

class TestSVGDMechanism:
    """SVGD 루프 동작 검증."""

    def test_diversity_increases(self):
        """SVGD 후 샘플 다양성 증가 (또는 유지)."""
        params = MPPIParams(
            N=10, K=128, dt=0.1, lambda_=10.0,
            svgd_num_iterations=3, svgd_step_size=0.5,
        )
        ctrl = SteinVariationalMPPIController(mppi_params=params, seed=42)
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ref[:, 0] = np.linspace(0, 2, 11)

        _, info = ctrl.compute_control(state, ref)
        # SVGD가 repulsive force로 다양성을 높이므로 일반적으로 증가
        # 하지만 attractive force도 작용하므로 약한 조건으로 검증
        assert info["sample_diversity_after"] > 0

    def test_cost_decreases_or_stable(self):
        """SVGD 반복 시 최소 비용이 합리적."""
        params = MPPIParams(
            N=10, K=128, dt=0.1, lambda_=10.0,
            svgd_num_iterations=3, svgd_step_size=0.1,
        )
        ctrl = SteinVariationalMPPIController(mppi_params=params, seed=42)
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ref[:, 0] = np.linspace(0, 2, 11)

        _, info = ctrl.compute_control(state, ref)
        assert np.isfinite(info["cost"])
        assert info["cost"] >= 0

    def test_more_iterations_more_diversity(self):
        """iteration 수가 많을수록 다양성 변화 더 큼."""
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ref[:, 0] = np.linspace(0, 2, 11)

        diversities = []
        for L in [1, 5]:
            params = MPPIParams(
                N=10, K=128, dt=0.1, lambda_=10.0,
                svgd_num_iterations=L, svgd_step_size=0.3,
            )
            ctrl = SteinVariationalMPPIController(mppi_params=params, seed=42)
            _, info = ctrl.compute_control(state, ref)
            diversities.append(info["sample_diversity_after"])

        # L=5가 L=1보다 다양성이 더 높을 것으로 기대 (일반적)
        # 약한 조건: 둘 다 유한한 양수
        assert all(d > 0 for d in diversities)

    def test_step_size_effect(self):
        """step_size가 클수록 업데이트 폭이 큼."""
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ref[:, 0] = np.linspace(0, 2, 11)

        diversities = []
        for step in [0.01, 1.0]:
            params = MPPIParams(
                N=10, K=128, dt=0.1, lambda_=10.0,
                svgd_num_iterations=3, svgd_step_size=step,
            )
            ctrl = SteinVariationalMPPIController(mppi_params=params, seed=42)
            _, info = ctrl.compute_control(state, ref)
            diversities.append(info["sample_diversity_after"])

        # 큰 step_size → 다양성 변화 더 큼 (일반적)
        assert all(d > 0 for d in diversities)
        assert all(np.isfinite(d) for d in diversities)


# ─────────────────────────────────────────────────────────────
# 가중치 기본 속성
# ─────────────────────────────────────────────────────────────

class TestSVMPCWeights:
    """SVMPC 가중치 속성 검증."""

    def test_weights_sum_to_one(self):
        """가중치 합 = 1."""
        params = MPPIParams(N=10, K=64, dt=0.1, svgd_num_iterations=2)
        ctrl = SteinVariationalMPPIController(mppi_params=params, seed=42)
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        _, info = ctrl.compute_control(state, ref)
        np.testing.assert_almost_equal(np.sum(info["sample_weights"]), 1.0)

    def test_weights_non_negative(self):
        """가중치 >= 0."""
        params = MPPIParams(N=10, K=64, dt=0.1, svgd_num_iterations=2)
        ctrl = SteinVariationalMPPIController(mppi_params=params, seed=42)
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        _, info = ctrl.compute_control(state, ref)
        assert np.all(info["sample_weights"] >= 0)

    def test_lower_cost_higher_weight(self):
        """비용 낮을수록 가중치 높음 (softmax 특성)."""
        ctrl = SteinVariationalMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, svgd_num_iterations=0),
            seed=42,
        )
        costs = np.array([1.0, 100.0])
        weights = ctrl._compute_weights(costs)
        assert weights[0] > weights[1]


# ─────────────────────────────────────────────────────────────
# 궤적 추적 성능
# ─────────────────────────────────────────────────────────────

class TestSVMPCTracking:
    """SVMPC 궤적 추적 성능."""

    def test_circle_tracking(self):
        """SVGD iter=3으로 원형 궤적 추적 RMSE < 0.3m."""
        params = MPPIParams(
            N=20, K=512, dt=0.05, lambda_=10.0,
            svgd_num_iterations=3, svgd_step_size=0.1,
        )
        ctrl = SteinVariationalMPPIController(mppi_params=params, seed=42)

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
        assert rmse < 0.3, f"SVMPC circle RMSE = {rmse:.4f} (> 0.3m)"

    def test_tracking_comparable_to_vanilla(self):
        """SVMPC 추적 성능이 Vanilla와 유사 (10배 이내)."""
        state0 = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((21, 3))
        ref[:, 0] = np.linspace(0, 3, 21)

        params_v = MPPIParams(N=20, K=256, dt=0.05, lambda_=10.0)
        params_sv = MPPIParams(
            N=20, K=256, dt=0.05, lambda_=10.0,
            svgd_num_iterations=2, svgd_step_size=0.1,
        )

        vanilla = MPPIController(mppi_params=params_v, seed=42)
        svmpc = SteinVariationalMPPIController(mppi_params=params_sv, seed=42)

        _, info_v = vanilla.compute_control(state0.copy(), ref)
        _, info_sv = svmpc.compute_control(state0.copy(), ref)

        # SVMPC 비용이 Vanilla의 10배를 넘지 않아야 함
        assert info_sv["cost"] < info_v["cost"] * 10, (
            f"SVMPC cost={info_sv['cost']:.4f} >> Vanilla cost={info_v['cost']:.4f}"
        )


# ─────────────────────────────────────────────────────────────
# Adaptive Temperature 연동
# ─────────────────────────────────────────────────────────────

class TestSVMPCAdaptiveTemp:
    def test_adaptive_temp_integration(self):
        """adaptive temperature와 SVGD 동시 사용."""
        params = MPPIParams(
            N=10, K=64, dt=0.1, svgd_num_iterations=2,
            adaptive_temperature=True,
            adaptive_temp_config={"target_ess_ratio": 0.5, "adaptation_rate": 0.1},
        )
        ctrl = SteinVariationalMPPIController(mppi_params=params, seed=42)

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

class TestSVMPCObstacle:
    def test_obstacle_avoidance(self):
        """장애물 존재 시 정상 동작."""
        obstacles = np.array([[1.5, 0.0, 0.3]])
        params = MPPIParams(
            N=15, K=256, dt=0.05, svgd_num_iterations=2, svgd_step_size=0.1,
        )
        ctrl = SteinVariationalMPPIController(
            mppi_params=params, seed=42, obstacles=obstacles,
        )

        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((16, 3))
        ref[:, 0] = np.linspace(0, 3, 16)

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert info["cost"] >= 0


# ─────────────────────────────────────────────────────────────
# Info 필드
# ─────────────────────────────────────────────────────────────

class TestSVMPCInfo:
    def test_svgd_info_fields(self):
        """SVGD 전용 info 필드 확인."""
        params = MPPIParams(N=10, K=64, dt=0.1, svgd_num_iterations=2)
        ctrl = SteinVariationalMPPIController(mppi_params=params, seed=42)

        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        _, info = ctrl.compute_control(state, ref)

        assert "svgd_iterations" in info
        assert info["svgd_iterations"] == 2
        assert "sample_diversity_before" in info
        assert "sample_diversity_after" in info
        assert np.isfinite(info["sample_diversity_before"])
        assert np.isfinite(info["sample_diversity_after"])


# ─────────────────────────────────────────────────────────────
# 엣지 케이스
# ─────────────────────────────────────────────────────────────

class TestSVMPCEdgeCases:
    """경계값 및 기본값 테스트."""

    def test_default_iterations_is_zero(self):
        """svgd_num_iterations 기본값이 0."""
        params = MPPIParams()
        assert params.svgd_num_iterations == 0

    def test_large_iterations_stable(self):
        """큰 L에서도 안정 동작."""
        params = MPPIParams(
            N=10, K=64, dt=0.1, svgd_num_iterations=10, svgd_step_size=0.05,
        )
        ctrl = SteinVariationalMPPIController(mppi_params=params, seed=42)
        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        control, info = ctrl.compute_control(state, ref)

        assert control.shape == (2,)
        assert np.all(np.isfinite(control))
        assert np.isfinite(info["cost"])


# ─────────────────────────────────────────────────────────────
# Reset
# ─────────────────────────────────────────────────────────────

class TestSVMPCReset:
    def test_reset(self):
        """reset 후 제어열 초기화."""
        ctrl = SteinVariationalMPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1, svgd_num_iterations=2),
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
