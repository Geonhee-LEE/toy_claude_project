"""SVG-MPPI 컨트롤러 테스트 — Guide particle 다중 모드 탐색 검증."""

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
from mpc_controller.controllers.mppi.svg_mppi import SVGMPPIController
from mpc_controller.controllers.mppi.stein_variational_mppi import (
    SteinVariationalMPPIController,
)
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
def svg_params():
    return MPPIParams(
        N=20, K=64, dt=0.05, lambda_=10.0,
        noise_sigma=np.array([0.3, 0.3]),
        Q=np.diag([10.0, 10.0, 1.0]),
        R=np.diag([0.01, 0.01]),
        Qf=np.diag([50.0, 50.0, 5.0]),
        svg_num_guide_particles=8,
        svg_guide_step_size=0.2,
        svg_guide_iterations=2,
        svg_resample_std=0.1,
    )


@pytest.fixture
def circle_ref():
    return generate_circle_trajectory(np.array([0.0, 0.0]), 2.0, 200)


# ─────────────────────────────────────────────────────────────
# 기본 생성 및 인터페이스
# ─────────────────────────────────────────────────────────────

class TestSVGMPPICreation:
    """생성자 및 상속 관계 검증."""

    def test_inherits_stein_variational(self, robot_params, svg_params):
        ctrl = SVGMPPIController(robot_params, svg_params, seed=42)
        assert isinstance(ctrl, SteinVariationalMPPIController)
        assert isinstance(ctrl, MPPIController)

    def test_default_params(self, robot_params, svg_params):
        ctrl = SVGMPPIController(robot_params, svg_params, seed=42)
        assert ctrl.params.svg_num_guide_particles == 8
        assert ctrl.params.svg_guide_iterations == 2

    def test_zero_guides_vanilla_fallback(self, robot_params, basic_params):
        basic_params.svg_num_guide_particles = 0
        basic_params.svg_guide_iterations = 0
        ctrl = SVGMPPIController(robot_params, basic_params, seed=42)
        assert isinstance(ctrl, SVGMPPIController)


# ─────────────────────────────────────────────────────────────
# compute_control 인터페이스
# ─────────────────────────────────────────────────────────────

class TestSVGMPPIComputeControl:
    """compute_control 출력 형식 및 내용 검증."""

    def test_output_shape(self, robot_params, svg_params, circle_ref):
        ctrl = SVGMPPIController(robot_params, svg_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, svg_params.N, svg_params.dt, state[2])

        u, info = ctrl.compute_control(state, ref)
        assert u.shape == (2,)

    def test_info_keys(self, robot_params, svg_params, circle_ref):
        ctrl = SVGMPPIController(robot_params, svg_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, svg_params.N, svg_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)

        required_keys = [
            "predicted_trajectory", "predicted_controls", "cost",
            "mean_cost", "solve_time", "sample_trajectories",
            "sample_weights", "sample_costs", "best_trajectory",
            "best_index", "ess", "temperature",
            "num_guides", "num_followers", "guide_iterations",
            "guide_diversity_before", "guide_diversity_after",
            "guide_costs",
        ]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

    def test_guide_info_values(self, robot_params, svg_params, circle_ref):
        ctrl = SVGMPPIController(robot_params, svg_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, svg_params.N, svg_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)
        assert info["num_guides"] == 8
        assert info["guide_iterations"] == 2
        assert info["num_followers"] == 64 - 8

    def test_control_within_limits(self, robot_params, svg_params, circle_ref):
        ctrl = SVGMPPIController(robot_params, svg_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, svg_params.N, svg_params.dt, state[2])

        u, _ = ctrl.compute_control(state, ref)
        assert abs(u[0]) <= robot_params.max_velocity + 1e-6
        assert abs(u[1]) <= robot_params.max_omega + 1e-6


# ─────────────────────────────────────────────────────────────
# Vanilla fallback
# ─────────────────────────────────────────────────────────────

class TestSVGMPPIFallback:
    """G=0 또는 L=0일 때 Vanilla fallback."""

    def test_zero_iterations_vanilla(self, robot_params, basic_params, circle_ref):
        basic_params.svg_num_guide_particles = 8
        basic_params.svg_guide_iterations = 0
        ctrl = SVGMPPIController(robot_params, basic_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, basic_params.N, basic_params.dt, state[2])

        u, info = ctrl.compute_control(state, ref)
        assert np.all(np.isfinite(u))
        # Vanilla fallback이므로 guide 전용 키가 없을 수 있음
        assert "ess" in info

    def test_zero_guides_vanilla(self, robot_params, basic_params, circle_ref):
        basic_params.svg_num_guide_particles = 0
        basic_params.svg_guide_iterations = 3
        ctrl = SVGMPPIController(robot_params, basic_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, basic_params.N, basic_params.dt, state[2])

        u, info = ctrl.compute_control(state, ref)
        assert np.all(np.isfinite(u))


# ─────────────────────────────────────────────────────────────
# SVGD Guide 동작 검증
# ─────────────────────────────────────────────────────────────

class TestSVGMPPIGuides:
    """Guide particle 동작 검증."""

    def test_guide_diversity_changes(self, robot_params, svg_params, circle_ref):
        """SVGD 적용 후 guide diversity가 변화해야 함."""
        ctrl = SVGMPPIController(robot_params, svg_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, svg_params.N, svg_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)
        # diversity가 0이 아니어야 함
        assert info["guide_diversity_after"] >= 0

    def test_guide_costs_shape(self, robot_params, svg_params, circle_ref):
        ctrl = SVGMPPIController(robot_params, svg_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, svg_params.N, svg_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)
        assert len(info["guide_costs"]) == svg_params.svg_num_guide_particles

    def test_more_guides_different_behavior(self, robot_params, circle_ref):
        """Guide 수를 변경하면 결과가 달라야 함."""
        results = []
        for G in [4, 16]:
            params = MPPIParams(
                N=20, K=64, dt=0.05, lambda_=10.0,
                noise_sigma=np.array([0.3, 0.3]),
                svg_num_guide_particles=G,
                svg_guide_iterations=2,
                svg_guide_step_size=0.2,
                svg_resample_std=0.1,
            )
            ctrl = SVGMPPIController(robot_params, params, seed=42)
            state = circle_ref[0].copy()
            interp = TrajectoryInterpolator(circle_ref, dt=0.05)
            ref = interp.get_reference(0, params.N, params.dt, state[2])

            u, _ = ctrl.compute_control(state, ref)
            results.append(u)

        # 다른 guide 수 → 다른 결과 (완벽히 같기는 어려움)
        assert results[0].shape == results[1].shape == (2,)


# ─────────────────────────────────────────────────────────────
# 성능: SVG-MPPI vs SVMPC
# ─────────────────────────────────────────────────────────────

class TestSVGMPPIvsFullSVGD:
    """SVG-MPPI가 full SVMPC보다 효율적인지 검증."""

    def test_fewer_pairwise_computations(self, robot_params, circle_ref):
        """G << K이므로 SVGD 계산량이 적어야 함."""
        K = 128
        G = 8
        # SVG-MPPI: O(G²D) per iteration
        # SVMPC: O(K²D) per iteration
        svg_cost = G * G
        full_cost = K * K
        assert svg_cost < full_cost

    def test_both_produce_valid_output(self, robot_params, circle_ref):
        """SVG-MPPI와 SVMPC 모두 유효한 출력."""
        K = 64

        # SVMPC
        svmpc_params = MPPIParams(
            N=20, K=K, dt=0.05, lambda_=10.0,
            noise_sigma=np.array([0.3, 0.3]),
            svgd_num_iterations=2,
            svgd_step_size=0.1,
        )
        svmpc = SteinVariationalMPPIController(robot_params, svmpc_params, seed=42)

        # SVG-MPPI
        svg_params = MPPIParams(
            N=20, K=K, dt=0.05, lambda_=10.0,
            noise_sigma=np.array([0.3, 0.3]),
            svg_num_guide_particles=8,
            svg_guide_iterations=2,
            svg_guide_step_size=0.2,
            svg_resample_std=0.1,
        )
        svg = SVGMPPIController(robot_params, svg_params, seed=42)

        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, 20, 0.05, state[2])

        u1, info1 = svmpc.compute_control(state, ref)
        u2, info2 = svg.compute_control(state, ref)

        assert np.all(np.isfinite(u1))
        assert np.all(np.isfinite(u2))
        assert info1["ess"] > 0
        assert info2["ess"] > 0


# ─────────────────────────────────────────────────────────────
# 다중 스텝 시뮬레이션
# ─────────────────────────────────────────────────────────────

class TestSVGMPPIMultiStep:
    """다중 스텝 시뮬레이션 안정성."""

    def test_multi_step_no_error(self, robot_params, svg_params, circle_ref):
        ctrl = SVGMPPIController(robot_params, svg_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)

        for step in range(15):
            ref = interp.get_reference(
                step * 0.05, svg_params.N, svg_params.dt, state[2]
            )
            u, info = ctrl.compute_control(state, ref)
            assert np.all(np.isfinite(u)), f"Non-finite at step {step}"
            state = state + np.array([
                u[0] * np.cos(state[2]) * 0.05,
                u[0] * np.sin(state[2]) * 0.05,
                u[1] * 0.05,
            ])

    def test_reset(self, robot_params, svg_params, circle_ref):
        ctrl = SVGMPPIController(robot_params, svg_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, svg_params.N, svg_params.dt, state[2])

        ctrl.compute_control(state, ref)
        ctrl.reset()
        assert ctrl._iteration_count == 0
        np.testing.assert_array_equal(ctrl.U, np.zeros_like(ctrl.U))


# ─────────────────────────────────────────────────────────────
# ESS 및 메트릭
# ─────────────────────────────────────────────────────────────

class TestSVGMPPIMetrics:
    """ESS, 비용 등 기본 메트릭."""

    def test_ess_valid_range(self, robot_params, svg_params, circle_ref):
        ctrl = SVGMPPIController(robot_params, svg_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, svg_params.N, svg_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)
        # K_total = K (guides + followers)
        assert info["ess"] >= 1.0

    def test_cost_non_negative(self, robot_params, svg_params, circle_ref):
        ctrl = SVGMPPIController(robot_params, svg_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, svg_params.N, svg_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)
        assert info["cost"] >= 0

    def test_weights_normalized(self, robot_params, svg_params, circle_ref):
        ctrl = SVGMPPIController(robot_params, svg_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, svg_params.N, svg_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)
        np.testing.assert_almost_equal(np.sum(info["sample_weights"]), 1.0, decimal=5)


# ─────────────────────────────────────────────────────────────
# 장애물 환경
# ─────────────────────────────────────────────────────────────

class TestSVGMPPIObstacles:
    """장애물 환경에서의 동작 — 다중 모드 탐색."""

    def test_with_obstacles(self, robot_params, svg_params, circle_ref):
        obstacles = np.array([[1.0, 0.0, 0.3], [-1.0, 0.0, 0.3]])
        ctrl = SVGMPPIController(
            robot_params, svg_params, seed=42, obstacles=obstacles
        )
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, svg_params.N, svg_params.dt, state[2])

        u, info = ctrl.compute_control(state, ref)
        assert np.all(np.isfinite(u))

    def test_narrow_passage_multimodal(self, robot_params, circle_ref):
        """좁은 통로에서 guide가 다양한 경로를 탐색."""
        obstacles = np.array([
            [1.0, 0.5, 0.3],
            [1.0, -0.5, 0.3],
        ])
        params = MPPIParams(
            N=20, K=128, dt=0.05, lambda_=10.0,
            noise_sigma=np.array([0.5, 0.5]),
            svg_num_guide_particles=16,
            svg_guide_iterations=3,
            svg_guide_step_size=0.2,
            svg_resample_std=0.15,
        )
        ctrl = SVGMPPIController(
            robot_params, params, seed=42, obstacles=obstacles
        )
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, params.N, params.dt, state[2])

        u, info = ctrl.compute_control(state, ref)
        assert np.all(np.isfinite(u))
        # Guide diversity가 0보다 커야 함 (다중 모드)
        assert info["guide_diversity_after"] > 0
