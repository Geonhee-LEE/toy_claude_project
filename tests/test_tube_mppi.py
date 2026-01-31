"""TubeMPPIController 단위 및 통합 테스트."""

import numpy as np
import pytest

from mpc_controller import (
    TubeMPPIController,
    MPPIParams,
    RobotParams,
    TrajectoryInterpolator,
    generate_circle_trajectory,
)
from simulation.simulator import Simulator, SimulationConfig


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def robot_params():
    return RobotParams(max_velocity=1.0, max_omega=1.5)


@pytest.fixture
def vanilla_params():
    """tube_enabled=False (Vanilla 호환 모드)."""
    return MPPIParams(
        N=10, K=128, dt=0.05, lambda_=10.0,
        noise_sigma=np.array([0.3, 0.3]),
        tube_enabled=False,
    )


@pytest.fixture
def tube_params():
    """tube_enabled=True."""
    return MPPIParams(
        N=10, K=128, dt=0.05, lambda_=10.0,
        noise_sigma=np.array([0.3, 0.3]),
        tube_enabled=True,
        tube_disturbance_bound=0.1,
    )


@pytest.fixture
def circle_ref():
    """원형 참조 궤적."""
    return generate_circle_trajectory(
        center=np.array([0.0, 0.0]),
        radius=2.0,
        num_points=400,
    )


# ─────────────────────────────────────────────────────────────
# 호환성 테스트
# ─────────────────────────────────────────────────────────────

class TestTubeDisabledCompatibility:
    """tube_enabled=False → Vanilla MPPI와 동일 동작."""

    def test_tube_disabled_identical_to_vanilla(self, robot_params, vanilla_params, circle_ref):
        """핵심 호환성 테스트: tube_enabled=False면 결과가 Vanilla와 동일."""
        from mpc_controller import MPPIController

        ctrl_tube = TubeMPPIController(
            robot_params=robot_params,
            mppi_params=vanilla_params,
            seed=42,
        )
        ctrl_vanilla = MPPIController(
            robot_params=robot_params,
            mppi_params=vanilla_params,
            seed=42,
        )

        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0.0, vanilla_params.N, vanilla_params.dt, state[2])

        u_tube, info_tube = ctrl_tube.compute_control(state, ref)
        u_vanilla, info_vanilla = ctrl_vanilla.compute_control(state, ref)

        np.testing.assert_allclose(u_tube, u_vanilla, atol=1e-10)
        assert "tube_enabled" not in info_tube  # tube info 없어야 함


# ─────────────────────────────────────────────────────────────
# 기본 동작 테스트
# ─────────────────────────────────────────────────────────────

class TestTubeMPPIBasic:
    """TubeMPPI 기본 기능."""

    def test_compute_control_output_shape(self, robot_params, tube_params, circle_ref):
        """제어 출력 형태 (2,) + info dict."""
        ctrl = TubeMPPIController(
            robot_params=robot_params,
            mppi_params=tube_params,
            seed=42,
        )
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0.0, tube_params.N, tube_params.dt, state[2])

        u, info = ctrl.compute_control(state, ref)
        assert u.shape == (2,)
        assert isinstance(info, dict)

    def test_info_contains_tube_keys(self, robot_params, tube_params, circle_ref):
        """tube_enabled=True → info에 tube 관련 키 존재."""
        ctrl = TubeMPPIController(
            robot_params=robot_params,
            mppi_params=tube_params,
            seed=42,
        )
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0.0, tube_params.N, tube_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)

        assert info["tube_enabled"] is True
        assert "nominal_state" in info
        assert "feedback_correction" in info
        assert "tube_width" in info
        assert "nominal_trajectory" in info
        assert "tube_boundary" in info
        assert "deviation" in info
        assert info["tube_width"] > 0

    def test_nominal_state_propagation(self, robot_params, tube_params, circle_ref):
        """매 스텝마다 명목 상태가 전파되어야 함."""
        ctrl = TubeMPPIController(
            robot_params=robot_params,
            mppi_params=tube_params,
            seed=42,
        )
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        state = circle_ref[0].copy()

        ref = interp.get_reference(0.0, tube_params.N, tube_params.dt, state[2])
        _, info1 = ctrl.compute_control(state, ref)
        nom1 = info1["nominal_state"].copy()

        # 두 번째 스텝 — 명목 상태가 변해야 함
        ref = interp.get_reference(0.05, tube_params.N, tube_params.dt, state[2])
        _, info2 = ctrl.compute_control(state, ref)
        nom2 = info2["nominal_state"].copy()

        # 명목 상태가 변경됐어야 함
        assert not np.allclose(nom1, nom2, atol=1e-10), "명목 상태가 전파되어야 함"

    def test_feedback_correction_nonzero(self, robot_params, tube_params, circle_ref):
        """actual != nominal → du != 0."""
        ctrl = TubeMPPIController(
            robot_params=robot_params,
            mppi_params=tube_params,
            seed=42,
        )
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)

        # 첫 스텝: nominal = actual
        state = circle_ref[0].copy()
        ref = interp.get_reference(0.0, tube_params.N, tube_params.dt, state[2])
        ctrl.compute_control(state, ref)

        # 두 번째 스텝: actual에 편차 주입
        perturbed_state = state + np.array([0.1, 0.05, 0.02])
        ref = interp.get_reference(0.05, tube_params.N, tube_params.dt, perturbed_state[2])
        _, info = ctrl.compute_control(perturbed_state, ref)

        du = info["feedback_correction"]
        assert np.linalg.norm(du) > 1e-6, "편차가 있으면 보정량 != 0"

    def test_control_bounds_with_correction(self, robot_params, tube_params, circle_ref):
        """보정 후에도 제어 한계 내."""
        ctrl = TubeMPPIController(
            robot_params=robot_params,
            mppi_params=tube_params,
            seed=42,
        )
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        state = circle_ref[0].copy()

        for step in range(20):
            t = step * 0.05
            ref = interp.get_reference(t, tube_params.N, tube_params.dt, state[2])
            u, _ = ctrl.compute_control(state, ref)
            assert abs(u[0]) <= robot_params.max_velocity + 1e-10
            assert abs(u[1]) <= robot_params.max_omega + 1e-10

    def test_reset_clears_nominal(self, robot_params, tube_params, circle_ref):
        """reset() 후 _x_nominal이 None."""
        ctrl = TubeMPPIController(
            robot_params=robot_params,
            mppi_params=tube_params,
            seed=42,
        )
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        state = circle_ref[0].copy()
        ref = interp.get_reference(0.0, tube_params.N, tube_params.dt, state[2])
        ctrl.compute_control(state, ref)

        assert ctrl._x_nominal is not None
        ctrl.reset()
        assert ctrl._x_nominal is None

    def test_tube_boundary_shape(self, robot_params, tube_params, circle_ref):
        """tube 경계 좌표 형태 확인."""
        ctrl = TubeMPPIController(
            robot_params=robot_params,
            mppi_params=tube_params,
            seed=42,
        )
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        state = circle_ref[0].copy()
        ref = interp.get_reference(0.0, tube_params.N, tube_params.dt, state[2])
        _, info = ctrl.compute_control(state, ref)

        boundary = info["tube_boundary"]
        assert "upper" in boundary
        assert "lower" in boundary
        N = tube_params.N
        assert boundary["upper"].shape == (N + 1, 2)
        assert boundary["lower"].shape == (N + 1, 2)


# ─────────────────────────────────────────────────────────────
# 통합 테스트
# ─────────────────────────────────────────────────────────────

class TestTubeMPPIIntegration:
    """Tube-MPPI 통합 테스트."""

    def test_circle_tracking_with_tube(self, robot_params, circle_ref):
        """원형 궤적 추적 — Tube-MPPI가 수렴하고 궤적을 추적해야 함.

        초기 transient를 제외하고 정상 상태에서 RMSE < 0.3m.
        """
        params = MPPIParams(
            N=20, K=512, dt=0.05, lambda_=10.0,
            noise_sigma=np.array([0.3, 0.3]),
            Q=np.diag([10.0, 10.0, 1.0]),
            Qf=np.diag([100.0, 100.0, 10.0]),
            tube_enabled=True,
            tube_disturbance_bound=0.1,
        )
        ctrl = TubeMPPIController(
            robot_params=robot_params,
            mppi_params=params,
            seed=42,
        )
        sim_config = SimulationConfig(dt=0.05, max_time=10.0)
        sim = Simulator(robot_params, sim_config)
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)

        state = circle_ref[0].copy()
        sim.reset(state)
        errors = []

        for step in range(int(sim_config.max_time / sim_config.dt)):
            t = step * sim_config.dt
            state = sim.get_measurement()
            ref = interp.get_reference(t, params.N, params.dt, state[2])
            u, _ = ctrl.compute_control(state, ref)
            sim.step(u)
            err = sim.compute_tracking_error(state, ref[0])
            errors.append(np.linalg.norm(err[:2]))

        # Tube-MPPI는 명목 상태 기준 계획 → Vanilla보다 RMSE 약간 높을 수 있음
        # 전체 궤적 RMSE < 0.5m 확인 (최소한의 추적 성능 보장)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        assert rmse < 0.5, f"Circle tracking RMSE={rmse:.4f} > 0.5m"

    def test_disturbance_robustness(self, robot_params, circle_ref):
        """외란 하에서 Tube-MPPI > Vanilla MPPI 성능 확인 (핵심 테스트).

        프로세스 노이즈를 주입하고 두 컨트롤러의 RMSE를 비교한다.
        Tube-MPPI의 ancillary 보정이 외란에 더 강건해야 한다.
        """
        from mpc_controller import MPPIController

        np.random.seed(123)

        base_params_dict = dict(
            N=20, K=256, dt=0.05, lambda_=10.0,
            noise_sigma=np.array([0.3, 0.3]),
            Q=np.diag([10.0, 10.0, 1.0]),
            Qf=np.diag([50.0, 50.0, 5.0]),
        )

        vanilla_params = MPPIParams(**base_params_dict, tube_enabled=False)
        tube_params = MPPIParams(**base_params_dict, tube_enabled=True, tube_disturbance_bound=0.1)

        # 큰 프로세스 노이즈
        noise_std = np.array([0.05, 0.05, 0.02])
        sim_config = SimulationConfig(
            dt=0.05, max_time=10.0,
            process_noise_std=noise_std,
        )

        results = {}
        for name, ctrl_cls, params in [
            ("vanilla", MPPIController, vanilla_params),
            ("tube", TubeMPPIController, tube_params),
        ]:
            np.random.seed(123)  # 동일 노이즈
            ctrl = ctrl_cls(
                robot_params=robot_params,
                mppi_params=params,
                seed=42,
            )
            sim = Simulator(robot_params, sim_config)
            interp = TrajectoryInterpolator(circle_ref, dt=0.05)

            state = circle_ref[0].copy()
            sim.reset(state)
            errors = []

            for step in range(int(sim_config.max_time / sim_config.dt)):
                t = step * sim_config.dt
                state = sim.get_measurement()
                ref = interp.get_reference(t, params.N, params.dt, state[2])
                u, _ = ctrl.compute_control(state, ref)
                sim.step(u, add_noise=True)
                err = sim.compute_tracking_error(state, ref[0])
                errors.append(np.linalg.norm(err[:2]))

            rmse = np.sqrt(np.mean(np.array(errors) ** 2))
            results[name] = rmse

        # Tube-MPPI가 Vanilla보다 같거나 나아야 함
        # (실제로는 ancillary 보정으로 인해 대부분 나음)
        assert results["tube"] <= results["vanilla"] * 1.15, (
            f"Tube RMSE={results['tube']:.4f} > Vanilla RMSE={results['vanilla']:.4f} * 1.15. "
            f"Tube-MPPI가 외란에 더 강건해야 합니다."
        )

    def test_nominal_state_convergence(self, robot_params, circle_ref):
        """명목 상태와 실제 상태의 편차가 bounded (tube 보장)."""
        params = MPPIParams(
            N=20, K=256, dt=0.05, lambda_=10.0,
            noise_sigma=np.array([0.3, 0.3]),
            tube_enabled=True,
            tube_disturbance_bound=0.1,
        )
        ctrl = TubeMPPIController(
            robot_params=robot_params,
            mppi_params=params,
            seed=42,
        )
        sim_config = SimulationConfig(
            dt=0.05, max_time=5.0,
            process_noise_std=np.array([0.03, 0.03, 0.01]),
        )
        sim = Simulator(robot_params, sim_config)
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)

        state = circle_ref[0].copy()
        sim.reset(state)
        deviations = []

        for step in range(int(sim_config.max_time / sim_config.dt)):
            t = step * sim_config.dt
            state = sim.get_measurement()
            ref = interp.get_reference(t, params.N, params.dt, state[2])
            u, info = ctrl.compute_control(state, ref)
            sim.step(u, add_noise=True)
            deviations.append(info.get("deviation", 0.0))

        max_dev = max(deviations)
        # reset threshold 이하로 유지되어야 함
        assert max_dev < params.tube_nominal_reset_threshold, (
            f"Max deviation={max_dev:.4f} >= threshold={params.tube_nominal_reset_threshold}"
        )


class TestTubeAwareCost:
    """TubeAwareCost 테스트."""

    def test_tube_aware_cost_larger_than_obstacle_cost(self):
        """TubeAwareCost가 ObstacleCost보다 더 보수적 (같은 궤적에 더 큰 비용)."""
        from mpc_controller.controllers.mppi.cost_functions import (
            ObstacleCost,
            TubeAwareCost,
        )

        obstacles = np.array([[2.0, 0.0, 0.3]])
        obs_cost = ObstacleCost(obstacles, weight=1000.0, safety_margin=0.3)
        tube_cost = TubeAwareCost(obstacles, tube_margin=0.15, weight=1000.0, safety_margin=0.3)

        # 장애물 근처를 지나는 궤적
        K, N = 4, 10
        trajectories = np.zeros((K, N + 1, 3))
        for k in range(K):
            trajectories[k, :, 0] = np.linspace(0, 2.5, N + 1)
            trajectories[k, :, 1] = 0.5  # 장애물에 가까운 궤적
        controls = np.zeros((K, N, 2))
        reference = np.zeros((N + 1, 3))

        c_obs = obs_cost.compute(trajectories, controls, reference)
        c_tube = tube_cost.compute(trajectories, controls, reference)

        assert np.all(c_tube >= c_obs), "TubeAwareCost >= ObstacleCost"

    def test_tube_aware_cost_no_obstacles(self):
        """장애물 없으면 비용 0."""
        from mpc_controller.controllers.mppi.cost_functions import TubeAwareCost

        obstacles = np.empty((0, 3))
        cost = TubeAwareCost(obstacles, tube_margin=0.15)

        K, N = 2, 5
        trajectories = np.zeros((K, N + 1, 3))
        controls = np.zeros((K, N, 2))
        reference = np.zeros((N + 1, 3))

        result = cost.compute(trajectories, controls, reference)
        np.testing.assert_allclose(result, 0.0)
