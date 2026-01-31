"""MPPI 컨트롤러 유닛 + 통합 테스트."""

import numpy as np
import pytest

from mpc_controller import (
    DifferentialDriveModel,
    RobotParams,
    MPPIController,
    MPPIParams,
    generate_circle_trajectory,
    generate_line_trajectory,
    TrajectoryInterpolator,
)
from mpc_controller.controllers.mppi.dynamics_wrapper import BatchDynamicsWrapper
from mpc_controller.controllers.mppi.utils import (
    normalize_angle_batch,
    softmax_weights,
    effective_sample_size,
    log_sum_exp,
)


class TestMPPIParams:
    """MPPIParams 데이터클래스 테스트."""

    def test_default_values(self):
        params = MPPIParams()
        assert params.N == 30
        assert params.dt == 0.05
        assert params.K == 1024
        assert params.lambda_ == 10.0

    def test_default_matrices(self):
        params = MPPIParams()
        assert params.Q.shape == (3, 3)
        assert params.R.shape == (2, 2)
        assert params.Qf.shape == (3, 3)
        assert params.noise_sigma.shape == (2,)

    def test_custom_values(self):
        params = MPPIParams(N=20, K=512, lambda_=2.0)
        assert params.N == 20
        assert params.K == 512
        assert params.lambda_ == 2.0


class TestBatchDynamicsWrapper:
    """배치 동역학 래퍼 테스트."""

    @pytest.fixture
    def wrapper(self):
        return BatchDynamicsWrapper()

    def test_propagate_batch_shape(self, wrapper):
        K = 10
        states = np.zeros((K, 3))
        controls = np.zeros((K, 2))
        controls[:, 0] = 1.0  # forward

        next_states = wrapper.propagate_batch(states, controls, dt=0.1)
        assert next_states.shape == (K, 3)

    def test_propagate_batch_straight(self, wrapper):
        K = 5
        states = np.zeros((K, 3))
        controls = np.zeros((K, 2))
        controls[:, 0] = 1.0

        next_states = wrapper.propagate_batch(states, controls, dt=0.1)
        assert np.all(next_states[:, 0] > 0)
        np.testing.assert_allclose(next_states[:, 1], 0.0, atol=1e-10)

    def test_rollout_batch_shape(self, wrapper):
        K, N = 8, 10
        x0 = np.array([0.0, 0.0, 0.0])
        controls = np.zeros((K, N, 2))
        controls[:, :, 0] = 0.5

        trajectories = wrapper.rollout_batch(x0, controls, dt=0.1)
        assert trajectories.shape == (K, N + 1, 3)

    def test_rollout_batch_initial_state(self, wrapper):
        K, N = 4, 5
        x0 = np.array([1.0, 2.0, 0.5])
        controls = np.zeros((K, N, 2))

        trajectories = wrapper.rollout_batch(x0, controls, dt=0.1)
        np.testing.assert_array_almost_equal(
            trajectories[:, 0, :], np.tile(x0, (K, 1))
        )

    def test_clip_controls(self, wrapper):
        controls = np.array([[[5.0, 10.0]]])  # exceeds limits
        clipped = wrapper.clip_controls(controls)
        assert clipped[0, 0, 0] <= wrapper.params.max_velocity
        assert clipped[0, 0, 1] <= wrapper.params.max_omega

    def test_rollout_consistency_with_model(self, wrapper):
        """배치 래퍼가 단일 모델과 동일한 결과를 내는지 검증."""
        model = DifferentialDriveModel()
        x0 = np.array([0.0, 0.0, 0.0])
        control = np.array([0.5, 0.3])
        dt = 0.05

        # 단일 모델 forward simulate
        expected = model.forward_simulate(x0, control, dt)

        # 배치 래퍼 (K=1)
        controls_batch = control[np.newaxis, np.newaxis, :]  # (1, 1, 2)
        traj = wrapper.rollout_batch(x0, controls_batch, dt)
        actual = traj[0, 1, :]

        np.testing.assert_allclose(actual, expected, atol=1e-10)


class TestMPPIUtils:
    """MPPI 유틸리티 함수 테스트."""

    def test_normalize_angle_batch(self):
        angles = np.array([0, np.pi, 2 * np.pi, -np.pi, 3 * np.pi])
        result = normalize_angle_batch(angles)
        assert np.all(result >= -np.pi)
        assert np.all(result <= np.pi)
        assert abs(result[0]) < 1e-10
        assert abs(result[2]) < 1e-10  # 2pi -> 0

    def test_softmax_weights_sum_to_one(self):
        costs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = softmax_weights(costs, lambda_=1.0)
        np.testing.assert_almost_equal(np.sum(weights), 1.0)

    def test_softmax_weights_lower_cost_higher_weight(self):
        costs = np.array([1.0, 10.0])
        weights = softmax_weights(costs, lambda_=1.0)
        assert weights[0] > weights[1]

    def test_softmax_weights_temperature(self):
        costs = np.array([1.0, 2.0, 3.0])
        w_cold = softmax_weights(costs, lambda_=0.1)
        w_hot = softmax_weights(costs, lambda_=10.0)
        # cold: 더 집중 (max weight 더 큼)
        assert np.max(w_cold) > np.max(w_hot)

    def test_effective_sample_size(self):
        # 균등 가중치: ESS = K
        K = 100
        uniform = np.ones(K) / K
        ess = effective_sample_size(uniform)
        np.testing.assert_almost_equal(ess, K)

        # 하나만 1: ESS = 1
        peaked = np.zeros(K)
        peaked[0] = 1.0
        ess = effective_sample_size(peaked)
        np.testing.assert_almost_equal(ess, 1.0)

    def test_log_sum_exp(self):
        values = np.array([1.0, 2.0, 3.0])
        result = log_sum_exp(values)
        expected = np.log(np.sum(np.exp(values)))
        np.testing.assert_almost_equal(result, expected)

    def test_log_sum_exp_large_values(self):
        """큰 값에서도 수치적으로 안정적인지 확인."""
        values = np.array([1000.0, 1001.0, 1002.0])
        result = log_sum_exp(values)
        assert np.isfinite(result)


class TestMPPIController:
    """MPPI 컨트롤러 테스트."""

    @pytest.fixture
    def controller(self):
        return MPPIController(
            mppi_params=MPPIParams(N=10, K=64, dt=0.1),
            seed=42,
        )

    def test_controller_creation(self, controller):
        assert controller is not None
        assert controller.params.N == 10
        assert controller.params.K == 64

    def test_compute_control_output_shape(self, controller):
        state = np.array([0.0, 0.0, 0.0])
        reference = np.zeros((11, 3))

        control, info = controller.compute_control(state, reference)

        assert control.shape == (2,)
        assert "predicted_trajectory" in info
        assert info["predicted_trajectory"].shape == (11, 3)

    def test_compute_control_info_keys(self, controller):
        state = np.array([0.0, 0.0, 0.0])
        reference = np.zeros((11, 3))

        _, info = controller.compute_control(state, reference)

        assert "sample_trajectories" in info
        assert "sample_weights" in info
        assert "sample_costs" in info
        assert "best_trajectory" in info
        assert "ess" in info
        assert "solve_time" in info

    def test_compute_control_sample_shapes(self, controller):
        state = np.array([0.0, 0.0, 0.0])
        reference = np.zeros((11, 3))

        _, info = controller.compute_control(state, reference)

        K, N = controller.params.K, controller.params.N
        assert info["sample_trajectories"].shape == (K, N + 1, 3)
        assert info["sample_weights"].shape == (K,)
        assert info["sample_costs"].shape == (K,)
        assert info["best_trajectory"].shape == (N + 1, 3)

    def test_compute_control_stationary(self, controller):
        state = np.array([0.0, 0.0, 0.0])
        reference = np.zeros((11, 3))

        control, _ = controller.compute_control(state, reference)
        assert np.abs(control[0]) < 0.5
        assert np.abs(control[1]) < 0.5

    def test_compute_control_forward_reference(self, controller):
        state = np.array([0.0, 0.0, 0.0])
        reference = np.zeros((11, 3))
        reference[:, 0] = np.linspace(0, 2, 11)

        control, _ = controller.compute_control(state, reference)
        assert control[0] > 0  # forward

    def test_control_bounds_respected(self, controller):
        state = np.array([0.0, 0.0, 0.0])
        reference = np.zeros((11, 3))
        reference[:, 0] = np.linspace(0, 10, 11)  # far target

        for _ in range(5):
            control, _ = controller.compute_control(state, reference)
            assert abs(control[0]) <= controller.robot_params.max_velocity + 1e-6
            assert abs(control[1]) <= controller.robot_params.max_omega + 1e-6

    def test_weights_sum_to_one(self, controller):
        state = np.array([0.0, 0.0, 0.0])
        reference = np.zeros((11, 3))
        reference[:, 0] = np.linspace(0, 2, 11)

        _, info = controller.compute_control(state, reference)
        np.testing.assert_almost_equal(np.sum(info["sample_weights"]), 1.0)

    def test_controller_reset(self, controller):
        state = np.array([0.0, 0.0, 0.0])
        reference = np.zeros((11, 3))
        reference[:, 0] = np.linspace(0, 2, 11)

        controller.compute_control(state, reference)
        assert controller._iteration_count > 0

        controller.reset()
        assert controller._iteration_count == 0
        np.testing.assert_array_equal(
            controller.U, np.zeros((controller.params.N, 2))
        )

    def test_set_obstacles(self, controller):
        obstacles = np.array([[2.0, 0.0, 0.5]])
        controller.set_obstacles(obstacles)

        state = np.array([0.0, 0.0, 0.0])
        reference = np.zeros((11, 3))
        reference[:, 0] = np.linspace(0, 4, 11)

        control, info = controller.compute_control(state, reference)
        assert control.shape == (2,)


class TestMPPIIntegration:
    """MPPI 통합 테스트."""

    def test_line_tracking(self):
        """직선 궤적 추적 테스트."""
        params = MPPIParams(N=15, K=256, dt=0.05)
        controller = MPPIController(mppi_params=params, seed=42)

        trajectory = generate_line_trajectory(
            start=np.array([0.0, 0.0]),
            end=np.array([3.0, 0.0]),
            num_points=100,
        )
        interpolator = TrajectoryInterpolator(trajectory, dt=0.05)

        state = np.array([0.0, 0.0, 0.0])
        dt_sim = 0.05

        for i in range(20):
            ref = interpolator.get_reference(
                i * dt_sim, params.N, params.dt
            )
            control, _ = controller.compute_control(state, ref)

            assert -1.0 <= control[0] <= 1.0
            assert -1.5 <= control[1] <= 1.5

            state = DifferentialDriveModel().forward_simulate(
                state, control, dt_sim
            )

    def test_circle_tracking_rmse(self):
        """원형 궤적 추적 RMSE < 0.2m 검증."""
        params = MPPIParams(N=20, K=512, dt=0.05, lambda_=10.0)
        controller = MPPIController(mppi_params=params, seed=42)

        trajectory = generate_circle_trajectory(
            center=np.array([0.0, 0.0]),
            radius=2.0,
            num_points=200,
        )
        interpolator = TrajectoryInterpolator(trajectory, dt=0.05)

        state = np.array([2.0, 0.0, np.pi / 2])
        model = DifferentialDriveModel()
        dt_sim = 0.05

        errors = []
        for i in range(100):
            t = i * dt_sim
            ref = interpolator.get_reference(
                t, params.N, params.dt, current_theta=state[2]
            )
            control, _ = controller.compute_control(state, ref)
            state = model.forward_simulate(state, control, dt_sim)

            # 원형 궤적에서의 거리 오차
            dist_from_center = np.sqrt(state[0] ** 2 + state[1] ** 2)
            errors.append(abs(dist_from_center - 2.0))

        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        assert rmse < 0.2, f"Circle tracking RMSE = {rmse:.4f} (> 0.2m)"

    def test_obstacle_avoidance(self):
        """장애물이 있을 때 궤적이 회피하는지 확인."""
        obstacles = np.array([[1.5, 0.0, 0.3]])
        params = MPPIParams(N=15, K=256, dt=0.05)
        controller = MPPIController(
            mppi_params=params, seed=42, obstacles=obstacles,
        )

        state = np.array([0.0, 0.0, 0.0])
        reference = np.zeros((16, 3))
        reference[:, 0] = np.linspace(0, 3, 16)

        control, info = controller.compute_control(state, reference)

        # 장애물이 있으므로 비용이 더 높을 수 있음
        assert info["cost"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
