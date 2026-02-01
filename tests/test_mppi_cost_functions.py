"""MPPI 비용 함수 테스트."""

import numpy as np
import pytest

from mpc_controller.controllers.mppi.cost_functions import (
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ControlRateCost,
    ObstacleCost,
    CompositeMPPICost,
)


class TestStateTrackingCost:
    """StateTrackingCost 테스트."""

    @pytest.fixture
    def cost_fn(self):
        Q = np.diag([10.0, 10.0, 1.0])
        return StateTrackingCost(Q)

    def test_output_shape(self, cost_fn):
        K, N = 8, 5
        traj = np.zeros((K, N + 1, 3))
        ctrl = np.zeros((K, N, 2))
        ref = np.zeros((N + 1, 3))

        costs = cost_fn.compute(traj, ctrl, ref)
        assert costs.shape == (K,)

    def test_zero_error_zero_cost(self, cost_fn):
        K, N = 4, 5
        ref = np.random.randn(N + 1, 3)
        traj = np.tile(ref, (K, 1, 1))
        ctrl = np.zeros((K, N, 2))

        costs = cost_fn.compute(traj, ctrl, ref)
        np.testing.assert_allclose(costs, 0.0, atol=1e-10)

    def test_larger_error_larger_cost(self, cost_fn):
        K, N = 2, 5
        ref = np.zeros((N + 1, 3))
        ctrl = np.zeros((K, N, 2))

        traj = np.zeros((K, N + 1, 3))
        traj[0, :, 0] = 0.1  # small error
        traj[1, :, 0] = 1.0  # large error

        costs = cost_fn.compute(traj, ctrl, ref)
        assert costs[1] > costs[0]

    def test_angle_wrapping(self, cost_fn):
        """theta 오차가 pi 경계에서 올바르게 처리되는지."""
        K, N = 2, 3
        ref = np.zeros((N + 1, 3))
        ref[:, 2] = np.pi - 0.1
        ctrl = np.zeros((K, N, 2))

        traj = np.zeros((K, N + 1, 3))
        traj[0, :, 2] = np.pi + 0.1  # 작은 각도 차이 (0.2)
        traj[1, :, 2] = 0.0  # 큰 각도 차이

        costs = cost_fn.compute(traj, ctrl, ref)
        assert costs[0] < costs[1]


class TestTerminalCost:
    """TerminalCost 테스트."""

    def test_output_shape(self):
        Qf = np.diag([100.0, 100.0, 10.0])
        cost_fn = TerminalCost(Qf)
        K, N = 4, 5
        traj = np.zeros((K, N + 1, 3))
        ctrl = np.zeros((K, N, 2))
        ref = np.zeros((N + 1, 3))

        costs = cost_fn.compute(traj, ctrl, ref)
        assert costs.shape == (K,)

    def test_only_terminal_state(self):
        Qf = np.diag([1.0, 1.0, 1.0])
        cost_fn = TerminalCost(Qf)
        K, N = 2, 5
        ref = np.zeros((N + 1, 3))
        ref[-1, 0] = 1.0  # terminal x=1
        ctrl = np.zeros((K, N, 2))

        traj = np.zeros((K, N + 1, 3))
        traj[0, -1, 0] = 1.0  # match terminal
        traj[1, -1, 0] = 0.0  # miss terminal

        costs = cost_fn.compute(traj, ctrl, ref)
        assert costs[0] < costs[1]


class TestControlEffortCost:
    """ControlEffortCost 테스트."""

    def test_output_shape(self):
        R = np.diag([0.1, 0.1])
        cost_fn = ControlEffortCost(R)
        K, N = 4, 5
        traj = np.zeros((K, N + 1, 3))
        ctrl = np.ones((K, N, 2))
        ref = np.zeros((N + 1, 3))

        costs = cost_fn.compute(traj, ctrl, ref)
        assert costs.shape == (K,)

    def test_zero_control_zero_cost(self):
        R = np.diag([0.1, 0.1])
        cost_fn = ControlEffortCost(R)
        K, N = 4, 5
        traj = np.zeros((K, N + 1, 3))
        ctrl = np.zeros((K, N, 2))
        ref = np.zeros((N + 1, 3))

        costs = cost_fn.compute(traj, ctrl, ref)
        np.testing.assert_allclose(costs, 0.0, atol=1e-10)

    def test_larger_control_larger_cost(self):
        R = np.diag([0.1, 0.1])
        cost_fn = ControlEffortCost(R)
        K, N = 2, 5
        traj = np.zeros((K, N + 1, 3))
        ref = np.zeros((N + 1, 3))

        ctrl = np.zeros((K, N, 2))
        ctrl[0, :, 0] = 0.1
        ctrl[1, :, 0] = 1.0

        costs = cost_fn.compute(traj, ctrl, ref)
        assert costs[1] > costs[0]


class TestControlRateCost:
    """ControlRateCost 테스트."""

    @pytest.fixture
    def cost_fn(self):
        R_rate = np.array([0.5, 0.5])
        return ControlRateCost(R_rate)

    def test_output_shape(self, cost_fn):
        K, N = 8, 5
        traj = np.zeros((K, N + 1, 3))
        ctrl = np.zeros((K, N, 2))
        ref = np.zeros((N + 1, 3))

        costs = cost_fn.compute(traj, ctrl, ref)
        assert costs.shape == (K,)

    def test_zero_change_zero_cost(self, cost_fn):
        """제어 변화가 없으면 비용 0."""
        K, N = 4, 5
        traj = np.zeros((K, N + 1, 3))
        ctrl = np.ones((K, N, 2)) * 0.5  # 일정한 제어
        ref = np.zeros((N + 1, 3))

        costs = cost_fn.compute(traj, ctrl, ref)
        np.testing.assert_allclose(costs, 0.0, atol=1e-10)

    def test_larger_change_larger_cost(self, cost_fn):
        """큰 제어 변화 → 큰 비용."""
        K, N = 2, 5
        traj = np.zeros((K, N + 1, 3))
        ref = np.zeros((N + 1, 3))

        ctrl = np.zeros((K, N, 2))
        # 샘플 0: 작은 변화
        ctrl[0, :, 0] = [0.0, 0.1, 0.2, 0.3, 0.4]
        # 샘플 1: 큰 변화 (진동)
        ctrl[1, :, 0] = [0.0, 1.0, 0.0, 1.0, 0.0]

        costs = cost_fn.compute(traj, ctrl, ref)
        assert costs[1] > costs[0]

    def test_matrix_input(self):
        """R_rate를 행렬로 전달해도 동작."""
        R_rate_mat = np.diag([0.5, 0.3])
        cost_fn = ControlRateCost(R_rate_mat)

        K, N = 4, 5
        traj = np.zeros((K, N + 1, 3))
        ctrl = np.zeros((K, N, 2))
        ctrl[:, :, 0] = np.linspace(0, 1, N)
        ref = np.zeros((N + 1, 3))

        costs = cost_fn.compute(traj, ctrl, ref)
        assert costs.shape == (K,)
        assert np.all(costs > 0)

    def test_single_step_horizon(self):
        """N=1 일 때 변화율 없음 → 비용 0."""
        R_rate = np.array([0.5, 0.5])
        cost_fn = ControlRateCost(R_rate)

        K, N = 4, 1
        traj = np.zeros((K, N + 1, 3))
        ctrl = np.ones((K, N, 2))
        ref = np.zeros((N + 1, 3))

        costs = cost_fn.compute(traj, ctrl, ref)
        np.testing.assert_allclose(costs, 0.0, atol=1e-10)


class TestObstacleCost:
    """ObstacleCost 테스트."""

    def test_no_obstacles(self):
        cost_fn = ObstacleCost(np.array([]).reshape(0, 3))
        K, N = 4, 5
        traj = np.zeros((K, N + 1, 3))
        ctrl = np.zeros((K, N, 2))
        ref = np.zeros((N + 1, 3))

        costs = cost_fn.compute(traj, ctrl, ref)
        np.testing.assert_allclose(costs, 0.0)

    def test_collision_high_cost(self):
        obstacles = np.array([[1.0, 0.0, 0.5]])  # x=1, y=0, r=0.5
        cost_fn = ObstacleCost(obstacles, weight=1000.0, safety_margin=0.3)
        K, N = 2, 5
        ctrl = np.zeros((K, N, 2))
        ref = np.zeros((N + 1, 3))

        traj = np.zeros((K, N + 1, 3))
        traj[0, :, 0] = 1.0  # 장애물 중심에 위치
        traj[1, :, 0] = 5.0  # 장애물과 멀리

        costs = cost_fn.compute(traj, ctrl, ref)
        assert costs[0] > costs[1]

    def test_outside_safety_zone_zero_cost(self):
        obstacles = np.array([[5.0, 5.0, 0.5]])
        cost_fn = ObstacleCost(obstacles, weight=1000.0, safety_margin=0.3)
        K, N = 4, 5
        traj = np.zeros((K, N + 1, 3))  # 원점 근처
        ctrl = np.zeros((K, N, 2))
        ref = np.zeros((N + 1, 3))

        costs = cost_fn.compute(traj, ctrl, ref)
        np.testing.assert_allclose(costs, 0.0, atol=1e-10)


class TestCompositeMPPICost:
    """CompositeMPPICost 테스트."""

    def test_empty_composite(self):
        composite = CompositeMPPICost()
        K, N = 4, 5
        traj = np.zeros((K, N + 1, 3))
        ctrl = np.zeros((K, N, 2))
        ref = np.zeros((N + 1, 3))

        costs = composite.compute(traj, ctrl, ref)
        np.testing.assert_allclose(costs, 0.0)

    def test_additive(self):
        Q = np.diag([1.0, 1.0, 1.0])
        R = np.diag([1.0, 1.0])

        state_cost = StateTrackingCost(Q)
        ctrl_cost = ControlEffortCost(R)

        composite = CompositeMPPICost()
        composite.add(state_cost).add(ctrl_cost)

        K, N = 4, 5
        traj = np.ones((K, N + 1, 3))
        ctrl = np.ones((K, N, 2))
        ref = np.zeros((N + 1, 3))

        total = composite.compute(traj, ctrl, ref)
        individual_state = state_cost.compute(traj, ctrl, ref)
        individual_ctrl = ctrl_cost.compute(traj, ctrl, ref)

        np.testing.assert_allclose(total, individual_state + individual_ctrl)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
