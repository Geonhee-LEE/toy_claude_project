"""CBF Barrier Function 단위 테스트."""

import numpy as np
import pytest

from mpc_controller.controllers.mppi.barrier_function import (
    CircleBarrier,
    BarrierFunctionSet,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def single_obstacle():
    """원점에 반경 0.5 장애물."""
    return np.array([0.0, 0.0, 0.5])


@pytest.fixture
def barrier(single_obstacle):
    """기본 CircleBarrier (robot_radius=0.2, margin=0.3)."""
    return CircleBarrier(single_obstacle, robot_radius=0.2, safety_margin=0.3)


@pytest.fixture
def multi_obstacles():
    """다중 장애물."""
    return np.array([
        [2.0, 0.0, 0.3],
        [4.0, 1.0, 0.4],
        [6.0, -1.0, 0.3],
    ])


@pytest.fixture
def barrier_set(multi_obstacles):
    return BarrierFunctionSet(
        multi_obstacles, robot_radius=0.2, safety_margin=0.3,
        activation_distance=3.0,
    )


# ─────────────────────────────────────────────────────────────
# CircleBarrier 테스트
# ─────────────────────────────────────────────────────────────

class TestCircleBarrier:

    def test_evaluate_inside_obstacle_negative(self, barrier):
        """장애물 안쪽 (d < d_safe) → h < 0."""
        state = np.array([0.3, 0.0, 0.0])  # d_safe = 0.5+0.2+0.3 = 1.0
        h = barrier.evaluate(state)
        assert h < 0, f"Expected h < 0, got {h}"

    def test_evaluate_outside_obstacle_positive(self, barrier):
        """장애물 바깥 (d > d_safe) → h > 0."""
        state = np.array([3.0, 0.0, 0.0])  # dist=3.0 > d_safe=1.0
        h = barrier.evaluate(state)
        assert h > 0, f"Expected h > 0, got {h}"

    def test_evaluate_on_boundary_zero(self, barrier):
        """안전 경계 위 → h ≈ 0."""
        # d_safe = 1.0, state at dist=1.0
        state = np.array([1.0, 0.0, 0.0])
        h = barrier.evaluate(state)
        np.testing.assert_allclose(h, 0.0, atol=1e-10)

    def test_gradient_direction_points_away(self, barrier):
        """gradient 방향이 장애물에서 멀어지는 방향."""
        state = np.array([0.5, 0.5, 0.0])
        grad = barrier.gradient(state)
        # ∇h = [2*(x-ox), 2*(y-oy), 0]
        assert grad[0] > 0, "gradient x should point away from obstacle"
        assert grad[1] > 0, "gradient y should point away from obstacle"
        assert grad[2] == 0.0, "gradient theta should be 0"

    def test_gradient_values(self, barrier):
        """gradient 수치 정확성."""
        state = np.array([2.0, 1.0, 0.5])
        grad = barrier.gradient(state)
        np.testing.assert_allclose(grad[0], 4.0, atol=1e-10)  # 2*(2-0)
        np.testing.assert_allclose(grad[1], 2.0, atol=1e-10)  # 2*(1-0)
        np.testing.assert_allclose(grad[2], 0.0, atol=1e-10)

    def test_batch_consistency(self, barrier):
        """evaluate_batch와 evaluate 결과 일치."""
        states = np.array([
            [0.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        h_batch = barrier.evaluate_batch(states)
        h_single = np.array([barrier.evaluate(s) for s in states])
        np.testing.assert_allclose(h_batch, h_single, atol=1e-10)


# ─────────────────────────────────────────────────────────────
# BarrierFunctionSet 테스트
# ─────────────────────────────────────────────────────────────

class TestBarrierFunctionSet:

    def test_active_barriers_filtering(self, barrier_set):
        """activation_distance 내의 장애물만 활성."""
        # (0,0) 근처 → obs[0]=(2,0)만 3.0 이내
        state = np.array([0.0, 0.0, 0.0])
        active = barrier_set.get_active_barriers(state)
        assert len(active) == 1
        np.testing.assert_allclose(active[0].obs_x, 2.0)

    def test_active_barriers_multiple(self, barrier_set):
        """여러 장애물이 활성화 거리 내."""
        # (3,0.5) → obs[0]=(2,0) dist≈1.12, obs[1]=(4,1) dist≈1.12
        state = np.array([3.0, 0.5, 0.0])
        active = barrier_set.get_active_barriers(state)
        assert len(active) >= 2

    def test_evaluate_all(self, barrier_set):
        """evaluate_all은 모든 barrier 값 반환."""
        state = np.array([0.0, 0.0, 0.0])
        h_all = barrier_set.evaluate_all(state)
        assert len(h_all) == 3

    def test_set_obstacles_updates(self, barrier_set):
        """set_obstacles로 장애물 갱신."""
        new_obs = np.array([[10.0, 10.0, 0.5]])
        barrier_set.set_obstacles(new_obs)
        assert len(barrier_set.barriers) == 1
        np.testing.assert_allclose(barrier_set.barriers[0].obs_x, 10.0)

    def test_empty_obstacles(self):
        """장애물 없는 경우."""
        bs = BarrierFunctionSet()
        h = bs.evaluate_all(np.array([0.0, 0.0, 0.0]))
        assert len(h) == 0
        active = bs.get_active_barriers(np.array([0.0, 0.0, 0.0]))
        assert len(active) == 0

    def test_multiple_obstacles_evaluate_all(self, multi_obstacles):
        """다중 장애물 evaluate_all 정확성."""
        bs = BarrierFunctionSet(multi_obstacles, robot_radius=0.2, safety_margin=0.3)
        state = np.array([2.0, 0.0, 0.0])
        h_all = bs.evaluate_all(state)
        assert len(h_all) == 3
        # obs[0]=(2,0,0.3): dist=0 → h = 0 - (0.3+0.2+0.3)² = -0.64
        d_safe_0 = 0.3 + 0.2 + 0.3
        expected_h0 = 0.0 - d_safe_0 ** 2
        np.testing.assert_allclose(h_all[0], expected_h0, atol=1e-10)
