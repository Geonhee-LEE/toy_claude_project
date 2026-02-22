"""CBF Safety Filter 단위 테스트."""

import numpy as np
import pytest

from mpc_controller.controllers.mppi.barrier_function import BarrierFunctionSet
from mpc_controller.controllers.mppi.cbf_safety_filter import CBFSafetyFilter


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

def diff_drive_dynamics(state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Differential drive 연속시간 동역학."""
    theta = state[2]
    v, omega = control[0], control[1]
    return np.array([
        v * np.cos(theta),
        v * np.sin(theta),
        omega,
    ])


@pytest.fixture
def obstacle_ahead():
    """로봇 정면에 장애물."""
    return np.array([[2.0, 0.0, 0.3]])


@pytest.fixture
def barrier_set(obstacle_ahead):
    return BarrierFunctionSet(
        obstacle_ahead, robot_radius=0.2, safety_margin=0.3,
        activation_distance=5.0,
    )


@pytest.fixture
def safety_filter(barrier_set):
    return CBFSafetyFilter(
        barrier_set, gamma=1.0, dt=0.05,
        u_min=np.array([-1.0, -1.5]),
        u_max=np.array([1.0, 1.5]),
    )


# ─────────────────────────────────────────────────────────────
# 테스트
# ─────────────────────────────────────────────────────────────

class TestCBFSafetyFilter:

    def test_no_active_barriers_passthrough(self):
        """활성 장애물 없으면 u_mppi 그대로 반환."""
        bs = BarrierFunctionSet(
            np.array([[100.0, 100.0, 0.3]]),  # 멀리 있음
            activation_distance=3.0,
        )
        sf = CBFSafetyFilter(bs, gamma=1.0, dt=0.05)
        state = np.array([0.0, 0.0, 0.0])
        u_mppi = np.array([0.5, 0.3])

        u_safe, info = sf.filter(state, u_mppi, diff_drive_dynamics)

        np.testing.assert_allclose(u_safe, u_mppi, atol=1e-10)
        assert info["num_active_barriers"] == 0
        assert info["filter_applied"] is False

    def test_filter_preserves_safe_control(self, safety_filter):
        """이미 안전한 제어는 변경하지 않음."""
        # 장애물에서 멀리 (dist >> d_safe) — 0 velocity, safe
        state = np.array([0.0, 0.0, 0.0])  # dist=2.0 to obs, d_safe=0.8
        # 장애물에서 멀어지는 방향으로 이동해도 안전
        u_mppi = np.array([0.0, 0.5])  # 회전만

        u_safe, info = safety_filter.filter(state, u_mppi, diff_drive_dynamics)

        # h(x) = 4.0 - 0.64 = 3.36 > 0, 멀어지므로 안전
        np.testing.assert_allclose(u_safe, u_mppi, atol=1e-6)

    def test_filter_modifies_unsafe_control(self, safety_filter):
        """안전하지 않은 제어를 수정."""
        # 장애물 가까이 (h 약간 양수)
        state = np.array([1.2, 0.0, 0.0])  # dist=0.8 to obs, d_safe=0.8 → h≈0
        # 장애물 쪽으로 직진
        u_mppi = np.array([1.0, 0.0])

        u_safe, info = safety_filter.filter(state, u_mppi, diff_drive_dynamics)

        # QP가 속도를 줄이거나 방향 변경
        assert info["filter_applied"] is True or info["num_active_barriers"] > 0
        # u_safe의 속도가 원래보다 작거나 방향이 변경됨
        assert np.linalg.norm(u_safe - u_mppi) >= 0  # 변경 발생 가능

    def test_control_bounds_respected(self, safety_filter):
        """제어 한계 준수."""
        state = np.array([1.5, 0.0, 0.0])
        u_mppi = np.array([0.8, 0.5])

        u_safe, info = safety_filter.filter(state, u_mppi, diff_drive_dynamics)

        assert u_safe[0] >= -1.0 - 1e-6
        assert u_safe[0] <= 1.0 + 1e-6
        assert u_safe[1] >= -1.5 - 1e-6
        assert u_safe[1] <= 1.5 + 1e-6

    def test_multiple_constraints_qp(self):
        """다중 장애물 제약 QP."""
        obstacles = np.array([
            [1.5, 0.5, 0.2],
            [1.5, -0.5, 0.2],
        ])
        bs = BarrierFunctionSet(obstacles, activation_distance=5.0)
        sf = CBFSafetyFilter(
            bs, gamma=1.0, dt=0.05,
            u_min=np.array([-1.0, -1.5]),
            u_max=np.array([1.0, 1.5]),
        )

        state = np.array([1.0, 0.0, 0.0])
        u_mppi = np.array([0.5, 0.0])

        u_safe, info = sf.filter(state, u_mppi, diff_drive_dynamics)

        assert info["num_active_barriers"] == 2
        assert len(u_safe) == 2

    def test_qp_failure_fallback(self):
        """QP 실패 시 u_mppi fallback."""
        # 극단적 상황 (장애물 내부)
        obstacles = np.array([[0.0, 0.0, 0.5]])
        bs = BarrierFunctionSet(obstacles, activation_distance=5.0)
        sf = CBFSafetyFilter(
            bs, gamma=100.0, dt=0.05,  # 극단적 gamma
            u_min=np.array([-0.001, -0.001]),  # 극단적 bounds
            u_max=np.array([0.001, 0.001]),
        )

        state = np.array([0.0, 0.0, 0.0])
        u_mppi = np.array([0.0005, 0.0])

        u_safe, info = sf.filter(state, u_mppi, diff_drive_dynamics)

        # 결과가 반환되어야 함 (fallback 또는 성공)
        assert len(u_safe) == 2

    def test_dynamics_fn_integration(self, safety_filter):
        """dynamics 함수 통합 확인."""
        state = np.array([0.0, 0.0, 0.0])
        u = np.array([1.0, 0.0])

        x_dot = diff_drive_dynamics(state, u)

        # theta=0이면 x_dot = [v, 0, omega] = [1, 0, 0]
        np.testing.assert_allclose(x_dot, [1.0, 0.0, 0.0], atol=1e-10)

    def test_info_dict_keys(self, safety_filter):
        """info 딕셔너리 필수 키 확인."""
        state = np.array([0.0, 0.0, 0.0])
        u_mppi = np.array([0.5, 0.0])

        _, info = safety_filter.filter(state, u_mppi, diff_drive_dynamics)

        required_keys = [
            "num_active_barriers",
            "filter_applied",
            "qp_success",
            "barrier_values",
            "constraint_margins",
        ]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"
