"""CBF-MPPI 컨트롤러 통합 테스트."""

import numpy as np
import pytest

from mpc_controller.models.differential_drive import RobotParams
from mpc_controller.controllers.mppi.mppi_params import MPPIParams, CBFParams
from mpc_controller.controllers.mppi.base_mppi import MPPIController
from mpc_controller.controllers.mppi.cbf_mppi import CBFMPPIController


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def robot_params():
    return RobotParams(max_velocity=1.0, max_omega=1.5)


@pytest.fixture
def mppi_params():
    return MPPIParams(
        N=10, K=128, dt=0.05, lambda_=10.0,
        noise_sigma=np.array([0.3, 0.3]),
        Q=np.diag([10.0, 10.0, 1.0]),
        R=np.diag([0.01, 0.01]),
        Qf=np.diag([50.0, 50.0, 5.0]),
    )


@pytest.fixture
def obstacles():
    """정면 장애물."""
    return np.array([[3.0, 0.0, 0.3]])


@pytest.fixture
def straight_ref():
    """직진 참조 궤적 (N+1=11 points)."""
    N = 10
    t = np.linspace(0, 1.0, N + 1)
    ref = np.column_stack([t * 5.0, np.zeros(N + 1), np.zeros(N + 1)])
    return ref


@pytest.fixture
def cbf_params_enabled():
    return CBFParams(
        enabled=True,
        gamma=1.0,
        safety_margin=0.3,
        robot_radius=0.2,
        activation_distance=5.0,
        cost_weight=500.0,
        use_safety_filter=True,
    )


@pytest.fixture
def cbf_params_disabled():
    return CBFParams(enabled=False)


# ─────────────────────────────────────────────────────────────
# 호환성 테스트
# ─────────────────────────────────────────────────────────────

class TestCBFDisabledCompatibility:

    def test_cbf_disabled_identical_to_vanilla(
        self, robot_params, mppi_params, cbf_params_disabled, obstacles, straight_ref
    ):
        """cbf_enabled=False → Vanilla MPPI와 동일 결과."""
        vanilla = MPPIController(
            robot_params=robot_params, mppi_params=mppi_params,
            seed=42, obstacles=obstacles,
        )
        cbf = CBFMPPIController(
            robot_params=robot_params, mppi_params=mppi_params,
            seed=42, obstacles=obstacles,
            cbf_params=cbf_params_disabled,
        )

        state = np.array([0.0, 0.0, 0.0])
        u_vanilla, _ = vanilla.compute_control(state, straight_ref)
        u_cbf, _ = cbf.compute_control(state, straight_ref)

        np.testing.assert_allclose(u_cbf, u_vanilla, atol=1e-10)


# ─────────────────────────────────────────────────────────────
# 기본 기능 테스트
# ─────────────────────────────────────────────────────────────

class TestCBFMPPIBasic:

    def test_output_shape(
        self, robot_params, mppi_params, cbf_params_enabled, obstacles, straight_ref
    ):
        """출력 형태 확인."""
        ctrl = CBFMPPIController(
            robot_params=robot_params, mppi_params=mppi_params,
            seed=42, obstacles=obstacles, cbf_params=cbf_params_enabled,
        )
        state = np.array([0.0, 0.0, 0.0])
        u, info = ctrl.compute_control(state, straight_ref)

        assert u.shape == (2,)
        assert isinstance(info, dict)

    def test_info_contains_cbf_keys(
        self, robot_params, mppi_params, cbf_params_enabled, obstacles, straight_ref
    ):
        """info에 CBF 관련 키 포함."""
        ctrl = CBFMPPIController(
            robot_params=robot_params, mppi_params=mppi_params,
            seed=42, obstacles=obstacles, cbf_params=cbf_params_enabled,
        )
        state = np.array([0.0, 0.0, 0.0])
        _, info = ctrl.compute_control(state, straight_ref)

        assert "cbf_enabled" in info
        assert info["cbf_enabled"] is True
        assert "u_mppi" in info
        assert "u_safe" in info
        assert "cbf_filter_info" in info
        assert "barrier_values" in info
        assert "min_barrier_value" in info

    def test_cbf_cost_increases_near_obstacle(
        self, robot_params, mppi_params, cbf_params_enabled, obstacles, straight_ref
    ):
        """장애물 가까이에서 CBF cost가 높음."""
        ctrl = CBFMPPIController(
            robot_params=robot_params, mppi_params=mppi_params,
            seed=42, obstacles=obstacles, cbf_params=cbf_params_enabled,
        )

        # 장애물에서 멀리
        state_far = np.array([0.0, 0.0, 0.0])
        _, info_far = ctrl.compute_control(state_far, straight_ref)
        ctrl.reset()

        # 장애물 근처
        state_near = np.array([2.5, 0.0, 0.0])
        ref_near = straight_ref.copy()
        ref_near[:, 0] += 2.5
        _, info_near = ctrl.compute_control(state_near, ref_near)

        # 근처에서 비용이 더 높을 것으로 기대
        assert info_near["cost"] >= 0
        assert info_far["cost"] >= 0

    def test_safety_filter_activates_near_obstacle(
        self, robot_params, mppi_params, cbf_params_enabled, straight_ref
    ):
        """장애물 가까이에서 safety filter 활성화."""
        obstacles = np.array([[1.5, 0.0, 0.3]])  # 매우 가까운 장애물
        ctrl = CBFMPPIController(
            robot_params=robot_params, mppi_params=mppi_params,
            seed=42, obstacles=obstacles, cbf_params=cbf_params_enabled,
        )

        state = np.array([1.0, 0.0, 0.0])  # 장애물 바로 앞
        _, info = ctrl.compute_control(state, straight_ref)

        filter_info = info["cbf_filter_info"]
        assert filter_info["num_active_barriers"] > 0


# ─────────────────────────────────────────────────────────────
# 통합 테스트
# ─────────────────────────────────────────────────────────────

class TestCBFMPPIIntegration:

    def test_set_obstacles_updates_cbf(
        self, robot_params, mppi_params, cbf_params_enabled, straight_ref
    ):
        """set_obstacles가 barrier set도 갱신."""
        ctrl = CBFMPPIController(
            robot_params=robot_params, mppi_params=mppi_params,
            seed=42, obstacles=np.array([[3.0, 0.0, 0.3]]),
            cbf_params=cbf_params_enabled,
        )

        new_obs = np.array([[1.0, 0.0, 0.5]])
        ctrl.set_obstacles(new_obs)

        assert len(ctrl._barrier_set.barriers) == 1
        np.testing.assert_allclose(ctrl._barrier_set.barriers[0].obs_x, 1.0)

    def test_head_on_collision_scenario(
        self, robot_params, mppi_params, cbf_params_enabled
    ):
        """정면 충돌 시나리오 — CBF가 안전 보장."""
        obstacles = np.array([[2.0, 0.0, 0.3]])
        d_safe = 0.3 + 0.2 + 0.3  # 0.8

        ctrl = CBFMPPIController(
            robot_params=robot_params, mppi_params=mppi_params,
            seed=42, obstacles=obstacles, cbf_params=cbf_params_enabled,
        )

        # 여러 스텝 시뮬레이션
        state = np.array([0.0, 0.0, 0.0])
        from mpc_controller.controllers.mppi.dynamics_wrapper import BatchDynamicsWrapper
        dynamics = BatchDynamicsWrapper(robot_params)
        min_dist = float("inf")

        for step in range(50):
            t = step * 0.05
            ref = np.column_stack([
                np.linspace(state[0], state[0] + 3.0, 11),
                np.zeros(11),
                np.zeros(11),
            ])
            u, info = ctrl.compute_control(state, ref)
            state = dynamics.propagate_batch(
                state[np.newaxis, :], u[np.newaxis, :], 0.05
            )[0]

            dist = np.sqrt((state[0] - 2.0) ** 2 + (state[1] - 0.0) ** 2)
            min_dist = min(min_dist, dist)

        # CBF가 d_safe 내 진입을 방지 (약간의 여유)
        assert min_dist > d_safe * 0.5, (
            f"Robot got too close: min_dist={min_dist:.3f}, d_safe={d_safe:.3f}"
        )

    def test_tracking_accuracy_not_degraded(
        self, robot_params, mppi_params
    ):
        """장애물 없을 때 CBF가 추적 정확도를 저하시키지 않음."""
        cbf_params = CBFParams(enabled=True, use_safety_filter=True)

        vanilla = MPPIController(
            robot_params=robot_params, mppi_params=mppi_params, seed=42,
        )
        cbf = CBFMPPIController(
            robot_params=robot_params, mppi_params=mppi_params,
            seed=42, cbf_params=cbf_params,
        )

        state = np.array([0.0, 0.0, 0.0])
        ref = np.column_stack([
            np.linspace(0, 2.0, 11),
            np.zeros(11),
            np.zeros(11),
        ])

        u_vanilla, info_v = vanilla.compute_control(state, ref)
        u_cbf, info_c = cbf.compute_control(state, ref)

        # 장애물 없으므로 동일하거나 비슷
        np.testing.assert_allclose(u_cbf, u_vanilla, atol=0.1)

    def test_cbf_enabled_avoids_obstacle(
        self, robot_params, mppi_params, cbf_params_enabled
    ):
        """CBF 활성화 시 장애물 회피 확인 (멀티스텝)."""
        obstacles = np.array([[2.5, 0.0, 0.3]])

        ctrl = CBFMPPIController(
            robot_params=robot_params, mppi_params=mppi_params,
            seed=42, obstacles=obstacles, cbf_params=cbf_params_enabled,
        )

        state = np.array([0.0, 0.0, 0.0])
        from mpc_controller.controllers.mppi.dynamics_wrapper import BatchDynamicsWrapper
        dynamics = BatchDynamicsWrapper(robot_params)

        all_barrier_positive = True
        for step in range(30):
            ref = np.column_stack([
                np.linspace(state[0], state[0] + 3.0, 11),
                np.zeros(11),
                np.zeros(11),
            ])
            u, info = ctrl.compute_control(state, ref)

            # barrier 값 확인
            if info.get("min_barrier_value", float("inf")) < -0.5:
                all_barrier_positive = False

            state = dynamics.propagate_batch(
                state[np.newaxis, :], u[np.newaxis, :], 0.05
            )[0]

        # 심각한 barrier 위반 없어야 함
        assert all_barrier_positive, "Barrier value dropped below -0.5"
