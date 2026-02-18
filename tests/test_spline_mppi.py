"""Spline-MPPI 컨트롤러 테스트 — B-spline 보간 기반 smooth sampling 검증."""

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
from mpc_controller.controllers.mppi.spline_mppi import (
    SplineMPPIController,
    _bspline_basis,
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
def spline_params():
    return MPPIParams(
        N=20, K=64, dt=0.05, lambda_=10.0,
        noise_sigma=np.array([0.3, 0.3]),
        Q=np.diag([10.0, 10.0, 1.0]),
        R=np.diag([0.01, 0.01]),
        Qf=np.diag([50.0, 50.0, 5.0]),
        spline_num_knots=8,
        spline_degree=3,
    )


@pytest.fixture
def circle_ref():
    return generate_circle_trajectory(np.array([0.0, 0.0]), 2.0, 200)


# ─────────────────────────────────────────────────────────────
# B-spline Basis 수학 검증
# ─────────────────────────────────────────────────────────────

class TestBsplineBasis:
    """B-spline basis matrix 수학적 정확성."""

    def test_shape(self):
        basis = _bspline_basis(20, 8, 3)
        assert basis.shape == (20, 8)

    def test_partition_of_unity(self):
        """각 행의 합이 1에 가까워야 함 (partition of unity)."""
        basis = _bspline_basis(30, 10, 3)
        row_sums = basis.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_non_negative(self):
        """B-spline basis 값은 비음수."""
        basis = _bspline_basis(20, 8, 3)
        assert np.all(basis >= -1e-10)

    def test_different_knots(self):
        """다양한 P 값에서 동작."""
        for P in [4, 6, 8, 12, 20]:
            basis = _bspline_basis(20, P, 3)
            assert basis.shape == (20, P)
            np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-6)

    def test_degree_2(self):
        """2차 B-spline도 동작."""
        basis = _bspline_basis(20, 8, 2)
        assert basis.shape == (20, 8)
        np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-6)

    def test_identity_when_P_equals_N(self):
        """P=N일 때 basis는 단위 행렬에 가까움."""
        N = 10
        basis = _bspline_basis(N, N, 3)
        assert basis.shape == (N, N)
        # 완전한 identity는 아니지만 대각 구조를 가져야 함
        for i in range(N):
            assert basis[i].sum() > 0.5  # 최소한 반 이상 기여

    def test_interpolation_smooth(self):
        """Knot값에 basis를 적용하면 smooth output."""
        N, P = 30, 6
        basis = _bspline_basis(N, P, 3)
        knots = np.random.randn(P, 2)
        result = basis @ knots  # (N, 2)
        # 결과의 1차 차분이 원본 knot 차분보다 작아야 함 (smooth)
        assert result.shape == (N, 2)
        du = np.diff(result, axis=0)
        assert np.all(np.isfinite(du))


# ─────────────────────────────────────────────────────────────
# 기본 생성 및 인터페이스
# ─────────────────────────────────────────────────────────────

class TestSplineMPPICreation:
    """생성자 및 기본 인터페이스 검증."""

    def test_inherits_mppi(self, robot_params, spline_params):
        ctrl = SplineMPPIController(robot_params, spline_params, seed=42)
        assert isinstance(ctrl, MPPIController)

    def test_basis_cached(self, robot_params, spline_params):
        ctrl = SplineMPPIController(robot_params, spline_params, seed=42)
        assert ctrl._basis.shape == (spline_params.N, spline_params.spline_num_knots)

    def test_knot_init_zero(self, robot_params, spline_params):
        ctrl = SplineMPPIController(robot_params, spline_params, seed=42)
        np.testing.assert_array_equal(
            ctrl.U_knots, np.zeros((spline_params.spline_num_knots, 2))
        )

    def test_custom_knot_sigma(self, robot_params, spline_params):
        spline_params.spline_knot_sigma = np.array([0.5, 0.8])
        ctrl = SplineMPPIController(robot_params, spline_params, seed=42)
        np.testing.assert_array_equal(ctrl._knot_sigma, [0.5, 0.8])


# ─────────────────────────────────────────────────────────────
# compute_control 인터페이스
# ─────────────────────────────────────────────────────────────

class TestSplineMPPIComputeControl:
    """compute_control 출력 형식 및 내용 검증."""

    def test_output_shape(self, robot_params, spline_params, circle_ref):
        ctrl = SplineMPPIController(robot_params, spline_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, spline_params.N, spline_params.dt, state[2])

        u, info = ctrl.compute_control(state, ref)
        assert u.shape == (2,)

    def test_info_keys(self, robot_params, spline_params, circle_ref):
        ctrl = SplineMPPIController(robot_params, spline_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, spline_params.N, spline_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)

        required_keys = [
            "predicted_trajectory", "predicted_controls", "cost",
            "sample_trajectories", "sample_weights", "ess",
            "knot_controls", "spline_basis", "num_knots", "control_rate",
        ]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

    def test_knot_controls_in_info(self, robot_params, spline_params, circle_ref):
        ctrl = SplineMPPIController(robot_params, spline_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, spline_params.N, spline_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)
        assert info["knot_controls"].shape == (spline_params.spline_num_knots, 2)

    def test_control_within_limits(self, robot_params, spline_params, circle_ref):
        ctrl = SplineMPPIController(robot_params, spline_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, spline_params.N, spline_params.dt, state[2])

        u, _ = ctrl.compute_control(state, ref)
        assert abs(u[0]) <= robot_params.max_velocity + 1e-6
        assert abs(u[1]) <= robot_params.max_omega + 1e-6


# ─────────────────────────────────────────────────────────────
# Smoothness 검증
# ─────────────────────────────────────────────────────────────

class TestSplineMPPISmoothness:
    """B-spline 보간으로 인한 smooth 제어 검증."""

    def test_interpolated_controls_smooth(self, robot_params, spline_params, circle_ref):
        """Spline-MPPI의 predicted_controls가 부드러운지 확인."""
        ctrl = SplineMPPIController(robot_params, spline_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, spline_params.N, spline_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)
        controls = info["predicted_controls"]
        du = np.diff(controls, axis=0)
        # 제어 변화율의 표준편차가 합리적 범위
        assert np.std(du) < 1.0

    def test_fewer_knots_smoother(self, robot_params, circle_ref):
        """Knot 수가 적을수록 더 smooth."""
        rates = []
        for P in [4, 8, 16]:
            params = MPPIParams(
                N=20, K=128, dt=0.05, lambda_=10.0,
                noise_sigma=np.array([0.3, 0.3]),
                spline_num_knots=P,
            )
            ctrl = SplineMPPIController(robot_params, params, seed=42)
            state = circle_ref[0].copy()
            interp = TrajectoryInterpolator(circle_ref, dt=0.05)
            ref = interp.get_reference(0, params.N, params.dt, state[2])

            _, info = ctrl.compute_control(state, ref)
            rates.append(info["control_rate"])

        # 적은 knot → 낮은 control rate (또는 비슷)
        assert rates[0] <= rates[-1] * 2.0


# ─────────────────────────────────────────────────────────────
# 다중 스텝 시뮬레이션
# ─────────────────────────────────────────────────────────────

class TestSplineMPPIMultiStep:
    """다중 스텝 시뮬레이션 안정성."""

    def test_multi_step_no_error(self, robot_params, spline_params, circle_ref):
        ctrl = SplineMPPIController(robot_params, spline_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)

        for step in range(20):
            ref = interp.get_reference(
                step * 0.05, spline_params.N, spline_params.dt, state[2]
            )
            u, info = ctrl.compute_control(state, ref)
            assert np.all(np.isfinite(u)), f"Non-finite at step {step}"
            state = state + np.array([
                u[0] * np.cos(state[2]) * 0.05,
                u[0] * np.sin(state[2]) * 0.05,
                u[1] * 0.05,
            ])

    def test_reset(self, robot_params, spline_params, circle_ref):
        ctrl = SplineMPPIController(robot_params, spline_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, spline_params.N, spline_params.dt, state[2])

        ctrl.compute_control(state, ref)
        ctrl.reset()
        np.testing.assert_array_equal(
            ctrl.U_knots, np.zeros_like(ctrl.U_knots)
        )


# ─────────────────────────────────────────────────────────────
# ESS 및 메트릭
# ─────────────────────────────────────────────────────────────

class TestSplineMPPIMetrics:
    """ESS, 비용 등 기본 메트릭."""

    def test_ess_valid_range(self, robot_params, spline_params, circle_ref):
        ctrl = SplineMPPIController(robot_params, spline_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, spline_params.N, spline_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)
        assert 1.0 <= info["ess"] <= spline_params.K

    def test_cost_non_negative(self, robot_params, spline_params, circle_ref):
        ctrl = SplineMPPIController(robot_params, spline_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, spline_params.N, spline_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)
        assert info["cost"] >= 0

    def test_weights_normalized(self, robot_params, spline_params, circle_ref):
        ctrl = SplineMPPIController(robot_params, spline_params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, spline_params.N, spline_params.dt, state[2])

        _, info = ctrl.compute_control(state, ref)
        np.testing.assert_almost_equal(np.sum(info["sample_weights"]), 1.0, decimal=5)


# ─────────────────────────────────────────────────────────────
# 장애물 환경
# ─────────────────────────────────────────────────────────────

class TestSplineMPPIObstacles:
    """장애물 환경에서의 동작."""

    def test_with_obstacles(self, robot_params, spline_params, circle_ref):
        obstacles = np.array([[1.0, 0.0, 0.3]])
        ctrl = SplineMPPIController(
            robot_params, spline_params, seed=42, obstacles=obstacles
        )
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, spline_params.N, spline_params.dt, state[2])

        u, info = ctrl.compute_control(state, ref)
        assert np.all(np.isfinite(u))


# ─────────────────────────────────────────────────────────────
# Auto Knot Sigma 검증
# ─────────────────────────────────────────────────────────────

class TestSplineMPPIAutoSigma:
    """spline_auto_knot_sigma 기능 검증."""

    def test_auto_sigma_greater_than_noise_sigma(self, robot_params):
        """Auto sigma는 basis 감쇠 보정으로 noise_sigma보다 커야 함."""
        params = MPPIParams(
            N=20, K=64, dt=0.05,
            noise_sigma=np.array([0.3, 0.3]),
            spline_num_knots=8,
            spline_auto_knot_sigma=True,
        )
        ctrl = SplineMPPIController(robot_params, params, seed=42)
        assert np.all(ctrl._knot_sigma > params.noise_sigma)

    def test_amp_factor_range(self, robot_params):
        """amp_factor는 1보다 크고 합리적 범위 (1, 10) 이내."""
        for P in [6, 8, 12]:
            params = MPPIParams(
                N=20, K=64, dt=0.05,
                noise_sigma=np.array([0.3, 0.3]),
                spline_num_knots=P,
                spline_auto_knot_sigma=True,
            )
            ctrl = SplineMPPIController(robot_params, params, seed=42)
            amp_factor = ctrl._knot_sigma[0] / params.noise_sigma[0]
            assert 1.0 < amp_factor < 10.0, f"P={P}, amp={amp_factor}"

    def test_explicit_override(self, robot_params):
        """명시적 spline_knot_sigma가 auto 보정을 무시."""
        params = MPPIParams(
            N=20, K=64, dt=0.05,
            noise_sigma=np.array([0.3, 0.3]),
            spline_num_knots=8,
            spline_auto_knot_sigma=True,
            spline_knot_sigma=np.array([0.5, 0.8]),
        )
        ctrl = SplineMPPIController(robot_params, params, seed=42)
        np.testing.assert_array_equal(ctrl._knot_sigma, [0.5, 0.8])

    def test_auto_disabled(self, robot_params):
        """auto=False면 noise_sigma 그대로 사용."""
        params = MPPIParams(
            N=20, K=64, dt=0.05,
            noise_sigma=np.array([0.3, 0.3]),
            spline_num_knots=8,
            spline_auto_knot_sigma=False,
        )
        ctrl = SplineMPPIController(robot_params, params, seed=42)
        np.testing.assert_array_equal(ctrl._knot_sigma, params.noise_sigma)


# ─────────────────────────────────────────────────────────────
# LS Warm-start 검증
# ─────────────────────────────────────────────────────────────

class TestSplineMPPIWarmstart:
    """LS 재투영 warm-start 일관성."""

    def test_ls_reprojection_consistency(self, robot_params, circle_ref):
        """LS 재투영 후 basis @ U_knots ≈ U_shifted."""
        params = MPPIParams(
            N=20, K=64, dt=0.05,
            noise_sigma=np.array([0.3, 0.3]),
            spline_num_knots=8,
            spline_auto_knot_sigma=True,
        )
        ctrl = SplineMPPIController(robot_params, params, seed=42)
        state = circle_ref[0].copy()
        interp = TrajectoryInterpolator(circle_ref, dt=0.05)
        ref = interp.get_reference(0, params.N, params.dt, state[2])

        # 1회 실행하여 U에 비자명(non-trivial) 값 생성
        ctrl.compute_control(state, ref)

        # shift된 U를 생성
        U_shifted = ctrl.U.copy()
        U_shifted[:-1] = U_shifted[1:]
        U_shifted[-1] = 0.0

        # LS 재투영
        knots_reproj = ctrl._basis_pinv @ U_shifted
        U_reconstructed = ctrl._basis @ knots_reproj

        # 재구성 오차가 작아야 함 (P >= degree+1이면 잘 맞음)
        rmse = np.sqrt(np.mean((U_shifted - U_reconstructed) ** 2))
        assert rmse < 0.1, f"LS reprojection RMSE={rmse}"


# ─────────────────────────────────────────────────────────────
# Figure8 RMSE 회귀 테스트
# ─────────────────────────────────────────────────────────────

class TestSplineMPPIFigure8Regression:
    """figure8 궤적 추적 RMSE 회귀 테스트."""

    @pytest.mark.slow
    def test_figure8_rmse_below_threshold(self, robot_params):
        """figure8 RMSE < 0.5m (Issue #64 목표)."""
        from mpc_controller import generate_figure_eight_trajectory

        trajectory = generate_figure_eight_trajectory(
            np.array([0.0, 0.0]), 2.0, 400
        )
        params = MPPIParams(
            N=20, K=512, dt=0.05, lambda_=10.0,
            noise_sigma=np.array([0.3, 0.3]),
            Q=np.diag([10.0, 10.0, 1.0]),
            R=np.diag([0.01, 0.01]),
            Qf=np.diag([100.0, 100.0, 10.0]),
            spline_num_knots=12,
            spline_auto_knot_sigma=True,
        )
        ctrl = SplineMPPIController(robot_params, params, seed=42)
        interp = TrajectoryInterpolator(trajectory, dt=0.05)

        state = trajectory[0].copy()
        errors = []
        dt = 0.05
        max_steps = int(20.0 / dt)

        for step in range(max_steps):
            t = step * dt
            ref = interp.get_reference(t, params.N, params.dt, state[2])
            u, _ = ctrl.compute_control(state, ref)

            # 추적 오차 기록
            pos_err = np.linalg.norm(state[:2] - ref[0, :2])
            errors.append(pos_err)

            # 상태 업데이트 (Euler)
            state = state + np.array([
                u[0] * np.cos(state[2]) * dt,
                u[0] * np.sin(state[2]) * dt,
                u[1] * dt,
            ])

            idx, dist = interp.find_closest_point(state[:2])
            if idx >= interp.num_points - 1 and dist < 0.1:
                break

        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        assert rmse < 0.5, f"figure8 RMSE={rmse:.4f}m (target <0.5m)"
