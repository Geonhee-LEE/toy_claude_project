"""
Tier 1: Core bindings tests — MPPIParams, MotionModel, Dynamics, Samplers, Utils.
15 tests total.
"""

import numpy as np
import pytest

from mpc_controller_ros2.mppi import (
    MPPIParams,
    DiffDriveModel,
    SwerveDriveModel,
    NonCoaxialSwerveModel,
    create_motion_model,
    BatchDynamicsWrapper,
    GaussianSampler,
    ColoredNoiseSampler,
    normalize_angle,
    softmax_weights,
    compute_ess,
    log_sum_exp,
)


# ============================================================================
# MPPIParams
# ============================================================================
class TestMPPIParams:
    def test_default_values(self):
        p = MPPIParams()
        assert p.N == 30
        assert p.K == 1024
        assert p.dt == pytest.approx(0.1)
        assert p.lambda_ == pytest.approx(10.0)
        assert p.motion_model == "diff_drive"

    def test_field_read_write(self):
        p = MPPIParams()
        p.N = 50
        p.K = 512
        p.lambda_ = 5.0
        p.motion_model = "swerve"
        assert p.N == 50
        assert p.K == 512
        assert p.lambda_ == pytest.approx(5.0)
        assert p.motion_model == "swerve"

    def test_eigen_numpy_conversion(self):
        p = MPPIParams()
        # Q is 3x3 for DiffDrive
        assert p.Q.shape == (3, 3)
        assert p.R.shape == (2, 2)
        assert p.noise_sigma.shape == (2,)
        # Modify via numpy
        p.Q = np.eye(3) * 20.0
        assert p.Q[0, 0] == pytest.approx(20.0)

    def test_get_feedback_gain_matrix(self):
        p = MPPIParams()
        K_fb = p.getFeedbackGainMatrix()
        assert K_fb.shape == (2, 3)  # nu=2, nx=3 for DiffDrive
        assert K_fb[0, 0] == pytest.approx(0.8)  # k_forward
        assert K_fb[1, 1] == pytest.approx(0.5)  # k_lateral
        assert K_fb[1, 2] == pytest.approx(1.0)  # k_angle

    def test_repr(self):
        p = MPPIParams()
        r = repr(p)
        assert "MPPIParams" in r
        assert "N=30" in r


# ============================================================================
# MotionModel
# ============================================================================
class TestMotionModel:
    def test_diff_drive_dims(self):
        m = DiffDriveModel(0.0, 1.0, -1.0, 1.0)
        assert m.stateDim() == 3
        assert m.controlDim() == 2
        assert m.isHolonomic() is False
        assert m.name() == "diff_drive"

    def test_swerve_dims(self):
        m = SwerveDriveModel(-1.0, 1.0, 0.5, 1.0)
        assert m.stateDim() == 3
        assert m.controlDim() == 3
        assert m.isHolonomic() is True
        assert m.name() == "swerve"

    def test_non_coaxial_dims(self):
        m = NonCoaxialSwerveModel(0.0, 1.0, 1.0, 2.0)
        assert m.stateDim() == 4
        assert m.controlDim() == 3
        assert m.isHolonomic() is False
        assert m.name() == "non_coaxial_swerve"

    def test_factory(self):
        p = MPPIParams()
        m = create_motion_model("diff_drive", p)
        assert m.stateDim() == 3
        assert m.controlDim() == 2

    def test_dynamics_batch_shape(self):
        m = DiffDriveModel(0.0, 1.0, -1.0, 1.0)
        M = 10
        states = np.zeros((M, 3))
        controls = np.ones((M, 2)) * 0.5
        x_dot = m.dynamicsBatch(states, controls)
        assert x_dot.shape == (M, 3)

    def test_clip_controls(self):
        m = DiffDriveModel(-0.5, 1.0, -1.0, 1.0)
        controls = np.array([[2.0, 3.0], [-1.0, -2.0]])
        clipped = m.clipControls(controls)
        assert clipped[0, 0] == pytest.approx(1.0)  # v clamped to v_max
        assert clipped[1, 1] == pytest.approx(-1.0)  # omega clamped to omega_min


# ============================================================================
# BatchDynamicsWrapper
# ============================================================================
class TestBatchDynamicsWrapper:
    def test_rollout_shape(self):
        p = MPPIParams()
        p.K = 8
        p.N = 5
        dyn = BatchDynamicsWrapper(p)
        x0 = np.array([0.0, 0.0, 0.0])
        controls = [np.random.randn(p.N, 2) * 0.1 for _ in range(p.K)]
        trajs = dyn.rolloutBatch(x0, controls, p.dt)
        assert len(trajs) == p.K
        assert trajs[0].shape == (p.N + 1, 3)

    def test_propagate_one_step(self):
        p = MPPIParams()
        dyn = BatchDynamicsWrapper(p)
        states = np.array([[0.0, 0.0, 0.0]])
        controls = np.array([[1.0, 0.0]])  # pure forward
        next_states = dyn.propagateBatch(states, controls, 0.1)
        assert next_states.shape == (1, 3)
        assert next_states[0, 0] > 0  # moved forward in x


# ============================================================================
# Samplers
# ============================================================================
class TestSamplers:
    def test_gaussian_shape(self):
        sigma = np.array([0.5, 0.3])
        sampler = GaussianSampler(sigma, seed=42)
        samples = sampler.sample(16, 10, 2)
        assert len(samples) == 16
        assert samples[0].shape == (10, 2)

    def test_colored_shape(self):
        sigma = np.array([0.5, 0.3])
        sampler = ColoredNoiseSampler(sigma, beta=2.0, seed=42)
        samples = sampler.sample(16, 10, 2)
        assert len(samples) == 16
        assert samples[0].shape == (10, 2)

    def test_seed_reproducibility(self):
        sigma = np.array([0.5, 0.3])
        s1 = GaussianSampler(sigma, seed=123)
        s2 = GaussianSampler(sigma, seed=123)
        a = s1.sample(4, 5, 2)
        b = s2.sample(4, 5, 2)
        for i in range(4):
            np.testing.assert_array_equal(a[i], b[i])


# ============================================================================
# NonCoaxialSwerve 60° Steering Constraint
# ============================================================================
class TestNonCoaxialSwerve60Deg:
    def test_non_coaxial_60deg_creation(self):
        """C++ 모델 60° 생성 및 차원 확인."""
        m = NonCoaxialSwerveModel(0.0, 1.5, 2.0, 2.0, np.pi / 3.0)
        assert m.stateDim() == 4
        assert m.controlDim() == 3
        assert m.name() == "non_coaxial_swerve"

    def test_non_coaxial_60deg_clamp(self):
        """clipControls로 제어 클리핑 및 propagate 결과 확인."""
        m = NonCoaxialSwerveModel(0.0, 1.5, 2.0, 2.0, np.pi / 3.0)
        # v=2.0 → 1.5로 클리핑 확인
        controls = np.array([[2.0, 0.0, 0.0]])
        clipped = m.clipControls(controls)
        assert abs(clipped[0, 0] - 1.5) < 1e-10

    def test_non_coaxial_60deg_propagate(self):
        """propagateBatch에서 60° 제약 준수 (rollout 기반)."""
        m = NonCoaxialSwerveModel(0.0, 1.5, 2.0, 2.0, np.pi / 3.0)
        x0 = np.array([0.0, 0.0, 0.0, 0.0])
        # 큰 delta_dot=5로 N=50 롤아웃
        controls_seq = [np.tile([[1.0, 0.0, 5.0]], (50, 1))]
        trajs = m.rolloutBatch(x0, controls_seq, 0.05)
        # 최종 delta가 60° 이내
        final_delta = trajs[0][-1, 3]
        assert final_delta <= np.pi / 3.0 + 1e-6


# ============================================================================
# Utils
# ============================================================================
class TestUtils:
    def test_normalize_angle(self):
        assert normalize_angle(0.0) == pytest.approx(0.0)
        assert normalize_angle(2 * np.pi) == pytest.approx(0.0, abs=1e-10)
        assert abs(normalize_angle(np.pi + 0.1) - (-np.pi + 0.1)) < 1e-10

    def test_softmax_weights_sum_one(self):
        costs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = softmax_weights(costs, 1.0)
        assert w.sum() == pytest.approx(1.0, abs=1e-10)
        assert w[0] > w[-1]  # lower cost → higher weight
