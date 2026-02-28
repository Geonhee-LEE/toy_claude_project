"""
Tier 3: Advanced features — AdaptiveTemperature, TubeMPPI, AncillaryController, CBF, SG Filter.
10 tests total.
"""

import numpy as np
import pytest

from mpc_controller_ros2.mppi import (
    MPPIParams,
    BatchDynamicsWrapper,
    AdaptiveTemperature,
    TubeMPPI,
    AncillaryController,
    CircleBarrier,
    BarrierFunctionSet,
    CBFSafetyFilter,
    SavitzkyGolayFilter,
)


# ============================================================================
# AdaptiveTemperature
# ============================================================================
class TestAdaptiveTemperature:
    def test_update(self):
        at = AdaptiveTemperature(10.0, 0.5, 0.1, 0.1, 100.0)
        assert at.getLambda() == pytest.approx(10.0)
        # Low ESS → lambda should increase
        lam = at.update(10.0, 1000)  # ESS/K = 0.01 << target 0.5
        assert lam > 10.0

    def test_reset(self):
        at = AdaptiveTemperature(10.0)
        at.update(10.0, 1000)
        at.reset(5.0)
        assert at.getLambda() == pytest.approx(5.0)


# ============================================================================
# TubeMPPI
# ============================================================================
class TestTubeMPPI:
    def test_corrected_control(self):
        p = MPPIParams()
        p.tube_enabled = True
        p.tube_width = 0.5
        tube = TubeMPPI(p)
        nominal_ctrl = np.array([0.5, 0.1])
        nominal_traj = np.zeros((p.N + 1, 3))
        nominal_traj[:, 0] = np.linspace(0, 1, p.N + 1)
        actual_state = np.array([0.05, 0.02, 0.01])  # slight offset
        corrected, info = tube.computeCorrectedControl(
            nominal_ctrl, nominal_traj, actual_state)
        assert corrected.shape == (2,)
        assert info.tube_width > 0

    def test_is_inside_tube(self):
        p = MPPIParams()
        p.tube_width = 1.0
        tube = TubeMPPI(p)
        nom = np.array([0.0, 0.0, 0.0])
        inside = np.array([0.1, 0.1, 0.0])
        outside = np.array([5.0, 5.0, 0.0])
        assert tube.isInsideTube(nom, inside) is True
        assert tube.isInsideTube(nom, outside) is False


# ============================================================================
# AncillaryController
# ============================================================================
class TestAncillaryController:
    def test_body_frame_error(self):
        ac = AncillaryController(0.8, 0.5, 1.0)
        nominal = np.array([1.0, 0.0, 0.0])
        actual = np.array([0.9, 0.1, 0.05])
        error = ac.computeBodyFrameError(nominal, actual)
        assert error.shape == (3,)

    def test_feedback_correction(self):
        ac = AncillaryController(0.8, 0.5, 1.0)
        nominal_ctrl = np.array([0.5, 0.1])
        nominal_state = np.array([1.0, 0.0, 0.0])
        actual_state = np.array([0.9, 0.1, 0.05])
        corrected = ac.computeCorrectedControl(
            nominal_ctrl, nominal_state, actual_state)
        assert corrected.shape == (2,)
        # Corrected should differ from nominal when there's error
        assert not np.allclose(corrected, nominal_ctrl)


# ============================================================================
# CBF
# ============================================================================
class TestCBF:
    def test_barrier_evaluate(self):
        barrier = CircleBarrier(5.0, 0.0, 0.5, 0.2, 0.3)
        state = np.array([0.0, 0.0, 0.0])
        h = barrier.evaluate(state)
        # Far from obstacle → h > 0 (safe)
        assert h > 0

    def test_barrier_set_filtering(self):
        bfs = BarrierFunctionSet(0.2, 0.3, 3.0)
        # Obstacle at (2,0) within activation distance, (10,0) outside
        obstacles = [np.array([2.0, 0.0, 0.3]), np.array([10.0, 0.0, 0.3])]
        bfs.setObstacles(obstacles)
        assert bfs.size() == 2
        state = np.array([0.0, 0.0, 0.0])
        n_active = bfs.getActiveBarriers(state)
        assert n_active == 1  # only (2,0) is within 3.0m

    def test_safety_filter_passthrough(self):
        bfs = BarrierFunctionSet(0.2, 0.3, 3.0)
        # No obstacles → filter should pass through
        p = MPPIParams()
        dyn = BatchDynamicsWrapper(p)
        u_min = np.array([-1.0, -1.0])
        u_max = np.array([1.0, 1.0])
        sf = CBFSafetyFilter(bfs, 1.0, 0.1, u_min, u_max)
        state = np.array([0.0, 0.0, 0.0])
        u_mppi = np.array([0.5, 0.1])
        u_safe, info = sf.filter(state, u_mppi, dyn)
        np.testing.assert_allclose(u_safe, u_mppi, atol=1e-10)
        assert info.filter_applied is False


# ============================================================================
# SavitzkyGolayFilter
# ============================================================================
class TestSavitzkyGolayFilter:
    def test_smoothing(self):
        sg = SavitzkyGolayFilter(3, 3, 2)
        assert sg.windowSize() == 7
        assert sg.halfWindow() == 3
        # Create noisy control sequence
        N = 10
        ctrl_seq = np.zeros((N, 2))
        ctrl_seq[:, 0] = 0.5 + np.random.randn(N) * 0.01
        ctrl_seq[:, 1] = 0.1
        # Push some history
        for i in range(3):
            sg.pushHistory(np.array([0.5, 0.1]))
        smoothed = sg.apply(ctrl_seq)
        assert smoothed.shape == (2,)
        # Reset and verify
        sg.reset()
        coeffs = sg.coefficients()
        assert len(coeffs) == 7
