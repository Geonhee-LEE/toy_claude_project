"""
Tier 2: Cost functions and weight computation tests.
10 tests total.
"""

import numpy as np
import pytest

from mpc_controller_ros2.mppi import (
    MPPIParams,
    BatchDynamicsWrapper,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ControlRateCost,
    PreferForwardCost,
    VelocityTrackingCost,
    CompositeMPPICost,
    VanillaMPPIWeights,
    LogMPPIWeights,
    TsallisMPPIWeights,
    RiskAwareMPPIWeights,
)


def _make_test_data(K=8, N=5, nx=3, nu=2):
    """Helper: create dummy trajectories, controls, reference."""
    x0 = np.zeros(nx)
    trajs = []
    controls = []
    for k in range(K):
        traj = np.zeros((N + 1, nx))
        traj[:, 0] = np.linspace(0, 1.0, N + 1)  # x moves forward
        trajs.append(traj)
        ctrl = np.zeros((N, nu))
        ctrl[:, 0] = 0.5  # constant v
        controls.append(ctrl)
    ref = np.zeros((N + 1, nx))
    ref[:, 0] = np.linspace(0, 1.0, N + 1)
    return trajs, controls, ref


# ============================================================================
# Individual cost functions
# ============================================================================
class TestCostFunctions:
    def test_state_tracking_zero_error(self):
        trajs, controls, ref = _make_test_data()
        Q = np.eye(3)
        cost = StateTrackingCost(Q)
        costs = cost.compute(trajs, controls, ref)
        assert costs.shape == (8,)
        # trajectories == reference → cost ≈ 0
        np.testing.assert_allclose(costs, 0.0, atol=1e-10)

    def test_terminal_cost(self):
        trajs, controls, ref = _make_test_data()
        # Shift reference terminal to create error
        ref[-1, :] = [10.0, 10.0, 0.0]
        Qf = np.eye(3)
        cost = TerminalCost(Qf)
        costs = cost.compute(trajs, controls, ref)
        assert costs.shape == (8,)
        assert np.all(costs > 0)

    def test_control_effort(self):
        trajs, controls, ref = _make_test_data()
        R = np.eye(2) * 0.1
        cost = ControlEffortCost(R)
        costs = cost.compute(trajs, controls, ref)
        assert costs.shape == (8,)
        assert np.all(costs > 0)

    def test_control_rate(self):
        trajs, controls, ref = _make_test_data()
        # Make controls vary to produce rate cost
        for ctrl in controls:
            ctrl[:, 0] = np.linspace(0, 1, len(ctrl))
        R_rate = np.eye(2)
        cost = ControlRateCost(R_rate)
        costs = cost.compute(trajs, controls, ref)
        assert costs.shape == (8,)
        assert np.all(costs > 0)

    def test_prefer_forward(self):
        trajs, controls, ref = _make_test_data()
        # Make some controls negative v
        for ctrl in controls[:4]:
            ctrl[:, 0] = -0.5
        cost = PreferForwardCost(5.0)
        costs = cost.compute(trajs, controls, ref)
        assert costs.shape == (8,)
        # Samples with negative v should have higher cost
        assert costs[0] > costs[4]

    def test_velocity_tracking(self):
        trajs, controls, ref = _make_test_data()
        cost = VelocityTrackingCost(10.0, 1.0, 0.1)
        costs = cost.compute(trajs, controls, ref)
        assert costs.shape == (8,)


# ============================================================================
# CompositeMPPICost
# ============================================================================
class TestCompositeCost:
    def test_composite_sum(self):
        trajs, controls, ref = _make_test_data()
        composite = CompositeMPPICost()
        composite.add_state_tracking(np.eye(3))
        composite.add_control_effort(np.eye(2) * 0.1)
        costs = composite.compute(trajs, controls, ref)
        assert costs.shape == (8,)

    def test_cost_breakdown(self):
        trajs, controls, ref = _make_test_data()
        composite = CompositeMPPICost()
        composite.add_state_tracking(np.eye(3))
        composite.add_control_effort(np.eye(2) * 0.1)
        breakdown = composite.computeDetailed(trajs, controls, ref)
        assert breakdown.total_costs.shape == (8,)
        assert "state_tracking" in breakdown.component_costs
        assert "control_effort" in breakdown.component_costs


# ============================================================================
# WeightComputation
# ============================================================================
class TestWeightComputation:
    def test_vanilla_sum_one(self):
        w = VanillaMPPIWeights()
        costs = np.array([1.0, 2.0, 5.0, 10.0])
        weights = w.compute(costs, 1.0)
        assert weights.sum() == pytest.approx(1.0, abs=1e-10)
        assert weights[0] > weights[-1]

    def test_log_approx_vanilla(self):
        costs = np.random.rand(100) * 10
        v = VanillaMPPIWeights().compute(costs, 5.0)
        l = LogMPPIWeights().compute(costs, 5.0)
        np.testing.assert_allclose(v, l, atol=1e-10)
