"""60° 스티어링 제한 NonCoaxialSwerve 테스트.

python -m pytest tests/test_steering_constraint.py -v --override-ini="addopts="
"""

import numpy as np
import pytest

from mpc_controller import (
    NonCoaxialSwerveDriveModel,
    NonCoaxialSwerveParams,
)
from examples.cpp_vs_python_benchmark.scenario import tight_turn_scenario, slalom_scenario


class TestSteeringConstraint:
    """60° 스티어링 제한 관련 테스트."""

    def test_60deg_clamp(self):
        """delta가 π/3으로 클램프되는지 확인."""
        params = NonCoaxialSwerveParams(max_steering_angle=np.pi / 3)
        model = NonCoaxialSwerveDriveModel(params)

        # delta=90° 상태에서 forward_simulate → 클램프 확인
        state = np.array([0.0, 0.0, 0.0, np.pi / 2])
        control = np.array([0.0, 0.0, 0.0])
        next_state = model.forward_simulate(state, control, 0.05)
        assert abs(next_state[3]) <= np.pi / 3 + 1e-10

    def test_60deg_state_bounds(self):
        """get_state_bounds가 60° 제한을 반영하는지 확인."""
        params = NonCoaxialSwerveParams(max_steering_angle=np.pi / 3)
        model = NonCoaxialSwerveDriveModel(params)

        lb, ub = model.get_state_bounds()
        assert abs(ub[3] - np.pi / 3) < 1e-10
        assert abs(lb[3] - (-np.pi / 3)) < 1e-10

    def test_60deg_control_bounds(self):
        """get_control_bounds 반환값 확인."""
        params = NonCoaxialSwerveParams(max_steering_angle=np.pi / 3)
        model = NonCoaxialSwerveDriveModel(params)

        lb, ub = model.get_control_bounds()
        assert len(lb) == 3
        assert len(ub) == 3
        # v, omega, delta_dot 제한 확인
        assert ub[0] == params.max_speed
        assert ub[1] == params.max_omega
        assert ub[2] == params.max_steering_rate

    def test_60deg_mpc_feasibility(self):
        """MPC가 60° 내에서 해를 찾는지 확인."""
        from mpc_controller import (
            NonCoaxialSwerveMPCController,
            NonCoaxialSwerveMPCParams,
        )

        params = NonCoaxialSwerveParams(max_steering_angle=np.pi / 3)
        mpc_params = NonCoaxialSwerveMPCParams(N=10, dt=0.1)
        controller = NonCoaxialSwerveMPCController(params, mpc_params)

        state = np.array([0.0, 0.0, 0.0, 0.0])
        ref = np.zeros((11, 3))
        ref[:, 0] = np.linspace(0, 1, 11)

        control, info = controller.compute_control(state, ref)
        assert len(control) == 3
        assert "predicted_trajectory" in info

    def test_90_vs_60_state_bounds(self):
        """90° vs 60° state bounds가 다른지 확인."""
        params_90 = NonCoaxialSwerveParams(max_steering_angle=np.pi / 2)
        params_60 = NonCoaxialSwerveParams(max_steering_angle=np.pi / 3)
        model_90 = NonCoaxialSwerveDriveModel(params_90)
        model_60 = NonCoaxialSwerveDriveModel(params_60)

        _, ub_90 = model_90.get_state_bounds()
        _, ub_60 = model_60.get_state_bounds()
        assert ub_90[3] > ub_60[3]

    def test_60deg_forward_simulate(self):
        """forward_simulate가 정상 동작하는지 확인."""
        params = NonCoaxialSwerveParams(max_steering_angle=np.pi / 3)
        model = NonCoaxialSwerveDriveModel(params)

        state = np.array([0.0, 0.0, 0.0, 0.0])
        control = np.array([1.0, 0.0, 0.5])  # v=1, omega=0, delta_dot=0.5
        next_state = model.forward_simulate(state, control, 0.1)

        assert len(next_state) == 4
        assert next_state[0] > 0  # 전진
        assert abs(next_state[3]) <= np.pi / 3 + 1e-10  # delta 제한

    def test_scenario_tight_turn(self):
        """tight_turn_scenario 생성 확인."""
        scenario = tight_turn_scenario(nx=4)
        assert scenario.name == "tight_turn"
        assert scenario.trajectory.shape[1] == 4
        assert scenario.initial_state.shape[0] == 4

    def test_scenario_slalom(self):
        """slalom_scenario 생성 확인."""
        scenario = slalom_scenario(nx=4)
        assert scenario.name == "slalom"
        assert scenario.trajectory.shape[1] == 4
        assert scenario.initial_state.shape[0] == 4
        assert scenario.trajectory.shape[0] == 300
