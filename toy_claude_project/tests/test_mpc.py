"""Tests for MPC controller."""

import numpy as np
import pytest

from mpc_controller import (
    DifferentialDriveModel,
    RobotParams,
    MPCController,
    MPCParams,
    TrajectoryInterpolator,
    generate_line_trajectory,
    generate_circle_trajectory,
)


class TestDifferentialDriveModel:
    """Tests for differential drive model."""

    def test_state_dimension(self):
        model = DifferentialDriveModel()
        assert model.STATE_DIM == 3
        assert model.CONTROL_DIM == 2

    def test_forward_simulate_straight(self):
        """Test forward simulation for straight motion."""
        model = DifferentialDriveModel()
        state = np.array([0.0, 0.0, 0.0])  # At origin, facing +x
        control = np.array([1.0, 0.0])  # Forward velocity, no rotation
        dt = 0.1

        next_state = model.forward_simulate(state, control, dt)

        # Should move in +x direction
        assert next_state[0] > 0
        assert abs(next_state[1]) < 1e-6
        assert abs(next_state[2]) < 1e-6

    def test_forward_simulate_rotation(self):
        """Test forward simulation for pure rotation."""
        model = DifferentialDriveModel()
        state = np.array([0.0, 0.0, 0.0])
        control = np.array([0.0, 1.0])  # No forward velocity, rotate
        dt = 0.1

        next_state = model.forward_simulate(state, control, dt)

        # Position should not change significantly
        assert abs(next_state[0]) < 1e-6
        assert abs(next_state[1]) < 1e-6
        # Heading should change
        assert next_state[2] > 0

    def test_control_bounds(self):
        """Test control bounds are returned correctly."""
        params = RobotParams(max_velocity=2.0, max_omega=3.0)
        model = DifferentialDriveModel(params)
        
        lb, ub = model.get_control_bounds()
        
        assert lb[0] == -2.0
        assert ub[0] == 2.0
        assert lb[1] == -3.0
        assert ub[1] == 3.0


class TestTrajectory:
    """Tests for trajectory generation."""

    def test_line_trajectory_shape(self):
        traj = generate_line_trajectory(
            start=np.array([0.0, 0.0]),
            end=np.array([10.0, 5.0]),
            num_points=100,
        )
        
        assert traj.shape == (100, 3)

    def test_line_trajectory_endpoints(self):
        start = np.array([1.0, 2.0])
        end = np.array([5.0, 8.0])
        traj = generate_line_trajectory(start, end, num_points=50)
        
        np.testing.assert_array_almost_equal(traj[0, :2], start)
        np.testing.assert_array_almost_equal(traj[-1, :2], end)

    def test_circle_trajectory_radius(self):
        center = np.array([0.0, 0.0])
        radius = 2.0
        traj = generate_circle_trajectory(center, radius, num_points=100)
        
        # Check all points are at correct radius
        distances = np.linalg.norm(traj[:, :2] - center, axis=1)
        np.testing.assert_array_almost_equal(distances, radius * np.ones(100))

    def test_trajectory_interpolator(self):
        traj = generate_line_trajectory(
            start=np.array([0.0, 0.0]),
            end=np.array([10.0, 0.0]),
            num_points=101,
        )
        dt = 0.1
        interpolator = TrajectoryInterpolator(traj, dt)
        
        # Get reference at t=0
        ref = interpolator.get_reference(0.0, horizon=10, mpc_dt=0.1)
        assert ref.shape == (11, 3)
        np.testing.assert_array_almost_equal(ref[0, :2], [0.0, 0.0])


class TestMPCController:
    """Tests for MPC controller."""

    @pytest.fixture
    def controller(self):
        """Create a test controller."""
        robot_params = RobotParams()
        mpc_params = MPCParams(N=10, dt=0.1)
        return MPCController(robot_params, mpc_params)

    def test_controller_creation(self, controller):
        """Test controller can be created."""
        assert controller is not None
        assert controller.params.N == 10

    def test_compute_control_output_shape(self, controller):
        """Test control output has correct shape."""
        state = np.array([0.0, 0.0, 0.0])
        reference = np.zeros((11, 3))  # N+1 reference points
        
        control, info = controller.compute_control(state, reference)
        
        assert control.shape == (2,)
        assert "predicted_trajectory" in info
        assert info["predicted_trajectory"].shape == (11, 3)

    def test_compute_control_stationary(self, controller):
        """Test control when already at reference."""
        state = np.array([0.0, 0.0, 0.0])
        reference = np.zeros((11, 3))
        
        control, _ = controller.compute_control(state, reference)
        
        # Control should be near zero when at reference
        assert np.abs(control[0]) < 0.1
        assert np.abs(control[1]) < 0.1

    def test_compute_control_forward_reference(self, controller):
        """Test control when reference is ahead."""
        state = np.array([0.0, 0.0, 0.0])  # At origin, facing +x
        reference = np.zeros((11, 3))
        reference[:, 0] = np.linspace(0, 2, 11)  # Reference moving in +x
        
        control, _ = controller.compute_control(state, reference)
        
        # Should command forward velocity
        assert control[0] > 0

    def test_controller_reset(self, controller):
        """Test controller reset."""
        state = np.array([0.0, 0.0, 0.0])
        reference = np.zeros((11, 3))
        reference[:, 0] = np.linspace(0, 2, 11)
        
        # Run once to populate warm start
        controller.compute_control(state, reference)
        assert controller.prev_solution is not None
        
        # Reset
        controller.reset()
        assert controller.prev_solution is None


class TestIntegration:
    """Integration tests."""

    def test_simple_tracking(self):
        """Test simple straight-line tracking."""
        robot_params = RobotParams()
        mpc_params = MPCParams(N=10, dt=0.1)
        controller = MPCController(robot_params, mpc_params)
        
        # Simple forward trajectory
        trajectory = generate_line_trajectory(
            start=np.array([0.0, 0.0]),
            end=np.array([5.0, 0.0]),
            num_points=50,
        )
        interpolator = TrajectoryInterpolator(trajectory, dt=0.1)
        
        state = np.array([0.0, 0.0, 0.0])
        
        # Run a few control steps
        for i in range(10):
            ref = interpolator.get_reference(i * 0.1, controller.params.N, controller.params.dt)
            control, _ = controller.compute_control(state, ref)
            
            # Check control is reasonable
            assert -robot_params.max_velocity <= control[0] <= robot_params.max_velocity
            assert -robot_params.max_omega <= control[1] <= robot_params.max_omega


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
