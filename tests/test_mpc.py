"""Tests for MPC controller."""

import numpy as np
import pytest

from mpc_controller import (
    DifferentialDriveModel,
    RobotParams,
    MPCController,
    MPCParams,
    SwerveDriveModel,
    SwerveParams,
    SwerveMPCController,
    SwerveMPCParams,
    NonCoaxialSwerveDriveModel,
    NonCoaxialSwerveParams,
    NonCoaxialSwerveMPCController,
    NonCoaxialSwerveMPCParams,
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


class TestSwerveDriveModel:
    """Tests for swerve drive model."""

    def test_state_dimension(self):
        model = SwerveDriveModel()
        assert model.STATE_DIM == 3
        assert model.CONTROL_DIM == 3

    def test_forward_simulate_straight(self):
        """Test forward simulation for straight motion."""
        model = SwerveDriveModel()
        state = np.array([0.0, 0.0, 0.0])  # At origin, facing +x
        control = np.array([1.0, 0.0, 0.0])  # Forward velocity, no lateral, no rotation
        dt = 0.1

        next_state = model.forward_simulate(state, control, dt)

        # Should move in +x direction
        assert next_state[0] > 0
        assert abs(next_state[1]) < 1e-6
        assert abs(next_state[2]) < 1e-6

    def test_forward_simulate_lateral(self):
        """Test forward simulation for lateral motion (swerve-specific)."""
        model = SwerveDriveModel()
        state = np.array([0.0, 0.0, 0.0])  # At origin, facing +x
        control = np.array([0.0, 1.0, 0.0])  # Lateral velocity only
        dt = 0.1

        next_state = model.forward_simulate(state, control, dt)

        # Should move in +y direction
        assert abs(next_state[0]) < 1e-6
        assert next_state[1] > 0
        assert abs(next_state[2]) < 1e-6

    def test_forward_simulate_rotation(self):
        """Test forward simulation for pure rotation."""
        model = SwerveDriveModel()
        state = np.array([0.0, 0.0, 0.0])
        control = np.array([0.0, 0.0, 1.0])  # Rotation only
        dt = 0.1

        next_state = model.forward_simulate(state, control, dt)

        # Position should not change significantly
        assert abs(next_state[0]) < 1e-6
        assert abs(next_state[1]) < 1e-6
        # Heading should change
        assert next_state[2] > 0

    def test_omnidirectional_motion(self):
        """Test that swerve drive can move diagonally while rotating."""
        model = SwerveDriveModel()
        state = np.array([0.0, 0.0, 0.0])
        control = np.array([1.0, 1.0, 0.5])  # Forward, lateral, and rotation
        dt = 0.1

        next_state = model.forward_simulate(state, control, dt)

        # Should move in both x and y, and rotate
        assert next_state[0] > 0
        assert next_state[1] > 0
        assert next_state[2] > 0

    def test_control_bounds(self):
        """Test control bounds are returned correctly."""
        params = SwerveParams(max_vx=2.0, max_vy=1.5, max_omega=3.0)
        model = SwerveDriveModel(params)

        lb, ub = model.get_control_bounds()

        assert lb[0] == -2.0
        assert ub[0] == 2.0
        assert lb[1] == -1.5
        assert ub[1] == 1.5
        assert lb[2] == -3.0
        assert ub[2] == 3.0

    def test_wheel_velocities(self):
        """Test wheel velocity computation."""
        model = SwerveDriveModel()

        # Pure forward motion
        wheel_states = model.compute_wheel_velocities(1.0, 0.0, 0.0)
        assert wheel_states.shape == (4, 2)
        # All wheels should have same speed and angle 0 for pure forward
        for i in range(4):
            assert abs(wheel_states[i, 0] - 1.0) < 1e-6
            assert abs(wheel_states[i, 1]) < 1e-6


class TestSwerveMPCController:
    """Tests for Swerve MPC controller."""

    @pytest.fixture
    def controller(self):
        """Create a test controller."""
        robot_params = SwerveParams()
        mpc_params = SwerveMPCParams(N=10, dt=0.1)
        return SwerveMPCController(robot_params, mpc_params)

    def test_controller_creation(self, controller):
        """Test controller can be created."""
        assert controller is not None
        assert controller.params.N == 10

    def test_compute_control_output_shape(self, controller):
        """Test control output has correct shape."""
        state = np.array([0.0, 0.0, 0.0])
        reference = np.zeros((11, 3))  # N+1 reference points

        control, info = controller.compute_control(state, reference)

        assert control.shape == (3,)  # vx, vy, omega
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
        assert np.abs(control[2]) < 0.1

    def test_compute_control_forward_reference(self, controller):
        """Test control when reference is ahead."""
        state = np.array([0.0, 0.0, 0.0])  # At origin, facing +x
        reference = np.zeros((11, 3))
        reference[:, 0] = np.linspace(0, 2, 11)  # Reference moving in +x

        control, _ = controller.compute_control(state, reference)

        # Should command forward velocity
        assert control[0] > 0

    def test_compute_control_lateral_reference(self, controller):
        """Test control when reference is to the side (swerve-specific)."""
        state = np.array([0.0, 0.0, 0.0])  # At origin, facing +x
        reference = np.zeros((11, 3))
        reference[:, 1] = np.linspace(0, 2, 11)  # Reference moving in +y

        control, _ = controller.compute_control(state, reference)

        # Should command lateral velocity (swerve can move sideways)
        assert control[1] > 0

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


class TestSwerveIntegration:
    """Integration tests for swerve drive."""

    def test_simple_tracking(self):
        """Test simple straight-line tracking with swerve drive."""
        robot_params = SwerveParams()
        mpc_params = SwerveMPCParams(N=10, dt=0.1)
        controller = SwerveMPCController(robot_params, mpc_params)

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
            assert -robot_params.max_vx <= control[0] <= robot_params.max_vx
            assert -robot_params.max_vy <= control[1] <= robot_params.max_vy
            assert -robot_params.max_omega <= control[2] <= robot_params.max_omega

    def test_lateral_tracking(self):
        """Test lateral movement tracking (unique to swerve drive)."""
        robot_params = SwerveParams()
        mpc_params = SwerveMPCParams(N=10, dt=0.1)
        controller = SwerveMPCController(robot_params, mpc_params)

        # Lateral trajectory (moving in y while facing x)
        trajectory = generate_line_trajectory(
            start=np.array([0.0, 0.0]),
            end=np.array([0.0, 5.0]),  # Move purely in y direction
            num_points=50,
        )
        # Override heading to stay at 0 (facing +x while moving +y)
        trajectory[:, 2] = 0.0
        interpolator = TrajectoryInterpolator(trajectory, dt=0.1)

        state = np.array([0.0, 0.0, 0.0])

        # Run a few control steps
        for i in range(10):
            ref = interpolator.get_reference(i * 0.1, controller.params.N, controller.params.dt)
            control, _ = controller.compute_control(state, ref)

            # Swerve drive should use lateral velocity for this motion
            # (unlike differential drive which would need to rotate)
            assert control.shape == (3,)


class TestNonCoaxialSwerveDriveModel:
    """Tests for non-coaxial swerve drive model."""

    def test_state_dimension(self):
        model = NonCoaxialSwerveDriveModel()
        assert model.STATE_DIM == 4  # [x, y, theta, delta]
        assert model.CONTROL_DIM == 3  # [v, omega, delta_dot]

    def test_forward_simulate_straight(self):
        """Test forward simulation for straight motion (delta=0)."""
        model = NonCoaxialSwerveDriveModel()
        state = np.array([0.0, 0.0, 0.0, 0.0])  # x, y, theta, delta
        control = np.array([1.0, 0.0, 0.0])  # v, omega, delta_dot
        dt = 0.1

        next_state = model.forward_simulate(state, control, dt)

        # Should move in +x direction (delta=0 means forward)
        assert next_state[0] > 0
        assert abs(next_state[1]) < 1e-6
        assert abs(next_state[2]) < 1e-6
        assert abs(next_state[3]) < 1e-6  # delta unchanged

    def test_forward_simulate_with_steering(self):
        """Test forward simulation with steering angle."""
        model = NonCoaxialSwerveDriveModel()
        delta = np.pi / 4  # 45 degrees
        state = np.array([0.0, 0.0, 0.0, delta])
        control = np.array([1.0, 0.0, 0.0])  # v=1, no rotation, no steering change
        dt = 0.1

        next_state = model.forward_simulate(state, control, dt)

        # Should move in diagonal direction (45 degrees)
        assert next_state[0] > 0  # x increases
        assert next_state[1] > 0  # y increases
        assert abs(next_state[3] - delta) < 1e-6  # delta unchanged

    def test_steering_rate(self):
        """Test steering angle rate control."""
        model = NonCoaxialSwerveDriveModel()
        state = np.array([0.0, 0.0, 0.0, 0.0])
        control = np.array([0.0, 0.0, 1.0])  # delta_dot = 1 rad/s
        dt = 0.1

        next_state = model.forward_simulate(state, control, dt)

        # Steering angle should change
        assert next_state[3] > 0

    def test_steering_angle_limits(self):
        """Test steering angle is clamped to limits."""
        params = NonCoaxialSwerveParams(max_steering_angle=np.pi / 2)
        model = NonCoaxialSwerveDriveModel(params)

        # Start at max steering and try to exceed
        state = np.array([0.0, 0.0, 0.0, np.pi / 2 - 0.01])
        control = np.array([0.0, 0.0, 1.0])  # Try to increase steering
        dt = 0.5  # Large dt to exceed limit

        next_state = model.forward_simulate(state, control, dt)

        # Steering should be clamped to max
        assert next_state[3] <= np.pi / 2

    def test_control_bounds(self):
        """Test control bounds are returned correctly."""
        params = NonCoaxialSwerveParams(
            max_speed=2.0,
            max_omega=3.0,
            max_steering_rate=1.5,
        )
        model = NonCoaxialSwerveDriveModel(params)

        lb, ub = model.get_control_bounds()

        assert lb[0] == -2.0  # v
        assert ub[0] == 2.0
        assert lb[1] == -3.0  # omega
        assert ub[1] == 3.0
        assert lb[2] == -1.5  # delta_dot
        assert ub[2] == 1.5

    def test_state_bounds(self):
        """Test state bounds (steering angle)."""
        params = NonCoaxialSwerveParams(max_steering_angle=np.pi / 2)
        model = NonCoaxialSwerveDriveModel(params)

        lb, ub = model.get_state_bounds()

        assert lb[3] == -np.pi / 2
        assert ub[3] == np.pi / 2

    def test_compute_body_velocity(self):
        """Test body velocity computation from speed and steering."""
        model = NonCoaxialSwerveDriveModel()

        # Forward (delta=0)
        vx, vy = model.compute_body_velocity(1.0, 0.0)
        assert abs(vx - 1.0) < 1e-6
        assert abs(vy) < 1e-6

        # 90 degrees (pure lateral)
        vx, vy = model.compute_body_velocity(1.0, np.pi / 2)
        assert abs(vx) < 1e-6
        assert abs(vy - 1.0) < 1e-6

    def test_required_steering_for_direction(self):
        """Test computing required steering for desired direction."""
        model = NonCoaxialSwerveDriveModel()

        # Forward motion (should be feasible)
        delta, v, feasible = model.required_steering_for_direction(1.0, 0.0)
        assert abs(delta) < 1e-6
        assert abs(v - 1.0) < 1e-6
        assert feasible

        # Lateral motion at 45 degrees (should be feasible with ±90° limit)
        delta, v, feasible = model.required_steering_for_direction(1.0, 1.0)
        assert abs(delta - np.pi / 4) < 1e-6
        assert feasible

        # Backward motion (requires |delta| > 90°, not feasible)
        delta, v, feasible = model.required_steering_for_direction(-1.0, 0.0)
        assert not feasible


class TestNonCoaxialSwerveMPCController:
    """Tests for Non-coaxial Swerve MPC controller."""

    @pytest.fixture
    def controller(self):
        """Create a test controller."""
        robot_params = NonCoaxialSwerveParams()
        mpc_params = NonCoaxialSwerveMPCParams(N=10, dt=0.1)
        return NonCoaxialSwerveMPCController(robot_params, mpc_params)

    def test_controller_creation(self, controller):
        """Test controller can be created."""
        assert controller is not None
        assert controller.params.N == 10

    def test_compute_control_output_shape(self, controller):
        """Test control output has correct shape."""
        state = np.array([0.0, 0.0, 0.0, 0.0])  # [x, y, theta, delta]
        reference = np.zeros((11, 3))  # N+1 reference points [x, y, theta]

        control, info = controller.compute_control(state, reference)

        assert control.shape == (3,)  # [v, omega, delta_dot]
        assert "predicted_trajectory" in info
        assert info["predicted_trajectory"].shape == (11, 4)  # State is 4D

    def test_compute_control_stationary(self, controller):
        """Test control when already at reference."""
        state = np.array([0.0, 0.0, 0.0, 0.0])
        reference = np.zeros((11, 3))

        control, _ = controller.compute_control(state, reference)

        # Control should be near zero when at reference
        assert np.abs(control[0]) < 0.1  # v
        assert np.abs(control[1]) < 0.1  # omega
        assert np.abs(control[2]) < 0.5  # delta_dot (may adjust steering)

    def test_compute_control_forward_reference(self, controller):
        """Test control when reference is ahead."""
        state = np.array([0.0, 0.0, 0.0, 0.0])  # At origin, delta=0
        reference = np.zeros((11, 3))
        reference[:, 0] = np.linspace(0, 2, 11)  # Reference moving in +x

        control, _ = controller.compute_control(state, reference)

        # Should command forward velocity
        assert control[0] > 0

    def test_steering_constraint_respected(self, controller):
        """Test that steering angle stays within limits."""
        # Start with steering at limit
        state = np.array([0.0, 0.0, 0.0, np.pi / 2 - 0.1])
        reference = np.zeros((11, 3))
        reference[:, 1] = np.linspace(0, 2, 11)  # Reference in +y (needs more steering)

        control, info = controller.compute_control(state, reference)

        # Check predicted trajectory respects steering limits
        predicted = info["predicted_trajectory"]
        max_delta = controller.robot.params.max_steering_angle
        for k in range(len(predicted)):
            assert abs(predicted[k, 3]) <= max_delta + 1e-4  # Small tolerance

    def test_controller_reset(self, controller):
        """Test controller reset."""
        state = np.array([0.0, 0.0, 0.0, 0.0])
        reference = np.zeros((11, 3))
        reference[:, 0] = np.linspace(0, 2, 11)

        # Run once to populate warm start
        controller.compute_control(state, reference)
        assert controller.prev_solution is not None

        # Reset
        controller.reset()
        assert controller.prev_solution is None


class TestNonCoaxialSwerveIntegration:
    """Integration tests for non-coaxial swerve drive."""

    def test_forward_tracking(self):
        """Test straight-line tracking with non-coaxial swerve drive."""
        robot_params = NonCoaxialSwerveParams()
        mpc_params = NonCoaxialSwerveMPCParams(N=10, dt=0.1)
        controller = NonCoaxialSwerveMPCController(robot_params, mpc_params)

        # Simple forward trajectory
        trajectory = generate_line_trajectory(
            start=np.array([0.0, 0.0]),
            end=np.array([5.0, 0.0]),
            num_points=50,
        )
        interpolator = TrajectoryInterpolator(trajectory, dt=0.1)

        state = np.array([0.0, 0.0, 0.0, 0.0])  # [x, y, theta, delta]

        # Run a few control steps
        for i in range(5):
            ref = interpolator.get_reference(i * 0.1, controller.params.N, controller.params.dt)
            control, _ = controller.compute_control(state, ref)

            # Check control is reasonable
            assert -robot_params.max_speed <= control[0] <= robot_params.max_speed
            assert -robot_params.max_omega <= control[1] <= robot_params.max_omega
            assert -robot_params.max_steering_rate <= control[2] <= robot_params.max_steering_rate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
