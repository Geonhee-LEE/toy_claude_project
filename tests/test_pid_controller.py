"""Unit tests for PID controller."""

import numpy as np
import pytest

from mpc_controller.controllers.pid_controller import PIDController, PIDGains
from mpc_controller.models.differential_drive import RobotParams


class TestPIDGains:
    """Test PIDGains dataclass."""

    def test_default_gains(self):
        """Test default gain values."""
        gains = PIDGains()
        assert gains.kp_linear == 1.0
        assert gains.ki_linear == 0.0
        assert gains.kd_linear == 0.1
        assert gains.kp_angular == 2.0
        assert gains.ki_angular == 0.0
        assert gains.kd_angular == 0.2

    def test_custom_gains(self):
        """Test custom gain values."""
        gains = PIDGains(
            kp_linear=2.0,
            ki_linear=0.1,
            kd_linear=0.5,
            kp_angular=3.0,
        )
        assert gains.kp_linear == 2.0
        assert gains.ki_linear == 0.1
        assert gains.kd_linear == 0.5
        assert gains.kp_angular == 3.0


class TestPIDController:
    """Test PIDController class."""

    @pytest.fixture
    def controller(self):
        """Create a PID controller instance."""
        return PIDController()

    @pytest.fixture
    def straight_trajectory(self):
        """Create a straight line trajectory."""
        N = 21
        x = np.linspace(0, 5, N)
        y = np.zeros(N)
        theta = np.zeros(N)
        return np.column_stack([x, y, theta])

    @pytest.fixture
    def circular_trajectory(self):
        """Create a circular trajectory."""
        N = 21
        t = np.linspace(0, np.pi, N)
        radius = 2.0
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        theta = t + np.pi / 2  # Tangent direction
        return np.column_stack([x, y, theta])

    def test_initialization(self, controller):
        """Test controller initialization."""
        assert controller._iteration_count == 0
        assert controller._linear_integral == 0.0
        assert controller._angular_integral == 0.0

    def test_initialization_with_params(self):
        """Test controller initialization with custom parameters."""
        robot_params = RobotParams(max_velocity=2.0, max_omega=3.0)
        pid_gains = PIDGains(kp_linear=1.5)
        controller = PIDController(
            robot_params=robot_params,
            pid_gains=pid_gains,
        )
        assert controller.gains.max_linear_velocity == 2.0
        assert controller.gains.max_angular_velocity == 3.0
        assert controller.gains.kp_linear == 1.5

    def test_compute_control_straight_line(self, controller, straight_trajectory):
        """Test control computation on straight trajectory."""
        # Start at origin, facing forward
        current_state = np.array([0.0, 0.0, 0.0])

        control, info = controller.compute_control(current_state, straight_trajectory)

        assert len(control) == 2
        assert "predicted_trajectory" in info
        assert "solve_time" in info
        assert "distance_error" in info
        assert "heading_error" in info

        # Should move forward (positive v)
        v, omega = control
        assert v > 0
        # Should have minimal angular velocity (straight line)
        assert abs(omega) < 0.5

    def test_compute_control_heading_error(self, controller, straight_trajectory):
        """Test control computation with heading error."""
        # Start at origin, facing 45 degrees
        current_state = np.array([0.0, 0.0, np.pi / 4])

        control, info = controller.compute_control(current_state, straight_trajectory)

        v, omega = control
        # Should turn right (negative omega) to face forward
        assert omega < 0

    def test_compute_control_circular(self, controller, circular_trajectory):
        """Test control computation on circular trajectory."""
        # Start at beginning of circle
        current_state = np.array([2.0, 0.0, np.pi / 2])

        control, info = controller.compute_control(current_state, circular_trajectory)

        v, omega = control
        assert v >= 0
        assert control.shape == (2,)

    def test_control_limits(self, controller, straight_trajectory):
        """Test that control outputs respect limits."""
        # Start far from trajectory to generate large errors
        current_state = np.array([0.0, 5.0, 0.0])

        control, _ = controller.compute_control(current_state, straight_trajectory)

        v, omega = control
        assert 0 <= v <= controller.gains.max_linear_velocity
        assert abs(omega) <= controller.gains.max_angular_velocity

    def test_reset(self, controller, straight_trajectory):
        """Test controller reset."""
        current_state = np.array([0.0, 0.0, 0.0])

        # Run a few iterations
        for _ in range(3):
            controller.compute_control(current_state, straight_trajectory)

        assert controller._iteration_count == 3
        assert controller._linear_integral != 0.0

        # Reset
        controller.reset()

        assert controller._iteration_count == 0
        assert controller._linear_integral == 0.0
        assert controller._angular_integral == 0.0
        assert controller._prev_linear_error == 0.0
        assert controller._prev_angular_error == 0.0

    def test_set_gains(self, controller):
        """Test dynamic gain setting."""
        controller.set_gains(kp_linear=3.0, kp_angular=4.0)

        assert controller.gains.kp_linear == 3.0
        assert controller.gains.kp_angular == 4.0
        # Other gains should remain unchanged
        assert controller.gains.ki_linear == 0.0

    def test_at_goal(self, controller, straight_trajectory):
        """Test behavior when at goal."""
        # Start at the goal position
        goal = straight_trajectory[-1]
        current_state = goal.copy()

        control, info = controller.compute_control(current_state, straight_trajectory)

        v, omega = control
        # Should have very low velocity at goal
        assert v < 0.1
        assert info["distance_error"] < 0.01

    def test_interface_compatibility_with_mpc(self, controller, straight_trajectory):
        """Test that PID has same interface as MPC."""
        current_state = np.array([0.0, 0.0, 0.0])

        control, info = controller.compute_control(current_state, straight_trajectory)

        # Check return types match MPC interface
        assert isinstance(control, np.ndarray)
        assert control.shape == (2,)
        assert isinstance(info, dict)

        # Check required info keys
        required_keys = [
            "predicted_trajectory",
            "predicted_controls",
            "cost",
            "solve_time",
            "solver_status",
        ]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"


class TestPIDvsMPCComparison:
    """Tests for comparing PID and MPC behavior."""

    def test_pid_faster_than_mpc(self):
        """PID should be faster to compute than MPC."""
        from mpc_controller.controllers.mpc import MPCController

        pid = PIDController()
        mpc = MPCController()

        trajectory = np.column_stack([
            np.linspace(0, 5, 21),
            np.zeros(21),
            np.zeros(21),
        ])
        current_state = np.array([0.0, 0.0, 0.0])

        # Compute with PID
        _, pid_info = pid.compute_control(current_state, trajectory)

        # Compute with MPC
        _, mpc_info = mpc.compute_control(current_state, trajectory)

        # PID should be faster
        assert pid_info["solve_time"] < mpc_info["solve_time"]

    def test_both_produce_valid_controls(self):
        """Both controllers should produce valid controls."""
        from mpc_controller.controllers.mpc import MPCController

        pid = PIDController()
        mpc = MPCController()

        trajectory = np.column_stack([
            np.linspace(0, 5, 21),
            np.zeros(21),
            np.zeros(21),
        ])
        current_state = np.array([0.0, 0.0, 0.0])

        pid_control, _ = pid.compute_control(current_state, trajectory)
        mpc_control, _ = mpc.compute_control(current_state, trajectory)

        # Both should produce 2D control vectors
        assert pid_control.shape == (2,)
        assert mpc_control.shape == (2,)

        # Both should produce forward motion for this trajectory
        assert pid_control[0] > 0  # v > 0
        assert mpc_control[0] > 0  # v > 0
