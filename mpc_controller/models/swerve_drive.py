"""Swerve drive robot kinematic model."""

from dataclasses import dataclass
from typing import Tuple

import casadi as ca
import numpy as np


@dataclass
class SwerveParams:
    """Swerve drive robot physical parameters."""

    # Robot geometry (rectangular chassis)
    length: float = 0.6  # Robot length [m]
    width: float = 0.5  # Robot width [m]

    # Velocity limits
    max_vx: float = 1.5  # Maximum x velocity [m/s]
    max_vy: float = 1.5  # Maximum y velocity [m/s]
    max_omega: float = 2.0  # Maximum angular velocity [rad/s]

    # Acceleration limits
    max_ax: float = 1.0  # Maximum x acceleration [m/s^2]
    max_ay: float = 1.0  # Maximum y acceleration [m/s^2]
    max_alpha: float = 1.5  # Maximum angular acceleration [rad/s^2]

    # Wheel parameters (for inverse kinematics if needed)
    wheel_radius: float = 0.05  # Wheel radius [m]
    max_wheel_speed: float = 10.0  # Maximum wheel angular speed [rad/s]


class SwerveDriveModel:
    """
    Swerve drive robot kinematic model.

    Swerve drive allows omnidirectional movement - the robot can move
    in any direction while independently controlling its orientation.

    State: [x, y, theta]
        - x, y: Position in world frame [m]
        - theta: Heading angle [rad]

    Control: [vx, vy, omega]
        - vx: Velocity in robot x-direction (forward) [m/s]
        - vy: Velocity in robot y-direction (left) [m/s]
        - omega: Angular velocity [rad/s]

    Kinematics (body velocities to world velocities):
        x_dot = vx * cos(theta) - vy * sin(theta)
        y_dot = vx * sin(theta) + vy * cos(theta)
        theta_dot = omega
    """

    STATE_DIM = 3  # [x, y, theta]
    CONTROL_DIM = 3  # [vx, vy, omega]

    def __init__(self, params: SwerveParams | None = None):
        self.params = params or SwerveParams()
        self._setup_casadi_model()

    def _setup_casadi_model(self) -> None:
        """Setup CasADi symbolic model for MPC."""
        # State variables
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        theta = ca.SX.sym("theta")
        self.state = ca.vertcat(x, y, theta)

        # Control variables (body frame velocities)
        vx = ca.SX.sym("vx")  # Forward velocity
        vy = ca.SX.sym("vy")  # Lateral velocity
        omega = ca.SX.sym("omega")  # Angular velocity
        self.control = ca.vertcat(vx, vy, omega)

        # Kinematics: transform body velocities to world frame
        x_dot = vx * ca.cos(theta) - vy * ca.sin(theta)
        y_dot = vx * ca.sin(theta) + vy * ca.cos(theta)
        theta_dot = omega
        self.state_dot = ca.vertcat(x_dot, y_dot, theta_dot)

        # CasADi function for dynamics
        self.dynamics = ca.Function(
            "dynamics",
            [self.state, self.control],
            [self.state_dot],
            ["state", "control"],
            ["state_dot"],
        )

    def discrete_dynamics(self, dt: float) -> ca.Function:
        """
        Get discrete-time dynamics using RK4 integration.

        Args:
            dt: Time step [s]

        Returns:
            CasADi function: state_next = f(state, control)
        """
        # RK4 integration
        k1 = self.dynamics(self.state, self.control)
        k2 = self.dynamics(self.state + dt / 2 * k1, self.control)
        k3 = self.dynamics(self.state + dt / 2 * k2, self.control)
        k4 = self.dynamics(self.state + dt * k3, self.control)

        state_next = self.state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # NOTE: theta is NOT normalized here to allow continuous angle tracking
        # across the ±π boundary. The MPC cost function handles angle wrapping.

        return ca.Function(
            "discrete_dynamics",
            [self.state, self.control],
            [state_next],
            ["state", "control"],
            ["state_next"],
        )

    def forward_simulate(
        self,
        state: np.ndarray,
        control: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Forward simulate one step (NumPy version for simulation).

        Args:
            state: Current state [x, y, theta]
            control: Control input [vx, vy, omega]
            dt: Time step [s]

        Returns:
            Next state [x, y, theta]
        """
        x, y, theta = state
        vx, vy, omega = control

        # RK4 integration
        def dynamics(s: np.ndarray, u: np.ndarray) -> np.ndarray:
            return np.array([
                u[0] * np.cos(s[2]) - u[1] * np.sin(s[2]),
                u[0] * np.sin(s[2]) + u[1] * np.cos(s[2]),
                u[2],
            ])

        k1 = dynamics(state, control)
        k2 = dynamics(state + dt / 2 * k1, control)
        k3 = dynamics(state + dt / 2 * k2, control)
        k4 = dynamics(state + dt * k3, control)

        state_next = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Normalize theta to [-pi, pi]
        state_next[2] = np.arctan2(np.sin(state_next[2]), np.cos(state_next[2]))

        return state_next

    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get control input bounds."""
        lb = np.array([
            -self.params.max_vx,
            -self.params.max_vy,
            -self.params.max_omega,
        ])
        ub = np.array([
            self.params.max_vx,
            self.params.max_vy,
            self.params.max_omega,
        ])
        return lb, ub

    def compute_wheel_velocities(
        self,
        vx: float,
        vy: float,
        omega: float,
    ) -> np.ndarray:
        """
        Compute individual wheel velocities for swerve drive.

        Assumes 4 wheels at corners of rectangular chassis.
        Returns wheel speeds and angles for each module.

        Args:
            vx: Body x velocity [m/s]
            vy: Body y velocity [m/s]
            omega: Angular velocity [rad/s]

        Returns:
            Array of shape (4, 2) with [speed, angle] for each wheel
            Wheel order: [front_left, front_right, rear_left, rear_right]
        """
        L = self.params.length / 2
        W = self.params.width / 2

        # Wheel positions relative to center (in body frame)
        wheel_positions = np.array([
            [L, W],    # Front left
            [L, -W],   # Front right
            [-L, W],   # Rear left
            [-L, -W],  # Rear right
        ])

        wheel_states = np.zeros((4, 2))

        for i, (wx, wy) in enumerate(wheel_positions):
            # Velocity contribution from rotation
            rot_vx = -omega * wy
            rot_vy = omega * wx

            # Total wheel velocity
            total_vx = vx + rot_vx
            total_vy = vy + rot_vy

            # Wheel speed and angle
            speed = np.sqrt(total_vx**2 + total_vy**2)
            angle = np.arctan2(total_vy, total_vx)

            wheel_states[i] = [speed, angle]

        return wheel_states
