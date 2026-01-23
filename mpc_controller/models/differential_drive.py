"""Differential drive robot kinematic model."""

from dataclasses import dataclass
from typing import Tuple

import casadi as ca
import numpy as np


@dataclass
class RobotParams:
    """Robot physical parameters."""

    wheel_base: float = 0.5  # Distance between wheels [m]
    max_velocity: float = 1.0  # Maximum linear velocity [m/s]
    max_omega: float = 1.5  # Maximum angular velocity [rad/s]
    max_acceleration: float = 0.5  # Maximum linear acceleration [m/s^2]
    max_alpha: float = 1.0  # Maximum angular acceleration [rad/s^2]


class DifferentialDriveModel:
    """
    Differential drive robot kinematic model.

    State: [x, y, theta]
    Control: [v, omega]

    Kinematics:
        x_dot = v * cos(theta)
        y_dot = v * sin(theta)
        theta_dot = omega
    """

    STATE_DIM = 3  # [x, y, theta]
    CONTROL_DIM = 2  # [v, omega]

    def __init__(self, params: RobotParams | None = None):
        self.params = params or RobotParams()
        self._setup_casadi_model()

    def _setup_casadi_model(self) -> None:
        """Setup CasADi symbolic model for MPC."""
        # State variables
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        theta = ca.SX.sym("theta")
        self.state = ca.vertcat(x, y, theta)

        # Control variables
        v = ca.SX.sym("v")
        omega = ca.SX.sym("omega")
        self.control = ca.vertcat(v, omega)

        # Kinematics (continuous-time)
        x_dot = v * ca.cos(theta)
        y_dot = v * ca.sin(theta)
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

        # Normalize theta to [-pi, pi]
        state_next[2] = ca.atan2(ca.sin(state_next[2]), ca.cos(state_next[2]))

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
            control: Control input [v, omega]
            dt: Time step [s]

        Returns:
            Next state [x, y, theta]
        """
        x, y, theta = state
        v, omega = control

        # RK4 integration
        def dynamics(s: np.ndarray, u: np.ndarray) -> np.ndarray:
            return np.array([
                u[0] * np.cos(s[2]),
                u[0] * np.sin(s[2]),
                u[1],
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
        lb = np.array([-self.params.max_velocity, -self.params.max_omega])
        ub = np.array([self.params.max_velocity, self.params.max_omega])
        return lb, ub
