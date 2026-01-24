"""
Non-coaxial swerve drive robot kinematic model.

Non-coaxial swerve drive에서는 각 휠 모듈이 제한된 스티어링 각도 범위를 가짐.
스티어링 각도가 상태에 포함되어 스티어링 각속도 제한을 적용할 수 있음.

┌─────────────────────────────────────────────────────────────────┐
│                    Non-Coaxial Swerve Drive                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│         Front                                                   │
│     ┌───┬───┬───┐                                              │
│     │ ↗ │   │ ↗ │  ← 각 휠 모듈이 독립적으로 스티어링           │
│     └───┴───┴───┘    (하지만 제한된 각도 범위: ±90°)           │
│     │           │                                               │
│     │   Robot   │    State: [x, y, θ, δ_fl, δ_fr, δ_rl, δ_rr] │
│     │   Body    │    Control: [v_fl, v_fr, v_rl, v_rr,         │
│     │           │              ω_fl, ω_fr, ω_rl, ω_rr]         │
│     ┌───┬───┬───┐                                              │
│     │ ↗ │   │ ↗ │                                              │
│     └───┴───┴───┘                                              │
│         Rear                                                    │
│                                                                 │
│  Simplified Control (for MPC):                                  │
│    State: [x, y, θ, δ_avg]  (평균 스티어링 각도)                │
│    Control: [v, δ_dot]  (속도, 스티어링 각속도)                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass
from typing import Tuple

import casadi as ca
import numpy as np


@dataclass
class NonCoaxialSwerveParams:
    """Non-coaxial swerve drive robot physical parameters."""

    # Robot geometry (rectangular chassis)
    length: float = 0.6  # Robot length [m]
    width: float = 0.5  # Robot width [m]

    # Velocity limits
    max_speed: float = 1.5  # Maximum wheel speed [m/s]
    max_omega: float = 2.0  # Maximum body angular velocity [rad/s]

    # Steering limits (non-coaxial constraint)
    max_steering_angle: float = np.pi / 2  # ±90° [rad]
    max_steering_rate: float = 2.0  # Maximum steering angular velocity [rad/s]

    # Acceleration limits
    max_accel: float = 1.0  # Maximum acceleration [m/s^2]
    max_steering_accel: float = 3.0  # Maximum steering acceleration [rad/s^2]

    # Wheel parameters
    wheel_radius: float = 0.05  # Wheel radius [m]


class NonCoaxialSwerveDriveModel:
    """
    Non-coaxial swerve drive robot kinematic model with steering constraints.

    각 휠 모듈의 스티어링 각도가 제한되어 있어 완전한 holonomic 이동이 불가능.
    스티어링 각도가 상태에 포함되어 연속적인 스티어링 제어가 가능.

    State: [x, y, theta, delta]
        - x, y: Position in world frame [m]
        - theta: Body heading angle [rad]
        - delta: Steering angle (average or reference) [rad]

    Control: [v, omega, delta_dot]
        - v: Velocity magnitude in steering direction [m/s]
        - omega: Body angular velocity [rad/s]
        - delta_dot: Steering angle rate [rad/s]

    Kinematics:
        스티어링 각도 delta는 body frame에서의 이동 방향을 결정:
        - vx_body = v * cos(delta)
        - vy_body = v * sin(delta)

        World frame velocities:
        - x_dot = vx_body * cos(theta) - vy_body * sin(theta)
        - y_dot = vx_body * sin(theta) + vy_body * cos(theta)
        - theta_dot = omega
        - delta_dot = steering rate (control input)
    """

    STATE_DIM = 4  # [x, y, theta, delta]
    CONTROL_DIM = 3  # [v, omega, delta_dot]

    def __init__(self, params: NonCoaxialSwerveParams | None = None):
        self.params = params or NonCoaxialSwerveParams()
        self._setup_casadi_model()

    def _setup_casadi_model(self) -> None:
        """Setup CasADi symbolic model for MPC."""
        # State variables
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        theta = ca.SX.sym("theta")  # Body heading
        delta = ca.SX.sym("delta")  # Steering angle
        self.state = ca.vertcat(x, y, theta, delta)

        # Control variables
        v = ca.SX.sym("v")  # Speed in steering direction
        omega = ca.SX.sym("omega")  # Body angular velocity
        delta_dot = ca.SX.sym("delta_dot")  # Steering rate
        self.control = ca.vertcat(v, omega, delta_dot)

        # Body frame velocities (determined by steering angle)
        vx_body = v * ca.cos(delta)
        vy_body = v * ca.sin(delta)

        # World frame velocities
        x_dot = vx_body * ca.cos(theta) - vy_body * ca.sin(theta)
        y_dot = vx_body * ca.sin(theta) + vy_body * ca.cos(theta)
        theta_dot = omega

        self.state_dot = ca.vertcat(x_dot, y_dot, theta_dot, delta_dot)

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

        # Clamp delta to steering limits
        max_delta = self.params.max_steering_angle
        state_next[3] = ca.fmax(-max_delta, ca.fmin(max_delta, state_next[3]))

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
            state: Current state [x, y, theta, delta]
            control: Control input [v, omega, delta_dot]
            dt: Time step [s]

        Returns:
            Next state [x, y, theta, delta]
        """
        x, y, theta, delta = state
        v, omega, delta_dot = control

        # RK4 integration
        def dynamics(s: np.ndarray, u: np.ndarray) -> np.ndarray:
            _, _, th, d = s
            vel, om, d_dot = u

            vx_body = vel * np.cos(d)
            vy_body = vel * np.sin(d)

            return np.array([
                vx_body * np.cos(th) - vy_body * np.sin(th),
                vx_body * np.sin(th) + vy_body * np.cos(th),
                om,
                d_dot,
            ])

        k1 = dynamics(state, control)
        k2 = dynamics(state + dt / 2 * k1, control)
        k3 = dynamics(state + dt / 2 * k2, control)
        k4 = dynamics(state + dt * k3, control)

        state_next = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Normalize theta to [-pi, pi]
        state_next[2] = np.arctan2(np.sin(state_next[2]), np.cos(state_next[2]))

        # Clamp delta to steering limits
        max_delta = self.params.max_steering_angle
        state_next[3] = np.clip(state_next[3], -max_delta, max_delta)

        return state_next

    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get control input bounds."""
        lb = np.array([
            -self.params.max_speed,
            -self.params.max_omega,
            -self.params.max_steering_rate,
        ])
        ub = np.array([
            self.params.max_speed,
            self.params.max_omega,
            self.params.max_steering_rate,
        ])
        return lb, ub

    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get state bounds (for steering angle constraint)."""
        max_delta = self.params.max_steering_angle
        lb = np.array([-np.inf, -np.inf, -np.inf, -max_delta])
        ub = np.array([np.inf, np.inf, np.inf, max_delta])
        return lb, ub

    def compute_body_velocity(
        self,
        v: float,
        delta: float,
    ) -> Tuple[float, float]:
        """
        Compute body frame velocities from speed and steering angle.

        Args:
            v: Speed in steering direction [m/s]
            delta: Steering angle [rad]

        Returns:
            Tuple of (vx_body, vy_body)
        """
        vx_body = v * np.cos(delta)
        vy_body = v * np.sin(delta)
        return vx_body, vy_body

    def compute_wheel_states(
        self,
        v: float,
        omega: float,
        delta: float,
    ) -> np.ndarray:
        """
        Compute individual wheel states for visualization/validation.

        Args:
            v: Speed in steering direction [m/s]
            omega: Body angular velocity [rad/s]
            delta: Steering angle [rad]

        Returns:
            Array of shape (4, 2) with [speed, angle] for each wheel
            Wheel order: [front_left, front_right, rear_left, rear_right]
        """
        L = self.params.length / 2
        W = self.params.width / 2

        # Body frame velocity
        vx_body = v * np.cos(delta)
        vy_body = v * np.sin(delta)

        # Wheel positions relative to center
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
            total_vx = vx_body + rot_vx
            total_vy = vy_body + rot_vy

            # Wheel speed and angle
            speed = np.sqrt(total_vx**2 + total_vy**2)
            angle = np.arctan2(total_vy, total_vx)

            wheel_states[i] = [speed, angle]

        return wheel_states

    def is_feasible(
        self,
        v: float,
        omega: float,
        delta: float,
    ) -> bool:
        """
        Check if the motion is feasible given steering constraints.

        Args:
            v: Speed [m/s]
            omega: Angular velocity [rad/s]
            delta: Steering angle [rad]

        Returns:
            True if motion is feasible
        """
        # Check steering angle limits
        if abs(delta) > self.params.max_steering_angle:
            return False

        # Check wheel states
        wheel_states = self.compute_wheel_states(v, omega, delta)

        for speed, angle in wheel_states:
            # Check wheel speed
            if speed > self.params.max_speed * 1.1:  # 10% margin
                return False

            # Check wheel steering angle (should be close to delta for swerve)
            angle_diff = abs(np.arctan2(np.sin(angle - delta), np.cos(angle - delta)))
            if angle_diff > self.params.max_steering_angle:
                return False

        return True

    def required_steering_for_direction(
        self,
        vx_desired: float,
        vy_desired: float,
    ) -> Tuple[float, float, bool]:
        """
        Compute required steering angle and speed for desired body velocities.

        Args:
            vx_desired: Desired body x velocity [m/s]
            vy_desired: Desired body y velocity [m/s]

        Returns:
            Tuple of (delta, v, is_feasible)
        """
        v = np.sqrt(vx_desired**2 + vy_desired**2)
        if v < 1e-6:
            return 0.0, 0.0, True

        delta = np.arctan2(vy_desired, vx_desired)

        # Check if within steering limits
        is_feasible = abs(delta) <= self.params.max_steering_angle

        # If not feasible, clamp to limits
        if not is_feasible:
            delta = np.clip(delta, -self.params.max_steering_angle,
                          self.params.max_steering_angle)

        return delta, v, is_feasible
