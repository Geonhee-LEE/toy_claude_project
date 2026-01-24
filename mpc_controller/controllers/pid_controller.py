"""PID Controller for path tracking comparison with MPC."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from mpc_controller.models.differential_drive import DifferentialDriveModel, RobotParams
from mpc_controller.utils.trajectory import normalize_angle

# Module-level logger
logger = logging.getLogger(__name__)


@dataclass
class PIDGains:
    """PID controller gains."""

    # Linear velocity PID gains
    kp_linear: float = 1.0
    ki_linear: float = 0.0
    kd_linear: float = 0.1

    # Angular velocity PID gains
    kp_angular: float = 2.0
    ki_angular: float = 0.0
    kd_angular: float = 0.2

    # Lookahead distance for path tracking
    lookahead_distance: float = 0.5

    # Control limits (will be overridden by robot params)
    max_linear_velocity: float = 1.0
    max_angular_velocity: float = 2.0


class PIDController:
    """
    PID controller for path tracking.

    Provides the same interface as MPCController for comparison purposes.
    Uses a Pure Pursuit-like approach with PID control.
    """

    def __init__(
        self,
        robot_params: RobotParams | None = None,
        pid_gains: PIDGains | None = None,
        log_file: str | Path | None = None,
    ):
        """
        Initialize PID controller.

        Args:
            robot_params: Robot physical parameters
            pid_gains: PID tuning parameters
            log_file: Optional log file path
        """
        self.robot_params = robot_params or RobotParams()
        self.gains = pid_gains or PIDGains()
        self._iteration_count = 0

        # Update control limits from robot params
        self.gains.max_linear_velocity = self.robot_params.max_velocity
        self.gains.max_angular_velocity = self.robot_params.max_omega

        # PID state variables
        self._linear_integral = 0.0
        self._angular_integral = 0.0
        self._prev_linear_error = 0.0
        self._prev_angular_error = 0.0
        self._prev_time = None

        self._setup_logging(log_file)

    def _setup_logging(self, log_file: str | Path | None = None) -> None:
        """Setup logging configuration."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        if log_file is not None:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            logger.info(f"PID logging to file: {log_file}")

    def _find_lookahead_point(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Find the lookahead point on the reference trajectory.

        Args:
            current_state: Current robot state [x, y, theta]
            reference_trajectory: Reference trajectory, shape (N+1, 3)

        Returns:
            Tuple of (lookahead_point, index)
        """
        x, y = current_state[0], current_state[1]

        # Find closest point
        distances = np.sqrt(
            (reference_trajectory[:, 0] - x) ** 2
            + (reference_trajectory[:, 1] - y) ** 2
        )
        closest_idx = np.argmin(distances)

        # Find lookahead point
        lookahead_idx = closest_idx
        for i in range(closest_idx, len(reference_trajectory)):
            dist = np.sqrt(
                (reference_trajectory[i, 0] - x) ** 2
                + (reference_trajectory[i, 1] - y) ** 2
            )
            if dist >= self.gains.lookahead_distance:
                lookahead_idx = i
                break
        else:
            # If no point found, use the last point
            lookahead_idx = len(reference_trajectory) - 1

        return reference_trajectory[lookahead_idx], lookahead_idx

    def _compute_errors(
        self,
        current_state: np.ndarray,
        target_point: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute distance and heading errors.

        Args:
            current_state: Current robot state [x, y, theta]
            target_point: Target point [x, y, theta]

        Returns:
            Tuple of (distance_error, heading_error)
        """
        x, y, theta = current_state

        # Distance to target
        dx = target_point[0] - x
        dy = target_point[1] - y
        distance_error = np.sqrt(dx**2 + dy**2)

        # Heading to target
        target_heading = np.arctan2(dy, dx)
        heading_error = normalize_angle(target_heading - theta)

        return distance_error, heading_error

    def compute_control(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        Compute control input using PID.

        Args:
            current_state: Current robot state [x, y, theta]
            reference_trajectory: Reference trajectory, shape (N+1, 3)

        Returns:
            Tuple of:
                - Control input [v, omega]
                - Info dict with tracking metrics
        """
        compute_start = time.perf_counter()

        # Find lookahead point
        lookahead_point, lookahead_idx = self._find_lookahead_point(
            current_state, reference_trajectory
        )

        # Compute errors
        distance_error, heading_error = self._compute_errors(
            current_state, lookahead_point
        )

        # Time delta for derivative term
        current_time = time.perf_counter()
        if self._prev_time is None:
            dt = 0.1  # Default dt
        else:
            dt = current_time - self._prev_time
            dt = max(dt, 0.001)  # Prevent division by zero
        self._prev_time = current_time

        # Linear velocity PID
        self._linear_integral += distance_error * dt
        linear_derivative = (distance_error - self._prev_linear_error) / dt
        self._prev_linear_error = distance_error

        v = (
            self.gains.kp_linear * distance_error
            + self.gains.ki_linear * self._linear_integral
            + self.gains.kd_linear * linear_derivative
        )

        # Reduce speed when heading error is large
        v *= np.cos(heading_error)

        # Angular velocity PID
        self._angular_integral += heading_error * dt
        angular_derivative = (heading_error - self._prev_angular_error) / dt
        self._prev_angular_error = heading_error

        omega = (
            self.gains.kp_angular * heading_error
            + self.gains.ki_angular * self._angular_integral
            + self.gains.kd_angular * angular_derivative
        )

        # Apply control limits
        v = np.clip(v, 0, self.gains.max_linear_velocity)
        omega = np.clip(
            omega,
            -self.gains.max_angular_velocity,
            self.gains.max_angular_velocity,
        )

        # If very close to goal, reduce velocity
        goal_point = reference_trajectory[-1]
        goal_distance = np.sqrt(
            (goal_point[0] - current_state[0]) ** 2
            + (goal_point[1] - current_state[1]) ** 2
        )
        if goal_distance < 0.1:
            v *= goal_distance / 0.1

        compute_time = time.perf_counter() - compute_start

        # Log iteration
        self._iteration_count += 1
        logger.info(
            f"PID iteration {self._iteration_count}: "
            f"compute_time={compute_time*1000:.2f}ms, "
            f"dist_error={distance_error:.4f}, "
            f"heading_error={np.degrees(heading_error):.2f}deg"
        )

        # Build predicted trajectory (simplified - just current state repeated)
        # PID doesn't predict future states like MPC
        predicted_trajectory = np.tile(current_state, (len(reference_trajectory), 1))

        info = {
            "predicted_trajectory": predicted_trajectory,
            "predicted_controls": np.array([[v, omega]]),
            "cost": distance_error**2 + heading_error**2,  # Pseudo-cost
            "solve_time": compute_time,
            "solver_status": "success",
            "distance_error": distance_error,
            "heading_error": heading_error,
            "lookahead_idx": lookahead_idx,
        }

        return np.array([v, omega]), info

    def reset(self) -> None:
        """Reset PID state and iteration count."""
        self._linear_integral = 0.0
        self._angular_integral = 0.0
        self._prev_linear_error = 0.0
        self._prev_angular_error = 0.0
        self._prev_time = None
        self._iteration_count = 0
        logger.debug("PID controller reset")

    def set_gains(
        self,
        kp_linear: float | None = None,
        ki_linear: float | None = None,
        kd_linear: float | None = None,
        kp_angular: float | None = None,
        ki_angular: float | None = None,
        kd_angular: float | None = None,
    ) -> None:
        """
        Update PID gains dynamically.

        Args:
            kp_linear: Proportional gain for linear velocity
            ki_linear: Integral gain for linear velocity
            kd_linear: Derivative gain for linear velocity
            kp_angular: Proportional gain for angular velocity
            ki_angular: Integral gain for angular velocity
            kd_angular: Derivative gain for angular velocity
        """
        if kp_linear is not None:
            self.gains.kp_linear = kp_linear
        if ki_linear is not None:
            self.gains.ki_linear = ki_linear
        if kd_linear is not None:
            self.gains.kd_linear = kd_linear
        if kp_angular is not None:
            self.gains.kp_angular = kp_angular
        if ki_angular is not None:
            self.gains.ki_angular = ki_angular
        if kd_angular is not None:
            self.gains.kd_angular = kd_angular

        logger.info(
            f"PID gains updated: "
            f"linear=({self.gains.kp_linear}, {self.gains.ki_linear}, {self.gains.kd_linear}), "
            f"angular=({self.gains.kp_angular}, {self.gains.ki_angular}, {self.gains.kd_angular})"
        )
