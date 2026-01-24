"""Trajectory generation utilities."""

from typing import Callable

import numpy as np


def normalize_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """
    Normalize angle to [-pi, pi].

    Args:
        angle: Angle(s) in radians

    Returns:
        Normalized angle(s) in [-pi, pi]
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def unwrap_angles(angles: np.ndarray) -> np.ndarray:
    """
    Unwrap angles to ensure continuity (no jumps > pi).

    연속적인 경로에서 각도가 -π ↔ π 경계를 넘을 때
    급격한 점프 없이 연속적으로 변하도록 함.

    Args:
        angles: Array of angles in radians

    Returns:
        Unwrapped angles (may exceed [-pi, pi] but continuous)
    """
    return np.unwrap(angles)


def angle_difference(angle1: float, angle2: float) -> float:
    """
    Compute the shortest angular difference from angle2 to angle1.

    Returns a value in [-pi, pi] representing the shortest rotation
    from angle2 to angle1.

    Args:
        angle1: Target angle
        angle2: Source angle

    Returns:
        Shortest angular difference (angle1 - angle2) in [-pi, pi]
    """
    diff = angle1 - angle2
    return np.arctan2(np.sin(diff), np.cos(diff))


def generate_line_trajectory(
    start: np.ndarray,
    end: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """
    Generate a straight line trajectory.

    Args:
        start: Start position [x, y]
        end: End position [x, y]
        num_points: Number of points

    Returns:
        Trajectory array, shape (num_points, 3) with [x, y, theta]
    """
    trajectory = np.zeros((num_points, 3))

    # Interpolate positions
    for i in range(2):
        trajectory[:, i] = np.linspace(start[i], end[i], num_points)

    # Compute heading angle
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    theta = np.arctan2(dy, dx)
    trajectory[:, 2] = theta

    return trajectory


def generate_circle_trajectory(
    center: np.ndarray,
    radius: float,
    num_points: int,
    start_angle: float = 0.0,
    end_angle: float = 2 * np.pi,
) -> np.ndarray:
    """
    Generate a circular trajectory.

    Args:
        center: Circle center [x, y]
        radius: Circle radius
        num_points: Number of points
        start_angle: Starting angle [rad]
        end_angle: Ending angle [rad]

    Returns:
        Trajectory array, shape (num_points, 3) with [x, y, theta]
        Note: theta is kept continuous (unwrapped) to avoid jumps at ±π boundary
    """
    trajectory = np.zeros((num_points, 3))
    angles = np.linspace(start_angle, end_angle, num_points)

    trajectory[:, 0] = center[0] + radius * np.cos(angles)
    trajectory[:, 1] = center[1] + radius * np.sin(angles)

    # Heading tangent to circle (90 degrees ahead of radial)
    # Keep angles continuous (unwrapped) to avoid discontinuity at ±π
    trajectory[:, 2] = angles + np.pi / 2

    # Do NOT normalize here - keep continuous for smooth interpolation
    # The MPC cost function will handle the angle difference properly

    return trajectory


def generate_figure_eight_trajectory(
    center: np.ndarray,
    scale: float,
    num_points: int,
) -> np.ndarray:
    """
    Generate a figure-8 (lemniscate) trajectory.

    Args:
        center: Center position [x, y]
        scale: Size scaling factor
        num_points: Number of points

    Returns:
        Trajectory array, shape (num_points, 3) with [x, y, theta]
        Note: theta is unwrapped for continuity
    """
    trajectory = np.zeros((num_points, 3))
    t = np.linspace(0, 2 * np.pi, num_points)

    # Lemniscate of Bernoulli parametrization
    trajectory[:, 0] = center[0] + scale * np.sin(t)
    trajectory[:, 1] = center[1] + scale * np.sin(t) * np.cos(t)

    # Compute heading from velocity direction
    dx = scale * np.cos(t)
    dy = scale * (np.cos(t) ** 2 - np.sin(t) ** 2)
    trajectory[:, 2] = np.arctan2(dy, dx)

    # Unwrap angles to ensure continuity
    trajectory[:, 2] = unwrap_angles(trajectory[:, 2])

    return trajectory


def generate_sinusoidal_trajectory(
    start: np.ndarray,
    length: float,
    amplitude: float,
    frequency: float,
    num_points: int,
) -> np.ndarray:
    """
    Generate a sinusoidal trajectory.

    Args:
        start: Start position [x, y]
        length: Total length in x direction
        amplitude: Sine wave amplitude
        frequency: Number of complete cycles
        num_points: Number of points

    Returns:
        Trajectory array, shape (num_points, 3) with [x, y, theta]
    """
    trajectory = np.zeros((num_points, 3))

    x = np.linspace(start[0], start[0] + length, num_points)
    trajectory[:, 0] = x
    trajectory[:, 1] = start[1] + amplitude * np.sin(2 * np.pi * frequency * (x - start[0]) / length)

    # Compute heading from derivative
    dy_dx = amplitude * 2 * np.pi * frequency / length * np.cos(
        2 * np.pi * frequency * (x - start[0]) / length
    )
    trajectory[:, 2] = np.arctan2(dy_dx, np.ones_like(dy_dx))

    return trajectory


class TrajectoryInterpolator:
    """
    Interpolates reference trajectory for MPC horizon.
    """

    def __init__(self, trajectory: np.ndarray, dt: float):
        """
        Args:
            trajectory: Full reference trajectory, shape (M, 3)
            dt: Time step between trajectory points
        """
        self.trajectory = trajectory
        self.dt = dt
        self.num_points = len(trajectory)
        self.total_time = (self.num_points - 1) * dt

    def get_reference(
        self,
        current_time: float,
        horizon: int,
        mpc_dt: float,
        current_theta: float | None = None,
    ) -> np.ndarray:
        """
        Get reference trajectory for MPC horizon.

        Args:
            current_time: Current time [s]
            horizon: MPC prediction horizon N
            mpc_dt: MPC time step
            current_theta: Current robot heading (optional, for angle continuity)

        Returns:
            Reference trajectory, shape (horizon+1, 3)
            Note: theta values are adjusted to be continuous with current_theta
        """
        reference = np.zeros((horizon + 1, 3))

        for k in range(horizon + 1):
            t = current_time + k * mpc_dt

            # Clamp to trajectory bounds
            if t >= self.total_time:
                reference[k] = self.trajectory[-1].copy()
            else:
                # Linear interpolation
                idx = t / self.dt
                idx_low = int(np.floor(idx))
                idx_high = min(idx_low + 1, self.num_points - 1)
                alpha = idx - idx_low

                # Interpolate position
                reference[k, :2] = (1 - alpha) * self.trajectory[idx_low, :2] + alpha * self.trajectory[idx_high, :2]

                # Interpolate angle - trajectory angles are already unwrapped/continuous
                theta_low = self.trajectory[idx_low, 2]
                theta_high = self.trajectory[idx_high, 2]
                reference[k, 2] = (1 - alpha) * theta_low + alpha * theta_high

        # Adjust reference angles to be continuous with current robot heading
        # This prevents the MPC from trying to take the "long way around"
        if current_theta is not None:
            # The reference trajectory angles may be outside [-π, π] (unwrapped).
            # We need to shift the entire reference to be in the same "wrapping"
            # as current_theta to minimize the error seen by MPC.
            ref_theta_0 = reference[0, 2]

            # Normalize both to [-π, π] to compute the minimal difference
            ref_theta_0_norm = normalize_angle(ref_theta_0)
            current_theta_norm = normalize_angle(current_theta)

            # Compute shortest angular difference
            diff = angle_difference(ref_theta_0_norm, current_theta_norm)

            # The target first reference angle should be current_theta + diff
            # This gives us the reference angle that's closest to current_theta
            target_ref_theta_0 = current_theta_norm + diff

            # Shift all reference angles to align
            offset = target_ref_theta_0 - ref_theta_0
            reference[:, 2] += offset

        return reference

    def find_closest_point(self, position: np.ndarray) -> tuple[int, float]:
        """
        Find the closest point on the trajectory to a given position.

        Args:
            position: Query position [x, y]

        Returns:
            Tuple of (index, distance)
        """
        distances = np.linalg.norm(self.trajectory[:, :2] - position[:2], axis=1)
        idx = np.argmin(distances)
        return idx, distances[idx]
