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


class LookaheadInterpolator:
    """로봇 위치 기반 closest-point + arc-length lookahead 참조 궤적 생성.

    C++ nav2 플러그인의 pathToReferenceTrajectory 로직을 Python으로 이식:
      1. closest point 탐색 (프루닝 윈도우)
      2. arc-length 누적
      3. lookahead 거리 내에서 N등분 선형 보간
      4. 경로 접선 theta + 각도 연속성 보정

    Parameters
    ----------
    trajectory : (M, nx) 전체 참조 궤적
    dt : float
        궤적 점 간 시간 간격 (완주 판정용)
    lookahead_dist : float
        Lookahead 거리 [m]. 0이면 자동 (v_max * N * mpc_dt)
    min_lookahead : float
        최소 lookahead [m] (goal 근처 수렴 보장)
    v_max : float
        최대 속도 [m/s] (자동 lookahead 계산용)
    """

    def __init__(
        self,
        trajectory: np.ndarray,
        dt: float = 0.05,
        lookahead_dist: float = 0.0,
        min_lookahead: float = 0.5,
        v_max: float = 1.0,
    ):
        self.trajectory = trajectory
        self.dt = dt
        self.num_points = len(trajectory)
        self.nx = trajectory.shape[1]
        self.lookahead_dist = lookahead_dist
        self.min_lookahead = min_lookahead
        self.v_max = v_max

        # arc-length 사전 계산
        diffs = np.diff(trajectory[:, :2], axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self.arc_lengths = np.zeros(self.num_points)
        self.arc_lengths[1:] = np.cumsum(seg_lengths)
        self.total_arc_length = self.arc_lengths[-1]

        # 프루닝 인덱스 (점진 탐색용)
        self._prune_idx = 0

    def reset(self):
        """프루닝 인덱스 초기화."""
        self._prune_idx = 0

    def find_closest_point(self, position: np.ndarray) -> tuple:
        """현재 위치에서 가장 가까운 경로 점 탐색 (윈도우 기반).

        Returns
        -------
        (index, distance)
        """
        search_end = min(self._prune_idx + 50, self.num_points)
        search_range = slice(self._prune_idx, search_end)
        dists = np.linalg.norm(
            self.trajectory[search_range, :2] - position[:2], axis=1
        )
        local_idx = np.argmin(dists)
        global_idx = self._prune_idx + local_idx
        self._prune_idx = global_idx
        return global_idx, float(dists[local_idx])

    def get_reference(
        self,
        current_position: np.ndarray,
        horizon: int,
        mpc_dt: float,
        current_theta: float = None,
    ) -> np.ndarray:
        """로봇 위치 기반 lookahead 참조 궤적 생성.

        Parameters
        ----------
        current_position : (2,) or (nx,) — x, y [, theta, ...]
        horizon : int — N
        mpc_dt : float
        current_theta : float, optional

        Returns
        -------
        reference : (horizon+1, nx)
        """
        # 1. Closest point
        closest_idx, _ = self.find_closest_point(current_position)

        # 2. Lookahead 거리 결정
        if self.lookahead_dist > 0:
            lookahead = self.lookahead_dist
        else:
            lookahead = self.v_max * horizon * mpc_dt

        # 남은 경로 길이
        remaining_arc = self.total_arc_length - self.arc_lengths[closest_idx]
        effective_lookahead = max(
            self.min_lookahead,
            min(lookahead, remaining_arc),
        )
        step_distance = effective_lookahead / horizon

        # 3. Arc-length stepping으로 참조 궤적 생성
        reference = np.zeros((horizon + 1, self.nx))
        base_arc = self.arc_lengths[closest_idx]
        path_idx = closest_idx

        for t in range(horizon + 1):
            target_arc = base_arc + t * step_distance

            # 목표 arc-length에 해당하는 구간 탐색
            while (path_idx < self.num_points - 1 and
                   self.arc_lengths[path_idx + 1] < target_arc):
                path_idx += 1

            if path_idx >= self.num_points - 1:
                # 경로 끝 도달
                reference[t, :] = self.trajectory[-1].copy()
            else:
                # 구간 내 선형 보간
                seg_len = self.arc_lengths[path_idx + 1] - self.arc_lengths[path_idx]
                if seg_len > 1e-6:
                    alpha = (target_arc - self.arc_lengths[path_idx]) / seg_len
                else:
                    alpha = 0.0

                reference[t, :2] = (
                    (1 - alpha) * self.trajectory[path_idx, :2]
                    + alpha * self.trajectory[path_idx + 1, :2]
                )

                # 경로 접선 theta
                dx = (self.trajectory[path_idx + 1, 0]
                      - self.trajectory[path_idx, 0])
                dy = (self.trajectory[path_idx + 1, 1]
                      - self.trajectory[path_idx, 1])
                tangent_len = np.sqrt(dx * dx + dy * dy)
                if tangent_len > 1e-6:
                    reference[t, 2] = np.arctan2(dy, dx)
                else:
                    reference[t, 2] = (reference[t - 1, 2] if t > 0 else 0.0)

                # nx > 3인 경우 나머지 상태는 보간
                if self.nx > 3:
                    reference[t, 3:] = (
                        (1 - alpha) * self.trajectory[path_idx, 3:]
                        + alpha * self.trajectory[path_idx + 1, 3:]
                    )

        # 4. 각도 연속성 보정
        # theta를 unwrap하여 급격한 점프 방지
        reference[:, 2] = np.unwrap(reference[:, 2])

        if current_theta is not None:
            ref_theta_0 = reference[0, 2]
            diff = np.arctan2(
                np.sin(ref_theta_0 - current_theta),
                np.cos(ref_theta_0 - current_theta),
            )
            target = current_theta + diff
            offset = target - ref_theta_0
            reference[:, 2] += offset

        return reference
