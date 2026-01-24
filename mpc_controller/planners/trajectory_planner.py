"""
Trajectory Planner Module.

MPC를 위한 경로 계획 및 궤적 생성을 담당합니다.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Waypoint:
    """경로점 데이터 클래스."""

    x: float
    y: float
    theta: Optional[float] = None
    velocity: Optional[float] = None


@dataclass
class Trajectory:
    """궤적 데이터 클래스."""

    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    velocity: np.ndarray
    timestamps: np.ndarray

    def __len__(self) -> int:
        return len(self.x)

    def get_state_at(self, index: int) -> tuple[float, float, float, float]:
        """특정 인덱스의 상태 반환."""
        return (
            self.x[index],
            self.y[index],
            self.theta[index],
            self.velocity[index],
        )


class TrajectoryPlanner:
    """
    경로 계획기 클래스.

    웨이포인트로부터 부드러운 궤적을 생성합니다.
    """

    def __init__(
        self,
        dt: float = 0.1,
        max_velocity: float = 1.0,
        max_acceleration: float = 0.5,
    ):
        """
        Initialize trajectory planner.

        Args:
            dt: Time step for trajectory generation
            max_velocity: Maximum allowed velocity
            max_acceleration: Maximum allowed acceleration
        """
        self.dt = dt
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

    def plan_from_waypoints(
        self,
        waypoints: list[Waypoint],
        interpolation_method: str = "cubic",
    ) -> Trajectory:
        """
        웨이포인트로부터 궤적을 생성합니다.

        Args:
            waypoints: List of waypoints
            interpolation_method: Interpolation method (linear, cubic)

        Returns:
            Generated trajectory
        """
        if len(waypoints) < 2:
            raise ValueError("At least 2 waypoints required")

        # 웨이포인트 좌표 추출
        wp_x = np.array([wp.x for wp in waypoints])
        wp_y = np.array([wp.y for wp in waypoints])

        # 웨이포인트 간 거리 계산
        distances = np.sqrt(np.diff(wp_x) ** 2 + np.diff(wp_y) ** 2)
        cumulative_dist = np.concatenate([[0], np.cumsum(distances)])
        total_distance = cumulative_dist[-1]

        # 속도 프로파일 계산
        num_points = int(total_distance / (self.max_velocity * self.dt)) + 1
        s = np.linspace(0, total_distance, num_points)

        # 보간
        if interpolation_method == "cubic":
            traj_x = self._cubic_interpolate(cumulative_dist, wp_x, s)
            traj_y = self._cubic_interpolate(cumulative_dist, wp_y, s)
        else:
            traj_x = np.interp(s, cumulative_dist, wp_x)
            traj_y = np.interp(s, cumulative_dist, wp_y)

        # 방향 계산
        theta = self._compute_heading(traj_x, traj_y)

        # 속도 프로파일 (사다리꼴)
        velocity = self._compute_velocity_profile(len(traj_x))

        # 타임스탬프
        timestamps = np.arange(len(traj_x)) * self.dt

        return Trajectory(
            x=traj_x,
            y=traj_y,
            theta=theta,
            velocity=velocity,
            timestamps=timestamps,
        )

    def plan_circle(
        self,
        center: tuple[float, float],
        radius: float,
        start_angle: float = 0.0,
        end_angle: float = 2 * np.pi,
    ) -> Trajectory:
        """
        원형 궤적을 생성합니다.

        Args:
            center: Circle center (x, y)
            radius: Circle radius
            start_angle: Starting angle
            end_angle: Ending angle

        Returns:
            Circular trajectory
        """
        arc_length = abs(end_angle - start_angle) * radius
        num_points = int(arc_length / (self.max_velocity * self.dt)) + 1
        angles = np.linspace(start_angle, end_angle, num_points)

        traj_x = center[0] + radius * np.cos(angles)
        traj_y = center[1] + radius * np.sin(angles)
        theta = angles + np.pi / 2  # 접선 방향

        velocity = self._compute_velocity_profile(num_points)
        timestamps = np.arange(num_points) * self.dt

        return Trajectory(
            x=traj_x,
            y=traj_y,
            theta=theta,
            velocity=velocity,
            timestamps=timestamps,
        )

    def plan_figure_eight(
        self,
        center: tuple[float, float],
        size: float,
    ) -> Trajectory:
        """
        8자 궤적을 생성합니다.

        Args:
            center: Center of figure eight
            size: Size of the figure eight

        Returns:
            Figure eight trajectory
        """
        t = np.linspace(0, 2 * np.pi, 200)

        traj_x = center[0] + size * np.sin(t)
        traj_y = center[1] + size * np.sin(t) * np.cos(t)

        theta = self._compute_heading(traj_x, traj_y)
        velocity = self._compute_velocity_profile(len(t))
        timestamps = np.arange(len(t)) * self.dt

        return Trajectory(
            x=traj_x,
            y=traj_y,
            theta=theta,
            velocity=velocity,
            timestamps=timestamps,
        )

    def _cubic_interpolate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_new: np.ndarray,
    ) -> np.ndarray:
        """Cubic spline interpolation."""
        from scipy.interpolate import CubicSpline

        cs = CubicSpline(x, y)
        return cs(x_new)

    def _compute_heading(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute heading angles from trajectory points."""
        dx = np.diff(x)
        dy = np.diff(y)
        theta = np.arctan2(dy, dx)
        # 마지막 점은 이전 방향 유지
        theta = np.append(theta, theta[-1])
        return theta

    def _compute_velocity_profile(self, num_points: int) -> np.ndarray:
        """
        사다리꼴 속도 프로파일 생성.

        가속 → 순항 → 감속 구간으로 구성됩니다.
        """
        velocity = np.ones(num_points) * self.max_velocity

        # 가속/감속 구간 길이
        accel_steps = int(self.max_velocity / (self.max_acceleration * self.dt))
        accel_steps = min(accel_steps, num_points // 3)

        # 가속 구간
        for i in range(accel_steps):
            velocity[i] = (i + 1) * self.max_acceleration * self.dt
            velocity[i] = min(velocity[i], self.max_velocity)

        # 감속 구간
        for i in range(accel_steps):
            velocity[-(i + 1)] = (i + 1) * self.max_acceleration * self.dt
            velocity[-(i + 1)] = min(velocity[-(i + 1)], self.max_velocity)

        return velocity

    def find_closest_point(
        self,
        trajectory: Trajectory,
        x: float,
        y: float,
        start_index: int = 0,
    ) -> tuple[int, float]:
        """
        궤적에서 현재 위치에 가장 가까운 점을 찾습니다.

        Args:
            trajectory: Reference trajectory
            x, y: Current position
            start_index: Starting search index

        Returns:
            (closest_index, distance)
        """
        # 검색 범위 제한 (효율성)
        search_range = min(50, len(trajectory) - start_index)
        end_index = start_index + search_range

        dx = trajectory.x[start_index:end_index] - x
        dy = trajectory.y[start_index:end_index] - y
        distances = np.sqrt(dx ** 2 + dy ** 2)

        min_idx = np.argmin(distances)
        return start_index + min_idx, distances[min_idx]

    def get_reference_states(
        self,
        trajectory: Trajectory,
        current_index: int,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        MPC horizon에 대한 참조 상태를 반환합니다.

        Args:
            trajectory: Reference trajectory
            current_index: Current trajectory index
            horizon: MPC prediction horizon

        Returns:
            (x_ref, y_ref, theta_ref, v_ref)
        """
        end_index = min(current_index + horizon, len(trajectory))

        x_ref = trajectory.x[current_index:end_index]
        y_ref = trajectory.y[current_index:end_index]
        theta_ref = trajectory.theta[current_index:end_index]
        v_ref = trajectory.velocity[current_index:end_index]

        # Horizon이 궤적 끝을 넘으면 마지막 값으로 패딩
        if len(x_ref) < horizon:
            pad_length = horizon - len(x_ref)
            x_ref = np.pad(x_ref, (0, pad_length), mode="edge")
            y_ref = np.pad(y_ref, (0, pad_length), mode="edge")
            theta_ref = np.pad(theta_ref, (0, pad_length), mode="edge")
            v_ref = np.pad(v_ref, (0, pad_length), mode="edge")

        return x_ref, y_ref, theta_ref, v_ref
