"""
Math Utilities Module.

로봇 제어에 필요한 수학 유틸리티 함수들을 제공합니다.
"""

import numpy as np
from typing import Tuple


def normalize_angle(angle: float) -> float:
    """
    각도를 [-pi, pi] 범위로 정규화합니다.

    Args:
        angle: Input angle in radians

    Returns:
        Normalized angle in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def normalize_angle_array(angles: np.ndarray) -> np.ndarray:
    """
    각도 배열을 [-pi, pi] 범위로 정규화합니다.

    Args:
        angles: Array of angles in radians

    Returns:
        Normalized angle array
    """
    return np.arctan2(np.sin(angles), np.cos(angles))


def angle_difference(angle1: float, angle2: float) -> float:
    """
    두 각도 사이의 최소 차이를 계산합니다.

    Args:
        angle1: First angle in radians
        angle2: Second angle in radians

    Returns:
        Smallest signed angle difference in [-pi, pi]
    """
    return normalize_angle(angle1 - angle2)


def euclidean_distance(
    x1: float, y1: float,
    x2: float, y2: float,
) -> float:
    """
    두 점 사이의 유클리드 거리를 계산합니다.

    Args:
        x1, y1: First point
        x2, y2: Second point

    Returns:
        Euclidean distance
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def rotation_matrix_2d(theta: float) -> np.ndarray:
    """
    2D 회전 행렬을 생성합니다.

    Args:
        theta: Rotation angle in radians

    Returns:
        2x2 rotation matrix
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def transform_to_local(
    x: float, y: float, theta: float,
    global_x: float, global_y: float,
) -> Tuple[float, float]:
    """
    전역 좌표를 로봇 로컬 좌표로 변환합니다.

    Args:
        x, y, theta: Robot pose in global frame
        global_x, global_y: Point in global frame

    Returns:
        (local_x, local_y) in robot frame
    """
    dx = global_x - x
    dy = global_y - y
    cos_theta = np.cos(-theta)
    sin_theta = np.sin(-theta)
    local_x = dx * cos_theta - dy * sin_theta
    local_y = dx * sin_theta + dy * cos_theta
    return local_x, local_y


def transform_to_global(
    x: float, y: float, theta: float,
    local_x: float, local_y: float,
) -> Tuple[float, float]:
    """
    로봇 로컬 좌표를 전역 좌표로 변환합니다.

    Args:
        x, y, theta: Robot pose in global frame
        local_x, local_y: Point in robot frame

    Returns:
        (global_x, global_y) in global frame
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    global_x = x + local_x * cos_theta - local_y * sin_theta
    global_y = y + local_x * sin_theta + local_y * cos_theta
    return global_x, global_y


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    값을 지정된 범위로 제한합니다.

    Args:
        value: Input value
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def interpolate_angle(
    angle1: float, angle2: float, t: float
) -> float:
    """
    두 각도 사이를 선형 보간합니다.

    최단 경로로 보간합니다.

    Args:
        angle1: Start angle
        angle2: End angle
        t: Interpolation factor [0, 1]

    Returns:
        Interpolated angle
    """
    diff = angle_difference(angle2, angle1)
    return normalize_angle(angle1 + t * diff)


def compute_curvature(
    x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    경로의 곡률을 계산합니다.

    Args:
        x, y: Path coordinates

    Returns:
        Curvature at each point
    """
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5

    # 무한대 방지
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)

    return curvature


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    이동 평균 필터를 적용합니다.

    Args:
        data: Input data array
        window_size: Size of moving window

    Returns:
        Filtered data
    """
    if window_size < 1:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode="same")


def low_pass_filter(
    current: float, target: float, alpha: float
) -> float:
    """
    1차 로우패스 필터를 적용합니다.

    Args:
        current: Current value
        target: Target value
        alpha: Filter coefficient (0 < alpha < 1)

    Returns:
        Filtered value
    """
    return current + alpha * (target - current)


class PIDController:
    """
    간단한 PID 컨트롤러.

    비교 및 테스트 목적으로 사용됩니다.
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        output_limits: Tuple[float, float] = (-float("inf"), float("inf")),
    ):
        """
        Initialize PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limits: (min, max) output limits
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits

        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        """
        PID 제어 출력을 계산합니다.

        Args:
            error: Current error
            dt: Time step

        Returns:
            Control output
        """
        # 적분 항
        self._integral += error * dt

        # 미분 항
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        # 출력 계산
        output = (
            self.kp * error +
            self.ki * self._integral +
            self.kd * derivative
        )

        # 출력 제한
        output = clamp(output, *self.output_limits)

        return output

    def reset(self) -> None:
        """컨트롤러 상태를 초기화합니다."""
        self._integral = 0.0
        self._prev_error = 0.0
