"""
Dynamic Obstacle Prediction Module.

동적 장애물의 미래 위치를 예측하는 알고리즘을 제공합니다.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional
import numpy as np


class PredictionModel(Enum):
    """예측 모델 타입."""

    CONSTANT_VELOCITY = "constant_velocity"  # 일정 속도 모델
    CONSTANT_ACCELERATION = "constant_acceleration"  # 일정 가속도 모델
    KALMAN_FILTER = "kalman_filter"  # 칼만 필터 기반 예측


@dataclass
class DynamicObstacleState:
    """동적 장애물 상태 정보."""

    x: float  # 위치 X [m]
    y: float  # 위치 Y [m]
    vx: float  # 속도 X [m/s]
    vy: float  # 속도 Y [m/s]
    ax: float = 0.0  # 가속도 X [m/s^2]
    ay: float = 0.0  # 가속도 Y [m/s^2]
    radius: float = 0.5  # 반경 [m]
    timestamp: float = 0.0  # 타임스탬프 [s]

    @property
    def speed(self) -> float:
        """속도 크기를 반환합니다."""
        return np.sqrt(self.vx**2 + self.vy**2)

    @property
    def heading(self) -> float:
        """진행 방향 각도를 반환합니다 [rad]."""
        return np.arctan2(self.vy, self.vx)


class DynamicObstaclePredictor:
    """
    동적 장애물 예측 클래스.

    다양한 예측 모델을 사용하여 동적 장애물의 미래 위치를 예측합니다.
    """

    def __init__(
        self,
        model: PredictionModel = PredictionModel.CONSTANT_VELOCITY,
        max_prediction_time: float = 3.0,
        uncertainty_growth_rate: float = 0.1,
    ):
        """
        동적 장애물 예측기 초기화.

        Args:
            model: 예측 모델 타입
            max_prediction_time: 최대 예측 시간 [s]
            uncertainty_growth_rate: 불확실성 증가율 [m/s]
        """
        self.model = model
        self.max_prediction_time = max_prediction_time
        self.uncertainty_growth_rate = uncertainty_growth_rate

        # 장애물 상태 히스토리 (칼만 필터용)
        self.state_history: dict[int, List[DynamicObstacleState]] = {}

    def predict_trajectory(
        self,
        obstacle_state: DynamicObstacleState,
        dt: float,
        steps: int,
    ) -> List[Tuple[float, float, float]]:
        """
        장애물의 미래 궤적을 예측합니다.

        Args:
            obstacle_state: 현재 장애물 상태
            dt: 시간 간격 [s]
            steps: 예측 스텝 수

        Returns:
            List of (x, y, radius_with_uncertainty) tuples
        """
        if self.model == PredictionModel.CONSTANT_VELOCITY:
            return self._predict_constant_velocity(obstacle_state, dt, steps)
        elif self.model == PredictionModel.CONSTANT_ACCELERATION:
            return self._predict_constant_acceleration(obstacle_state, dt, steps)
        elif self.model == PredictionModel.KALMAN_FILTER:
            return self._predict_kalman_filter(obstacle_state, dt, steps)
        else:
            raise ValueError(f"Unknown prediction model: {self.model}")

    def _predict_constant_velocity(
        self,
        state: DynamicObstacleState,
        dt: float,
        steps: int,
    ) -> List[Tuple[float, float, float]]:
        """
        일정 속도 모델로 예측.

        가장 간단한 모델: p(t) = p(0) + v * t

        Args:
            state: 현재 상태
            dt: 시간 간격
            steps: 예측 스텝 수

        Returns:
            List of (x, y, radius) tuples
        """
        predictions = []

        for i in range(1, steps + 1):
            t = i * dt

            # 위치 예측
            pred_x = state.x + state.vx * t
            pred_y = state.y + state.vy * t

            # 불확실성 증가 (시간에 비례)
            uncertainty = self.uncertainty_growth_rate * t
            pred_radius = state.radius + uncertainty

            predictions.append((pred_x, pred_y, pred_radius))

        return predictions

    def _predict_constant_acceleration(
        self,
        state: DynamicObstacleState,
        dt: float,
        steps: int,
    ) -> List[Tuple[float, float, float]]:
        """
        일정 가속도 모델로 예측.

        운동 방정식: p(t) = p(0) + v(0)*t + 0.5*a*t^2

        Args:
            state: 현재 상태
            dt: 시간 간격
            steps: 예측 스텝 수

        Returns:
            List of (x, y, radius) tuples
        """
        predictions = []

        for i in range(1, steps + 1):
            t = i * dt

            # 위치 예측 (등가속도 운동)
            pred_x = state.x + state.vx * t + 0.5 * state.ax * t**2
            pred_y = state.y + state.vy * t + 0.5 * state.ay * t**2

            # 불확실성 증가 (시간 제곱에 비례)
            uncertainty = self.uncertainty_growth_rate * (t**1.5)
            pred_radius = state.radius + uncertainty

            predictions.append((pred_x, pred_y, pred_radius))

        return predictions

    def _predict_kalman_filter(
        self,
        state: DynamicObstacleState,
        dt: float,
        steps: int,
    ) -> List[Tuple[float, float, float]]:
        """
        칼만 필터 기반 예측.

        히스토리 기반 상태 추정 및 예측.
        (추후 구현: 현재는 constant velocity와 동일)

        Args:
            state: 현재 상태
            dt: 시간 간격
            steps: 예측 스텝 수

        Returns:
            List of (x, y, radius) tuples
        """
        # TODO: 칼만 필터 구현
        # 현재는 일정 속도 모델과 동일하게 동작
        return self._predict_constant_velocity(state, dt, steps)

    def update_state_history(
        self,
        obstacle_id: int,
        state: DynamicObstacleState,
        max_history: int = 10,
    ) -> None:
        """
        장애물 상태 히스토리 업데이트.

        칼만 필터 등 히스토리 기반 예측에 사용됩니다.

        Args:
            obstacle_id: 장애물 ID
            state: 현재 상태
            max_history: 최대 히스토리 개수
        """
        if obstacle_id not in self.state_history:
            self.state_history[obstacle_id] = []

        self.state_history[obstacle_id].append(state)

        # 최대 히스토리 개수 유지
        if len(self.state_history[obstacle_id]) > max_history:
            self.state_history[obstacle_id].pop(0)

    def estimate_acceleration(
        self,
        obstacle_id: int,
    ) -> Optional[Tuple[float, float]]:
        """
        히스토리를 기반으로 가속도를 추정합니다.

        Args:
            obstacle_id: 장애물 ID

        Returns:
            (ax, ay) or None if insufficient history
        """
        if obstacle_id not in self.state_history:
            return None

        history = self.state_history[obstacle_id]

        if len(history) < 2:
            return None

        # 최근 2개 상태로 가속도 추정
        recent = history[-1]
        previous = history[-2]

        dt = recent.timestamp - previous.timestamp

        if dt <= 0.001:
            return None

        ax = (recent.vx - previous.vx) / dt
        ay = (recent.vy - previous.vy) / dt

        return (ax, ay)

    def predict_collision_time(
        self,
        obstacle_state: DynamicObstacleState,
        robot_x: float,
        robot_y: float,
        robot_radius: float,
        safety_margin: float = 0.3,
    ) -> Optional[float]:
        """
        로봇과 장애물의 충돌 예상 시간을 계산합니다.

        Args:
            obstacle_state: 장애물 상태
            robot_x, robot_y: 로봇 위치
            robot_radius: 로봇 반경
            safety_margin: 안전 마진

        Returns:
            충돌 예상 시간 [s] or None (충돌 없음)
        """
        # 상대 위치 및 속도
        rel_x = obstacle_state.x - robot_x
        rel_y = obstacle_state.y - robot_y
        rel_vx = obstacle_state.vx
        rel_vy = obstacle_state.vy

        # 현재 거리
        current_dist = np.sqrt(rel_x**2 + rel_y**2)

        # 충돌 거리
        collision_dist = obstacle_state.radius + robot_radius + safety_margin

        # 이미 충돌 거리 안에 있으면
        if current_dist <= collision_dist:
            return 0.0

        # 속도가 너무 작으면 정적 상태로 판단
        if abs(rel_vx) < 1e-6 and abs(rel_vy) < 1e-6:
            # 정적 상태이고 충돌 거리 밖에 있으면 충돌 없음
            return None

        # 2차 방정식 계수
        # ||p(t)||^2 = collision_dist^2
        # p(t) = (rel_x + rel_vx*t, rel_y + rel_vy*t)
        a = rel_vx**2 + rel_vy**2
        b = 2 * (rel_x * rel_vx + rel_y * rel_vy)
        c = rel_x**2 + rel_y**2 - collision_dist**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return None  # 충돌 없음

        # 양의 시간 해 찾기
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)

        # 가장 빠른 양의 시간 반환 (예측 시간 범위 내)
        valid_times = [t for t in [t1, t2] if t > 0 and t < self.max_prediction_time]

        if valid_times:
            return min(valid_times)

        return None

    def predict_multiple_obstacles(
        self,
        obstacle_states: List[DynamicObstacleState],
        dt: float,
        steps: int,
    ) -> List[List[Tuple[float, float, float]]]:
        """
        여러 장애물의 궤적을 예측합니다.

        Args:
            obstacle_states: 장애물 상태 리스트
            dt: 시간 간격
            steps: 예측 스텝 수

        Returns:
            각 장애물별 예측 궤적 리스트
        """
        return [
            self.predict_trajectory(obs, dt, steps)
            for obs in obstacle_states
        ]
