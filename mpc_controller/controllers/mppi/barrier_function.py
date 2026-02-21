"""Control Barrier Function (CBF) 모듈 — 원형 장애물 barrier 함수.

h_i(x) = ||p - p_obs_i||² - d_safe_i²
d_safe_i = r_obs_i + r_robot + margin

h > 0: 안전 영역 (장애물 밖)
h = 0: 안전 경계
h < 0: 위험 영역 (장애물 안)
"""

from typing import List, Optional

import numpy as np


class CircleBarrier:
    """원형 장애물에 대한 barrier function.

    h(x) = ||p - p_obs||² - d_safe²

    이차(squared) 형태를 사용하여:
    - gradient 연속성 보장 (sqrt 미분 불연속 회피)
    - 배치 계산 효율적
    """

    def __init__(
        self,
        obstacle: np.ndarray,
        robot_radius: float = 0.2,
        safety_margin: float = 0.3,
    ):
        """
        Args:
            obstacle: (3,) [x, y, radius] 장애물 정보
            robot_radius: 로봇 반경 [m]
            safety_margin: 추가 안전 마진 [m]
        """
        self.obs_x = obstacle[0]
        self.obs_y = obstacle[1]
        self.obs_radius = obstacle[2]
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.d_safe = self.obs_radius + self.robot_radius + self.safety_margin
        self.d_safe_sq = self.d_safe ** 2

    def evaluate(self, state: np.ndarray) -> float:
        """단일 상태에서 barrier 값 계산.

        Args:
            state: (nx,) 상태 벡터 [x, y, ...]

        Returns:
            h(x) = ||p - p_obs||² - d_safe²
        """
        dx = state[0] - self.obs_x
        dy = state[1] - self.obs_y
        return dx ** 2 + dy ** 2 - self.d_safe_sq

    def evaluate_batch(self, states: np.ndarray) -> np.ndarray:
        """배치 상태에서 barrier 값 계산.

        Args:
            states: (M, nx) 상태 배열

        Returns:
            (M,) barrier 값 배열
        """
        dx = states[:, 0] - self.obs_x
        dy = states[:, 1] - self.obs_y
        return dx ** 2 + dy ** 2 - self.d_safe_sq

    def gradient(self, state: np.ndarray) -> np.ndarray:
        """barrier function의 상태 gradient.

        ∇h = [2(x - x_obs), 2(y - y_obs), 0, ...]

        Args:
            state: (nx,) 상태 벡터

        Returns:
            (nx,) gradient 벡터
        """
        nx = len(state)
        grad = np.zeros(nx)
        grad[0] = 2.0 * (state[0] - self.obs_x)
        grad[1] = 2.0 * (state[1] - self.obs_y)
        return grad


class BarrierFunctionSet:
    """다중 장애물 barrier 함수 집합.

    activation_distance 내의 장애물만 활성 barrier로 선택하여
    QP 제약 수를 줄이고 효율성을 높임.
    """

    def __init__(
        self,
        obstacles: Optional[np.ndarray] = None,
        robot_radius: float = 0.2,
        safety_margin: float = 0.3,
        activation_distance: float = 3.0,
    ):
        """
        Args:
            obstacles: (M, 3) 장애물 배열 [x, y, radius]
            robot_radius: 로봇 반경 [m]
            safety_margin: 추가 안전 마진 [m]
            activation_distance: 활성화 거리 [m] — 이 거리 내 장애물만 CBF 적용
        """
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.activation_distance = activation_distance
        self.barriers: List[CircleBarrier] = []

        if obstacles is not None and len(obstacles) > 0:
            self.set_obstacles(obstacles)

    def set_obstacles(self, obstacles: np.ndarray) -> None:
        """장애물 목록 갱신.

        Args:
            obstacles: (M, 3) 장애물 배열 [x, y, radius]
        """
        self.barriers = []
        for obs in obstacles:
            self.barriers.append(
                CircleBarrier(obs, self.robot_radius, self.safety_margin)
            )

    def get_active_barriers(self, state: np.ndarray) -> List[CircleBarrier]:
        """activation_distance 내의 활성 barrier 반환.

        Args:
            state: (nx,) 현재 상태 [x, y, ...]

        Returns:
            활성 CircleBarrier 리스트
        """
        active = []
        for barrier in self.barriers:
            dist = np.sqrt(
                (state[0] - barrier.obs_x) ** 2
                + (state[1] - barrier.obs_y) ** 2
            )
            if dist <= self.activation_distance:
                active.append(barrier)
        return active

    def evaluate_all(self, state: np.ndarray) -> np.ndarray:
        """모든 barrier의 h(x) 값 계산.

        Args:
            state: (nx,) 상태 벡터

        Returns:
            (M,) barrier 값 배열. 장애물이 없으면 빈 배열.
        """
        if len(self.barriers) == 0:
            return np.array([])
        return np.array([b.evaluate(state) for b in self.barriers])
