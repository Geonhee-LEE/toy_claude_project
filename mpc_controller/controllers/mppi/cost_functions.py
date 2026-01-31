"""MPPI 비용 함수 모듈 (순수 NumPy 배치 처리).

모든 비용 함수:
  입력: trajectories (K, N+1, nx), controls (K, N, nu), reference (N+1, nx)
  출력: costs (K,)
"""

from typing import List, Optional

import numpy as np

from mpc_controller.controllers.mppi.utils import normalize_angle_batch


class MPPICostFunction:
    """MPPI 비용 함수 기본 클래스."""

    def compute(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """비용 계산.

        Args:
            trajectories: (K, N+1, nx) 상태 궤적
            controls: (K, N, nu) 제어 입력
            reference: (N+1, nx) 참조 궤적

        Returns:
            (K,) 비용 배열
        """
        raise NotImplementedError


class StateTrackingCost(MPPICostFunction):
    """참조 궤적 추적 비용.

    cost_k = sum_{t=0}^{N-1} (x_t - ref_t)^T Q (x_t - ref_t)
    각도 오차는 normalize_angle_batch로 처리.
    """

    def __init__(self, Q: np.ndarray):
        """
        Args:
            Q: (nx, nx) 상태 가중 행렬
        """
        self.Q = Q
        self.q_diag = np.diag(Q)

    def compute(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        # (K, N+1, nx) - (N+1, nx) -> (K, N+1, nx)
        error = trajectories[:, :-1, :] - reference[np.newaxis, :-1, :]
        # 각도 오차 정규화
        error[:, :, 2] = normalize_angle_batch(error[:, :, 2])
        # (K, N, nx) * (nx,) -> (K, N, nx) -> sum -> (K,)
        weighted = error ** 2 * self.q_diag[np.newaxis, np.newaxis, :]
        return np.sum(weighted, axis=(1, 2))


class TerminalCost(MPPICostFunction):
    """터미널 상태 비용.

    cost_k = (x_N - ref_N)^T Qf (x_N - ref_N)
    """

    def __init__(self, Qf: np.ndarray):
        self.Qf = Qf
        self.qf_diag = np.diag(Qf)

    def compute(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        error = trajectories[:, -1, :] - reference[-1, :]
        error[:, 2] = normalize_angle_batch(error[:, 2])
        weighted = error ** 2 * self.qf_diag[np.newaxis, :]
        return np.sum(weighted, axis=1)


class ControlEffortCost(MPPICostFunction):
    """제어 입력 크기 비용.

    cost_k = sum_{t=0}^{N-1} u_t^T R u_t
    """

    def __init__(self, R: np.ndarray):
        self.R = R
        self.r_diag = np.diag(R)

    def compute(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        weighted = controls ** 2 * self.r_diag[np.newaxis, np.newaxis, :]
        return np.sum(weighted, axis=(1, 2))


class ControlRateCost(MPPICostFunction):
    """제어 입력 변화율 비용 — 부드러운 제어 유도.

    cost_k = sum_{t=0}^{N-2} (u_{t+1} - u_t)^T R_rate (u_{t+1} - u_t)

    MPC의 Rd 행렬에 대응하며, 제어 입력 진동을 억제한다.
    """

    def __init__(self, R_rate: np.ndarray):
        """
        Args:
            R_rate: (nu, nu) 또는 (nu,) 변화율 가중 행렬/벡터
        """
        self.r_diag = np.diag(R_rate) if R_rate.ndim == 2 else R_rate

    def compute(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        # du = u_{t+1} - u_t : (K, N-1, nu)
        du = controls[:, 1:, :] - controls[:, :-1, :]
        # 가중 제곱합: (K, N-1, nu) -> (K,)
        weighted = du ** 2 * self.r_diag[np.newaxis, np.newaxis, :]
        return np.sum(weighted, axis=(1, 2))


class ObstacleCost(MPPICostFunction):
    """장애물 회피 비용.

    각 장애물과의 거리가 safety_margin 이내이면 높은 비용 부과.
    cost = weight * sum_t max(0, safety_dist - dist)^2
    """

    def __init__(
        self,
        obstacles: np.ndarray,
        weight: float = 1000.0,
        safety_margin: float = 0.3,
    ):
        """
        Args:
            obstacles: (M, 3) 장애물 [x, y, radius]
            weight: 장애물 비용 가중치
            safety_margin: 안전 마진 [m]
        """
        self.obstacles = obstacles
        self.weight = weight
        self.safety_margin = safety_margin

    def compute(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        if len(self.obstacles) == 0:
            return np.zeros(trajectories.shape[0])

        K, N_plus_1, _ = trajectories.shape
        positions = trajectories[:, :, :2]  # (K, N+1, 2)
        total_cost = np.zeros(K)

        for obs in self.obstacles:
            ox, oy, radius = obs[0], obs[1], obs[2]
            safety_dist = radius + self.safety_margin

            # (K, N+1)
            dx = positions[:, :, 0] - ox
            dy = positions[:, :, 1] - oy
            dist = np.sqrt(dx ** 2 + dy ** 2)

            penetration = np.maximum(0.0, safety_dist - dist)
            total_cost += self.weight * np.sum(penetration ** 2, axis=1)

        return total_cost


class TubeAwareCost(MPPICostFunction):
    """Tube-aware 장애물 회피 비용.

    ObstacleCost의 safety_margin에 tube_margin을 추가하여
    명목 궤적이 보수적으로 계획되도록 유도한다.

    명목 궤적은 실제 상태와 tube_width만큼 편차가 있을 수 있으므로
    장애물과의 안전 거리를 tube_width만큼 확장한다.

    effective_safety = safety_margin + tube_margin
    """

    def __init__(
        self,
        obstacles: np.ndarray,
        tube_margin: float = 0.15,
        weight: float = 1000.0,
        safety_margin: float = 0.3,
    ):
        """
        Args:
            obstacles: (M, 3) 장애물 [x, y, radius]
            tube_margin: tube 폭에 의한 추가 안전 마진 [m]
            weight: 장애물 비용 가중치
            safety_margin: 기본 안전 마진 [m]
        """
        self.obstacles = obstacles
        self.tube_margin = tube_margin
        self.weight = weight
        self.safety_margin = safety_margin

    def compute(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        if len(self.obstacles) == 0:
            return np.zeros(trajectories.shape[0])

        K = trajectories.shape[0]
        positions = trajectories[:, :, :2]  # (K, N+1, 2)
        total_cost = np.zeros(K)
        effective_margin = self.safety_margin + self.tube_margin

        for obs in self.obstacles:
            ox, oy, radius = obs[0], obs[1], obs[2]
            safety_dist = radius + effective_margin

            dx = positions[:, :, 0] - ox
            dy = positions[:, :, 1] - oy
            dist = np.sqrt(dx ** 2 + dy ** 2)

            penetration = np.maximum(0.0, safety_dist - dist)
            total_cost += self.weight * np.sum(penetration ** 2, axis=1)

        return total_cost


class CompositeMPPICost:
    """여러 비용 함수를 합산하는 복합 비용."""

    def __init__(self):
        self.cost_functions: List[MPPICostFunction] = []

    def add(self, cost_fn: MPPICostFunction) -> "CompositeMPPICost":
        self.cost_functions.append(cost_fn)
        return self

    def compute(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """모든 비용 함수의 합산 비용 계산.

        Returns:
            (K,) 총 비용 배열
        """
        K = trajectories.shape[0]
        total = np.zeros(K)
        for cost_fn in self.cost_functions:
            total += cost_fn.compute(trajectories, controls, reference)
        return total
