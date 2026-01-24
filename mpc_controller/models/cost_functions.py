"""
MPC Cost Functions Module.

MPC 최적화를 위한 다양한 비용 함수를 정의합니다.
"""

from typing import Optional
import casadi as ca
import numpy as np


class CostFunction:
    """Base class for MPC cost functions."""

    def __init__(self, weight: float = 1.0):
        """
        Initialize cost function.

        Args:
            weight: Cost weight factor
        """
        self.weight = weight

    def compute(self, *args, **kwargs) -> ca.SX:
        """Compute the cost value."""
        raise NotImplementedError


class PositionCost(CostFunction):
    """
    위치 추적 비용 함수.

    로봇의 현재 위치와 목표 위치 간의 오차를 계산합니다.
    """

    def compute(
        self,
        x: ca.SX,
        y: ca.SX,
        x_ref: ca.SX,
        y_ref: ca.SX,
    ) -> ca.SX:
        """
        Compute position tracking cost.

        Args:
            x, y: Current position
            x_ref, y_ref: Reference position

        Returns:
            Position tracking cost
        """
        return self.weight * ((x - x_ref) ** 2 + (y - y_ref) ** 2)


class OrientationCost(CostFunction):
    """
    방향 추적 비용 함수.

    각도 차이를 [-pi, pi] 범위로 정규화하여 계산합니다.
    """

    def compute(
        self,
        theta: ca.SX,
        theta_ref: ca.SX,
    ) -> ca.SX:
        """
        Compute orientation tracking cost.

        Args:
            theta: Current orientation
            theta_ref: Reference orientation

        Returns:
            Orientation tracking cost
        """
        # 각도 차이 정규화
        angle_diff = theta - theta_ref
        angle_diff = ca.atan2(ca.sin(angle_diff), ca.cos(angle_diff))
        return self.weight * (angle_diff ** 2)


class ControlEffortCost(CostFunction):
    """
    제어 입력 비용 함수.

    제어 입력의 크기를 최소화하여 에너지 효율을 높입니다.
    """

    def compute(
        self,
        v: ca.SX,
        omega: ca.SX,
        v_weight: float = 1.0,
        omega_weight: float = 1.0,
    ) -> ca.SX:
        """
        Compute control effort cost.

        Args:
            v: Linear velocity
            omega: Angular velocity
            v_weight: Weight for linear velocity
            omega_weight: Weight for angular velocity

        Returns:
            Control effort cost
        """
        return self.weight * (v_weight * v ** 2 + omega_weight * omega ** 2)


class ControlSmoothnessCost(CostFunction):
    """
    제어 입력 변화율 비용 함수.

    제어 입력의 급격한 변화를 방지하여 부드러운 움직임을 유도합니다.
    """

    def compute(
        self,
        u_current: ca.SX,
        u_previous: ca.SX,
    ) -> ca.SX:
        """
        Compute control smoothness cost.

        Args:
            u_current: Current control input
            u_previous: Previous control input

        Returns:
            Control smoothness cost
        """
        delta_u = u_current - u_previous
        return self.weight * ca.sumsqr(delta_u)


class ObstacleAvoidanceCost(CostFunction):
    """
    장애물 회피 비용 함수.

    장애물과의 거리에 따른 페널티를 부여합니다.
    """

    def __init__(
        self,
        weight: float = 1.0,
        safety_margin: float = 0.3,
        influence_distance: float = 2.0,
    ):
        """
        Initialize obstacle avoidance cost.

        Args:
            weight: Cost weight
            safety_margin: Minimum safe distance from obstacle
            influence_distance: Distance at which obstacle starts affecting cost
        """
        super().__init__(weight)
        self.safety_margin = safety_margin
        self.influence_distance = influence_distance

    def compute(
        self,
        x: ca.SX,
        y: ca.SX,
        obs_x: float,
        obs_y: float,
        obs_radius: float,
    ) -> ca.SX:
        """
        Compute obstacle avoidance cost.

        Args:
            x, y: Robot position
            obs_x, obs_y: Obstacle center position
            obs_radius: Obstacle radius

        Returns:
            Obstacle avoidance cost
        """
        # 장애물까지의 거리
        dist = ca.sqrt((x - obs_x) ** 2 + (y - obs_y) ** 2)
        safe_dist = obs_radius + self.safety_margin

        # 영향 범위 내에서만 비용 적용
        cost = ca.if_else(
            dist < self.influence_distance,
            self.weight / (dist - safe_dist + 0.01) ** 2,
            0.0,
        )
        return cost


class TerminalCost(CostFunction):
    """
    터미널 비용 함수.

    예측 horizon 끝에서의 상태 오차에 대한 비용입니다.
    """

    def compute(
        self,
        state: ca.SX,
        ref_state: ca.SX,
        Q_terminal: np.ndarray,
    ) -> ca.SX:
        """
        Compute terminal cost.

        Args:
            state: Final predicted state
            ref_state: Reference final state
            Q_terminal: Terminal cost weight matrix

        Returns:
            Terminal cost
        """
        error = state - ref_state
        return self.weight * ca.mtimes([error.T, Q_terminal, error])


class CompositeCost:
    """
    복합 비용 함수.

    여러 비용 함수를 조합하여 총 비용을 계산합니다.
    """

    def __init__(self):
        """Initialize composite cost."""
        self.costs: list[CostFunction] = []

    def add_cost(self, cost: CostFunction) -> "CompositeCost":
        """
        Add a cost function to the composite.

        Args:
            cost: Cost function to add

        Returns:
            Self for method chaining
        """
        self.costs.append(cost)
        return self

    def compute_total(self, *args, **kwargs) -> ca.SX:
        """
        Compute total cost from all components.

        Returns:
            Total cost value
        """
        total = 0.0
        for cost in self.costs:
            total += cost.compute(*args, **kwargs)
        return total
