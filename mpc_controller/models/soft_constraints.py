"""
Soft Constraints Module for MPC.

MPC에서 사용하는 소프트 제약조건을 정의합니다.
하드 제약조건과 달리 위반 시 패널티 비용을 부여하여
feasibility를 보장합니다.

┌─────────────────────────────────────────────────────────────┐
│                    Soft Constraint Flow                      │
├─────────────────────────────────────────────────────────────┤
│  Constraint: g(x) <= 0                                       │
│       │                                                      │
│       ▼                                                      │
│  Add slack variable: g(x) <= s,  s >= 0                     │
│       │                                                      │
│       ▼                                                      │
│  Add penalty to cost: J += w * s^p                          │
│       │                                                      │
│       ▼                                                      │
│  Optimize: min J  s.t. g(x) <= s, s >= 0                    │
└─────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import casadi as ca
import numpy as np


class ConstraintType(Enum):
    """제약조건 유형."""

    VELOCITY = "velocity"           # 속도 제약
    ACCELERATION = "acceleration"   # 가속도 제약
    POSITION = "position"           # 위치 제약 (경계)
    OBSTACLE = "obstacle"           # 장애물 회피
    CUSTOM = "custom"               # 사용자 정의


class PenaltyType(Enum):
    """패널티 함수 유형."""

    QUADRATIC = "quadratic"   # L2 norm: w * s^2
    LINEAR = "linear"         # L1 norm: w * s
    HUBER = "huber"          # Huber loss: smooth L1


@dataclass
class SoftConstraintParams:
    """소프트 제약조건 매개변수."""

    # 기본 가중치
    velocity_weight: float = 100.0      # 속도 제약 위반 패널티
    acceleration_weight: float = 50.0    # 가속도 제약 위반 패널티
    position_weight: float = 200.0       # 위치 제약 위반 패널티
    obstacle_weight: float = 500.0       # 장애물 제약 위반 패널티

    # 패널티 유형
    penalty_type: PenaltyType = PenaltyType.QUADRATIC

    # Huber loss delta (penalty_type이 HUBER일 때 사용)
    huber_delta: float = 1.0

    # 소프트 제약조건 활성화 여부
    enabled: bool = True

    # 제약조건별 활성화
    enable_velocity_soft: bool = True
    enable_acceleration_soft: bool = True
    enable_position_soft: bool = False
    enable_obstacle_soft: bool = True


@dataclass
class ConstraintViolation:
    """제약조건 위반 정보."""

    constraint_type: ConstraintType
    violation_amount: float  # 위반량 (양수: 위반, 0: 만족)
    slack_value: float       # slack variable 값
    penalty_cost: float      # 패널티 비용
    timestep: int            # 위반 발생 시점

    @property
    def is_violated(self) -> bool:
        """제약조건 위반 여부."""
        return self.violation_amount > 1e-6


@dataclass
class SoftConstraintResult:
    """소프트 제약조건 최적화 결과."""

    total_penalty: float = 0.0
    violations: List[ConstraintViolation] = field(default_factory=list)
    slack_values: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def has_violations(self) -> bool:
        """위반 존재 여부."""
        return any(v.is_violated for v in self.violations)

    @property
    def max_violation(self) -> float:
        """최대 위반량."""
        if not self.violations:
            return 0.0
        return max(v.violation_amount for v in self.violations)

    def get_violations_by_type(
        self, constraint_type: ConstraintType
    ) -> List[ConstraintViolation]:
        """유형별 위반 목록."""
        return [v for v in self.violations if v.constraint_type == constraint_type]


class SoftConstraint:
    """
    소프트 제약조건 클래스.

    하드 제약조건을 소프트 제약조건으로 변환하여
    최적화 문제의 feasibility를 보장합니다.
    """

    def __init__(
        self,
        constraint_type: ConstraintType,
        weight: float,
        penalty_type: PenaltyType = PenaltyType.QUADRATIC,
        huber_delta: float = 1.0,
        name: Optional[str] = None,
    ):
        """
        Initialize soft constraint.

        Args:
            constraint_type: 제약조건 유형
            weight: 패널티 가중치
            penalty_type: 패널티 함수 유형
            huber_delta: Huber loss delta 값
            name: 제약조건 이름 (디버깅용)
        """
        self.constraint_type = constraint_type
        self.weight = weight
        self.penalty_type = penalty_type
        self.huber_delta = huber_delta
        self.name = name or f"soft_{constraint_type.value}"

    def compute_penalty(self, slack: ca.SX) -> ca.SX:
        """
        Slack variable에 대한 패널티 비용 계산.

        Args:
            slack: Slack variable (s >= 0)

        Returns:
            패널티 비용
        """
        if self.penalty_type == PenaltyType.QUADRATIC:
            return self.weight * slack ** 2
        elif self.penalty_type == PenaltyType.LINEAR:
            return self.weight * slack
        elif self.penalty_type == PenaltyType.HUBER:
            # Huber loss: smooth approximation of L1
            delta = self.huber_delta
            return self.weight * ca.if_else(
                slack <= delta,
                0.5 * slack ** 2,
                delta * (slack - 0.5 * delta)
            )
        else:
            return self.weight * slack ** 2

    def create_slack_variable(
        self, dim: int, horizon: int
    ) -> Tuple[ca.SX, ca.SX, ca.SX]:
        """
        Slack variable 생성.

        Args:
            dim: 제약조건 차원
            horizon: 예측 horizon

        Returns:
            (slack variables, lower bounds, upper bounds)
        """
        slack = ca.SX.sym(f"slack_{self.name}", dim, horizon)
        lb = ca.DM.zeros(dim, horizon)  # s >= 0
        ub = ca.DM.inf(dim, horizon)    # s <= inf
        return slack, lb, ub


class VelocitySoftConstraint(SoftConstraint):
    """속도 소프트 제약조건."""

    def __init__(
        self,
        v_max: float,
        omega_max: float,
        weight: float = 100.0,
        penalty_type: PenaltyType = PenaltyType.QUADRATIC,
    ):
        """
        Initialize velocity soft constraint.

        Args:
            v_max: 최대 선속도 [m/s]
            omega_max: 최대 각속도 [rad/s]
            weight: 패널티 가중치
            penalty_type: 패널티 유형
        """
        super().__init__(
            ConstraintType.VELOCITY,
            weight,
            penalty_type,
            name="velocity"
        )
        self.v_max = v_max
        self.omega_max = omega_max

    def apply(
        self,
        v: ca.SX,
        omega: ca.SX,
        slack_v: ca.SX,
        slack_omega: ca.SX,
    ) -> Tuple[ca.SX, ca.SX, ca.SX]:
        """
        속도 소프트 제약조건 적용.

        Args:
            v: 선속도
            omega: 각속도
            slack_v: 선속도 slack
            slack_omega: 각속도 slack

        Returns:
            (constraint expressions, lower bounds, upper bounds)
        """
        # |v| <= v_max + slack_v
        # |omega| <= omega_max + slack_omega
        constraints = ca.vertcat(
            v - self.v_max - slack_v,
            -v - self.v_max - slack_v,
            omega - self.omega_max - slack_omega,
            -omega - self.omega_max - slack_omega,
        )
        lb = ca.DM([-ca.inf, -ca.inf, -ca.inf, -ca.inf])
        ub = ca.DM([0, 0, 0, 0])

        return constraints, lb, ub

    def get_penalty(self, slack_v: ca.SX, slack_omega: ca.SX) -> ca.SX:
        """패널티 비용 계산."""
        return self.compute_penalty(slack_v) + self.compute_penalty(slack_omega)


class AccelerationSoftConstraint(SoftConstraint):
    """가속도 소프트 제약조건."""

    def __init__(
        self,
        a_max: float,
        alpha_max: float,
        dt: float,
        weight: float = 50.0,
        penalty_type: PenaltyType = PenaltyType.QUADRATIC,
    ):
        """
        Initialize acceleration soft constraint.

        Args:
            a_max: 최대 선가속도 [m/s^2]
            alpha_max: 최대 각가속도 [rad/s^2]
            dt: 시간 간격 [s]
            weight: 패널티 가중치
            penalty_type: 패널티 유형
        """
        super().__init__(
            ConstraintType.ACCELERATION,
            weight,
            penalty_type,
            name="acceleration"
        )
        self.a_max = a_max
        self.alpha_max = alpha_max
        self.dt = dt

    def apply(
        self,
        v_curr: ca.SX,
        v_prev: ca.SX,
        omega_curr: ca.SX,
        omega_prev: ca.SX,
        slack_a: ca.SX,
        slack_alpha: ca.SX,
    ) -> Tuple[ca.SX, ca.SX, ca.SX]:
        """
        가속도 소프트 제약조건 적용.

        Args:
            v_curr, v_prev: 현재/이전 선속도
            omega_curr, omega_prev: 현재/이전 각속도
            slack_a: 선가속도 slack
            slack_alpha: 각가속도 slack

        Returns:
            (constraint expressions, lower bounds, upper bounds)
        """
        # 가속도 계산
        a = (v_curr - v_prev) / self.dt
        alpha = (omega_curr - omega_prev) / self.dt

        # |a| <= a_max + slack_a
        # |alpha| <= alpha_max + slack_alpha
        constraints = ca.vertcat(
            a - self.a_max - slack_a,
            -a - self.a_max - slack_a,
            alpha - self.alpha_max - slack_alpha,
            -alpha - self.alpha_max - slack_alpha,
        )
        lb = ca.DM([-ca.inf, -ca.inf, -ca.inf, -ca.inf])
        ub = ca.DM([0, 0, 0, 0])

        return constraints, lb, ub

    def get_penalty(self, slack_a: ca.SX, slack_alpha: ca.SX) -> ca.SX:
        """패널티 비용 계산."""
        return self.compute_penalty(slack_a) + self.compute_penalty(slack_alpha)


class ObstacleSoftConstraint(SoftConstraint):
    """장애물 회피 소프트 제약조건."""

    def __init__(
        self,
        safety_margin: float = 0.3,
        weight: float = 500.0,
        penalty_type: PenaltyType = PenaltyType.QUADRATIC,
    ):
        """
        Initialize obstacle soft constraint.

        Args:
            safety_margin: 안전 여유 거리 [m]
            weight: 패널티 가중치
            penalty_type: 패널티 유형
        """
        super().__init__(
            ConstraintType.OBSTACLE,
            weight,
            penalty_type,
            name="obstacle"
        )
        self.safety_margin = safety_margin

    def apply(
        self,
        x: ca.SX,
        y: ca.SX,
        obs_x: float,
        obs_y: float,
        obs_radius: float,
        slack: ca.SX,
    ) -> Tuple[ca.SX, ca.SX, ca.SX]:
        """
        장애물 회피 소프트 제약조건 적용.

        dist(robot, obstacle) >= obs_radius + safety_margin - slack

        Args:
            x, y: 로봇 위치
            obs_x, obs_y: 장애물 중심
            obs_radius: 장애물 반경
            slack: Slack variable

        Returns:
            (constraint expressions, lower bounds, upper bounds)
        """
        # 거리 계산
        dist = ca.sqrt((x - obs_x) ** 2 + (y - obs_y) ** 2)
        min_dist = obs_radius + self.safety_margin

        # dist >= min_dist - slack
        # => min_dist - dist <= slack
        # => min_dist - dist - slack <= 0
        constraint = min_dist - dist - slack
        lb = ca.DM([-ca.inf])
        ub = ca.DM([0])

        return constraint, lb, ub

    def get_penalty(self, slack: ca.SX) -> ca.SX:
        """패널티 비용 계산."""
        return self.compute_penalty(slack)


class PositionSoftConstraint(SoftConstraint):
    """위치 경계 소프트 제약조건."""

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        weight: float = 200.0,
        penalty_type: PenaltyType = PenaltyType.QUADRATIC,
    ):
        """
        Initialize position soft constraint.

        Args:
            x_min, x_max: X 좌표 범위
            y_min, y_max: Y 좌표 범위
            weight: 패널티 가중치
            penalty_type: 패널티 유형
        """
        super().__init__(
            ConstraintType.POSITION,
            weight,
            penalty_type,
            name="position"
        )
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def apply(
        self,
        x: ca.SX,
        y: ca.SX,
        slack_x: ca.SX,
        slack_y: ca.SX,
    ) -> Tuple[ca.SX, ca.SX, ca.SX]:
        """
        위치 경계 소프트 제약조건 적용.

        Args:
            x, y: 로봇 위치
            slack_x, slack_y: Slack variables

        Returns:
            (constraint expressions, lower bounds, upper bounds)
        """
        # x_min - slack <= x <= x_max + slack
        # y_min - slack <= y <= y_max + slack
        constraints = ca.vertcat(
            self.x_min - slack_x - x,  # x >= x_min - slack
            x - self.x_max - slack_x,   # x <= x_max + slack
            self.y_min - slack_y - y,  # y >= y_min - slack
            y - self.y_max - slack_y,   # y <= y_max + slack
        )
        lb = ca.DM([-ca.inf, -ca.inf, -ca.inf, -ca.inf])
        ub = ca.DM([0, 0, 0, 0])

        return constraints, lb, ub

    def get_penalty(self, slack_x: ca.SX, slack_y: ca.SX) -> ca.SX:
        """패널티 비용 계산."""
        return self.compute_penalty(slack_x) + self.compute_penalty(slack_y)


class SoftConstraintManager:
    """
    소프트 제약조건 관리자.

    여러 소프트 제약조건을 관리하고 MPC 문제에 통합합니다.
    """

    def __init__(self, params: Optional[SoftConstraintParams] = None):
        """
        Initialize soft constraint manager.

        Args:
            params: 소프트 제약조건 매개변수
        """
        self.params = params or SoftConstraintParams()
        self.constraints: List[SoftConstraint] = []
        self._result: Optional[SoftConstraintResult] = None

    @property
    def enabled(self) -> bool:
        """소프트 제약조건 활성화 여부."""
        return self.params.enabled

    @property
    def result(self) -> Optional[SoftConstraintResult]:
        """최근 최적화 결과."""
        return self._result

    def add_constraint(self, constraint: SoftConstraint) -> "SoftConstraintManager":
        """
        소프트 제약조건 추가.

        Args:
            constraint: 추가할 제약조건

        Returns:
            Self for method chaining
        """
        self.constraints.append(constraint)
        return self

    def add_velocity_constraint(
        self,
        v_max: float,
        omega_max: float,
        weight: Optional[float] = None,
    ) -> "SoftConstraintManager":
        """속도 소프트 제약조건 추가."""
        if not self.params.enable_velocity_soft:
            return self

        constraint = VelocitySoftConstraint(
            v_max=v_max,
            omega_max=omega_max,
            weight=weight or self.params.velocity_weight,
            penalty_type=self.params.penalty_type,
        )
        return self.add_constraint(constraint)

    def add_acceleration_constraint(
        self,
        a_max: float,
        alpha_max: float,
        dt: float,
        weight: Optional[float] = None,
    ) -> "SoftConstraintManager":
        """가속도 소프트 제약조건 추가."""
        if not self.params.enable_acceleration_soft:
            return self

        constraint = AccelerationSoftConstraint(
            a_max=a_max,
            alpha_max=alpha_max,
            dt=dt,
            weight=weight or self.params.acceleration_weight,
            penalty_type=self.params.penalty_type,
        )
        return self.add_constraint(constraint)

    def add_obstacle_constraint(
        self,
        safety_margin: float = 0.3,
        weight: Optional[float] = None,
    ) -> "SoftConstraintManager":
        """장애물 소프트 제약조건 추가."""
        if not self.params.enable_obstacle_soft:
            return self

        constraint = ObstacleSoftConstraint(
            safety_margin=safety_margin,
            weight=weight or self.params.obstacle_weight,
            penalty_type=self.params.penalty_type,
        )
        return self.add_constraint(constraint)

    def add_position_constraint(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        weight: Optional[float] = None,
    ) -> "SoftConstraintManager":
        """위치 경계 소프트 제약조건 추가."""
        if not self.params.enable_position_soft:
            return self

        constraint = PositionSoftConstraint(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            weight=weight or self.params.position_weight,
            penalty_type=self.params.penalty_type,
        )
        return self.add_constraint(constraint)

    def get_total_slack_dim(self, horizon: int) -> int:
        """
        전체 slack variable 차원 계산.

        Args:
            horizon: 예측 horizon

        Returns:
            Total slack variable dimension
        """
        total_dim = 0
        for constraint in self.constraints:
            if isinstance(constraint, VelocitySoftConstraint):
                total_dim += 2 * horizon  # v, omega
            elif isinstance(constraint, AccelerationSoftConstraint):
                total_dim += 2 * (horizon - 1)  # a, alpha (k>0)
            elif isinstance(constraint, ObstacleSoftConstraint):
                total_dim += horizon  # one per timestep
            elif isinstance(constraint, PositionSoftConstraint):
                total_dim += 2 * horizon  # x, y
        return total_dim

    def process_solution(
        self,
        slack_values: np.ndarray,
        horizon: int,
    ) -> SoftConstraintResult:
        """
        최적화 결과에서 제약조건 위반 정보 추출.

        Args:
            slack_values: Slack variable 최적해
            horizon: 예측 horizon

        Returns:
            소프트 제약조건 결과
        """
        result = SoftConstraintResult()
        idx = 0

        for constraint in self.constraints:
            if isinstance(constraint, VelocitySoftConstraint):
                n = 2 * horizon
                slacks = slack_values[idx:idx + n]
                result.slack_values[constraint.name] = slacks

                # 위반 분석
                for k in range(horizon):
                    sv = slacks[k]  # velocity slack
                    so = slacks[horizon + k]  # omega slack

                    if sv > 1e-6:
                        result.violations.append(ConstraintViolation(
                            constraint_type=ConstraintType.VELOCITY,
                            violation_amount=sv,
                            slack_value=sv,
                            penalty_cost=constraint.weight * sv ** 2,
                            timestep=k,
                        ))
                    if so > 1e-6:
                        result.violations.append(ConstraintViolation(
                            constraint_type=ConstraintType.VELOCITY,
                            violation_amount=so,
                            slack_value=so,
                            penalty_cost=constraint.weight * so ** 2,
                            timestep=k,
                        ))

                idx += n

            elif isinstance(constraint, AccelerationSoftConstraint):
                n = 2 * (horizon - 1)
                slacks = slack_values[idx:idx + n]
                result.slack_values[constraint.name] = slacks
                idx += n

            elif isinstance(constraint, ObstacleSoftConstraint):
                n = horizon
                slacks = slack_values[idx:idx + n]
                result.slack_values[constraint.name] = slacks

                for k in range(horizon):
                    s = slacks[k]
                    if s > 1e-6:
                        result.violations.append(ConstraintViolation(
                            constraint_type=ConstraintType.OBSTACLE,
                            violation_amount=s,
                            slack_value=s,
                            penalty_cost=constraint.weight * s ** 2,
                            timestep=k,
                        ))

                idx += n

            elif isinstance(constraint, PositionSoftConstraint):
                n = 2 * horizon
                slacks = slack_values[idx:idx + n]
                result.slack_values[constraint.name] = slacks
                idx += n

        # 총 패널티 계산
        result.total_penalty = sum(v.penalty_cost for v in result.violations)

        self._result = result
        return result

    def get_visualization_data(self) -> Dict:
        """
        RVIZ 시각화를 위한 데이터 반환.

        Returns:
            시각화 데이터 딕셔너리
        """
        if self._result is None:
            return {}

        return {
            "has_violations": self._result.has_violations,
            "max_violation": self._result.max_violation,
            "total_penalty": self._result.total_penalty,
            "violations": [
                {
                    "type": v.constraint_type.value,
                    "amount": v.violation_amount,
                    "timestep": v.timestep,
                }
                for v in self._result.violations
            ],
        }
