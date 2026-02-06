"""
Realtime Replanner Module.

실시간 환경 변화에 대응하여 경로를 재계획하는 모듈입니다.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from enum import Enum
import numpy as np
from scipy.interpolate import CubicSpline

from mpc_controller.planners.trajectory_planner import (
    Trajectory,
    TrajectoryPlanner,
    Waypoint,
)
from simulation.environments import Environment, Obstacle


class ReplanningTrigger(Enum):
    """재계획 트리거 타입."""

    COLLISION_RISK = "collision_risk"  # 충돌 위험 감지
    PATH_DEVIATION = "path_deviation"  # 경로 이탈
    NEW_OBSTACLE = "new_obstacle"  # 신규 장애물 출현
    GOAL_CHANGE = "goal_change"  # 목표점 변경
    MANUAL = "manual"  # 수동 트리거


@dataclass
class ReplanningConfig:
    """재계획 설정 파라미터."""

    # 충돌 위험 임계값
    collision_check_distance: float = 3.0  # 충돌 체크 거리 [m]
    safety_margin: float = 0.5  # 안전 마진 [m]
    min_obstacle_distance: float = 0.8  # 최소 장애물 거리 [m]

    # 경로 이탈 임계값
    max_path_deviation: float = 1.0  # 최대 경로 이탈 거리 [m]

    # 재계획 주기 제한
    min_replanning_interval: float = 0.5  # 최소 재계획 간격 [s]

    # 궤적 생성 파라미터
    lookahead_distance: float = 5.0  # 전방 주시 거리 [m]
    trajectory_horizon: float = 5.0  # 궤적 시간 범위 [s]

    # 부드러운 연결 파라미터
    smoothing_window: int = 10  # 부드러운 연결을 위한 윈도우 크기
    blend_ratio: float = 0.3  # 기존 궤적과의 블렌딩 비율 (0~1)


@dataclass
class ReplanningEvent:
    """재계획 이벤트 정보."""

    trigger: ReplanningTrigger  # 트리거 타입
    timestamp: float  # 이벤트 발생 시각 [s]
    reason: str  # 재계획 사유
    old_trajectory: Optional[Trajectory] = None  # 이전 궤적
    new_trajectory: Optional[Trajectory] = None  # 새 궤적


class RealtimeReplanner:
    """
    실시간 경로 재계획 클래스.

    환경 변화를 감지하고 필요시 경로를 재계획합니다.
    """

    def __init__(
        self,
        trajectory_planner: TrajectoryPlanner,
        environment: Environment,
        config: Optional[ReplanningConfig] = None,
    ):
        """
        실시간 재계획기 초기화.

        Args:
            trajectory_planner: 궤적 계획기
            environment: 환경 객체
            config: 재계획 설정
        """
        self.planner = trajectory_planner
        self.environment = environment
        self.config = config or ReplanningConfig()

        # 상태 변수
        self.current_trajectory: Optional[Trajectory] = None
        self.goal_state: Optional[np.ndarray] = None
        self.last_replanning_time: float = -np.inf
        self.replanning_history: List[ReplanningEvent] = []

        # 환경 변화 추적
        self.previous_obstacle_count: int = 0
        self.previous_obstacles_hash: Optional[int] = None

    def initialize(
        self,
        initial_state: np.ndarray,
        goal_state: np.ndarray,
    ) -> Trajectory:
        """
        초기 궤적을 생성합니다.

        Args:
            initial_state: 초기 상태 [x, y, theta]
            goal_state: 목표 상태 [x, y, theta]

        Returns:
            초기 궤적
        """
        self.goal_state = goal_state

        # 웨이포인트 기반 초기 궤적 생성
        waypoints = [
            Waypoint(x=initial_state[0], y=initial_state[1], theta=initial_state[2]),
            Waypoint(x=goal_state[0], y=goal_state[1], theta=goal_state[2]),
        ]

        self.current_trajectory = self.planner.plan_from_waypoints(
            waypoints, interpolation_method="cubic"
        )

        return self.current_trajectory

    def check_replanning_needed(
        self,
        current_state: np.ndarray,
        current_time: float,
    ) -> Tuple[bool, Optional[ReplanningTrigger], str]:
        """
        재계획 필요 여부를 확인합니다.

        Args:
            current_state: 현재 상태 [x, y, theta]
            current_time: 현재 시각 [s]

        Returns:
            (재계획 필요 여부, 트리거 타입, 사유)
        """
        # 최소 재계획 간격 체크
        if (
            current_time - self.last_replanning_time
            < self.config.min_replanning_interval
        ):
            return False, None, ""

        # 1. 충돌 위험 체크
        collision_risk, collision_reason = self._check_collision_risk(current_state)
        if collision_risk:
            return True, ReplanningTrigger.COLLISION_RISK, collision_reason

        # 2. 경로 이탈 체크
        path_deviation, deviation_reason = self._check_path_deviation(current_state)
        if path_deviation:
            return True, ReplanningTrigger.PATH_DEVIATION, deviation_reason

        # 3. 신규 장애물 체크
        new_obstacle, obstacle_reason = self._check_new_obstacles()
        if new_obstacle:
            return True, ReplanningTrigger.NEW_OBSTACLE, obstacle_reason

        return False, None, ""

    def _check_collision_risk(
        self, current_state: np.ndarray
    ) -> Tuple[bool, str]:
        """
        충돌 위험을 체크합니다.

        Args:
            current_state: 현재 상태 [x, y, theta]

        Returns:
            (충돌 위험 여부, 사유)
        """
        if self.current_trajectory is None:
            return False, ""

        # 현재 위치에서 전방 lookahead_distance 구간의 궤적 체크
        closest_idx, _ = self.planner.find_closest_point(
            self.current_trajectory, current_state[0], current_state[1]
        )

        # 전방 구간 추출
        lookahead_points = int(
            self.config.lookahead_distance / self.planner.dt / self.planner.max_velocity
        )
        end_idx = min(closest_idx + lookahead_points, len(self.current_trajectory))

        # 각 포인트에서 장애물까지의 거리 체크
        for i in range(closest_idx, end_idx):
            traj_point = np.array([self.current_trajectory.x[i], self.current_trajectory.y[i], 0])
            min_dist = self.environment.min_obstacle_distance(traj_point)

            if min_dist < self.config.min_obstacle_distance:
                return (
                    True,
                    f"Trajectory point at ({self.current_trajectory.x[i]:.2f}, "
                    f"{self.current_trajectory.y[i]:.2f}) too close to obstacle "
                    f"(distance: {min_dist:.2f}m < {self.config.min_obstacle_distance}m)",
                )

        return False, ""

    def _check_path_deviation(
        self, current_state: np.ndarray
    ) -> Tuple[bool, str]:
        """
        경로 이탈을 체크합니다.

        Args:
            current_state: 현재 상태 [x, y, theta]

        Returns:
            (경로 이탈 여부, 사유)
        """
        if self.current_trajectory is None:
            return False, ""

        _, distance = self.planner.find_closest_point(
            self.current_trajectory, current_state[0], current_state[1]
        )

        if distance > self.config.max_path_deviation:
            return (
                True,
                f"Path deviation too large: {distance:.2f}m > "
                f"{self.config.max_path_deviation}m",
            )

        return False, ""

    def _check_new_obstacles(self) -> Tuple[bool, str]:
        """
        신규 장애물 출현을 체크합니다.

        Returns:
            (신규 장애물 여부, 사유)
        """
        current_obstacle_count = len(self.environment.obstacles)

        # 장애물 개수 변화 감지
        if current_obstacle_count != self.previous_obstacle_count:
            reason = (
                f"Obstacle count changed: {self.previous_obstacle_count} -> "
                f"{current_obstacle_count}"
            )
            self.previous_obstacle_count = current_obstacle_count
            return True, reason

        # 장애물 상태 변화 감지 (해시 비교)
        current_hash = self._compute_obstacles_hash()
        if (
            self.previous_obstacles_hash is not None
            and current_hash != self.previous_obstacles_hash
        ):
            self.previous_obstacles_hash = current_hash
            return True, "Obstacle configuration changed"

        self.previous_obstacles_hash = current_hash
        return False, ""

    def _compute_obstacles_hash(self) -> int:
        """
        현재 장애물 상태의 해시값을 계산합니다.

        Returns:
            해시값
        """
        # 간단한 해시: 각 장애물의 중심 좌표를 문자열로 변환 후 해시
        obstacle_str = ""
        for obs in self.environment.obstacles:
            # CircleObstacle의 경우
            if hasattr(obs, "center"):
                obstacle_str += f"{obs.center[0]:.2f},{obs.center[1]:.2f};"
            # RectangleObstacle의 경우
            elif hasattr(obs, "center"):
                obstacle_str += f"{obs.center[0]:.2f},{obs.center[1]:.2f};"

        return hash(obstacle_str)

    def replan(
        self,
        current_state: np.ndarray,
        current_time: float,
        trigger: ReplanningTrigger,
        reason: str,
    ) -> Trajectory:
        """
        경로를 재계획합니다.

        Args:
            current_state: 현재 상태 [x, y, theta]
            current_time: 현재 시각 [s]
            trigger: 재계획 트리거
            reason: 재계획 사유

        Returns:
            새로운 궤적
        """
        if self.goal_state is None:
            raise ValueError("Goal state not set. Call initialize() first.")

        # 이전 궤적 저장
        old_trajectory = self.current_trajectory

        # 간단한 재계획: 현재 위치에서 목표까지 직선 경로
        # TODO: 더 정교한 경로 계획 알고리즘 (RRT*, A* 등) 통합 가능
        waypoints = self._generate_waypoints_avoiding_obstacles(
            current_state, self.goal_state
        )

        # 새 궤적 생성
        new_trajectory = self.planner.plan_from_waypoints(
            waypoints, interpolation_method="cubic"
        )

        # 기존 궤적과 부드럽게 연결
        if old_trajectory is not None and self.config.blend_ratio > 0:
            new_trajectory = self._smooth_transition(
                old_trajectory, new_trajectory, current_state, current_time
            )

        # 상태 업데이트
        self.current_trajectory = new_trajectory
        self.last_replanning_time = current_time

        # 이벤트 기록
        event = ReplanningEvent(
            trigger=trigger,
            timestamp=current_time,
            reason=reason,
            old_trajectory=old_trajectory,
            new_trajectory=new_trajectory,
        )
        self.replanning_history.append(event)

        return new_trajectory

    def _generate_waypoints_avoiding_obstacles(
        self,
        start_state: np.ndarray,
        goal_state: np.ndarray,
    ) -> List[Waypoint]:
        """
        장애물을 회피하는 웨이포인트를 생성합니다.

        간단한 구현: 직선 경로 상의 장애물을 우회하는 웨이포인트 추가

        Args:
            start_state: 시작 상태 [x, y, theta]
            goal_state: 목표 상태 [x, y, theta]

        Returns:
            웨이포인트 리스트
        """
        waypoints = [
            Waypoint(x=start_state[0], y=start_state[1], theta=start_state[2])
        ]

        # 직선 경로 상의 장애물 체크 및 우회 포인트 생성
        num_check_points = 20
        for i in range(1, num_check_points):
            t = i / num_check_points
            x = start_state[0] + t * (goal_state[0] - start_state[0])
            y = start_state[1] + t * (goal_state[1] - start_state[1])

            point = np.array([x, y, 0])
            min_dist = self.environment.min_obstacle_distance(point)

            # 충돌 위험이 있으면 우회 포인트 추가
            if min_dist < self.config.min_obstacle_distance:
                # 수직 방향으로 우회
                dx = goal_state[0] - start_state[0]
                dy = goal_state[1] - start_state[1]
                norm = np.sqrt(dx**2 + dy**2)
                if norm > 0:
                    # 좌측 우회 시도
                    detour_x = x + (-dy / norm) * (self.config.min_obstacle_distance + 0.5)
                    detour_y = y + (dx / norm) * (self.config.min_obstacle_distance + 0.5)

                    detour_point = np.array([detour_x, detour_y, 0])
                    detour_dist = self.environment.min_obstacle_distance(detour_point)

                    # 좌측 우회가 안전하면 추가
                    if detour_dist >= self.config.min_obstacle_distance:
                        waypoints.append(Waypoint(x=detour_x, y=detour_y))
                    else:
                        # 우측 우회 시도
                        detour_x = x - (-dy / norm) * (self.config.min_obstacle_distance + 0.5)
                        detour_y = y - (dx / norm) * (self.config.min_obstacle_distance + 0.5)
                        waypoints.append(Waypoint(x=detour_x, y=detour_y))

        # 목표점 추가
        waypoints.append(
            Waypoint(x=goal_state[0], y=goal_state[1], theta=goal_state[2])
        )

        return waypoints

    def _smooth_transition(
        self,
        old_trajectory: Trajectory,
        new_trajectory: Trajectory,
        current_state: np.ndarray,
        current_time: float,
    ) -> Trajectory:
        """
        이전 궤적과 새 궤적을 부드럽게 연결합니다.

        Args:
            old_trajectory: 이전 궤적
            new_trajectory: 새 궤적
            current_state: 현재 상태
            current_time: 현재 시각

        Returns:
            부드럽게 연결된 궤적
        """
        # 블렌딩 윈도우 크기
        window = self.config.smoothing_window

        # 새 궤적이 충분히 길지 않으면 그대로 반환
        if len(new_trajectory) <= window:
            return new_trajectory

        # 블렌딩: 처음 window 포인트를 선형 보간
        blend_ratio = self.config.blend_ratio
        for i in range(min(window, len(new_trajectory), len(old_trajectory))):
            alpha = (1 - blend_ratio) * (i / window)  # 0 -> (1-blend_ratio)

            # 이전 궤적의 해당 포인트 찾기
            if i < len(old_trajectory):
                new_trajectory.x[i] = (
                    alpha * new_trajectory.x[i] + (1 - alpha) * old_trajectory.x[i]
                )
                new_trajectory.y[i] = (
                    alpha * new_trajectory.y[i] + (1 - alpha) * old_trajectory.y[i]
                )

        return new_trajectory

    def update(
        self,
        current_state: np.ndarray,
        current_time: float,
    ) -> Tuple[Optional[Trajectory], Optional[ReplanningEvent]]:
        """
        상태를 업데이트하고 필요시 재계획을 수행합니다.

        Args:
            current_state: 현재 상태 [x, y, theta]
            current_time: 현재 시각 [s]

        Returns:
            (업데이트된 궤적 (재계획 발생시만), 재계획 이벤트 (발생시만))
        """
        # 재계획 필요 여부 확인
        need_replan, trigger, reason = self.check_replanning_needed(
            current_state, current_time
        )

        if need_replan and trigger is not None:
            # 재계획 수행
            new_trajectory = self.replan(current_state, current_time, trigger, reason)
            event = self.replanning_history[-1]  # 가장 최근 이벤트
            return new_trajectory, event

        return None, None

    def get_current_trajectory(self) -> Optional[Trajectory]:
        """
        현재 궤적을 반환합니다.

        Returns:
            현재 궤적
        """
        return self.current_trajectory

    def get_replanning_history(self) -> List[ReplanningEvent]:
        """
        재계획 히스토리를 반환합니다.

        Returns:
            재계획 이벤트 리스트
        """
        return self.replanning_history

    def reset(self) -> None:
        """재계획기를 리셋합니다."""
        self.current_trajectory = None
        self.goal_state = None
        self.last_replanning_time = -np.inf
        self.replanning_history.clear()
        self.previous_obstacle_count = 0
        self.previous_obstacles_hash = None
