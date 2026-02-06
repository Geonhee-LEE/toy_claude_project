"""
Realtime Replanner 단위 테스트.

실시간 경로 재계획 기능을 검증합니다.
"""

import pytest
import numpy as np
from mpc_controller.planners.realtime_replanner import (
    RealtimeReplanner,
    ReplanningConfig,
    ReplanningTrigger,
)
from mpc_controller.planners.trajectory_planner import TrajectoryPlanner
from simulation.environments import (
    EmptyEnvironment,
    CircleObstacle,
    ObstacleFieldEnvironment,
)


@pytest.fixture
def trajectory_planner() -> TrajectoryPlanner:
    """궤적 계획기 fixture."""
    return TrajectoryPlanner(dt=0.1, max_velocity=1.0, max_acceleration=0.5)


@pytest.fixture
def empty_environment() -> EmptyEnvironment:
    """빈 환경 fixture."""
    return EmptyEnvironment(bounds=(-10, 10, -10, 10))


@pytest.fixture
def obstacle_environment() -> ObstacleFieldEnvironment:
    """장애물 환경 fixture."""
    env = ObstacleFieldEnvironment(
        bounds=(-5, 5, -5, 5),
        num_obstacles=3,
        min_radius=0.3,
        max_radius=0.5,
        seed=42,
    )
    return env


@pytest.fixture
def replanner(
    trajectory_planner: TrajectoryPlanner, empty_environment: EmptyEnvironment
) -> RealtimeReplanner:
    """재계획기 fixture."""
    return RealtimeReplanner(
        trajectory_planner=trajectory_planner,
        environment=empty_environment,
        config=ReplanningConfig(),
    )


class TestReplanningConfig:
    """ReplanningConfig 테스트."""

    def test_default_config(self) -> None:
        """기본 설정값 검증."""
        config = ReplanningConfig()

        assert config.collision_check_distance == 3.0
        assert config.safety_margin == 0.5
        assert config.min_obstacle_distance == 0.8
        assert config.max_path_deviation == 1.0
        assert config.min_replanning_interval == 0.5
        assert config.lookahead_distance == 5.0
        assert config.trajectory_horizon == 5.0
        assert config.smoothing_window == 10
        assert config.blend_ratio == 0.3

    def test_custom_config(self) -> None:
        """커스텀 설정값 검증."""
        config = ReplanningConfig(
            collision_check_distance=5.0,
            safety_margin=0.8,
            min_obstacle_distance=1.0,
            max_path_deviation=0.5,
        )

        assert config.collision_check_distance == 5.0
        assert config.safety_margin == 0.8
        assert config.min_obstacle_distance == 1.0
        assert config.max_path_deviation == 0.5


class TestRealtimeReplanner:
    """RealtimeReplanner 테스트."""

    def test_initialization(self, replanner: RealtimeReplanner) -> None:
        """재계획기 초기화 검증."""
        assert replanner.current_trajectory is None
        assert replanner.goal_state is None
        assert replanner.last_replanning_time == -np.inf
        assert len(replanner.replanning_history) == 0

    def test_initialize_trajectory(self, replanner: RealtimeReplanner) -> None:
        """초기 궤적 생성 검증."""
        initial_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 5.0, np.pi / 4])

        trajectory = replanner.initialize(initial_state, goal_state)

        assert trajectory is not None
        assert len(trajectory) > 0
        assert replanner.current_trajectory is trajectory
        assert np.allclose(replanner.goal_state, goal_state)

        # 궤적의 시작점과 끝점 확인
        assert np.isclose(trajectory.x[0], initial_state[0], atol=0.1)
        assert np.isclose(trajectory.y[0], initial_state[1], atol=0.1)
        assert np.isclose(trajectory.x[-1], goal_state[0], atol=0.1)
        assert np.isclose(trajectory.y[-1], goal_state[1], atol=0.1)

    def test_no_replanning_on_good_path(self, replanner: RealtimeReplanner) -> None:
        """정상 경로에서 재계획 불필요 검증."""
        initial_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 5.0, np.pi / 4])

        replanner.initialize(initial_state, goal_state)

        # 궤적 상의 포인트에서 재계획 필요 여부 확인
        current_state = np.array([1.0, 1.0, np.pi / 4])
        current_time = 1.0

        need_replan, trigger, reason = replanner.check_replanning_needed(
            current_state, current_time
        )

        assert not need_replan
        assert trigger is None
        assert reason == ""

    def test_path_deviation_triggers_replanning(
        self, replanner: RealtimeReplanner
    ) -> None:
        """경로 이탈 시 재계획 트리거 검증."""
        initial_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 0.0, 0.0])

        replanner.initialize(initial_state, goal_state)

        # 경로에서 크게 벗어난 위치
        deviated_state = np.array([2.5, 2.0, 0.0])  # 직선 경로에서 2m 이탈
        current_time = 1.0

        need_replan, trigger, reason = replanner.check_replanning_needed(
            deviated_state, current_time
        )

        assert need_replan
        assert trigger == ReplanningTrigger.PATH_DEVIATION
        assert "deviation" in reason.lower()

    def test_new_obstacle_triggers_replanning(
        self,
        trajectory_planner: TrajectoryPlanner,
        empty_environment: EmptyEnvironment,
    ) -> None:
        """신규 장애물 출현 시 재계획 트리거 검증."""
        replanner = RealtimeReplanner(
            trajectory_planner=trajectory_planner,
            environment=empty_environment,
            config=ReplanningConfig(),
        )

        initial_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 0.0, 0.0])

        replanner.initialize(initial_state, goal_state)

        # 초기 체크 (장애물 없음)
        current_state = np.array([1.0, 0.0, 0.0])
        current_time = 1.0

        need_replan, _, _ = replanner.check_replanning_needed(
            current_state, current_time
        )
        assert not need_replan

        # 신규 장애물 추가
        empty_environment.add_obstacle(
            CircleObstacle(center=np.array([2.5, 0.0]), radius=0.5)
        )

        # 재체크 (장애물 추가 후)
        current_time = 2.0
        need_replan, trigger, reason = replanner.check_replanning_needed(
            current_state, current_time
        )

        assert need_replan
        assert trigger == ReplanningTrigger.NEW_OBSTACLE
        assert "obstacle" in reason.lower()

    def test_collision_risk_triggers_replanning(
        self, trajectory_planner: TrajectoryPlanner
    ) -> None:
        """충돌 위험 시 재계획 트리거 검증."""
        # 경로 상에 장애물이 있는 환경
        env = EmptyEnvironment(bounds=(-10, 10, -10, 10))
        env.add_obstacle(CircleObstacle(center=np.array([2.5, 0.0]), radius=0.6))

        replanner = RealtimeReplanner(
            trajectory_planner=trajectory_planner,
            environment=env,
            config=ReplanningConfig(min_obstacle_distance=0.8),
        )

        # 장애물을 통과하는 직선 경로
        initial_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 0.0, 0.0])

        replanner.initialize(initial_state, goal_state)

        # 장애물 근처에서 체크
        current_state = np.array([1.0, 0.0, 0.0])
        current_time = 1.0

        need_replan, trigger, reason = replanner.check_replanning_needed(
            current_state, current_time
        )

        assert need_replan
        assert trigger == ReplanningTrigger.COLLISION_RISK
        assert "obstacle" in reason.lower() or "close" in reason.lower()

    def test_min_replanning_interval(self, replanner: RealtimeReplanner) -> None:
        """최소 재계획 간격 검증."""
        initial_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 0.0, 0.0])

        replanner.initialize(initial_state, goal_state)

        # 첫 재계획
        deviated_state = np.array([2.5, 2.0, 0.0])
        current_time = 1.0

        need_replan1, _, _ = replanner.check_replanning_needed(
            deviated_state, current_time
        )
        assert need_replan1

        # 재계획 수행
        replanner.replan(
            deviated_state, current_time, ReplanningTrigger.PATH_DEVIATION, "test"
        )

        # 최소 간격 내에 재체크 (0.2초 후)
        current_time = 1.2
        need_replan2, _, _ = replanner.check_replanning_needed(
            deviated_state, current_time
        )

        # 최소 간격(0.5s)이 지나지 않아 재계획 불필요
        assert not need_replan2

        # 최소 간격 후 재체크 (0.6초 후)
        current_time = 1.6
        need_replan3, _, _ = replanner.check_replanning_needed(
            deviated_state, current_time
        )

        # 최소 간격이 지났으므로 재계획 필요
        assert need_replan3

    def test_replan_generates_new_trajectory(
        self, replanner: RealtimeReplanner
    ) -> None:
        """재계획 시 새 궤적 생성 검증."""
        initial_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 5.0, 0.0])

        old_trajectory = replanner.initialize(initial_state, goal_state)

        # 재계획 수행
        current_state = np.array([2.0, 2.0, 0.0])
        current_time = 1.0

        new_trajectory = replanner.replan(
            current_state,
            current_time,
            ReplanningTrigger.MANUAL,
            "test replanning",
        )

        assert new_trajectory is not None
        assert new_trajectory is not old_trajectory
        assert replanner.current_trajectory is new_trajectory
        assert len(replanner.replanning_history) == 1

        # 이벤트 검증
        event = replanner.replanning_history[0]
        assert event.trigger == ReplanningTrigger.MANUAL
        assert event.timestamp == current_time
        assert event.reason == "test replanning"
        assert event.old_trajectory is old_trajectory
        assert event.new_trajectory is new_trajectory

    def test_update_with_no_replanning(self, replanner: RealtimeReplanner) -> None:
        """정상 상태에서 update 호출 검증."""
        initial_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 5.0, 0.0])

        replanner.initialize(initial_state, goal_state)

        # 정상 경로 상의 포인트
        current_state = np.array([1.0, 1.0, np.pi / 4])
        current_time = 1.0

        new_trajectory, event = replanner.update(current_state, current_time)

        assert new_trajectory is None
        assert event is None

    def test_update_with_replanning(
        self, trajectory_planner: TrajectoryPlanner
    ) -> None:
        """재계획 발생 시 update 호출 검증."""
        # 장애물 환경
        env = EmptyEnvironment(bounds=(-10, 10, -10, 10))

        replanner = RealtimeReplanner(
            trajectory_planner=trajectory_planner,
            environment=env,
            config=ReplanningConfig(),
        )

        initial_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 0.0, 0.0])

        replanner.initialize(initial_state, goal_state)

        # 경로 이탈
        deviated_state = np.array([2.5, 2.0, 0.0])
        current_time = 1.0

        new_trajectory, event = replanner.update(deviated_state, current_time)

        assert new_trajectory is not None
        assert event is not None
        assert event.trigger == ReplanningTrigger.PATH_DEVIATION

    def test_reset(self, replanner: RealtimeReplanner) -> None:
        """재계획기 리셋 검증."""
        initial_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 5.0, 0.0])

        replanner.initialize(initial_state, goal_state)
        replanner.replan(
            np.array([1.0, 1.0, 0.0]), 1.0, ReplanningTrigger.MANUAL, "test"
        )

        # 리셋 전 상태 확인
        assert replanner.current_trajectory is not None
        assert replanner.goal_state is not None
        assert len(replanner.replanning_history) > 0

        # 리셋
        replanner.reset()

        # 리셋 후 상태 확인
        assert replanner.current_trajectory is None
        assert replanner.goal_state is None
        assert replanner.last_replanning_time == -np.inf
        assert len(replanner.replanning_history) == 0

    def test_get_current_trajectory(self, replanner: RealtimeReplanner) -> None:
        """현재 궤적 조회 검증."""
        assert replanner.get_current_trajectory() is None

        initial_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 5.0, 0.0])

        trajectory = replanner.initialize(initial_state, goal_state)

        assert replanner.get_current_trajectory() is trajectory

    def test_get_replanning_history(self, replanner: RealtimeReplanner) -> None:
        """재계획 히스토리 조회 검증."""
        assert len(replanner.get_replanning_history()) == 0

        initial_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 5.0, 0.0])

        replanner.initialize(initial_state, goal_state)

        # 첫 재계획
        replanner.replan(
            np.array([1.0, 1.0, 0.0]), 1.0, ReplanningTrigger.MANUAL, "test1"
        )

        history = replanner.get_replanning_history()
        assert len(history) == 1
        assert history[0].reason == "test1"

        # 두 번째 재계획
        replanner.replan(
            np.array([2.0, 2.0, 0.0]), 2.0, ReplanningTrigger.MANUAL, "test2"
        )

        history = replanner.get_replanning_history()
        assert len(history) == 2
        assert history[1].reason == "test2"


class TestWaypointGeneration:
    """웨이포인트 생성 테스트."""

    def test_waypoint_generation_without_obstacles(
        self, replanner: RealtimeReplanner
    ) -> None:
        """장애물이 없을 때 웨이포인트 생성 검증."""
        start_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 5.0, 0.0])

        waypoints = replanner._generate_waypoints_avoiding_obstacles(
            start_state, goal_state
        )

        # 시작점과 목표점만 있어야 함
        assert len(waypoints) == 2
        assert np.isclose(waypoints[0].x, start_state[0])
        assert np.isclose(waypoints[0].y, start_state[1])
        assert np.isclose(waypoints[-1].x, goal_state[0])
        assert np.isclose(waypoints[-1].y, goal_state[1])

    def test_waypoint_generation_with_obstacles(
        self, trajectory_planner: TrajectoryPlanner
    ) -> None:
        """장애물이 있을 때 웨이포인트 생성 검증."""
        # 경로 상에 장애물 배치
        env = EmptyEnvironment(bounds=(-10, 10, -10, 10))
        env.add_obstacle(CircleObstacle(center=np.array([2.5, 2.5]), radius=0.5))

        replanner = RealtimeReplanner(
            trajectory_planner=trajectory_planner,
            environment=env,
            config=ReplanningConfig(min_obstacle_distance=0.8),
        )

        start_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 5.0, 0.0])

        waypoints = replanner._generate_waypoints_avoiding_obstacles(
            start_state, goal_state
        )

        # 우회 포인트가 추가되어야 함
        assert len(waypoints) > 2
