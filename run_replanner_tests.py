#!/usr/bin/env python3
"""실시간 재계획 테스트 실행 스크립트."""

import sys
sys.path.insert(0, '.')

from tests.planners.test_realtime_replanner import *

if __name__ == "__main__":
    # 테스트 인스턴스 생성
    planner = TrajectoryPlanner(dt=0.1, max_velocity=1.0, max_acceleration=0.5)
    env = EmptyEnvironment(bounds=(-10, 10, -10, 10))
    replanner = RealtimeReplanner(
        trajectory_planner=planner,
        environment=env,
        config=ReplanningConfig(),
    )

    print("=" * 70)
    print("실시간 재계획 테스트 실행")
    print("=" * 70)

    # 테스트 1: 초기화
    print("\n[TEST 1] 재계획기 초기화 검증")
    try:
        assert replanner.current_trajectory is None
        assert replanner.goal_state is None
        assert replanner.last_replanning_time == -np.inf
        assert len(replanner.replanning_history) == 0
        print("✓ PASS: 초기화 성공")
    except AssertionError as e:
        print(f"✗ FAIL: {e}")

    # 테스트 2: 초기 궤적 생성
    print("\n[TEST 2] 초기 궤적 생성 검증")
    try:
        initial_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 5.0, np.pi / 4])

        trajectory = replanner.initialize(initial_state, goal_state)

        assert trajectory is not None
        assert len(trajectory) > 0
        assert replanner.current_trajectory is trajectory
        assert np.allclose(replanner.goal_state, goal_state)
        assert np.isclose(trajectory.x[0], initial_state[0], atol=0.1)
        assert np.isclose(trajectory.y[0], initial_state[1], atol=0.1)
        assert np.isclose(trajectory.x[-1], goal_state[0], atol=0.1)
        assert np.isclose(trajectory.y[-1], goal_state[1], atol=0.1)
        print(f"✓ PASS: 궤적 생성 성공 (길이: {len(trajectory)})")
    except AssertionError as e:
        print(f"✗ FAIL: {e}")

    # 테스트 3: 정상 경로에서 재계획 불필요
    print("\n[TEST 3] 정상 경로에서 재계획 불필요 검증")
    try:
        current_state = np.array([1.0, 1.0, np.pi / 4])
        current_time = 1.0

        need_replan, trigger, reason = replanner.check_replanning_needed(
            current_state, current_time
        )

        assert not need_replan
        assert trigger is None
        assert reason == ""
        print("✓ PASS: 재계획 불필요 확인")
    except AssertionError as e:
        print(f"✗ FAIL: {e}")

    # 테스트 4: 경로 이탈 시 재계획 트리거
    print("\n[TEST 4] 경로 이탈 시 재계획 트리거 검증")
    try:
        replanner.reset()
        initial_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 0.0, 0.0])
        replanner.initialize(initial_state, goal_state)

        deviated_state = np.array([2.5, 2.0, 0.0])
        current_time = 1.0

        need_replan, trigger, reason = replanner.check_replanning_needed(
            deviated_state, current_time
        )

        assert need_replan
        assert trigger == ReplanningTrigger.PATH_DEVIATION
        assert "deviation" in reason.lower()
        print(f"✓ PASS: 경로 이탈 감지 (사유: {reason})")
    except AssertionError as e:
        print(f"✗ FAIL: {e}")

    # 테스트 5: 충돌 위험 시 재계획 트리거
    print("\n[TEST 5] 충돌 위험 시 재계획 트리거 검증")
    try:
        env2 = EmptyEnvironment(bounds=(-10, 10, -10, 10))
        env2.add_obstacle(CircleObstacle(center=np.array([2.5, 0.0]), radius=0.6))

        replanner2 = RealtimeReplanner(
            trajectory_planner=planner,
            environment=env2,
            config=ReplanningConfig(min_obstacle_distance=0.8),
        )

        initial_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 0.0, 0.0])
        replanner2.initialize(initial_state, goal_state)

        current_state = np.array([1.0, 0.0, 0.0])
        current_time = 1.0

        need_replan, trigger, reason = replanner2.check_replanning_needed(
            current_state, current_time
        )

        assert need_replan
        assert trigger == ReplanningTrigger.COLLISION_RISK
        print(f"✓ PASS: 충돌 위험 감지 (사유: {reason[:50]}...)")
    except AssertionError as e:
        print(f"✗ FAIL: {e}")

    # 테스트 6: 재계획 수행
    print("\n[TEST 6] 재계획 수행 검증")
    try:
        replanner.reset()
        initial_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 5.0, 0.0])

        old_trajectory = replanner.initialize(initial_state, goal_state)
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

        event = replanner.replanning_history[0]
        assert event.trigger == ReplanningTrigger.MANUAL
        assert event.timestamp == current_time
        assert event.reason == "test replanning"
        print(f"✓ PASS: 재계획 성공 (새 궤적 길이: {len(new_trajectory)})")
    except AssertionError as e:
        print(f"✗ FAIL: {e}")

    # 테스트 7: update 메서드
    print("\n[TEST 7] update 메서드 검증")
    try:
        replanner.reset()
        initial_state = np.array([0.0, 0.0, 0.0])
        goal_state = np.array([5.0, 0.0, 0.0])
        replanner.initialize(initial_state, goal_state)

        # 정상 상태
        current_state = np.array([1.0, 0.0, 0.0])
        current_time = 1.0
        new_traj, event = replanner.update(current_state, current_time)
        assert new_traj is None
        assert event is None

        # 경로 이탈
        deviated_state = np.array([2.5, 2.0, 0.0])
        current_time = 2.0
        new_traj, event = replanner.update(deviated_state, current_time)
        assert new_traj is not None
        assert event is not None
        assert event.trigger == ReplanningTrigger.PATH_DEVIATION
        print(f"✓ PASS: update 메서드 정상 동작")
    except AssertionError as e:
        print(f"✗ FAIL: {e}")

    # 테스트 8: reset
    print("\n[TEST 8] reset 메서드 검증")
    try:
        assert replanner.current_trajectory is not None
        assert replanner.goal_state is not None
        assert len(replanner.replanning_history) > 0

        replanner.reset()

        assert replanner.current_trajectory is None
        assert replanner.goal_state is None
        assert replanner.last_replanning_time == -np.inf
        assert len(replanner.replanning_history) == 0
        print("✓ PASS: reset 성공")
    except AssertionError as e:
        print(f"✗ FAIL: {e}")

    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)
