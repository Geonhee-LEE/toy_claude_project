#!/usr/bin/env python3
"""
Realtime Replanning Demo: 실시간 경로 재계획 시연

환경 변화에 대응하여 실시간으로 경로를 재계획하는 데모입니다.

시나리오:
1. 로봇이 초기 경로를 따라 이동
2. 중간에 신규 장애물 출현
3. 충돌 위험 감지 → 자동 경로 재계획
4. 재계획된 경로를 따라 목표 도달

┌──────────────────────────────────────────────────────┐
│            실시간 경로 재계획 흐름도                  │
└──────────────────────────────────────────────────────┘

  초기 상태 (0, 0)
      │
      ▼
┌─────────────────┐
│ 초기 경로 계획  │ ─────► 직선 경로 (0,0) → (10,10)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 로봇 이동 시작  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 환경 변화 감지  │ ◀──── 신규 장애물 출현 (5,5)
│ - 장애물 추가   │
│ - 충돌 위험     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 재계획 트리거   │ ─────► 충돌 위험 OR 경로 이탈
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 경로 재계획     │ ─────► 장애물 회피 경로 생성
│ - 장애물 회피   │
│ - 부드러운 연결 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 새 경로 추종    │
└────────┬────────┘
         │
         ▼
    목표 도달 (10,10)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle
from matplotlib.animation import FuncAnimation
import time

from mpc_controller.planners.trajectory_planner import TrajectoryPlanner
from mpc_controller.planners.realtime_replanner import (
    RealtimeReplanner,
    ReplanningConfig,
    ReplanningTrigger,
)
from simulation.environments import EmptyEnvironment, CircleObstacle


def create_dynamic_scenario():
    """
    동적 장애물 시나리오 생성.

    Returns:
        (초기 상태, 목표 상태, 환경, 장애물 추가 시점 리스트)
    """
    initial_state = np.array([0.0, 0.0, np.pi / 4])
    goal_state = np.array([10.0, 10.0, np.pi / 4])

    # 빈 환경에서 시작
    environment = EmptyEnvironment(bounds=(-2, 12, -2, 12))

    # 장애물 추가 이벤트 (시간, 장애물)
    obstacle_events = [
        (2.0, CircleObstacle(center=np.array([5.0, 5.0]), radius=0.8)),
        (5.0, CircleObstacle(center=np.array([7.0, 7.0]), radius=0.6)),
        (8.0, CircleObstacle(center=np.array([3.0, 8.0]), radius=0.5)),
    ]

    return initial_state, goal_state, environment, obstacle_events


def simulate_realtime_replanning(
    initial_state,
    goal_state,
    environment,
    obstacle_events,
    dt=0.1,
    max_time=20.0,
):
    """
    실시간 재계획 시뮬레이션 실행.

    Args:
        initial_state: 초기 상태
        goal_state: 목표 상태
        environment: 환경 객체
        obstacle_events: 장애물 추가 이벤트 리스트
        dt: 시간 간격
        max_time: 최대 시뮬레이션 시간

    Returns:
        시뮬레이션 결과 딕셔너리
    """
    # 궤적 계획기 및 재계획기 초기화
    planner = TrajectoryPlanner(dt=dt, max_velocity=1.5, max_acceleration=0.8)

    config = ReplanningConfig(
        collision_check_distance=4.0,
        min_obstacle_distance=1.2,
        max_path_deviation=1.5,
        min_replanning_interval=0.5,
        lookahead_distance=6.0,
        smoothing_window=15,
        blend_ratio=0.4,
    )

    replanner = RealtimeReplanner(
        trajectory_planner=planner, environment=environment, config=config
    )

    # 초기 궤적 생성
    initial_trajectory = replanner.initialize(initial_state, goal_state)

    # 시뮬레이션 변수
    current_state = initial_state.copy()
    current_time = 0.0
    trajectory_history = [initial_trajectory]
    state_history = [current_state.copy()]
    replanning_events = []

    # 장애물 이벤트 큐
    obstacle_queue = sorted(obstacle_events, key=lambda x: x[0])
    next_obstacle_idx = 0

    print("\n" + "=" * 70)
    print("실시간 경로 재계획 시뮬레이션 시작")
    print("=" * 70)
    print(f"초기 상태: ({initial_state[0]:.1f}, {initial_state[1]:.1f})")
    print(f"목표 상태: ({goal_state[0]:.1f}, {goal_state[1]:.1f})")
    print(f"초기 궤적 길이: {len(initial_trajectory)} 포인트")
    print("-" * 70)

    # 시뮬레이션 루프
    while current_time < max_time:
        # 장애물 추가 이벤트 처리
        while (
            next_obstacle_idx < len(obstacle_queue)
            and current_time >= obstacle_queue[next_obstacle_idx][0]
        ):
            _, obstacle = obstacle_queue[next_obstacle_idx]
            environment.add_obstacle(obstacle)
            print(
                f"\n[{current_time:.1f}s] 신규 장애물 출현! "
                f"위치: ({obstacle.center[0]:.1f}, {obstacle.center[1]:.1f}), "
                f"반경: {obstacle.radius:.1f}m"
            )
            next_obstacle_idx += 1

        # 재계획 필요 여부 확인 및 실행
        new_trajectory, event = replanner.update(current_state, current_time)

        if event is not None:
            print(
                f"\n[{current_time:.1f}s] 재계획 발생!"
                f"\n  트리거: {event.trigger.value}"
                f"\n  사유: {event.reason}"
            )
            trajectory_history.append(new_trajectory)
            replanning_events.append(event)

        # 현재 궤적을 따라 이동 (간단한 경로 추종)
        current_trajectory = replanner.get_current_trajectory()
        if current_trajectory is not None:
            # 현재 위치에서 가장 가까운 궤적 포인트 찾기
            closest_idx, _ = planner.find_closest_point(
                current_trajectory, current_state[0], current_state[1]
            )

            # 다음 목표 포인트 (lookahead)
            lookahead_idx = min(closest_idx + 5, len(current_trajectory) - 1)

            target_x = current_trajectory.x[lookahead_idx]
            target_y = current_trajectory.y[lookahead_idx]

            # 목표 방향으로 이동
            dx = target_x - current_state[0]
            dy = target_y - current_state[1]
            distance = np.sqrt(dx**2 + dy**2)

            if distance > 0.01:
                # 속도 제어 (간단한 비례 제어)
                velocity = min(1.5, distance * 2.0)
                current_state[0] += (dx / distance) * velocity * dt
                current_state[1] += (dy / distance) * velocity * dt
                current_state[2] = np.arctan2(dy, dx)

        state_history.append(current_state.copy())
        current_time += dt

        # 목표 도달 확인
        dist_to_goal = np.linalg.norm(current_state[:2] - goal_state[:2])
        if dist_to_goal < 0.3:
            print(
                f"\n[{current_time:.1f}s] 목표 도달!"
                f" (최종 거리: {dist_to_goal:.2f}m)"
            )
            break

    print("\n" + "=" * 70)
    print("시뮬레이션 완료")
    print(f"총 시뮬레이션 시간: {current_time:.2f}s")
    print(f"재계획 발생 횟수: {len(replanning_events)}회")
    print("=" * 70)

    return {
        "states": np.array(state_history),
        "trajectories": trajectory_history,
        "replanning_events": replanning_events,
        "environment": environment,
        "initial_state": initial_state,
        "goal_state": goal_state,
        "total_time": current_time,
    }


def plot_simulation_result(result):
    """
    시뮬레이션 결과 시각화.

    Args:
        result: 시뮬레이션 결과 딕셔너리
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. 궤적 및 재계획 이력
    ax = axes[0]

    # 환경 장애물
    for obs in result["environment"].obstacles:
        circle = Circle(
            obs.center, obs.radius, color="red", alpha=0.3, label="Obstacle"
        )
        ax.add_patch(circle)

    # 초기 궤적
    initial_traj = result["trajectories"][0]
    ax.plot(
        initial_traj.x,
        initial_traj.y,
        "k--",
        linewidth=1.5,
        alpha=0.4,
        label="Initial Trajectory",
    )

    # 재계획된 궤적들
    colors = plt.cm.rainbow(np.linspace(0, 1, len(result["trajectories"])))
    for i, traj in enumerate(result["trajectories"][1:], start=1):
        ax.plot(
            traj.x,
            traj.y,
            "--",
            linewidth=1.0,
            alpha=0.5,
            color=colors[i],
            label=f"Replanned {i}",
        )

    # 실제 주행 경로
    states = result["states"]
    ax.plot(
        states[:, 0],
        states[:, 1],
        "b-",
        linewidth=2.5,
        label="Actual Path",
        zorder=10,
    )

    # 재계획 발생 지점 표시
    for i, event in enumerate(result["replanning_events"]):
        # 이벤트 발생 시점의 상태 찾기
        event_idx = int(event.timestamp / 0.1)
        if event_idx < len(states):
            ax.plot(
                states[event_idx, 0],
                states[event_idx, 1],
                "r*",
                markersize=15,
                markeredgewidth=1.5,
                markeredgecolor="darkred",
                zorder=20,
            )
            ax.text(
                states[event_idx, 0] + 0.3,
                states[event_idx, 1] + 0.3,
                f"R{i+1}",
                fontsize=10,
                color="darkred",
                fontweight="bold",
            )

    # 시작점과 목표점
    ax.plot(
        result["initial_state"][0],
        result["initial_state"][1],
        "go",
        markersize=15,
        label="Start",
        zorder=15,
    )
    ax.plot(
        result["goal_state"][0],
        result["goal_state"][1],
        "r^",
        markersize=15,
        label="Goal",
        zorder=15,
    )

    ax.set_xlabel("X [m]", fontsize=12)
    ax.set_ylabel("Y [m]", fontsize=12)
    ax.set_title(
        "Realtime Replanning Trajectories", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # 2. 재계획 이벤트 타임라인
    ax = axes[1]

    if result["replanning_events"]:
        event_times = [e.timestamp for e in result["replanning_events"]]
        event_triggers = [e.trigger.value for e in result["replanning_events"]]

        # 트리거 타입별 색상
        trigger_colors = {
            "collision_risk": "red",
            "path_deviation": "orange",
            "new_obstacle": "purple",
            "manual": "blue",
        }

        for i, (t, trigger) in enumerate(zip(event_times, event_triggers)):
            color = trigger_colors.get(trigger, "gray")
            ax.barh(
                i,
                0.5,
                left=t,
                height=0.8,
                color=color,
                alpha=0.7,
                edgecolor="black",
            )
            ax.text(
                t + 0.6,
                i,
                f"{trigger}\n({t:.1f}s)",
                fontsize=9,
                va="center",
            )

        ax.set_yticks(range(len(event_times)))
        ax.set_yticklabels([f"Event {i+1}" for i in range(len(event_times))])
        ax.set_xlabel("Time [s]", fontsize=12)
        ax.set_title("Replanning Events Timeline", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_xlim(0, result["total_time"])
    else:
        ax.text(
            0.5,
            0.5,
            "No Replanning Events",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("realtime_replanning_demo.png", dpi=150, bbox_inches="tight")
    print("\n그림 저장 완료: realtime_replanning_demo.png")


def print_summary(result):
    """시뮬레이션 요약 출력."""
    print("\n" + "=" * 70)
    print("시뮬레이션 요약")
    print("=" * 70)

    print(f"\n[경로 정보]")
    print(f"  초기 궤적 길이: {len(result['trajectories'][0])} 포인트")
    print(f"  최종 궤적 길이: {len(result['trajectories'][-1])} 포인트")
    print(f"  실제 이동 거리: {len(result['states']) * 0.1:.2f}m (근사)")

    print(f"\n[재계획 정보]")
    print(f"  재계획 발생 횟수: {len(result['replanning_events'])}회")

    for i, event in enumerate(result["replanning_events"], 1):
        print(
            f"  {i}. [{event.timestamp:.1f}s] {event.trigger.value}: {event.reason[:50]}..."
        )

    print(f"\n[환경 정보]")
    print(f"  장애물 개수: {len(result['environment'].obstacles)}개")

    print(f"\n[성능 지표]")
    final_dist = np.linalg.norm(
        result["states"][-1][:2] - result["goal_state"][:2]
    )
    print(f"  최종 목표 거리: {final_dist:.3f}m")
    print(f"  총 시뮬레이션 시간: {result['total_time']:.2f}s")

    print("=" * 70)


def main():
    """메인 함수."""
    print(
        """
╔═══════════════════════════════════════════════════════════════════════╗
║         Realtime Replanning Demo: 실시간 경로 재계획 시연            ║
╠═══════════════════════════════════════════════════════════════════════╣
║  환경 변화에 대응하여 실시간으로 경로를 재계획합니다.                ║
║                                                                       ║
║  시나리오:                                                            ║
║    1. 초기 직선 경로 계획                                             ║
║    2. t=2s: 첫 번째 장애물 출현 → 재계획                             ║
║    3. t=5s: 두 번째 장애물 출현 → 재계획                             ║
║    4. t=8s: 세 번째 장애물 출현 → 재계획                             ║
║    5. 목표 도달                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
        """
    )

    # 시나리오 생성
    initial_state, goal_state, environment, obstacle_events = (
        create_dynamic_scenario()
    )

    # 시뮬레이션 실행
    result = simulate_realtime_replanning(
        initial_state=initial_state,
        goal_state=goal_state,
        environment=environment,
        obstacle_events=obstacle_events,
        dt=0.1,
        max_time=20.0,
    )

    # 결과 요약
    print_summary(result)

    # 시각화
    plot_simulation_result(result)
    plt.show()

    print("\n데모 완료!")


if __name__ == "__main__":
    main()
