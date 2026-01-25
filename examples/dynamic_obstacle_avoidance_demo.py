#!/usr/bin/env python3
"""
Dynamic Obstacle Avoidance Demo: 동적 장애물 회피 시뮬레이션

MPC를 사용하여 움직이는 장애물을 피해 목표점까지 이동하는 시나리오입니다.

┌─────────────────────────────────────────────────────────────────────┐
│                      시나리오 개요                                   │
├─────────────────────────────────────────────────────────────────────┤
│ 1. 시작점: (0, 0)                                                   │
│ 2. 목표점: (8, 8)                                                   │
│ 3. 정적 장애물: 1-2개 고정 장애물                                    │
│ 4. 동적 장애물: 2-3개 움직이는 장애물                                │
│ 5. MPC가 동적 예측을 통해 충돌 회피                                  │
└─────────────────────────────────────────────────────────────────────┘

작동 흐름:
┌──────────────┐
│   초기 상태  │
└──────┬───────┘
       │
       ▼
┌──────────────┐      ┌───────────────────┐
│  MPC 제어    │─────▶│ 장애물 감지       │
└──────┬───────┘      └────────┬──────────┘
       │                       │
       │   ┌───────────────────┘
       │   │ 동적 장애물 예측 (미래 위치)
       ▼   ▼
┌──────────────┐
│ 회피 경로    │
│ 생성         │
└──────┬───────┘
       │
       ▼
┌──────────────┐      ┌───────────────────┐
│ 상태 업데이트│─────▶│ 장애물 상태 업데이트│
└──────┬───────┘      └────────┬──────────┘
       │                       │
       │                       │
       │   목표 도달?          │
       └─────▶ Yes: 종료      │
              No: 반복◀────────┘
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

from mpc_controller.controllers.mpc import MPCController, MPCParams
from mpc_controller.models.differential_drive import RobotParams
from mpc_controller.models.soft_constraints import (
    SoftConstraintParams,
    PenaltyType,
)
from mpc_controller.planners.obstacle_avoidance import Obstacle, ObstacleAvoidance
from mpc_controller.planners.dynamic_obstacle_predictor import PredictionModel


@dataclass
class SimulationResult:
    """시뮬레이션 결과 데이터."""

    states: np.ndarray  # 상태 궤적 [N x 3]
    controls: np.ndarray  # 제어 입력 [N x 2]
    obstacle_states: List[List[Tuple[float, float]]]  # 장애물 궤적
    costs: List[float]  # 각 단계별 비용
    solve_times: List[float]  # 각 단계별 솔버 시간
    total_time: float  # 전체 소요 시간
    success: bool  # 목표 도달 여부
    collision_detected: bool  # 충돌 발생 여부
    final_distance_to_goal: float  # 최종 목표까지의 거리
    collision_risks: List[List[Tuple[int, Optional[float]]]]  # 충돌 위험 기록


def create_dynamic_obstacle_scenario() -> Tuple[
    np.ndarray, np.ndarray, List[Obstacle]
]:
    """
    동적 장애물 회피 시나리오 생성.

    Returns:
        (초기 상태, 목표 상태, 장애물 리스트)
    """
    # 시작점과 목표점
    initial_state = np.array([0.0, 0.0, np.pi / 4])  # (x, y, theta)
    goal_state = np.array([8.0, 8.0, np.pi / 4])

    # 장애물 배치
    obstacles = [
        # 정적 장애물 1개
        Obstacle(
            x=4.0,
            y=3.0,
            radius=0.5,
            velocity_x=0.0,
            velocity_y=0.0,
            obstacle_type="static",
        ),
        # 동적 장애물 1: 오른쪽으로 이동
        Obstacle(
            x=2.0,
            y=4.0,
            radius=0.4,
            velocity_x=0.5,  # 0.5 m/s
            velocity_y=0.0,
            obstacle_type="dynamic",
        ),
        # 동적 장애물 2: 위로 이동
        Obstacle(
            x=6.0,
            y=2.0,
            radius=0.4,
            velocity_x=0.0,
            velocity_y=0.6,  # 0.6 m/s
            obstacle_type="dynamic",
        ),
        # 동적 장애물 3: 대각선으로 이동
        Obstacle(
            x=5.0,
            y=6.0,
            radius=0.3,
            velocity_x=-0.3,
            velocity_y=-0.3,
            obstacle_type="dynamic",
        ),
    ]

    return initial_state, goal_state, obstacles


def generate_reference_trajectory(
    start: np.ndarray,
    goal: np.ndarray,
    n_points: int = 150,
) -> np.ndarray:
    """
    시작점에서 목표점까지의 직선 참조 궤적 생성.

    Args:
        start: 시작 상태 [x, y, theta]
        goal: 목표 상태 [x, y, theta]
        n_points: 궤적 포인트 수

    Returns:
        참조 궤적 [n_points x 3]
    """
    # 직선 보간
    x = np.linspace(start[0], goal[0], n_points)
    y = np.linspace(start[1], goal[1], n_points)

    # 방향각 계산
    theta = np.zeros(n_points)
    for i in range(n_points):
        if i < n_points - 1:
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            theta[i] = np.arctan2(dy, dx)
        else:
            theta[i] = theta[i - 1]

    return np.column_stack([x, y, theta])


def update_dynamic_obstacles(
    obstacles: List[Obstacle],
    dt: float,
) -> None:
    """
    동적 장애물의 위치를 업데이트합니다.

    Args:
        obstacles: 장애물 리스트
        dt: 시간 간격
    """
    for obs in obstacles:
        if obs.obstacle_type == "dynamic":
            obs.x += obs.velocity_x * dt
            obs.y += obs.velocity_y * dt


def run_dynamic_obstacle_avoidance_simulation(
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    obstacles: List[Obstacle],
    prediction_model: PredictionModel = PredictionModel.CONSTANT_VELOCITY,
    dt: float = 0.1,
    max_steps: int = 300,
) -> SimulationResult:
    """
    동적 장애물 회피 시뮬레이션 실행.

    Args:
        initial_state: 초기 상태 [x, y, theta]
        goal_state: 목표 상태 [x, y, theta]
        obstacles: 장애물 리스트 (동적 장애물 포함)
        prediction_model: 동적 장애물 예측 모델
        dt: 시간 간격 [s]
        max_steps: 최대 스텝 수

    Returns:
        시뮬레이션 결과
    """
    # 참조 궤적 생성 (직선)
    reference = generate_reference_trajectory(initial_state, goal_state)

    # 로봇 파라미터
    robot_params = RobotParams(
        max_velocity=1.0,
        max_omega=2.0,
    )

    # MPC 파라미터 (동적 장애물 회피를 위한 설정)
    soft_params = SoftConstraintParams(
        velocity_weight=50.0,
        acceleration_weight=30.0,
        obstacle_weight=800.0,  # 동적 장애물을 위해 더 높은 가중치
        penalty_type=PenaltyType.QUADRATIC,
        enable_velocity_soft=True,
        enable_acceleration_soft=True,
        enable_obstacle_soft=True,
    )

    mpc_params = MPCParams(
        N=25,  # 더 긴 예측 horizon (동적 장애물 예측을 위해)
        dt=dt,
        a_max=0.8,
        alpha_max=1.5,
        soft_constraints=soft_params,
    )

    controller = MPCController(
        robot_params=robot_params,
        mpc_params=mpc_params,
        enable_soft_constraints=True,
    )

    # 장애물 회피 모듈 초기화
    obstacle_avoidance = ObstacleAvoidance(
        safety_margin=0.4,
        detection_range=6.0,
        prediction_horizon=3.0,
        prediction_model=prediction_model,
    )

    # 시뮬레이션 변수
    states = [initial_state.copy()]
    controls = []
    costs = []
    solve_times = []
    collision_detected = False
    obstacle_states_history = [[] for _ in obstacles]
    collision_risks_history = []

    state = initial_state.copy()
    controller.reset()

    start_time = time.perf_counter()

    # 시뮬레이션 루프
    for step in range(max_steps):
        # 장애물 상태 업데이트
        obstacle_avoidance.update_obstacles(obstacles)

        # 장애물 위치 기록
        for i, obs in enumerate(obstacles):
            obstacle_states_history[i].append((obs.x, obs.y))

        # 동적 장애물 충돌 위험 평가
        collision_risks = obstacle_avoidance.check_dynamic_collision_risk(
            state[0], state[1], robot_radius=0.3
        )
        collision_risks_history.append(
            [(obstacles.index(obs), t) for obs, t in collision_risks]
        )

        # 현재 참조 궤적 추출
        start_idx = min(step, len(reference) - controller.N - 1)
        ref_window = reference[start_idx : start_idx + controller.N + 1]

        if len(ref_window) < controller.N + 1:
            # 끝에 도달하면 목표점으로 패딩
            padding = np.tile(goal_state, (controller.N + 1 - len(ref_window), 1))
            ref_window = np.vstack([ref_window, padding])

        try:
            # MPC 제어 입력 계산
            control, info = controller.compute_control(state, ref_window)

            costs.append(info.get("cost", 0))
            solve_times.append(info.get("solve_time", 0))

            # 상태 업데이트 (간단한 오일러 적분)
            x, y, theta = state
            v, omega = control

            next_state = np.array(
                [
                    x + v * np.cos(theta) * dt,
                    y + v * np.sin(theta) * dt,
                    theta + omega * dt,
                ]
            )

            # 충돌 체크
            is_collision, _ = obstacle_avoidance.check_collision(
                next_state[0], next_state[1], robot_radius=0.3
            )
            if is_collision:
                collision_detected = True
                print(
                    f"  [WARNING] 충돌 감지! Step {step}, 위치: ({next_state[0]:.2f}, {next_state[1]:.2f})"
                )

            state = next_state
            states.append(state.copy())
            controls.append(control.copy())

            # 동적 장애물 위치 업데이트
            update_dynamic_obstacles(obstacles, dt)

            # 목표 도달 확인
            dist_to_goal = np.linalg.norm(state[:2] - goal_state[:2])
            if dist_to_goal < 0.2:
                print(f"  [SUCCESS] 목표 도달! Step {step}")
                break

        except Exception as e:
            print(f"  [ERROR] Step {step} 실패: {e}")
            break

    total_time = time.perf_counter() - start_time

    # 최종 결과
    states_array = np.array(states)
    controls_array = np.array(controls) if controls else np.array([]).reshape(0, 2)
    final_distance = np.linalg.norm(state[:2] - goal_state[:2])
    success = final_distance < 0.2

    return SimulationResult(
        states=states_array,
        controls=controls_array,
        obstacle_states=obstacle_states_history,
        costs=costs,
        solve_times=solve_times,
        total_time=total_time,
        success=success,
        collision_detected=collision_detected,
        final_distance_to_goal=final_distance,
        collision_risks=collision_risks_history,
    )


def plot_dynamic_obstacle_avoidance(
    result: SimulationResult,
    obstacles_initial: List[Obstacle],
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    reference: np.ndarray,
    save_path: str = None,
) -> plt.Figure:
    """
    동적 장애물 회피 시뮬레이션 결과 시각화.

    Args:
        result: 시뮬레이션 결과
        obstacles_initial: 초기 장애물 상태
        initial_state: 초기 상태
        goal_state: 목표 상태
        reference: 참조 궤적
        save_path: 저장 경로

    Returns:
        Figure 객체
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. 궤적 및 장애물 경로
    ax = axes[0, 0]

    # 참조 궤적
    ax.plot(
        reference[:, 0],
        reference[:, 1],
        "k--",
        linewidth=1.5,
        label="Reference",
        alpha=0.5,
    )

    # 로봇 실제 궤적
    ax.plot(
        result.states[:, 0],
        result.states[:, 1],
        "b-",
        linewidth=2.5,
        label="Robot Trajectory",
    )

    # 동적 장애물 궤적 그리기
    for i, (obs, obs_traj) in enumerate(zip(obstacles_initial, result.obstacle_states)):
        if obs.obstacle_type == "dynamic":
            traj_array = np.array(obs_traj)
            ax.plot(
                traj_array[:, 0],
                traj_array[:, 1],
                "--",
                linewidth=1.5,
                alpha=0.6,
                label=f"Dynamic Obs {i+1} Path",
            )

            # 초기 위치
            circle_start = Circle(
                obs_traj[0], obs.radius, color="orange", alpha=0.3, linestyle="--"
            )
            ax.add_patch(circle_start)

            # 최종 위치
            circle_end = Circle(obs_traj[-1], obs.radius, color="red", alpha=0.5)
            ax.add_patch(circle_end)

            # 속도 벡터 표시 (초기 위치에서)
            if abs(obs.velocity_x) > 0.01 or abs(obs.velocity_y) > 0.01:
                arrow = FancyArrow(
                    obs_traj[0][0],
                    obs_traj[0][1],
                    obs.velocity_x * 2,
                    obs.velocity_y * 2,
                    width=0.1,
                    head_width=0.3,
                    head_length=0.2,
                    fc="orange",
                    ec="orange",
                    alpha=0.7,
                )
                ax.add_patch(arrow)

        else:
            # 정적 장애물
            circle = Circle(obs_traj[0], obs.radius, color="gray", alpha=0.6)
            ax.add_patch(circle)

    # 시작점과 목표점
    ax.plot(initial_state[0], initial_state[1], "go", markersize=12, label="Start")
    ax.plot(goal_state[0], goal_state[1], "r*", markersize=18, label="Goal")

    ax.set_xlabel("X [m]", fontsize=12)
    ax.set_ylabel("Y [m]", fontsize=12)
    ax.set_title("Dynamic Obstacle Avoidance Trajectory", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # 2. 제어 입력 (선속도)
    ax = axes[0, 1]

    if len(result.controls) > 0:
        time_steps = np.arange(len(result.controls)) * 0.1
        ax.plot(time_steps, result.controls[:, 0], "b-", linewidth=2, label="Linear Velocity")
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="v_max")

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Linear Velocity [m/s]", fontsize=12)
    ax.set_title("Linear Velocity Profile", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 제어 입력 (각속도)
    ax = axes[1, 0]

    if len(result.controls) > 0:
        time_steps = np.arange(len(result.controls)) * 0.1
        ax.plot(
            time_steps, result.controls[:, 1], "g-", linewidth=2, label="Angular Velocity"
        )
        ax.axhline(y=2.0, color="red", linestyle="--", alpha=0.5, label="ω_max")
        ax.axhline(y=-2.0, color="red", linestyle="--", alpha=0.5)

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Angular Velocity [rad/s]", fontsize=12)
    ax.set_title("Angular Velocity Profile", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 충돌 위험 시간 그래프
    ax = axes[1, 1]

    # 충돌 위험 시간 시각화
    if result.collision_risks:
        time_steps = np.arange(len(result.collision_risks)) * 0.1

        # 각 동적 장애물별로 충돌 위험 시간 표시
        dynamic_obs_indices = [
            i for i, obs in enumerate(obstacles_initial) if obs.obstacle_type == "dynamic"
        ]

        for obs_idx in dynamic_obs_indices:
            collision_times = []
            for step_risks in result.collision_risks:
                # 해당 장애물의 충돌 시간 찾기
                matching_risk = [t for idx, t in step_risks if idx == obs_idx]
                if matching_risk and matching_risk[0] is not None:
                    collision_times.append(matching_risk[0])
                else:
                    collision_times.append(None)

            # None이 아닌 값만 플롯
            valid_times = [(t, ct) for t, ct in zip(time_steps, collision_times) if ct is not None]
            if valid_times:
                times, cts = zip(*valid_times)
                ax.plot(times, cts, "-o", markersize=3, label=f"Dynamic Obs {obs_idx+1}")

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Critical Time (1s)")
    ax.set_xlabel("Simulation Time [s]", fontsize=12)
    ax.set_ylabel("Time to Collision [s]", fontsize=12)
    ax.set_title("Collision Risk Assessment", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n그림 저장 완료: {save_path}")

    return fig


def print_simulation_results(result: SimulationResult) -> None:
    """시뮬레이션 결과 출력."""

    print("\n" + "=" * 70)
    print("                동적 장애물 회피 시뮬레이션 결과")
    print("=" * 70)

    headers = ["Metric", "Value"]
    row_format = "{:<40}{:<30}"

    print(row_format.format(*headers))
    print("-" * 70)

    # 성공 여부
    print(row_format.format("Goal Reached", "Yes" if result.success else "No"))

    # 충돌 여부
    print(
        row_format.format(
            "Collision Detected", "Yes" if result.collision_detected else "No"
        )
    )

    # 최종 거리
    print(
        row_format.format(
            "Final Distance to Goal [m]", f"{result.final_distance_to_goal:.3f}"
        )
    )

    # 스텝 수
    print(row_format.format("Total Steps", str(len(result.controls))))

    # 총 시간
    print(row_format.format("Total Time [s]", f"{result.total_time:.3f}"))

    # 평균 솔버 시간
    avg_solve = np.mean(result.solve_times) * 1000 if result.solve_times else 0
    print(row_format.format("Avg Solve Time [ms]", f"{avg_solve:.2f}"))

    # 평균 비용
    avg_cost = np.mean(result.costs) if result.costs else 0
    print(row_format.format("Avg Cost", f"{avg_cost:.2f}"))

    print("=" * 70)


def main():
    """메인 함수."""

    print(
        """
╔═══════════════════════════════════════════════════════════════════════╗
║      Dynamic Obstacle Avoidance Demo: 동적 장애물 회피 시뮬레이션     ║
╠═══════════════════════════════════════════════════════════════════════╣
║  MPC를 사용하여 움직이는 장애물을 피해 목표점까지 이동합니다.        ║
║                                                                       ║
║  시나리오:                                                            ║
║    - 시작점: (0, 0)                                                  ║
║    - 목표점: (8, 8)                                                  ║
║    - 정적 장애물: 1개                                                ║
║    - 동적 장애물: 3개 (각각 다른 방향으로 이동)                      ║
║                                                                       ║
║  특징:                                                                ║
║    - 동적 장애물의 미래 위치 예측                                    ║
║    - 충돌 위험 시간 계산                                             ║
║    - 불확실성을 고려한 안전 마진 증가                                ║
╚═══════════════════════════════════════════════════════════════════════╝
    """
    )

    # 시나리오 생성
    initial_state, goal_state, obstacles = create_dynamic_obstacle_scenario()

    print(f"\n초기 상태: ({initial_state[0]:.1f}, {initial_state[1]:.1f})")
    print(f"목표 상태: ({goal_state[0]:.1f}, {goal_state[1]:.1f})")
    print(f"\n장애물 정보:")
    for i, obs in enumerate(obstacles, 1):
        if obs.obstacle_type == "static":
            print(
                f"  장애물 {i} [정적]: 중심=({obs.x:.1f}, {obs.y:.1f}), 반경={obs.radius:.2f}m"
            )
        else:
            print(
                f"  장애물 {i} [동적]: 중심=({obs.x:.1f}, {obs.y:.1f}), 반경={obs.radius:.2f}m, "
                f"속도=({obs.velocity_x:.2f}, {obs.velocity_y:.2f}) m/s"
            )

    # 참조 궤적
    reference = generate_reference_trajectory(initial_state, goal_state)

    # 시뮬레이션 실행
    print("\n동적 장애물 회피 시뮬레이션 실행 중...")
    print("예측 모델: Constant Velocity with Uncertainty")

    result = run_dynamic_obstacle_avoidance_simulation(
        initial_state=initial_state,
        goal_state=goal_state,
        obstacles=obstacles.copy(),  # 복사본 사용 (원본 유지)
        prediction_model=PredictionModel.CONSTANT_VELOCITY,
    )

    # 결과 출력
    print_simulation_results(result)

    # 시각화
    fig = plot_dynamic_obstacle_avoidance(
        result=result,
        obstacles_initial=obstacles,
        initial_state=initial_state,
        goal_state=goal_state,
        reference=reference,
        save_path="dynamic_obstacle_avoidance_demo.png",
    )

    plt.show()

    print("\n테스트 완료!")


if __name__ == "__main__":
    main()
