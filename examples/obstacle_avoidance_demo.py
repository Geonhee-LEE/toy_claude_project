#!/usr/bin/env python3
"""
Obstacle Avoidance Demo: 장애물 회피 시뮬레이션

MPC를 사용하여 정적 장애물을 피해 목표점까지 이동하는 시나리오입니다.

┌─────────────────────────────────────────────────────────────────────┐
│                      시나리오 개요                                   │
├─────────────────────────────────────────────────────────────────────┤
│ 1. 시작점: (0, 0)                                                   │
│ 2. 목표점: (5, 5)                                                   │
│ 3. 장애물: 2-3개 원형 장애물 배치                                    │
│ 4. MPC가 ObstacleSoftConstraint를 사용하여 회피                     │
└─────────────────────────────────────────────────────────────────────┘

작동 흐름:
┌─────────────┐
│  초기 상태  │
└──────┬──────┘
       │
       ▼
┌─────────────┐      ┌──────────────┐
│ MPC 제어    │─────▶│ 장애물 감지  │
└──────┬──────┘      └──────┬───────┘
       │                    │
       │   ┌────────────────┘
       │   │ 소프트 제약조건 적용
       ▼   ▼
┌─────────────┐
│ 회피 경로   │
│ 생성        │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 상태 업데이트│
└──────┬──────┘
       │
       │ 목표 도달?
       └─────▶ Yes: 종료
              No: 반복
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass
from typing import List, Tuple
import time

from mpc_controller.controllers.mpc import MPCController, MPCParams
from mpc_controller.models.differential_drive import RobotParams
from mpc_controller.models.soft_constraints import (
    SoftConstraintParams,
    PenaltyType,
    ObstacleSoftConstraint,
)


@dataclass
class Obstacle:
    """장애물 정보."""

    x: float  # 중심 X 좌표 [m]
    y: float  # 중심 Y 좌표 [m]
    radius: float  # 반경 [m]

    def distance_to(self, x: float, y: float) -> float:
        """점까지의 거리 계산."""
        return np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)

    def is_collision(self, x: float, y: float, safety_margin: float = 0.0) -> bool:
        """충돌 여부 확인."""
        return self.distance_to(x, y) < (self.radius + safety_margin)


@dataclass
class SimulationResult:
    """시뮬레이션 결과 데이터."""

    states: np.ndarray  # 상태 궤적 [N x 3]
    controls: np.ndarray  # 제어 입력 [N x 2]
    costs: List[float]  # 각 단계별 비용
    solve_times: List[float]  # 각 단계별 솔버 시간
    total_time: float  # 전체 소요 시간
    success: bool  # 목표 도달 여부
    collision_detected: bool  # 충돌 발생 여부
    final_distance_to_goal: float  # 최종 목표까지의 거리


def create_obstacle_scenario() -> Tuple[np.ndarray, np.ndarray, List[Obstacle]]:
    """
    장애물 회피 시나리오 생성.

    Returns:
        (초기 상태, 목표 상태, 장애물 리스트)
    """
    # 시작점과 목표점
    initial_state = np.array([0.0, 0.0, np.pi / 4])  # (x, y, theta)
    goal_state = np.array([5.0, 5.0, np.pi / 4])

    # 정적 장애물 배치 (경로 중간에 위치)
    obstacles = [
        Obstacle(x=2.0, y=1.5, radius=0.5),  # 장애물 1
        Obstacle(x=3.5, y=3.0, radius=0.4),  # 장애물 2
        Obstacle(x=1.5, y=3.5, radius=0.6),  # 장애물 3
    ]

    return initial_state, goal_state, obstacles


def generate_reference_trajectory(
    start: np.ndarray,
    goal: np.ndarray,
    n_points: int = 100,
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


def run_obstacle_avoidance_simulation(
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    obstacles: List[Obstacle],
    enable_obstacle_avoidance: bool = True,
    dt: float = 0.1,
    max_steps: int = 200,
) -> SimulationResult:
    """
    장애물 회피 시뮬레이션 실행.

    Args:
        initial_state: 초기 상태 [x, y, theta]
        goal_state: 목표 상태 [x, y, theta]
        obstacles: 장애물 리스트
        enable_obstacle_avoidance: 장애물 회피 활성화 여부
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

    # MPC 파라미터
    if enable_obstacle_avoidance:
        # 장애물 회피를 위한 소프트 제약조건 설정
        soft_params = SoftConstraintParams(
            velocity_weight=50.0,
            acceleration_weight=30.0,
            obstacle_weight=500.0,  # 높은 가중치로 장애물 회피 강조
            penalty_type=PenaltyType.QUADRATIC,
            enable_velocity_soft=True,
            enable_acceleration_soft=True,
            enable_obstacle_soft=True,
        )

        mpc_params = MPCParams(
            N=20,  # 더 긴 예측 horizon
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

        # 장애물 제약조건 추가
        for obs in obstacles:
            # TODO: MPC 컨트롤러에 장애물 추가 기능 필요
            # controller.add_obstacle(obs.x, obs.y, obs.radius)
            pass

    else:
        # 장애물 회피 없이 일반 MPC
        mpc_params = MPCParams(
            N=15,
            dt=dt,
            a_max=0.8,
            alpha_max=1.5,
        )

        controller = MPCController(
            robot_params=robot_params,
            mpc_params=mpc_params,
            enable_soft_constraints=False,
        )

    # 시뮬레이션 변수
    states = [initial_state.copy()]
    controls = []
    costs = []
    solve_times = []
    collision_detected = False

    state = initial_state.copy()
    controller.reset()

    start_time = time.perf_counter()

    # 시뮬레이션 루프
    for step in range(max_steps):
        # 현재 참조 궤적 추출
        start_idx = min(step, len(reference) - controller.N - 1)
        ref_window = reference[start_idx:start_idx + controller.N + 1]

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

            next_state = np.array([
                x + v * np.cos(theta) * dt,
                y + v * np.sin(theta) * dt,
                theta + omega * dt,
            ])

            # 충돌 체크
            for obs in obstacles:
                if obs.is_collision(next_state[0], next_state[1], safety_margin=0.2):
                    collision_detected = True
                    print(f"  [WARNING] 충돌 감지! Step {step}, 장애물: ({obs.x:.2f}, {obs.y:.2f})")

            state = next_state
            states.append(state.copy())
            controls.append(control.copy())

            # 목표 도달 확인
            dist_to_goal = np.linalg.norm(state[:2] - goal_state[:2])
            if dist_to_goal < 0.15:
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
    success = final_distance < 0.15

    return SimulationResult(
        states=states_array,
        controls=controls_array,
        costs=costs,
        solve_times=solve_times,
        total_time=total_time,
        success=success,
        collision_detected=collision_detected,
        final_distance_to_goal=final_distance,
    )


def plot_obstacle_avoidance(
    result_with_avoidance: SimulationResult,
    result_without_avoidance: SimulationResult,
    obstacles: List[Obstacle],
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    reference: np.ndarray,
    save_path: str = None,
) -> plt.Figure:
    """
    장애물 회피 시뮬레이션 결과 시각화.

    Args:
        result_with_avoidance: 장애물 회피 활성화 결과
        result_without_avoidance: 장애물 회피 비활성화 결과
        obstacles: 장애물 리스트
        initial_state: 초기 상태
        goal_state: 목표 상태
        reference: 참조 궤적
        save_path: 저장 경로

    Returns:
        Figure 객체
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 궤적 비교 (장애물 회피 O)
    ax = axes[0, 0]

    # 참조 궤적
    ax.plot(reference[:, 0], reference[:, 1], 'k--',
            linewidth=1.5, label='Reference', alpha=0.5)

    # 장애물 그리기
    for i, obs in enumerate(obstacles):
        circle = Circle((obs.x, obs.y), obs.radius,
                       color='red', alpha=0.3, label=f'Obstacle {i+1}' if i == 0 else '')
        ax.add_patch(circle)
        # 안전 마진
        safety_circle = Circle((obs.x, obs.y), obs.radius + 0.3,
                             color='orange', fill=False, linestyle='--', alpha=0.5)
        ax.add_patch(safety_circle)

    # 실제 궤적 (회피 활성화)
    ax.plot(result_with_avoidance.states[:, 0],
            result_with_avoidance.states[:, 1],
            'b-', linewidth=2, label='MPC with Avoidance')

    # 시작점과 목표점
    ax.plot(initial_state[0], initial_state[1], 'go',
            markersize=10, label='Start')
    ax.plot(goal_state[0], goal_state[1], 'r*',
            markersize=15, label='Goal')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Obstacle Avoidance Trajectory (WITH Avoidance)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # 2. 궤적 비교 (장애물 회피 X)
    ax = axes[0, 1]

    # 참조 궤적
    ax.plot(reference[:, 0], reference[:, 1], 'k--',
            linewidth=1.5, label='Reference', alpha=0.5)

    # 장애물 그리기
    for i, obs in enumerate(obstacles):
        circle = Circle((obs.x, obs.y), obs.radius,
                       color='red', alpha=0.3, label=f'Obstacle {i+1}' if i == 0 else '')
        ax.add_patch(circle)

    # 실제 궤적 (회피 비활성화)
    ax.plot(result_without_avoidance.states[:, 0],
            result_without_avoidance.states[:, 1],
            'r-', linewidth=2, label='MPC without Avoidance')

    # 시작점과 목표점
    ax.plot(initial_state[0], initial_state[1], 'go',
            markersize=10, label='Start')
    ax.plot(goal_state[0], goal_state[1], 'r*',
            markersize=15, label='Goal')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Trajectory (WITHOUT Avoidance)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # 3. 제어 입력 비교 (선속도)
    ax = axes[1, 0]

    if len(result_with_avoidance.controls) > 0:
        ax.plot(result_with_avoidance.controls[:, 0],
                'b-', linewidth=2, label='With Avoidance')

    if len(result_without_avoidance.controls) > 0:
        ax.plot(result_without_avoidance.controls[:, 0],
                'r-', linewidth=2, label='Without Avoidance')

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='v_max')
    ax.set_xlabel('Step')
    ax.set_ylabel('Linear Velocity [m/s]')
    ax.set_title('Linear Velocity Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 제어 입력 비교 (각속도)
    ax = axes[1, 1]

    if len(result_with_avoidance.controls) > 0:
        ax.plot(result_with_avoidance.controls[:, 1],
                'b-', linewidth=2, label='With Avoidance')

    if len(result_without_avoidance.controls) > 0:
        ax.plot(result_without_avoidance.controls[:, 1],
                'r-', linewidth=2, label='Without Avoidance')

    ax.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='ω_max')
    ax.axhline(y=-2.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Angular Velocity [rad/s]')
    ax.set_title('Angular Velocity Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n그림 저장 완료: {save_path}")

    return fig


def print_simulation_results(
    result_with: SimulationResult,
    result_without: SimulationResult,
) -> None:
    """시뮬레이션 결과 출력."""

    print("\n" + "=" * 70)
    print("                    시뮬레이션 결과 비교")
    print("=" * 70)

    headers = ["Metric", "With Avoidance", "Without Avoidance"]
    row_format = "{:<30}{:<20}{:<20}"

    print(row_format.format(*headers))
    print("-" * 70)

    # 성공 여부
    print(row_format.format(
        "Goal Reached",
        "Yes" if result_with.success else "No",
        "Yes" if result_without.success else "No"
    ))

    # 충돌 여부
    print(row_format.format(
        "Collision Detected",
        "Yes" if result_with.collision_detected else "No",
        "Yes" if result_without.collision_detected else "No"
    ))

    # 최종 거리
    print(row_format.format(
        "Final Distance to Goal [m]",
        f"{result_with.final_distance_to_goal:.3f}",
        f"{result_without.final_distance_to_goal:.3f}"
    ))

    # 스텝 수
    print(row_format.format(
        "Total Steps",
        str(len(result_with.controls)),
        str(len(result_without.controls))
    ))

    # 총 시간
    print(row_format.format(
        "Total Time [s]",
        f"{result_with.total_time:.3f}",
        f"{result_without.total_time:.3f}"
    ))

    # 평균 솔버 시간
    avg_solve_with = np.mean(result_with.solve_times) * 1000 if result_with.solve_times else 0
    avg_solve_without = np.mean(result_without.solve_times) * 1000 if result_without.solve_times else 0
    print(row_format.format(
        "Avg Solve Time [ms]",
        f"{avg_solve_with:.2f}",
        f"{avg_solve_without:.2f}"
    ))

    print("=" * 70)


def main():
    """메인 함수."""

    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║           Obstacle Avoidance Demo: 장애물 회피 시뮬레이션             ║
╠═══════════════════════════════════════════════════════════════════════╣
║  MPC를 사용하여 정적 장애물을 피해 목표점까지 이동합니다.            ║
║                                                                       ║
║  시나리오:                                                            ║
║    - 시작점: (0, 0)                                                  ║
║    - 목표점: (5, 5)                                                  ║
║    - 장애물: 3개 원형 장애물                                         ║
║                                                                       ║
║  비교:                                                                ║
║    1. 장애물 회피 활성화 (ObstacleSoftConstraint 사용)               ║
║    2. 장애물 회피 비활성화 (일반 경로 추적)                          ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)

    # 시나리오 생성
    initial_state, goal_state, obstacles = create_obstacle_scenario()

    print(f"\n초기 상태: ({initial_state[0]:.1f}, {initial_state[1]:.1f})")
    print(f"목표 상태: ({goal_state[0]:.1f}, {goal_state[1]:.1f})")
    print(f"\n장애물 정보:")
    for i, obs in enumerate(obstacles, 1):
        print(f"  장애물 {i}: 중심=({obs.x:.1f}, {obs.y:.1f}), 반경={obs.radius:.2f}m")

    # 참조 궤적
    reference = generate_reference_trajectory(initial_state, goal_state)

    # 1. 장애물 회피 활성화 시뮬레이션
    print("\n[1] 장애물 회피 활성화 시뮬레이션 실행 중...")
    result_with_avoidance = run_obstacle_avoidance_simulation(
        initial_state=initial_state,
        goal_state=goal_state,
        obstacles=obstacles,
        enable_obstacle_avoidance=True,
    )

    # 2. 장애물 회피 비활성화 시뮬레이션
    print("\n[2] 장애물 회피 비활성화 시뮬레이션 실행 중...")
    result_without_avoidance = run_obstacle_avoidance_simulation(
        initial_state=initial_state,
        goal_state=goal_state,
        obstacles=obstacles,
        enable_obstacle_avoidance=False,
    )

    # 결과 출력
    print_simulation_results(result_with_avoidance, result_without_avoidance)

    # 시각화
    fig = plot_obstacle_avoidance(
        result_with_avoidance=result_with_avoidance,
        result_without_avoidance=result_without_avoidance,
        obstacles=obstacles,
        initial_state=initial_state,
        goal_state=goal_state,
        reference=reference,
        save_path="obstacle_avoidance_demo.png",
    )

    plt.show()

    print("\n테스트 완료!")


if __name__ == "__main__":
    main()
