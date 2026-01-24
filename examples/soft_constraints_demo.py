#!/usr/bin/env python3
"""
Soft Constraints Demo: 소프트 제약조건 비교 데모

하드 제약조건만 사용하는 MPC와 소프트 제약조건을 사용하는 MPC를 비교합니다.

┌─────────────────────────────────────────────────────────────────────┐
│                      비교 시나리오                                   │
├─────────────────────────────────────────────────────────────────────┤
│ 1. 정상 주행: 제약조건 내에서 동작                                   │
│ 2. 급격한 회전: 가속도 제약 테스트                                   │
│ 3. 고속 주행: 속도 제약 테스트                                       │
└─────────────────────────────────────────────────────────────────────┘
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import time

from mpc_controller.controllers.mpc import MPCController, MPCParams
from mpc_controller.models.differential_drive import RobotParams
from mpc_controller.models.soft_constraints import SoftConstraintParams, PenaltyType


@dataclass
class TestResult:
    """테스트 결과 데이터."""
    name: str
    states: np.ndarray
    controls: np.ndarray
    costs: List[float]
    solve_times: List[float]
    violations: List[dict]
    feasible: bool
    total_time: float


def generate_aggressive_trajectory(n_points: int = 100) -> np.ndarray:
    """
    급격한 방향 전환이 필요한 경로 생성.

    S자 곡선으로 가속도 제약을 테스트합니다.
    """
    t = np.linspace(0, 2 * np.pi, n_points)
    x = t * 0.5  # 전진
    y = np.sin(t) * 0.5  # S자 곡선
    theta = np.arctan2(np.gradient(y), np.gradient(x))

    return np.column_stack([x, y, theta])


def generate_high_speed_trajectory(n_points: int = 100) -> np.ndarray:
    """
    고속 직선 주행 경로 생성.

    속도 제약을 테스트합니다.
    """
    t = np.linspace(0, 5, n_points)
    x = t * 2.0  # 빠른 전진 (2m/s 요구)
    y = np.zeros_like(t)
    theta = np.zeros_like(t)

    return np.column_stack([x, y, theta])


def run_mpc_test(
    controller: MPCController,
    reference: np.ndarray,
    initial_state: np.ndarray,
    dt: float = 0.1,
    max_steps: int = 200,
    name: str = "MPC",
) -> TestResult:
    """
    MPC 컨트롤러 테스트 실행.
    """
    states = [initial_state.copy()]
    controls = []
    costs = []
    solve_times = []
    violations = []

    state = initial_state.copy()
    controller.reset()

    start_time = time.perf_counter()
    feasible = True

    for step in range(max_steps):
        # 현재 참조 궤적 추출
        start_idx = min(step, len(reference) - controller.N - 1)
        ref_window = reference[start_idx:start_idx + controller.N + 1]

        if len(ref_window) < controller.N + 1:
            # 끝에 도달하면 마지막 점으로 패딩
            padding = np.tile(reference[-1:], (controller.N + 1 - len(ref_window), 1))
            ref_window = np.vstack([ref_window, padding])

        try:
            control, info = controller.compute_control(state, ref_window)

            costs.append(info.get("cost", 0))
            solve_times.append(info.get("solve_time", 0))

            # 소프트 제약조건 위반 정보
            soft_info = info.get("soft_constraints", {})
            if soft_info:
                violations.append({
                    "step": step,
                    "has_violations": soft_info.get("has_violations", False),
                    "max_vel": soft_info.get("max_velocity_violation", 0),
                    "max_acc": soft_info.get("max_acceleration_violation", 0),
                })

            # 상태 업데이트 (간단한 오일러 적분)
            x, y, theta = state
            v, omega = control

            state = np.array([
                x + v * np.cos(theta) * dt,
                y + v * np.sin(theta) * dt,
                theta + omega * dt,
            ])

            states.append(state.copy())
            controls.append(control.copy())

            # 목표 도달 확인
            dist_to_goal = np.linalg.norm(state[:2] - reference[-1, :2])
            if dist_to_goal < 0.1:
                break

        except Exception as e:
            print(f"  [{name}] Step {step} failed: {e}")
            feasible = False
            break

    total_time = time.perf_counter() - start_time

    return TestResult(
        name=name,
        states=np.array(states),
        controls=np.array(controls) if controls else np.array([]).reshape(0, 2),
        costs=costs,
        solve_times=solve_times,
        violations=violations,
        feasible=feasible,
        total_time=total_time,
    )


def plot_comparison(
    results: List[TestResult],
    reference: np.ndarray,
    title: str = "MPC Comparison",
    save_path: str = None,
) -> plt.Figure:
    """
    비교 결과 시각화.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    colors = ["blue", "red", "green", "orange"]

    # 1. 궤적 비교
    ax = axes[0, 0]
    ax.plot(reference[:, 0], reference[:, 1], "k--", linewidth=2, label="Reference", alpha=0.7)
    for result, color in zip(results, colors):
        ax.plot(result.states[:, 0], result.states[:, 1],
                color=color, linewidth=2, label=result.name)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Trajectory Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # 2. 위치 오차
    ax = axes[0, 1]
    for result, color in zip(results, colors):
        if len(result.states) > 1:
            # 참조와의 거리 계산
            errors = []
            for i, state in enumerate(result.states):
                ref_idx = min(i, len(reference) - 1)
                error = np.linalg.norm(state[:2] - reference[ref_idx, :2])
                errors.append(error)
            ax.plot(errors, color=color, linewidth=2, label=f"{result.name}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Position Error [m]")
    ax.set_title("Tracking Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 제어 입력 (속도)
    ax = axes[0, 2]
    for result, color in zip(results, colors):
        if len(result.controls) > 0:
            ax.plot(result.controls[:, 0], color=color, linewidth=2, label=f"{result.name} v")
    ax.axhline(y=1.0, color="gray", linestyle="--", label="v_max")
    ax.axhline(y=-1.0, color="gray", linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Linear Velocity [m/s]")
    ax.set_title("Linear Velocity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 제어 입력 (각속도)
    ax = axes[1, 0]
    for result, color in zip(results, colors):
        if len(result.controls) > 0:
            ax.plot(result.controls[:, 1], color=color, linewidth=2, label=f"{result.name} ω")
    ax.axhline(y=2.0, color="gray", linestyle="--", label="ω_max")
    ax.axhline(y=-2.0, color="gray", linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Angular Velocity [rad/s]")
    ax.set_title("Angular Velocity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Solve Time
    ax = axes[1, 1]
    for result, color in zip(results, colors):
        if result.solve_times:
            ax.plot([t * 1000 for t in result.solve_times],
                   color=color, linewidth=2, label=result.name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Solve Time [ms]")
    ax.set_title("Solver Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. 소프트 제약조건 위반
    ax = axes[1, 2]
    for result, color in zip(results, colors):
        if result.violations:
            vel_violations = [v.get("max_vel", 0) for v in result.violations]
            acc_violations = [v.get("max_acc", 0) for v in result.violations]
            ax.plot(vel_violations, color=color, linewidth=2,
                   label=f"{result.name} vel", linestyle="-")
            ax.plot(acc_violations, color=color, linewidth=2,
                   label=f"{result.name} acc", linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Violation Amount")
    ax.set_title("Soft Constraint Violations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def print_metrics(results: List[TestResult]) -> None:
    """결과 메트릭 출력."""
    print("\n" + "=" * 70)
    print("                         테스트 결과 비교")
    print("=" * 70)

    headers = ["Metric", *[r.name for r in results]]
    row_format = "{:<25}" + "{:<20}" * len(results)

    print(row_format.format(*headers))
    print("-" * 70)

    # Feasibility
    feasible = ["Yes" if r.feasible else "No" for r in results]
    print(row_format.format("Feasible", *feasible))

    # Total Time
    times = [f"{r.total_time:.3f} s" for r in results]
    print(row_format.format("Total Time", *times))

    # Average Solve Time
    avg_solve = [f"{np.mean(r.solve_times)*1000:.2f} ms" if r.solve_times else "N/A"
                 for r in results]
    print(row_format.format("Avg Solve Time", *avg_solve))

    # Total Steps
    steps = [str(len(r.controls)) for r in results]
    print(row_format.format("Steps Completed", *steps))

    # Violation Count
    violation_counts = []
    for r in results:
        if r.violations:
            count = sum(1 for v in r.violations if v.get("has_violations", False))
            violation_counts.append(str(count))
        else:
            violation_counts.append("N/A (hard)")
    print(row_format.format("Violation Steps", *violation_counts))

    # Max Velocity Violation
    max_vel = []
    for r in results:
        if r.violations:
            max_v = max((v.get("max_vel", 0) for v in r.violations), default=0)
            max_vel.append(f"{max_v:.4f}")
        else:
            max_vel.append("N/A")
    print(row_format.format("Max Vel Violation", *max_vel))

    # Max Acceleration Violation
    max_acc = []
    for r in results:
        if r.violations:
            max_a = max((v.get("max_acc", 0) for v in r.violations), default=0)
            max_acc.append(f"{max_a:.4f}")
        else:
            max_acc.append("N/A")
    print(row_format.format("Max Acc Violation", *max_acc))

    print("=" * 70)


def main():
    """메인 함수."""
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║           Soft Constraints Demo: MPC 비교 테스트                      ║
╠═══════════════════════════════════════════════════════════════════════╣
║  - Hard Constraints Only: 하드 제약조건만 사용                        ║
║  - Soft Constraints (Quadratic): L2 패널티 함수                       ║
║  - Soft Constraints (Linear): L1 패널티 함수                          ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)

    # 로봇 파라미터 (제한적인 가속도)
    robot_params = RobotParams(
        max_velocity=1.0,
        max_omega=2.0,
    )

    # === 시나리오 1: 급격한 S자 곡선 ===
    print("\n[시나리오 1] 급격한 S자 곡선 주행")
    print("-" * 50)

    reference = generate_aggressive_trajectory(100)
    initial_state = np.array([0.0, 0.0, 0.0])

    # 1. Hard Constraints Only
    print("  Testing: Hard Constraints Only...")
    mpc_hard = MPCController(
        robot_params=robot_params,
        mpc_params=MPCParams(
            N=15,
            dt=0.1,
            a_max=0.5,  # 낮은 가속도 제한
            alpha_max=1.0,
        ),
        enable_soft_constraints=False,
    )
    result_hard = run_mpc_test(mpc_hard, reference, initial_state, name="Hard Only")

    # 2. Soft Constraints (Quadratic)
    print("  Testing: Soft Constraints (Quadratic)...")
    soft_params_quad = SoftConstraintParams(
        velocity_weight=100.0,
        acceleration_weight=50.0,
        penalty_type=PenaltyType.QUADRATIC,
    )
    mpc_soft_quad = MPCController(
        robot_params=robot_params,
        mpc_params=MPCParams(
            N=15,
            dt=0.1,
            a_max=0.5,
            alpha_max=1.0,
            soft_constraints=soft_params_quad,
        ),
        enable_soft_constraints=True,
    )
    result_soft_quad = run_mpc_test(mpc_soft_quad, reference, initial_state,
                                     name="Soft (Quadratic)")

    # 3. Soft Constraints (Linear)
    print("  Testing: Soft Constraints (Linear)...")
    soft_params_lin = SoftConstraintParams(
        velocity_weight=100.0,
        acceleration_weight=50.0,
        penalty_type=PenaltyType.LINEAR,
    )
    mpc_soft_lin = MPCController(
        robot_params=robot_params,
        mpc_params=MPCParams(
            N=15,
            dt=0.1,
            a_max=0.5,
            alpha_max=1.0,
            soft_constraints=soft_params_lin,
        ),
        enable_soft_constraints=True,
    )
    result_soft_lin = run_mpc_test(mpc_soft_lin, reference, initial_state,
                                    name="Soft (Linear)")

    results = [result_hard, result_soft_quad, result_soft_lin]

    # 결과 출력
    print_metrics(results)

    # 시각화
    fig = plot_comparison(
        results,
        reference,
        title="Scenario 1: Aggressive S-Curve Trajectory",
        save_path="soft_constraints_comparison.png",
    )

    plt.show()

    print("\n테스트 완료!")


if __name__ == "__main__":
    main()
