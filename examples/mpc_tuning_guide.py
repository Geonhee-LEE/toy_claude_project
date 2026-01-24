#!/usr/bin/env python3
"""
MPC 파라미터 튜닝 가이드

이 예제는 다양한 MPC 파라미터 조합을 비교하고, 각 파라미터가 성능에 미치는 영향을 시각화합니다.

┌─────────────────────────────────────────────────────────────────────────┐
│                          파라미터 영향 분석                              │
├─────────────────────────────────────────────────────────────────────────┤
│ N  (Prediction Horizon): 예측 구간 길이                                 │
│ dt (Time Step):          이산화 시간 간격                                │
│ Q  (State Weights):      상태 오차 가중치 [x, y, theta]                 │
│ R  (Control Weights):    제어 입력 가중치 [v, omega]                     │
│ Qf (Terminal Weights):   최종 상태 오차 가중치                           │
│ Rd (Control Rate):       제어 입력 변화율 가중치 (부드러움)              │
└─────────────────────────────────────────────────────────────────────────┘
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time

from mpc_controller.controllers.mpc import MPCController, MPCParams
from mpc_controller.models.differential_drive import RobotParams


@dataclass
class TuningResult:
    """파라미터 튜닝 결과 데이터."""
    name: str
    params: MPCParams
    states: np.ndarray
    controls: np.ndarray
    costs: List[float]
    solve_times: List[float]
    tracking_errors: List[float]
    control_smoothness: float  # 제어 입력 변화량의 평균
    total_time: float


def generate_test_trajectory(trajectory_type: str = "sine", n_points: int = 100) -> np.ndarray:
    """
    테스트용 궤적 생성.

    Args:
        trajectory_type: 궤적 타입 ("sine", "circle", "straight")
        n_points: 궤적 점 개수

    Returns:
        궤적 배열 [n_points, 3] (x, y, theta)
    """
    if trajectory_type == "sine":
        # 정현파 궤적 - 주파수 응답 테스트에 적합
        t = np.linspace(0, 4 * np.pi, n_points)
        x = t * 0.3
        y = np.sin(t) * 0.8
        dx = np.gradient(x)
        dy = np.gradient(y)
        theta = np.arctan2(dy, dx)

    elif trajectory_type == "circle":
        # 원형 궤적 - 곡선 주행 테스트에 적합
        t = np.linspace(0, 2 * np.pi, n_points)
        radius = 2.0
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        theta = t + np.pi / 2

    elif trajectory_type == "straight":
        # 직선 궤적 - 직진 성능 테스트에 적합
        t = np.linspace(0, 5, n_points)
        x = t
        y = np.zeros_like(t)
        theta = np.zeros_like(t)

    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    return np.column_stack([x, y, theta])


def run_mpc_tuning_test(
    controller: MPCController,
    reference: np.ndarray,
    initial_state: np.ndarray,
    dt: float = 0.1,
    max_steps: int = 200,
    name: str = "MPC",
) -> TuningResult:
    """
    MPC 파라미터 튜닝 테스트 실행.

    Args:
        controller: MPC 컨트롤러
        reference: 참조 궤적
        initial_state: 초기 상태
        dt: 시뮬레이션 시간 간격
        max_steps: 최대 스텝 수
        name: 테스트 이름

    Returns:
        TuningResult 객체
    """
    states = [initial_state.copy()]
    controls = []
    costs = []
    solve_times = []
    tracking_errors = []

    state = initial_state.copy()
    controller.reset()

    start_time = time.perf_counter()

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

            # 추적 오차 계산
            ref_idx = min(step, len(reference) - 1)
            tracking_error = np.linalg.norm(state[:2] - reference[ref_idx, :2])
            tracking_errors.append(tracking_error)

            # 상태 업데이트 (오일러 적분)
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
            print(f"  [{name}] Step {step} 실패: {e}")
            break

    total_time = time.perf_counter() - start_time

    # 제어 입력 부드러움 계산 (제어 변화량의 RMS)
    controls_arr = np.array(controls) if controls else np.array([]).reshape(0, 2)
    if len(controls_arr) > 1:
        control_diff = np.diff(controls_arr, axis=0)
        control_smoothness = np.sqrt(np.mean(control_diff ** 2))
    else:
        control_smoothness = 0.0

    return TuningResult(
        name=name,
        params=controller.params,
        states=np.array(states),
        controls=controls_arr,
        costs=costs,
        solve_times=solve_times,
        tracking_errors=tracking_errors,
        control_smoothness=control_smoothness,
        total_time=total_time,
    )


def compare_prediction_horizon(
    reference: np.ndarray,
    initial_state: np.ndarray,
) -> List[TuningResult]:
    """
    예측 구간(N) 변화에 따른 성능 비교.

    N이 크면:
      - 장점: 더 먼 미래를 고려하여 최적화, 성능 향상 가능
      - 단점: 계산 시간 증가, 실시간성 저하

    N이 작으면:
      - 장점: 빠른 계산, 실시간 제어에 유리
      - 단점: 근시안적 최적화, 성능 저하 가능
    """
    print("\n[테스트 1] Prediction Horizon (N) 비교")
    print("-" * 60)

    robot_params = RobotParams(max_velocity=1.0, max_omega=2.0)
    N_values = [5, 10, 20, 30]
    results = []

    for N in N_values:
        print(f"  Testing N={N}...")
        mpc_params = MPCParams(
            N=N,
            dt=0.1,
            Q=np.diag([10.0, 10.0, 1.0]),
            R=np.diag([0.1, 0.1]),
        )
        controller = MPCController(robot_params, mpc_params, enable_soft_constraints=False)
        result = run_mpc_tuning_test(
            controller, reference, initial_state, name=f"N={N}"
        )
        results.append(result)

    return results


def compare_time_step(
    reference: np.ndarray,
    initial_state: np.ndarray,
) -> List[TuningResult]:
    """
    시간 간격(dt) 변화에 따른 성능 비교.

    dt가 작으면:
      - 장점: 더 정확한 이산화, 비선형 시스템에 유리
      - 단점: 같은 N으로 더 짧은 미래만 볼 수 있음

    dt가 크면:
      - 장점: 같은 N으로 더 먼 미래를 볼 수 있음
      - 단점: 이산화 오차 증가, 비선형성 무시 가능
    """
    print("\n[테스트 2] Time Step (dt) 비교")
    print("-" * 60)

    robot_params = RobotParams(max_velocity=1.0, max_omega=2.0)
    dt_values = [0.05, 0.1, 0.2, 0.3]
    results = []

    for dt in dt_values:
        print(f"  Testing dt={dt}...")
        mpc_params = MPCParams(
            N=20,
            dt=dt,
            Q=np.diag([10.0, 10.0, 1.0]),
            R=np.diag([0.1, 0.1]),
        )
        controller = MPCController(robot_params, mpc_params, enable_soft_constraints=False)
        result = run_mpc_tuning_test(
            controller, reference, initial_state, name=f"dt={dt}"
        )
        results.append(result)

    return results


def compare_state_weights(
    reference: np.ndarray,
    initial_state: np.ndarray,
) -> List[TuningResult]:
    """
    상태 가중치(Q) 변화에 따른 성능 비교.

    Q가 크면:
      - 장점: 궤적 추적 정확도 향상
      - 단점: 제어 입력이 커질 수 있음, 과도한 반응

    Q가 작으면:
      - 장점: 부드러운 제어
      - 단점: 궤적 추적 정확도 저하
    """
    print("\n[테스트 3] State Weights (Q) 비교")
    print("-" * 60)

    robot_params = RobotParams(max_velocity=1.0, max_omega=2.0)
    Q_configs = [
        ("Low Q", np.diag([1.0, 1.0, 0.1])),
        ("Medium Q", np.diag([10.0, 10.0, 1.0])),
        ("High Q", np.diag([100.0, 100.0, 10.0])),
    ]
    results = []

    for name, Q in Q_configs:
        print(f"  Testing {name}...")
        mpc_params = MPCParams(
            N=20,
            dt=0.1,
            Q=Q,
            R=np.diag([0.1, 0.1]),
        )
        controller = MPCController(robot_params, mpc_params, enable_soft_constraints=False)
        result = run_mpc_tuning_test(
            controller, reference, initial_state, name=name
        )
        results.append(result)

    return results


def compare_control_weights(
    reference: np.ndarray,
    initial_state: np.ndarray,
) -> List[TuningResult]:
    """
    제어 가중치(R) 변화에 따른 성능 비교.

    R이 크면:
      - 장점: 제어 에너지 절약, 부드러운 제어
      - 단점: 궤적 추적 정확도 저하

    R이 작으면:
      - 장점: 적극적인 제어, 빠른 응답
      - 단점: 제어 입력이 커질 수 있음, 에너지 소모 증가
    """
    print("\n[테스트 4] Control Weights (R) 비교")
    print("-" * 60)

    robot_params = RobotParams(max_velocity=1.0, max_omega=2.0)
    R_configs = [
        ("Low R", np.diag([0.01, 0.01])),
        ("Medium R", np.diag([0.1, 0.1])),
        ("High R", np.diag([1.0, 1.0])),
    ]
    results = []

    for name, R in R_configs:
        print(f"  Testing {name}...")
        mpc_params = MPCParams(
            N=20,
            dt=0.1,
            Q=np.diag([10.0, 10.0, 1.0]),
            R=R,
        )
        controller = MPCController(robot_params, mpc_params, enable_soft_constraints=False)
        result = run_mpc_tuning_test(
            controller, reference, initial_state, name=name
        )
        results.append(result)

    return results


def compare_control_rate_weights(
    reference: np.ndarray,
    initial_state: np.ndarray,
) -> List[TuningResult]:
    """
    제어 변화율 가중치(Rd) 변화에 따른 성능 비교.

    Rd가 크면:
      - 장점: 매우 부드러운 제어, 급격한 변화 방지
      - 단점: 느린 응답, 동적 성능 저하

    Rd가 작으면:
      - 장점: 빠른 응답, 동적 성능 향상
      - 단점: 제어 입력 떨림 발생 가능
    """
    print("\n[테스트 5] Control Rate Weights (Rd) 비교")
    print("-" * 60)

    robot_params = RobotParams(max_velocity=1.0, max_omega=2.0)
    Rd_configs = [
        ("No Rd", np.diag([0.0, 0.0])),
        ("Low Rd", np.diag([0.1, 0.1])),
        ("Medium Rd", np.diag([0.5, 0.5])),
        ("High Rd", np.diag([2.0, 2.0])),
    ]
    results = []

    for name, Rd in Rd_configs:
        print(f"  Testing {name}...")
        mpc_params = MPCParams(
            N=20,
            dt=0.1,
            Q=np.diag([10.0, 10.0, 1.0]),
            R=np.diag([0.1, 0.1]),
            Rd=Rd,
        )
        controller = MPCController(robot_params, mpc_params, enable_soft_constraints=False)
        result = run_mpc_tuning_test(
            controller, reference, initial_state, name=name
        )
        results.append(result)

    return results


def plot_tuning_results(
    results: List[TuningResult],
    reference: np.ndarray,
    title: str = "MPC Tuning Comparison",
    save_path: str = None,
) -> plt.Figure:
    """
    파라미터 튜닝 결과 시각화.

    6개 서브플롯:
      1. 궤적 비교
      2. 추적 오차
      3. 선속도
      4. 각속도
      5. 솔버 시간
      6. 비용 함수
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # 1. 궤적 비교
    ax = axes[0, 0]
    ax.plot(reference[:, 0], reference[:, 1], "k--", linewidth=2.5,
            label="Reference", alpha=0.7, zorder=1)
    for result, color in zip(results, colors):
        ax.plot(result.states[:, 0], result.states[:, 1],
                color=color, linewidth=2, label=result.name, alpha=0.8)
    ax.set_xlabel("X [m]", fontsize=11)
    ax.set_ylabel("Y [m]", fontsize=11)
    ax.set_title("Trajectory Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # 2. 추적 오차
    ax = axes[0, 1]
    for result, color in zip(results, colors):
        if result.tracking_errors:
            ax.plot(result.tracking_errors, color=color, linewidth=2,
                   label=result.name, alpha=0.8)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Tracking Error [m]", fontsize=11)
    ax.set_title("Position Tracking Error", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. 선속도
    ax = axes[0, 2]
    for result, color in zip(results, colors):
        if len(result.controls) > 0:
            ax.plot(result.controls[:, 0], color=color, linewidth=2,
                   label=result.name, alpha=0.8)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5,
              label="v_max", alpha=0.5)
    ax.axhline(y=-1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Linear Velocity [m/s]", fontsize=11)
    ax.set_title("Linear Velocity", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. 각속도
    ax = axes[1, 0]
    for result, color in zip(results, colors):
        if len(result.controls) > 0:
            ax.plot(result.controls[:, 1], color=color, linewidth=2,
                   label=result.name, alpha=0.8)
    ax.axhline(y=2.0, color="gray", linestyle="--", linewidth=1.5,
              label="ω_max", alpha=0.5)
    ax.axhline(y=-2.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Angular Velocity [rad/s]", fontsize=11)
    ax.set_title("Angular Velocity", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 5. 솔버 시간
    ax = axes[1, 1]
    for result, color in zip(results, colors):
        if result.solve_times:
            solve_times_ms = [t * 1000 for t in result.solve_times]
            ax.plot(solve_times_ms, color=color, linewidth=2,
                   label=result.name, alpha=0.8)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Solve Time [ms]", fontsize=11)
    ax.set_title("Solver Computation Time", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 6. 비용 함수
    ax = axes[1, 2]
    for result, color in zip(results, colors):
        if result.costs:
            ax.plot(result.costs, color=color, linewidth=2,
                   label=result.name, alpha=0.8)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Cost", fontsize=11)
    ax.set_title("Optimization Cost", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"그림 저장됨: {save_path}")

    return fig


def print_tuning_metrics(results: List[TuningResult]) -> None:
    """튜닝 결과 메트릭 출력."""
    print("\n" + "=" * 90)
    print("                           파라미터 튜닝 결과 비교")
    print("=" * 90)

    headers = ["Metric", *[r.name for r in results]]
    col_width = max(15, max(len(r.name) for r in results) + 2)
    row_format = "{:<30}" + ("{:<" + str(col_width) + "}") * len(results)

    print(row_format.format(*headers))
    print("-" * 90)

    # 평균 추적 오차
    avg_errors = [f"{np.mean(r.tracking_errors):.4f} m" if r.tracking_errors else "N/A"
                  for r in results]
    print(row_format.format("평균 추적 오차", *avg_errors))

    # 최대 추적 오차
    max_errors = [f"{np.max(r.tracking_errors):.4f} m" if r.tracking_errors else "N/A"
                  for r in results]
    print(row_format.format("최대 추적 오차", *max_errors))

    # 제어 부드러움 (낮을수록 좋음)
    smoothness = [f"{r.control_smoothness:.4f}" for r in results]
    print(row_format.format("제어 부드러움 (RMS)", *smoothness))

    # 평균 솔버 시간
    avg_solve = [f"{np.mean(r.solve_times)*1000:.2f} ms" if r.solve_times else "N/A"
                 for r in results]
    print(row_format.format("평균 솔버 시간", *avg_solve))

    # 최대 솔버 시간
    max_solve = [f"{np.max(r.solve_times)*1000:.2f} ms" if r.solve_times else "N/A"
                 for r in results]
    print(row_format.format("최대 솔버 시간", *max_solve))

    # 전체 시간
    total_times = [f"{r.total_time:.3f} s" for r in results]
    print(row_format.format("전체 실행 시간", *total_times))

    # 완료된 스텝 수
    steps = [str(len(r.controls)) for r in results]
    print(row_format.format("완료 스텝 수", *steps))

    # 평균 비용
    avg_costs = [f"{np.mean(r.costs):.2f}" if r.costs else "N/A" for r in results]
    print(row_format.format("평균 비용", *avg_costs))

    print("=" * 90)


def print_tuning_recommendations() -> None:
    """파라미터 튜닝 권장사항 출력."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                        MPC 파라미터 튜닝 권장사항                         ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  1. Prediction Horizon (N):                                               ║
║     - 시작점: N = 20 (일반적인 기본값)                                    ║
║     - 증가: 더 나은 성능, 더 긴 계산 시간                                 ║
║     - 감소: 더 빠른 계산, 성능 저하 가능                                  ║
║     - 권장: 실시간 제약 내에서 가능한 한 크게                             ║
║                                                                           ║
║  2. Time Step (dt):                                                       ║
║     - 시작점: dt = 0.1 s (일반적인 제어 주기)                             ║
║     - N * dt = 예측 시간 범위 (예: N=20, dt=0.1 → 2초 예측)               ║
║     - 권장: 제어 루프 주기와 같거나 약간 크게                             ║
║                                                                           ║
║  3. State Weights (Q):                                                    ║
║     - 시작점: Q = diag([10, 10, 1])                                       ║
║     - 위치(x,y)를 각도(θ)보다 크게 → 위치 추적 우선                       ║
║     - 증가: 더 정확한 추적, 더 공격적인 제어                              ║
║     - 감소: 더 부드러운 제어, 추적 오차 증가                              ║
║                                                                           ║
║  4. Control Weights (R):                                                  ║
║     - 시작점: R = diag([0.1, 0.1])                                        ║
║     - Q/R 비율이 추적 정확도를 결정                                       ║
║     - 증가: 에너지 절약, 부드러운 제어                                    ║
║     - 감소: 빠른 응답, 에너지 소모 증가                                   ║
║                                                                           ║
║  5. Terminal Weights (Qf):                                                ║
║     - 시작점: Qf = Q * 10 (Q보다 크게)                                    ║
║     - 최종 목표 도달 정확도 강조                                          ║
║     - 권장: Q보다 5~10배 크게                                             ║
║                                                                           ║
║  6. Control Rate Weights (Rd):                                            ║
║     - 시작점: Rd = diag([0.5, 0.5])                                       ║
║     - 제어 입력의 부드러움 조절                                           ║
║     - 증가: 매우 부드러운 제어, 느린 응답                                 ║
║     - 감소: 빠른 응답, 떨림 발생 가능                                     ║
║                                                                           ║
║  튜닝 순서:                                                               ║
║    ① N, dt 선택 (계산 시간 고려)                                         ║
║    ② Q, R 튜닝 (추적 vs 에너지 트레이드오프)                             ║
║    ③ Qf 조정 (최종 목표 도달 강조)                                       ║
║    ④ Rd 조정 (제어 부드러움)                                             ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)


def main():
    """메인 함수."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║              MPC Parameter Tuning Guide (파라미터 튜닝 가이드)             ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  다양한 MPC 파라미터 조합을 비교하고, 각 파라미터가 성능에 미치는        ║
║  영향을 분석합니다.                                                       ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    # 테스트 궤적 생성
    reference = generate_test_trajectory("sine", n_points=150)
    initial_state = np.array([0.0, 0.0, 0.0])

    # ========================================================================
    # 테스트 1: Prediction Horizon (N) 비교
    # ========================================================================
    results_N = compare_prediction_horizon(reference, initial_state)
    print_tuning_metrics(results_N)

    fig1 = plot_tuning_results(
        results_N,
        reference,
        title="Parameter Tuning: Prediction Horizon (N) Comparison",
        save_path="tuning_N_comparison.png",
    )

    # ========================================================================
    # 테스트 2: Time Step (dt) 비교
    # ========================================================================
    results_dt = compare_time_step(reference, initial_state)
    print_tuning_metrics(results_dt)

    fig2 = plot_tuning_results(
        results_dt,
        reference,
        title="Parameter Tuning: Time Step (dt) Comparison",
        save_path="tuning_dt_comparison.png",
    )

    # ========================================================================
    # 테스트 3: State Weights (Q) 비교
    # ========================================================================
    results_Q = compare_state_weights(reference, initial_state)
    print_tuning_metrics(results_Q)

    fig3 = plot_tuning_results(
        results_Q,
        reference,
        title="Parameter Tuning: State Weights (Q) Comparison",
        save_path="tuning_Q_comparison.png",
    )

    # ========================================================================
    # 테스트 4: Control Weights (R) 비교
    # ========================================================================
    results_R = compare_control_weights(reference, initial_state)
    print_tuning_metrics(results_R)

    fig4 = plot_tuning_results(
        results_R,
        reference,
        title="Parameter Tuning: Control Weights (R) Comparison",
        save_path="tuning_R_comparison.png",
    )

    # ========================================================================
    # 테스트 5: Control Rate Weights (Rd) 비교
    # ========================================================================
    results_Rd = compare_control_rate_weights(reference, initial_state)
    print_tuning_metrics(results_Rd)

    fig5 = plot_tuning_results(
        results_Rd,
        reference,
        title="Parameter Tuning: Control Rate Weights (Rd) Comparison",
        save_path="tuning_Rd_comparison.png",
    )

    # ========================================================================
    # 튜닝 권장사항 출력
    # ========================================================================
    print_tuning_recommendations()

    print("\n✅ 모든 튜닝 테스트 완료!")
    print("📊 생성된 그래프:")
    print("   - tuning_N_comparison.png")
    print("   - tuning_dt_comparison.png")
    print("   - tuning_Q_comparison.png")
    print("   - tuning_R_comparison.png")
    print("   - tuning_Rd_comparison.png")

    plt.show()


if __name__ == "__main__":
    main()
