#!/usr/bin/env python3
"""
MPC 컨트롤러 성능 벤치마크

다양한 시나리오에서 MPC 컨트롤러의 성능을 측정하고 비교합니다.

┌─────────────────────────────────────────────────────────────────────┐
│                      벤치마크 시나리오                               │
├─────────────────────────────────────────────────────────────────────┤
│ 1. 원형 궤적: 일정한 곡률의 경로 추종 성능                           │
│ 2. S자 궤적: 급격한 방향 전환 성능                                   │
│ 3. 직선 궤적: 직선 주행 및 속도 제어 성능                            │
│ 4. 파라미터 변화: N, Q, R, dt 등의 영향 분석                        │
│ 5. 소프트 제약조건: 제약조건 유무에 따른 성능 비교                   │
└─────────────────────────────────────────────────────────────────────┘

측정 지표:
- Solve Time: 최적화 문제 풀이 시간
- Tracking Error: 경로 추종 오차 (위치 오차)
- Control Effort: 제어 입력 크기 (에너지 효율)
- Smoothness: 제어 입력 변화율 (승차감)
- Success Rate: 목표 도달 성공률
"""

import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from mpc_controller.controllers.mpc import MPCController, MPCParams
from mpc_controller.models.differential_drive import RobotParams
from mpc_controller.models.soft_constraints import SoftConstraintParams
from mpc_controller.utils import (
    generate_circle_trajectory,
    generate_figure_eight_trajectory,
    generate_sinusoidal_trajectory,
    normalize_angle,
)


@dataclass
class BenchmarkResult:
    """벤치마크 결과 저장 구조체."""

    scenario_name: str
    config_name: str

    # 상태 및 제어 입력 히스토리
    states: np.ndarray
    controls: np.ndarray
    reference: np.ndarray

    # 성능 지표
    solve_times: np.ndarray
    tracking_errors: np.ndarray
    control_efforts: np.ndarray
    control_smoothness: np.ndarray

    # 통계
    avg_solve_time: float = 0.0
    max_solve_time: float = 0.0
    std_solve_time: float = 0.0

    avg_tracking_error: float = 0.0
    max_tracking_error: float = 0.0
    std_tracking_error: float = 0.0

    total_control_effort: float = 0.0
    avg_control_smoothness: float = 0.0

    success: bool = True
    total_time: float = 0.0
    num_steps: int = 0

    def __post_init__(self):
        """통계 계산."""
        if len(self.solve_times) > 0:
            self.avg_solve_time = float(np.mean(self.solve_times))
            self.max_solve_time = float(np.max(self.solve_times))
            self.std_solve_time = float(np.std(self.solve_times))

        if len(self.tracking_errors) > 0:
            self.avg_tracking_error = float(np.mean(self.tracking_errors))
            self.max_tracking_error = float(np.max(self.tracking_errors))
            self.std_tracking_error = float(np.std(self.tracking_errors))

        if len(self.control_efforts) > 0:
            self.total_control_effort = float(np.sum(self.control_efforts))

        if len(self.control_smoothness) > 0:
            self.avg_control_smoothness = float(np.mean(self.control_smoothness))

        self.num_steps = len(self.states) if self.states is not None else 0


@dataclass
class BenchmarkConfig:
    """벤치마크 설정."""

    name: str
    mpc_params: MPCParams
    description: str = ""


def generate_benchmark_trajectories() -> Dict[str, np.ndarray]:
    """
    벤치마크용 다양한 참조 궤적 생성.

    Returns:
        시나리오 이름을 키로 하는 궤적 딕셔너리
    """
    trajectories = {}

    # 1. 원형 궤적 (일정한 곡률)
    trajectories["circle"] = generate_circle_trajectory(
        center=np.array([0.0, 0.0]),
        radius=2.0,
        num_points=200
    )

    # 2. S자 궤적 (급격한 방향 전환)
    trajectories["figure8"] = generate_figure_eight_trajectory(
        center=np.array([0.0, 0.0]),
        scale=2.0,
        num_points=200
    )

    # 3. 직선 궤적 (속도 제어)
    t = np.linspace(0, 10, 200)
    x = t
    y = np.zeros_like(t)
    theta = np.zeros_like(t)
    trajectories["straight"] = np.column_stack([x, y, theta])

    # 4. 정현파 궤적 (부드러운 곡선)
    trajectories["sine"] = generate_sinusoidal_trajectory(
        start=np.array([0.0, 0.0]),
        length=10.0,
        amplitude=1.5,
        frequency=0.5,
        num_points=200
    )

    return trajectories


def create_benchmark_configs() -> List[BenchmarkConfig]:
    """
    벤치마크용 다양한 MPC 설정 생성.

    Returns:
        BenchmarkConfig 리스트
    """
    configs = []

    # 1. 기본 설정
    configs.append(BenchmarkConfig(
        name="baseline",
        mpc_params=MPCParams(
            N=20,
            dt=0.1,
            Q=np.diag([10.0, 10.0, 1.0]),
            R=np.diag([0.1, 0.1]),
            Qf=np.diag([100.0, 100.0, 10.0]),
            Rd=np.diag([0.5, 0.5]),
        ),
        description="기본 MPC 설정 (N=20, dt=0.1)"
    ))

    # 2. 짧은 예측 호라이즌
    configs.append(BenchmarkConfig(
        name="short_horizon",
        mpc_params=MPCParams(
            N=10,
            dt=0.1,
            Q=np.diag([10.0, 10.0, 1.0]),
            R=np.diag([0.1, 0.1]),
        ),
        description="짧은 예측 호라이즌 (N=10)"
    ))

    # 3. 긴 예측 호라이즌
    configs.append(BenchmarkConfig(
        name="long_horizon",
        mpc_params=MPCParams(
            N=30,
            dt=0.1,
            Q=np.diag([10.0, 10.0, 1.0]),
            R=np.diag([0.1, 0.1]),
        ),
        description="긴 예측 호라이즌 (N=30)"
    ))

    # 4. 높은 제어 가중치 (부드러운 제어)
    configs.append(BenchmarkConfig(
        name="smooth_control",
        mpc_params=MPCParams(
            N=20,
            dt=0.1,
            Q=np.diag([10.0, 10.0, 1.0]),
            R=np.diag([1.0, 1.0]),  # 높은 R
            Rd=np.diag([2.0, 2.0]),  # 높은 Rd
        ),
        description="부드러운 제어 (높은 R, Rd)"
    ))

    # 5. 높은 추적 정확도
    configs.append(BenchmarkConfig(
        name="high_accuracy",
        mpc_params=MPCParams(
            N=20,
            dt=0.1,
            Q=np.diag([50.0, 50.0, 5.0]),  # 높은 Q
            R=np.diag([0.05, 0.05]),  # 낮은 R
            Qf=np.diag([200.0, 200.0, 20.0]),
        ),
        description="높은 추적 정확도 (높은 Q, 낮은 R)"
    ))

    # 6. 빠른 샘플링
    configs.append(BenchmarkConfig(
        name="fast_sampling",
        mpc_params=MPCParams(
            N=20,
            dt=0.05,  # 빠른 샘플링
            Q=np.diag([10.0, 10.0, 1.0]),
            R=np.diag([0.1, 0.1]),
        ),
        description="빠른 샘플링 (dt=0.05)"
    ))

    # 7. 소프트 제약조건 활성화
    configs.append(BenchmarkConfig(
        name="soft_constraints",
        mpc_params=MPCParams(
            N=20,
            dt=0.1,
            Q=np.diag([10.0, 10.0, 1.0]),
            R=np.diag([0.1, 0.1]),
            soft_constraints=SoftConstraintParams(
                enable_velocity_soft=True,
                enable_acceleration_soft=True,
                velocity_weight=100.0,
                acceleration_weight=100.0,
            ),
        ),
        description="소프트 제약조건 활성화"
    ))

    return configs


def simulate_mpc(
    controller: MPCController,
    reference: np.ndarray,
    initial_state: np.ndarray,
    scenario_name: str,
    config_name: str,
    dt: float = 0.05,
    max_steps: int = 500,
) -> BenchmarkResult:
    """
    MPC 시뮬레이션 실행 및 성능 측정.

    Args:
        controller: MPC 컨트롤러
        reference: 참조 궤적 (N, 3)
        initial_state: 초기 상태 [x, y, theta]
        scenario_name: 시나리오 이름
        config_name: 설정 이름
        dt: 시뮬레이션 타임스텝
        max_steps: 최대 스텝 수

    Returns:
        BenchmarkResult
    """
    # 상태 및 제어 입력 저장
    states = [initial_state.copy()]
    controls = []
    solve_times = []
    tracking_errors = []
    control_efforts = []
    control_smoothness = []

    current_state = initial_state.copy()
    controller.reset()

    start_time = time.perf_counter()
    success = True

    for step in range(max_steps):
        # 가장 가까운 참조점 찾기
        distances = np.sqrt(
            (reference[:, 0] - current_state[0])**2 +
            (reference[:, 1] - current_state[1])**2
        )
        closest_idx = np.argmin(distances)

        # MPC 호라이즌만큼 참조 궤적 추출
        horizon = controller.N + 1
        ref_segment = np.zeros((horizon, 3))

        if closest_idx + horizon <= len(reference):
            ref_segment = reference[closest_idx:closest_idx + horizon]
        else:
            # 끝에 도달하면 마지막 점으로 패딩
            remaining = len(reference) - closest_idx
            ref_segment[:remaining] = reference[closest_idx:]
            ref_segment[remaining:] = reference[-1]

        # 제어 입력 계산
        try:
            control, info = controller.compute_control(current_state, ref_segment)
            solve_times.append(info['solve_time'])
        except Exception as e:
            print(f"  ⚠️  최적화 실패: {e}")
            success = False
            break

        # 추적 오차 계산 (위치 오차)
        error = np.sqrt(
            (current_state[0] - reference[closest_idx, 0])**2 +
            (current_state[1] - reference[closest_idx, 1])**2
        )
        tracking_errors.append(error)

        # 제어 노력 계산 (v^2 + omega^2)
        effort = control[0]**2 + control[1]**2
        control_efforts.append(effort)

        # 제어 평활도 계산 (이전 제어와의 차이)
        if len(controls) > 0:
            smoothness = np.sqrt(
                (control[0] - controls[-1][0])**2 +
                (control[1] - controls[-1][1])**2
            )
            control_smoothness.append(smoothness)

        # 상태 업데이트 (Euler integration)
        v, omega = control
        current_state[0] += v * np.cos(current_state[2]) * dt
        current_state[1] += v * np.sin(current_state[2]) * dt
        current_state[2] += omega * dt
        current_state[2] = normalize_angle(current_state[2])

        states.append(current_state.copy())
        controls.append(control)

        # 목표 도달 확인
        goal_dist = np.sqrt(
            (current_state[0] - reference[-1, 0])**2 +
            (current_state[1] - reference[-1, 1])**2
        )

        if goal_dist < 0.1 and closest_idx > len(reference) - 10:
            break

    total_time = time.perf_counter() - start_time

    # BenchmarkResult 생성
    result = BenchmarkResult(
        scenario_name=scenario_name,
        config_name=config_name,
        states=np.array(states),
        controls=np.array(controls),
        reference=reference,
        solve_times=np.array(solve_times),
        tracking_errors=np.array(tracking_errors),
        control_efforts=np.array(control_efforts),
        control_smoothness=np.array(control_smoothness),
        success=success,
        total_time=total_time,
    )

    return result


def plot_benchmark_summary(
    results: List[BenchmarkResult],
    save_path: str = None
) -> None:
    """
    벤치마크 결과 종합 시각화.

    Args:
        results: 벤치마크 결과 리스트
        save_path: 저장 경로 (None이면 화면에만 표시)
    """
    if not results:
        print("⚠️  시각화할 결과가 없습니다.")
        return

    # 시나리오별 그룹화
    scenarios = {}
    for result in results:
        if result.scenario_name not in scenarios:
            scenarios[result.scenario_name] = []
        scenarios[result.scenario_name].append(result)

    # 각 시나리오마다 그래프 생성
    for scenario_name, scenario_results in scenarios.items():
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # 색상 맵
        colors = plt.cm.tab10(np.linspace(0, 1, len(scenario_results)))

        # 1. 궤적 비교
        ax1 = fig.add_subplot(gs[0, :2])
        ref = scenario_results[0].reference
        ax1.plot(ref[:, 0], ref[:, 1], 'k--', linewidth=2.5, label='Reference', alpha=0.7)

        for i, result in enumerate(scenario_results):
            ax1.plot(result.states[:, 0], result.states[:, 1],
                    color=colors[i], linewidth=1.5, label=result.config_name, alpha=0.8)

        ax1.scatter([ref[0, 0]], [ref[0, 1]], c='green', s=150, marker='o',
                   zorder=5, label='Start', edgecolors='black', linewidths=1.5)
        ax1.scatter([ref[-1, 0]], [ref[-1, 1]], c='red', s=200, marker='*',
                   zorder=5, label='Goal', edgecolors='black', linewidths=1.5)
        ax1.set_xlabel('X [m]', fontsize=11)
        ax1.set_ylabel('Y [m]', fontsize=11)
        ax1.set_title(f'{scenario_name.capitalize()} Trajectory Comparison', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)

        # 2. Solve Time 비교
        ax2 = fig.add_subplot(gs[0, 2])
        config_names = [r.config_name for r in scenario_results]
        solve_time_avg = [r.avg_solve_time * 1000 for r in scenario_results]
        solve_time_std = [r.std_solve_time * 1000 for r in scenario_results]

        bars = ax2.bar(range(len(config_names)), solve_time_avg,
                      yerr=solve_time_std, capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(config_names)))
        ax2.set_xticklabels(config_names, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Avg Solve Time [ms]', fontsize=10)
        ax2.set_title('Solve Time Comparison', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # 값 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        # 3. Tracking Error 비교
        ax3 = fig.add_subplot(gs[0, 3])
        error_avg = [r.avg_tracking_error for r in scenario_results]
        error_max = [r.max_tracking_error for r in scenario_results]

        x_pos = np.arange(len(config_names))
        width = 0.35
        bars1 = ax3.bar(x_pos - width/2, error_avg, width, label='Avg Error',
                       color=colors, alpha=0.7, edgecolor='black')
        bars2 = ax3.bar(x_pos + width/2, error_max, width, label='Max Error',
                       color=colors, alpha=0.5, edgecolor='black', hatch='//')

        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(config_names, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Tracking Error [m]', fontsize=10)
        ax3.set_title('Tracking Error Comparison', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Tracking Error 시간 변화
        ax4 = fig.add_subplot(gs[1, :2])
        for i, result in enumerate(scenario_results):
            time_vec = np.arange(len(result.tracking_errors)) * 0.05
            ax4.plot(time_vec, result.tracking_errors,
                    color=colors[i], linewidth=1.5, label=result.config_name, alpha=0.8)

        ax4.set_xlabel('Time [s]', fontsize=11)
        ax4.set_ylabel('Tracking Error [m]', fontsize=11)
        ax4.set_title('Tracking Error over Time', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9, loc='best')
        ax4.grid(True, alpha=0.3)

        # 5. Solve Time 시간 변화
        ax5 = fig.add_subplot(gs[1, 2:])
        for i, result in enumerate(scenario_results):
            time_vec = np.arange(len(result.solve_times)) * 0.05
            ax5.plot(time_vec, result.solve_times * 1000,
                    color=colors[i], linewidth=1.5, label=result.config_name, alpha=0.8)

        ax5.set_xlabel('Time [s]', fontsize=11)
        ax5.set_ylabel('Solve Time [ms]', fontsize=11)
        ax5.set_title('Solve Time over Time', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9, loc='best')
        ax5.grid(True, alpha=0.3)

        # 6. Control Effort 비교
        ax6 = fig.add_subplot(gs[2, 0])
        effort = [r.total_control_effort for r in scenario_results]
        bars = ax6.bar(range(len(config_names)), effort,
                      color=colors, alpha=0.7, edgecolor='black')
        ax6.set_xticks(range(len(config_names)))
        ax6.set_xticklabels(config_names, rotation=45, ha='right', fontsize=8)
        ax6.set_ylabel('Total Control Effort', fontsize=10)
        ax6.set_title('Control Effort Comparison', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

        # 7. Control Smoothness 비교
        ax7 = fig.add_subplot(gs[2, 1])
        smoothness = [r.avg_control_smoothness for r in scenario_results]
        bars = ax7.bar(range(len(config_names)), smoothness,
                      color=colors, alpha=0.7, edgecolor='black')
        ax7.set_xticks(range(len(config_names)))
        ax7.set_xticklabels(config_names, rotation=45, ha='right', fontsize=8)
        ax7.set_ylabel('Avg Control Smoothness', fontsize=10)
        ax7.set_title('Control Smoothness Comparison', fontsize=11, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')

        # 8. 성능 요약 테이블
        ax8 = fig.add_subplot(gs[2, 2:])
        ax8.axis('off')

        # 테이블 데이터 준비
        table_data = []
        headers = ['Config', 'Solve [ms]', 'Error [m]', 'Effort', 'Steps', 'Time [s]']

        for result in scenario_results:
            row = [
                result.config_name[:15],
                f'{result.avg_solve_time*1000:.2f}',
                f'{result.avg_tracking_error:.4f}',
                f'{result.total_control_effort:.1f}',
                f'{result.num_steps}',
                f'{result.total_time:.2f}'
            ]
            table_data.append(row)

        # 테이블 생성
        table = ax8.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # 헤더 스타일
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # 행 색상 교대
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        # 전체 제목
        fig.suptitle(f'MPC Benchmark: {scenario_name.capitalize()} Scenario',
                    fontsize=16, fontweight='bold', y=0.98)

        # 저장
        if save_path:
            scenario_save_path = save_path.replace('.png', f'_{scenario_name}.png')
            plt.savefig(scenario_save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 그래프 저장: {scenario_save_path}")

    plt.show()


def print_benchmark_table(results: List[BenchmarkResult]) -> None:
    """
    벤치마크 결과를 표 형식으로 출력.

    Args:
        results: 벤치마크 결과 리스트
    """
    if not results:
        print("⚠️  출력할 결과가 없습니다.")
        return

    print("\n" + "="*120)
    print("                                      MPC CONTROLLER BENCHMARK RESULTS")
    print("="*120)

    # 시나리오별 그룹화
    scenarios = {}
    for result in results:
        if result.scenario_name not in scenarios:
            scenarios[result.scenario_name] = []
        scenarios[result.scenario_name].append(result)

    # 각 시나리오별 출력
    for scenario_name, scenario_results in scenarios.items():
        print(f"\n┌─ Scenario: {scenario_name.upper()} " + "─" * (105 - len(scenario_name)))
        print("│")

        # 테이블 헤더
        header = (
            f"│ {'Config':<20} │ "
            f"{'Solve Time [ms]':<20} │ "
            f"{'Tracking Error [m]':<25} │ "
            f"{'Control':<15} │ "
            f"{'Steps':<8} │"
        )
        print(header)
        print("│" + "─" * 117 + "│")

        subheader = (
            f"│ {'':<20} │ "
            f"{'Avg':>8}  {'Max':>8}    │ "
            f"{'Avg':>10}  {'Max':>10}    │ "
            f"{'Effort':>15} │ "
            f"{'':<8} │"
        )
        print(subheader)
        print("│" + "─" * 117 + "│")

        # 각 설정별 결과 출력
        for result in scenario_results:
            status = "✓" if result.success else "✗"
            row = (
                f"│ {status} {result.config_name:<17} │ "
                f"{result.avg_solve_time*1000:8.2f}  {result.max_solve_time*1000:8.2f}    │ "
                f"{result.avg_tracking_error:10.4f}  {result.max_tracking_error:10.4f}    │ "
                f"{result.total_control_effort:15.1f} │ "
                f"{result.num_steps:8d} │"
            )
            print(row)

        print("└" + "─" * 117 + "┘")

        # 각 시나리오별 베스트 설정 찾기
        best_solve = min(scenario_results, key=lambda r: r.avg_solve_time)
        best_error = min(scenario_results, key=lambda r: r.avg_tracking_error)
        best_effort = min(scenario_results, key=lambda r: r.total_control_effort)

        print(f"\n  🏆 Best Solve Time:      {best_solve.config_name:<20} ({best_solve.avg_solve_time*1000:.2f} ms)")
        print(f"  🎯 Best Tracking Error:  {best_error.config_name:<20} ({best_error.avg_tracking_error:.4f} m)")
        print(f"  ⚡ Best Control Effort:  {best_effort.config_name:<20} ({best_effort.total_control_effort:.1f})")

    print("\n" + "="*120 + "\n")


def run_benchmark(
    scenarios: List[str] = None,
    configs: List[str] = None,
    plot: bool = True,
    save_path: str = None,
) -> List[BenchmarkResult]:
    """
    전체 벤치마크 실행.

    Args:
        scenarios: 실행할 시나리오 리스트 (None이면 전체)
        configs: 실행할 설정 리스트 (None이면 전체)
        plot: 그래프 출력 여부
        save_path: 그래프 저장 경로

    Returns:
        벤치마크 결과 리스트
    """
    # 궤적 및 설정 생성
    trajectories = generate_benchmark_trajectories()
    all_configs = create_benchmark_configs()

    # 필터링
    if scenarios:
        trajectories = {k: v for k, v in trajectories.items() if k in scenarios}
    if configs:
        all_configs = [c for c in all_configs if c.name in configs]

    print("\n" + "="*80)
    print("                        MPC CONTROLLER BENCHMARK")
    print("="*80)
    print(f"\n시나리오: {len(trajectories)}개")
    print(f"설정: {len(all_configs)}개")
    print(f"총 실행: {len(trajectories) * len(all_configs)}개\n")

    # ASCII 플로우차트
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│                         Benchmark Flow                              │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print("│  1. Generate Trajectories  →  [Circle, Figure8, Straight, Sine]    │")
    print("│  2. Create MPC Configs     →  [Baseline, Short/Long Horizon, ...]  │")
    print("│  3. Run Simulations        →  Measure Performance                   │")
    print("│  4. Collect Metrics        →  [Solve Time, Error, Effort, ...]     │")
    print("│  5. Analyze & Visualize    →  Tables + Graphs                       │")
    print("└─────────────────────────────────────────────────────────────────────┘\n")

    results = []
    total_runs = len(trajectories) * len(all_configs)
    current_run = 0

    # 모든 조합 실행
    for scenario_name, reference in trajectories.items():
        print(f"\n▶ Scenario: {scenario_name.upper()}")
        print("─" * 80)

        initial_state = reference[0].copy()

        for config in all_configs:
            current_run += 1
            print(f"  [{current_run}/{total_runs}] {config.name:<20} ... ", end='', flush=True)

            # MPC 컨트롤러 생성
            controller = MPCController(
                mpc_params=config.mpc_params,
                enable_soft_constraints=(config.name == "soft_constraints"),
            )

            # 시뮬레이션 실행
            result = simulate_mpc(
                controller=controller,
                reference=reference,
                initial_state=initial_state.copy(),
                scenario_name=scenario_name,
                config_name=config.name,
                dt=0.05,
                max_steps=500,
            )

            results.append(result)

            # 결과 출력
            status = "✓" if result.success else "✗"
            print(
                f"{status}  "
                f"Solve: {result.avg_solve_time*1000:6.2f}ms  "
                f"Error: {result.avg_tracking_error:7.4f}m  "
                f"Steps: {result.num_steps:4d}"
            )

    # 결과 출력
    print_benchmark_table(results)

    # 시각화
    if plot:
        print("📊 그래프 생성 중...\n")
        plot_benchmark_summary(results, save_path)

    return results


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="MPC 컨트롤러 성능 벤치마크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전체 벤치마크 실행
  python mpc_benchmark.py

  # 특정 시나리오만 실행
  python mpc_benchmark.py --scenarios circle figure8

  # 특정 설정만 실행
  python mpc_benchmark.py --configs baseline soft_constraints

  # 결과 그래프 저장
  python mpc_benchmark.py --save benchmark_results.png

  # 그래프 없이 표만 출력
  python mpc_benchmark.py --no-plot
        """
    )

    parser.add_argument(
        '--scenarios',
        nargs='+',
        choices=['circle', 'figure8', 'straight', 'sine'],
        help='실행할 시나리오 선택 (기본: 전체)',
    )

    parser.add_argument(
        '--configs',
        nargs='+',
        choices=[
            'baseline', 'short_horizon', 'long_horizon',
            'smooth_control', 'high_accuracy', 'fast_sampling',
            'soft_constraints'
        ],
        help='실행할 MPC 설정 선택 (기본: 전체)',
    )

    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='결과 그래프 저장 경로',
    )

    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='그래프 출력 생략 (표만 출력)',
    )

    args = parser.parse_args()

    # 벤치마크 실행
    results = run_benchmark(
        scenarios=args.scenarios,
        configs=args.configs,
        plot=not args.no_plot,
        save_path=args.save,
    )

    print("✓ 벤치마크 완료!\n")

    return results


if __name__ == "__main__":
    main()
