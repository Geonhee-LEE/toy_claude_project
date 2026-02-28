#!/usr/bin/env python3
"""Vanilla MPPI vs CBF-MPPI 비교 데모 + 벤치마크.

Vanilla MPPI (soft cost만)와 CBF-MPPI (Hybrid: soft CBF cost + QP safety filter)를
동일 조건에서 실행하고 안전성 + 성능을 정량 비교합니다.

┌──────────────────────────────────────────────────────────────────┐
│  Vanilla MPPI:   soft obstacle cost만 → 안전 보장 없음          │
│                                                                  │
│  CBF-MPPI:       MPPI + CBFCost (soft 유도)                     │
│                      ↓                                           │
│                  CBF Safety Filter (QP)  → u_safe (hard 보장)    │
│                  min ‖u - u_mppi‖²                               │
│                  s.t. ḣ(x,u) + γ·h(x) ≥ 0                      │
└──────────────────────────────────────────────────────────────────┘

4가지 시나리오:
  1. Head-on    : 경로 정면에 장애물
  2. Narrow     : 좁은 통로 통과
  3. Multi      : 다중 장애물 사인 궤적
  4. Dense      : 밀집 장애물 필드

실행:
    python examples/mppi_vs_cbf_mppi_demo.py
    python examples/mppi_vs_cbf_mppi_demo.py --scenario narrow
    python examples/mppi_vs_cbf_mppi_demo.py --scenario multi --live
    python examples/mppi_vs_cbf_mppi_demo.py --benchmark          # 전 시나리오 일괄
    python examples/mppi_vs_cbf_mppi_demo.py --save comparison.png
"""

import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from mpc_controller import (
    DifferentialDriveModel,
    MPPIController,
    MPPIParams,
    RobotParams,
    TrajectoryInterpolator,
    generate_sinusoidal_trajectory,
)
from mpc_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mpc_controller.controllers.mppi.mppi_params import CBFParams
from simulation.simulator import Simulator, SimulationConfig


# ─────────────────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    """단일 컨트롤러 시뮬레이션 결과."""

    name: str
    states: np.ndarray
    controls: np.ndarray
    references: np.ndarray
    tracking_errors: np.ndarray
    solve_times: np.ndarray
    costs: np.ndarray
    time_array: np.ndarray
    total_wall_time: float
    ess_history: np.ndarray
    temp_history: np.ndarray
    safety_violations: int = 0
    collisions: int = 0
    min_barrier_values: np.ndarray = field(default_factory=lambda: np.array([]))
    min_obstacle_distances: np.ndarray = field(default_factory=lambda: np.array([]))
    info_history: Optional[List[dict]] = None

    @property
    def position_rmse(self) -> float:
        pos_err = self.tracking_errors[:, :2]
        return float(np.sqrt(np.mean(np.sum(pos_err ** 2, axis=1))))

    @property
    def heading_rmse(self) -> float:
        return float(np.sqrt(np.mean(self.tracking_errors[:, 2] ** 2)))

    @property
    def control_rate(self) -> float:
        if len(self.controls) < 2:
            return 0.0
        du = np.diff(self.controls, axis=0)
        return float(np.sqrt(np.mean(du ** 2)))

    @property
    def jerk(self) -> float:
        """제어 jerk (d²u/dt²) RMS."""
        if len(self.controls) < 3:
            return 0.0
        ddu = np.diff(self.controls, axis=0, n=2)
        return float(np.sqrt(np.mean(ddu ** 2)))

    @property
    def min_surface_distance(self) -> float:
        """최소 장애물 표면 거리 (음수 = 충돌)."""
        if len(self.min_obstacle_distances) == 0:
            return float("inf")
        return float(np.min(self.min_obstacle_distances))

    @property
    def barrier_violation_ratio(self) -> float:
        """Barrier 값이 음수인 비율 (h(x) < 0)."""
        if len(self.min_barrier_values) == 0:
            return 0.0
        return float(np.mean(self.min_barrier_values < 0))

    @property
    def path_length(self) -> float:
        """실제 이동 경로 길이 [m]."""
        if len(self.states) < 2:
            return 0.0
        diffs = np.diff(self.states[:, :2], axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    @property
    def avg_solve_ms(self) -> float:
        return float(np.mean(self.solve_times)) * 1000


# ─────────────────────────────────────────────────────────────
# 장애물 시나리오
# ─────────────────────────────────────────────────────────────

SCENARIO_NAMES = ["head_on", "narrow", "multi", "dense"]


def get_scenario(name: str) -> Dict:
    """시나리오별 장애물/궤적 생성."""
    if name == "head_on":
        trajectory = np.column_stack([
            np.linspace(0, 8, 300),
            np.zeros(300),
            np.zeros(300),
        ])
        obstacles = np.array([
            [4.0, 0.0, 0.3],
        ])
        return {"trajectory": trajectory, "obstacles": obstacles,
                "label": "Head-on Collision"}

    elif name == "narrow":
        trajectory = np.column_stack([
            np.linspace(0, 8, 300),
            np.zeros(300),
            np.zeros(300),
        ])
        obstacles = np.array([
            [4.0, 0.8, 0.3],
            [4.0, -0.8, 0.3],
        ])
        return {"trajectory": trajectory, "obstacles": obstacles,
                "label": "Narrow Passage"}

    elif name == "dense":
        trajectory = np.column_stack([
            np.linspace(0, 10, 400),
            np.zeros(400),
            np.zeros(400),
        ])
        obstacles = np.array([
            [2.0, 0.3, 0.25],
            [3.5, -0.4, 0.3],
            [5.0, 0.2, 0.25],
            [6.5, -0.3, 0.3],
            [8.0, 0.4, 0.25],
            [9.0, -0.2, 0.2],
        ])
        return {"trajectory": trajectory, "obstacles": obstacles,
                "label": "Dense Obstacle Field"}

    else:  # multi
        trajectory = generate_sinusoidal_trajectory(
            start=np.array([0.0, 0.0]),
            length=10.0,
            amplitude=1.5,
            frequency=0.3,
            num_points=400,
        )
        obstacles = np.array([
            [2.0, 0.5, 0.3],
            [4.0, -0.3, 0.4],
            [6.0, 0.8, 0.3],
            [8.0, -0.5, 0.35],
        ])
        return {"trajectory": trajectory, "obstacles": obstacles,
                "label": "Multi-obstacle Field"}


# ─────────────────────────────────────────────────────────────
# 안전 메트릭 계산
# ─────────────────────────────────────────────────────────────

def compute_min_surface_distance(
    state: np.ndarray,
    obstacles: np.ndarray,
    robot_radius: float = 0.2,
) -> float:
    """최소 장애물 표면 거리 (음수 = 관통)."""
    if obstacles is None or len(obstacles) == 0:
        return float("inf")
    dists = np.linalg.norm(state[:2] - obstacles[:, :2], axis=1) - obstacles[:, 2] - robot_radius
    return float(np.min(dists))


def check_collision(
    state: np.ndarray,
    obstacles: np.ndarray,
    robot_radius: float = 0.2,
) -> bool:
    """로봇이 장애물과 물리적으로 충돌했는지 확인."""
    if obstacles is None or len(obstacles) == 0:
        return False
    dists = np.linalg.norm(state[:2] - obstacles[:, :2], axis=1) - obstacles[:, 2] - robot_radius
    return bool(np.any(dists < 0))


def check_safety_violation(
    state: np.ndarray,
    obstacles: np.ndarray,
    robot_radius: float = 0.2,
    safety_margin: float = 0.3,
) -> bool:
    """로봇이 장애물 안전 영역 내에 있는지 확인 (safety_margin 포함)."""
    if obstacles is None or len(obstacles) == 0:
        return False
    dists = np.linalg.norm(state[:2] - obstacles[:, :2], axis=1) - obstacles[:, 2] - robot_radius - safety_margin
    return bool(np.any(dists < 0))


def compute_min_barrier(
    state: np.ndarray,
    obstacles: np.ndarray,
    robot_radius: float = 0.2,
    safety_margin: float = 0.3,
) -> float:
    """모든 장애물에 대한 최소 barrier 값 계산 h(x) = d² - d_safe²."""
    min_h = float("inf")
    for obs in obstacles:
        dx = state[0] - obs[0]
        dy = state[1] - obs[1]
        d_safe = obs[2] + robot_radius + safety_margin
        h = dx ** 2 + dy ** 2 - d_safe ** 2
        min_h = min(min_h, h)
    return min_h


# ─────────────────────────────────────────────────────────────
# 컨트롤러 생성
# ─────────────────────────────────────────────────────────────

def create_controllers(
    robot_params: RobotParams,
    obstacles: np.ndarray,
    K: int = 512,
    seed: int = 42,
) -> Tuple:
    """Vanilla MPPI + CBF-MPPI 컨트롤러 쌍 생성."""
    mppi_params = MPPIParams(
        N=20, K=K, dt=0.05, lambda_=10.0,
        noise_sigma=np.array([0.3, 0.3]),
        Q=np.diag([10.0, 10.0, 1.0]),
        R=np.diag([0.01, 0.01]),
        Qf=np.diag([100.0, 100.0, 10.0]),
    )

    vanilla = MPPIController(
        robot_params=robot_params,
        mppi_params=mppi_params,
        seed=seed,
        obstacles=obstacles,
    )

    cbf_params = CBFParams(
        enabled=True,
        gamma=1.0,
        safety_margin=0.3,
        robot_radius=0.2,
        activation_distance=3.0,
        cost_weight=500.0,
        use_safety_filter=True,
    )
    cbf = CBFMPPIController(
        robot_params=robot_params,
        mppi_params=mppi_params,
        seed=seed,
        obstacles=obstacles,
        cbf_params=cbf_params,
    )

    return vanilla, cbf, cbf_params


# ─────────────────────────────────────────────────────────────
# 시뮬레이션 루프
# ─────────────────────────────────────────────────────────────

def simulate(
    controller,
    name: str,
    interpolator: TrajectoryInterpolator,
    initial_state: np.ndarray,
    sim_config: SimulationConfig,
    robot_params: RobotParams,
    obstacles: np.ndarray,
    store_info: bool = False,
) -> RunResult:
    """시뮬레이션 루프 + 안전 메트릭 수집."""
    sim = Simulator(robot_params, sim_config)
    sim.reset(initial_state)
    controller.reset()

    num_steps = int(sim_config.max_time / sim_config.dt)

    states = [initial_state.copy()]
    controls_list: List[np.ndarray] = []
    references_list: List[np.ndarray] = []
    errors_list: List[np.ndarray] = []
    solve_times_list: List[float] = []
    costs_list: List[float] = []
    time_list: List[float] = []
    ess_list: List[float] = []
    temp_list: List[float] = []
    barrier_list: List[float] = []
    surface_dist_list: List[float] = []
    info_list: Optional[List[dict]] = [] if store_info else None

    safety_violations = 0
    collisions = 0

    wall_start = time.perf_counter()

    for step in range(num_steps):
        t = step * sim_config.dt
        state = sim.get_measurement()

        ref = interpolator.get_reference(
            t, controller.params.N, controller.params.dt,
            current_theta=state[2],
        )

        control, info = controller.compute_control(state, ref)
        next_state = sim.step(control)
        error = sim.compute_tracking_error(state, ref[0])

        # 안전 메트릭
        if check_safety_violation(next_state, obstacles):
            safety_violations += 1
        if check_collision(next_state, obstacles):
            collisions += 1

        min_h = compute_min_barrier(state, obstacles)
        min_d = compute_min_surface_distance(state, obstacles)

        time_list.append(t)
        states.append(next_state.copy())
        controls_list.append(control.copy())
        references_list.append(ref[0].copy())
        errors_list.append(error)
        solve_times_list.append(info["solve_time"])
        costs_list.append(info["cost"])
        ess_list.append(info.get("ess", 0))
        temp_list.append(info.get("temperature", controller.params.lambda_))
        barrier_list.append(min_h)
        surface_dist_list.append(min_d)

        if store_info:
            info_list.append(info)

        idx, dist = interpolator.find_closest_point(state[:2])
        if idx >= interpolator.num_points - 1 and dist < 0.1:
            break

    total_wall_time = time.perf_counter() - wall_start

    return RunResult(
        name=name,
        states=np.array(states),
        controls=np.array(controls_list),
        references=np.array(references_list),
        tracking_errors=np.array(errors_list),
        solve_times=np.array(solve_times_list),
        costs=np.array(costs_list),
        time_array=np.array(time_list),
        total_wall_time=total_wall_time,
        ess_history=np.array(ess_list),
        temp_history=np.array(temp_list),
        safety_violations=safety_violations,
        collisions=collisions,
        min_barrier_values=np.array(barrier_list),
        min_obstacle_distances=np.array(surface_dist_list),
        info_history=info_list,
    )


# ─────────────────────────────────────────────────────────────
# ASCII 요약표
# ─────────────────────────────────────────────────────────────

def print_summary(vanilla: RunResult, cbf: RunResult, scenario_label: str) -> None:
    """단일 시나리오 비교 요약표."""
    w = 70
    cw = 15

    def row(label, v_val, c_val, fmt=".4f"):
        v_str = f"{v_val:{cw}{fmt}}" if isinstance(v_val, float) else f"{v_val:>{cw}}"
        c_str = f"{c_val:{cw}{fmt}}" if isinstance(c_val, float) else f"{c_val:>{cw}}"
        return f"| {label:<36s}{v_str}{c_str} |"

    lines = [
        "",
        "+" + "=" * w + "+",
        f"|{'Vanilla MPPI vs CBF-MPPI: ' + scenario_label:^{w}s}|",
        "+" + "=" * w + "+",
        f"| {'Metric':<36s}{'Vanilla':>{cw}s}{'CBF-MPPI':>{cw}s} |",
        "+" + "-" * w + "+",
        f"|{'── Tracking Performance ──':^{w}s}|",
        "+" + "-" * w + "+",
        row("Position RMSE [m]", vanilla.position_rmse, cbf.position_rmse),
        row("Heading RMSE [rad]", vanilla.heading_rmse, cbf.heading_rmse),
        row("Path Length [m]", vanilla.path_length, cbf.path_length),
        "+" + "-" * w + "+",
        f"|{'── Control Quality ──':^{w}s}|",
        "+" + "-" * w + "+",
        row("Control Rate RMS", vanilla.control_rate, cbf.control_rate),
        row("Control Jerk RMS", vanilla.jerk, cbf.jerk),
        row("Avg Solve Time [ms]", vanilla.avg_solve_ms, cbf.avg_solve_ms, ".2f"),
        "+" + "-" * w + "+",
        f"|{'── Safety Metrics ──':^{w}s}|",
        "+" + "-" * w + "+",
        row("Collisions (physical)", vanilla.collisions, cbf.collisions, "d"),
        row("Safety Violations (margin)", vanilla.safety_violations, cbf.safety_violations, "d"),
        row("Min Surface Distance [m]", vanilla.min_surface_distance, cbf.min_surface_distance),
        row("Min Barrier Value h(x)", float(np.min(vanilla.min_barrier_values)) if len(vanilla.min_barrier_values) else float("inf"),
            float(np.min(cbf.min_barrier_values)) if len(cbf.min_barrier_values) else float("inf")),
        row("Barrier Violation Ratio", vanilla.barrier_violation_ratio, cbf.barrier_violation_ratio),
        row("Simulation Steps", len(vanilla.time_array), len(cbf.time_array), "d"),
        "+" + "=" * w + "+",
    ]

    # 판정
    safety_winner = "CBF-MPPI" if cbf.collisions <= vanilla.collisions else "Vanilla"
    tracking_winner = "CBF-MPPI" if cbf.position_rmse < vanilla.position_rmse else "Vanilla"
    smooth_winner = "CBF-MPPI" if cbf.control_rate < vanilla.control_rate else "Vanilla"

    lines += [
        f"|  Safety winner     : {'CBF-MPPI' if cbf.collisions <= vanilla.collisions else 'Vanilla':<{w-23}s}|",
        f"|  Tracking winner   : {tracking_winner:<{w-23}s}|",
        f"|  Smoothness winner : {smooth_winner:<{w-23}s}|",
        "+" + "=" * w + "+",
        "",
    ]

    print("\n".join(lines))


def print_benchmark_summary(all_results: Dict[str, Tuple[RunResult, RunResult]]) -> None:
    """전 시나리오 종합 벤치마크 요약표."""
    scenarios = list(all_results.keys())
    n = len(scenarios)

    w = 26 + 20 * n + 2
    cw = 20

    def header_cols():
        return "".join(f"{s:^{cw}s}" for s in scenarios)

    def row_split(label, get_vanilla, get_cbf, fmt=".4f"):
        v_cols = ""
        c_cols = ""
        for s in scenarios:
            van, cbf = all_results[s]
            v_val = get_vanilla(van)
            c_val = get_cbf(cbf)
            v_str = f"{v_val:{fmt}}" if isinstance(v_val, float) else str(v_val)
            c_str = f"{c_val:{fmt}}" if isinstance(c_val, float) else str(c_val)
            v_cols += f"{v_str:>{cw//2}s}"
            c_cols += f"{c_str:>{cw//2}s}"
        return (
            f"| {label + ' (V)':<26s}" + "".join(
                f"{get_vanilla(all_results[s][0]):{cw}{fmt}}" if isinstance(get_vanilla(all_results[s][0]), float) else f"{get_vanilla(all_results[s][0]):>{cw}}"
                for s in scenarios
            ) + " |",
            f"| {label + ' (C)':<26s}" + "".join(
                f"{get_cbf(all_results[s][1]):{cw}{fmt}}" if isinstance(get_cbf(all_results[s][1]), float) else f"{get_cbf(all_results[s][1]):>{cw}}"
                for s in scenarios
            ) + " |",
        )

    lines = [
        "",
        "+" + "=" * w + "+",
        f"|{'MPPI vs CBF-MPPI : All Scenarios Benchmark':^{w}s}|",
        "+" + "=" * w + "+",
        f"| {'Metric':<26s}" + header_cols() + " |",
        f"| {'(V=Vanilla, C=CBF-MPPI)':<26s}" + "".join(f"{'─' * cw}" for _ in scenarios) + " |",
        "+" + "-" * w + "+",
    ]

    metrics = [
        ("Pos RMSE [m]", lambda r: r.position_rmse, ".4f"),
        ("Collisions", lambda r: r.collisions, "d"),
        ("Safety Violations", lambda r: r.safety_violations, "d"),
        ("Min Surface Dist [m]", lambda r: r.min_surface_distance, ".3f"),
        ("Control Rate", lambda r: r.control_rate, ".4f"),
        ("Avg Solve [ms]", lambda r: r.avg_solve_ms, ".2f"),
    ]

    for label, getter, fmt in metrics:
        for tag, idx in [("V", 0), ("C", 1)]:
            row_str = f"| {label + f' ({tag})':<26s}"
            for s in scenarios:
                val = getter(all_results[s][idx])
                if isinstance(val, float):
                    row_str += f"{val:>{cw}{fmt}}"
                else:
                    row_str += f"{val:>{cw}}"
            row_str += " |"
            lines.append(row_str)
        lines.append("+" + "-" * w + "+")

    # 종합 판정
    lines.append(f"|{'── Summary ──':^{w}s}|")
    lines.append("+" + "-" * w + "+")

    total_collisions_v = sum(all_results[s][0].collisions for s in scenarios)
    total_collisions_c = sum(all_results[s][1].collisions for s in scenarios)
    total_violations_v = sum(all_results[s][0].safety_violations for s in scenarios)
    total_violations_c = sum(all_results[s][1].safety_violations for s in scenarios)
    avg_rmse_v = np.mean([all_results[s][0].position_rmse for s in scenarios])
    avg_rmse_c = np.mean([all_results[s][1].position_rmse for s in scenarios])
    avg_solve_v = np.mean([all_results[s][0].avg_solve_ms for s in scenarios])
    avg_solve_c = np.mean([all_results[s][1].avg_solve_ms for s in scenarios])

    lines += [
        f"| {'Total Collisions':<26s}{'Vanilla: ' + str(total_collisions_v):>{w//2}s}{'CBF-MPPI: ' + str(total_collisions_c):>{w//2 - 26}s} |",
        f"| {'Total Safety Violations':<26s}{'Vanilla: ' + str(total_violations_v):>{w//2}s}{'CBF-MPPI: ' + str(total_violations_c):>{w//2 - 26}s} |",
        f"| {'Avg Position RMSE [m]':<26s}{'Vanilla: ' + f'{avg_rmse_v:.4f}':>{w//2}s}{'CBF-MPPI: ' + f'{avg_rmse_c:.4f}':>{w//2 - 26}s} |",
        f"| {'Avg Solve Time [ms]':<26s}{'Vanilla: ' + f'{avg_solve_v:.2f}':>{w//2}s}{'CBF-MPPI: ' + f'{avg_solve_c:.2f}':>{w//2 - 26}s} |",
        "+" + "=" * w + "+",
    ]

    # CBF 효과 판정
    collision_reduction = total_collisions_v - total_collisions_c
    violation_reduction = total_violations_v - total_violations_c
    rmse_overhead = (avg_rmse_c - avg_rmse_v) / max(avg_rmse_v, 1e-6) * 100
    solve_overhead = (avg_solve_c - avg_solve_v) / max(avg_solve_v, 1e-6) * 100

    lines += [
        "",
        "  CBF-MPPI Effect:",
        f"    Collision reduction  : {collision_reduction:+d} ({total_collisions_v} -> {total_collisions_c})",
        f"    Violation reduction  : {violation_reduction:+d} ({total_violations_v} -> {total_violations_c})",
        f"    RMSE overhead        : {rmse_overhead:+.1f}%",
        f"    Solve time overhead  : {solve_overhead:+.1f}%",
        "",
    ]

    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────
# 시각화 — 3x3 비교 그래프
# ─────────────────────────────────────────────────────────────

def plot_comparison(
    vanilla: RunResult,
    cbf: RunResult,
    reference: np.ndarray,
    obstacles: np.ndarray,
    scenario_label: str,
    save_path: Optional[str] = None,
):
    """3x3 비교 그래프."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    cv, cc = "tab:blue", "tab:red"

    # ── (0,0) 궤적 비교 ──
    ax = axes[0, 0]
    ax.plot(reference[:, 0], reference[:, 1],
            "k--", linewidth=1.5, alpha=0.4, label="Reference")
    ax.plot(vanilla.states[:, 0], vanilla.states[:, 1],
            color=cv, linewidth=1.5, label="Vanilla")
    ax.plot(cbf.states[:, 0], cbf.states[:, 1],
            color=cc, linewidth=1.5, label="CBF-MPPI")
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2],
                             color="gray", alpha=0.6)
        safety = plt.Circle((obs[0], obs[1]), obs[2] + 0.5,
                             color="orange", alpha=0.1, linestyle="--", fill=True)
        ax.add_patch(safety)
        ax.add_patch(circle)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Trajectory Comparison")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # ── (0,1) 위치 오차 ──
    ax = axes[0, 1]
    v_err = np.linalg.norm(vanilla.tracking_errors[:, :2], axis=1)
    c_err = np.linalg.norm(cbf.tracking_errors[:, :2], axis=1)
    ax.plot(vanilla.time_array, v_err, color=cv, linewidth=1, label="Vanilla")
    ax.plot(cbf.time_array, c_err, color=cc, linewidth=1, label="CBF-MPPI")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [m]")
    ax.set_title("Position Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (0,2) Barrier 값 ──
    ax = axes[0, 2]
    ax.plot(vanilla.time_array, vanilla.min_barrier_values,
            color=cv, linewidth=1, label="Vanilla")
    ax.plot(cbf.time_array, cbf.min_barrier_values,
            color=cc, linewidth=1, label="CBF-MPPI")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Safety boundary (h=0)")
    ax.fill_between(vanilla.time_array, ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -1, 0,
                     alpha=0.05, color="red")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("min h(x)")
    ax.set_title("Minimum Barrier Value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (1,0) 장애물 표면 거리 ──
    ax = axes[1, 0]
    ax.plot(vanilla.time_array, vanilla.min_obstacle_distances,
            color=cv, linewidth=1, label="Vanilla")
    ax.plot(cbf.time_array, cbf.min_obstacle_distances,
            color=cc, linewidth=1, label="CBF-MPPI")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Collision boundary")
    ax.axhline(y=0.3, color="orange", linestyle=":", alpha=0.5, label="Safety margin")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Min Surface Distance [m]")
    ax.set_title("Obstacle Distance")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── (1,1) 선속도 ──
    ax = axes[1, 1]
    ax.plot(vanilla.time_array[:len(vanilla.controls)], vanilla.controls[:, 0],
            color=cv, linewidth=1, label="Vanilla")
    ax.plot(cbf.time_array[:len(cbf.controls)], cbf.controls[:, 0],
            color=cc, linewidth=1, label="CBF-MPPI")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("v [m/s]")
    ax.set_title("Linear Velocity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (1,2) 각속도 ──
    ax = axes[1, 2]
    ax.plot(vanilla.time_array[:len(vanilla.controls)], vanilla.controls[:, 1],
            color=cv, linewidth=1, label="Vanilla")
    ax.plot(cbf.time_array[:len(cbf.controls)], cbf.controls[:, 1],
            color=cc, linewidth=1, label="CBF-MPPI")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("omega [rad/s]")
    ax.set_title("Angular Velocity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (2,0) 제어 변화율 + Jerk ──
    ax = axes[2, 0]
    if len(vanilla.controls) > 1:
        v_du = np.linalg.norm(np.diff(vanilla.controls, axis=0), axis=1)
        c_du = np.linalg.norm(np.diff(cbf.controls, axis=0), axis=1)
        ax.plot(vanilla.time_array[:len(v_du)], v_du,
                color=cv, linewidth=0.8, alpha=0.7, label="Vanilla |du|")
        ax.plot(cbf.time_array[:len(c_du)], c_du,
                color=cc, linewidth=0.8, alpha=0.7, label="CBF |du|")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("|Δu|")
    ax.set_title("Control Rate (Smoothness)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (2,1) Solve time ──
    ax = axes[2, 1]
    ax.plot(vanilla.time_array, vanilla.solve_times * 1000,
            color=cv, linewidth=1, alpha=0.7, label="Vanilla")
    ax.plot(cbf.time_array, cbf.solve_times * 1000,
            color=cc, linewidth=1, alpha=0.7, label="CBF-MPPI")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Solve Time [ms]")
    ax.set_title("Computation Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (2,2) 종합 바 차트 ──
    ax = axes[2, 2]
    labels = ["RMSE\n[m]", "Ctrl Rate", "Jerk", "Collisions", "Violations"]
    v_vals = [vanilla.position_rmse, vanilla.control_rate, vanilla.jerk,
              vanilla.collisions, vanilla.safety_violations]
    c_vals = [cbf.position_rmse, cbf.control_rate, cbf.jerk,
              cbf.collisions, cbf.safety_violations]

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width/2, v_vals, width, label="Vanilla",
                   color=cv, alpha=0.7, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, c_vals, width, label="CBF-MPPI",
                   color=cc, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Summary Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(f"Vanilla MPPI vs CBF-MPPI: {scenario_label}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────
# Live 리플레이
# ─────────────────────────────────────────────────────────────

def live_replay(
    trajectory: np.ndarray,
    obstacles: np.ndarray,
    initial_state: np.ndarray,
    sim_config: SimulationConfig,
    robot_params: RobotParams,
    scenario_label: str,
    K: int = 256,
):
    """Vanilla vs CBF-MPPI 동시 실시간 리플레이."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    vanilla, cbf, cbf_params = create_controllers(robot_params, obstacles, K=K)

    # 시뮬레이터 2개 (동일 초기 상태)
    sim_v = Simulator(robot_params, sim_config)
    sim_c = Simulator(robot_params, sim_config)
    sim_v.reset(initial_state.copy())
    sim_c.reset(initial_state.copy())
    vanilla.reset()
    cbf.reset()

    interp_v = TrajectoryInterpolator(trajectory, dt=sim_config.dt)
    interp_c = TrajectoryInterpolator(trajectory, dt=sim_config.dt)

    plt.ion()
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1])

    # 상단 좌: 궤적 뷰 (둘 다 오버레이)
    ax_traj = fig.add_subplot(gs[0, 0:2])
    ax_traj.set_title(f"Live: Vanilla vs CBF-MPPI ({scenario_label})", fontsize=13)
    ax_traj.set_xlabel("X [m]")
    ax_traj.set_ylabel("Y [m]")
    ax_traj.grid(True, alpha=0.2)
    ax_traj.set_aspect("equal")

    # 참조 궤적 + 장애물
    ax_traj.plot(trajectory[:, 0], trajectory[:, 1],
                 "k--", lw=1.5, alpha=0.3, label="Reference")
    for obs in obstacles:
        c_obs = plt.Circle((obs[0], obs[1]), obs[2], color="gray", alpha=0.6)
        c_safe = plt.Circle((obs[0], obs[1]), obs[2] + 0.5,
                             color="orange", alpha=0.08, fill=True)
        ax_traj.add_patch(c_safe)
        ax_traj.add_patch(c_obs)

    x_margin, y_margin = 2.0, 2.0
    ax_traj.set_xlim(trajectory[:, 0].min() - x_margin,
                      trajectory[:, 0].max() + x_margin)
    ax_traj.set_ylim(trajectory[:, 1].min() - y_margin,
                      trajectory[:, 1].max() + y_margin)

    # 궤적 라인
    cv, cc = "tab:blue", "tab:red"
    (trace_v,) = ax_traj.plot([], [], color=cv, lw=2.0, alpha=0.8, label="Vanilla")
    (trace_c,) = ax_traj.plot([], [], color=cc, lw=2.0, alpha=0.8, label="CBF-MPPI")

    # 예측 궤적 라인
    (pred_v,) = ax_traj.plot([], [], color=cv, lw=1.5, alpha=0.4, linestyle=":")
    (pred_c,) = ax_traj.plot([], [], color=cc, lw=1.5, alpha=0.4, linestyle=":")

    # 샘플 궤적 (Vanilla/CBF 각각)
    MAX_SAMPLES = 15
    sample_lines_v = [ax_traj.plot([], [], "-", color=cv, alpha=0.05, lw=0.3)[0]
                      for _ in range(MAX_SAMPLES)]
    sample_lines_c = [ax_traj.plot([], [], "-", color=cc, alpha=0.05, lw=0.3)[0]
                      for _ in range(MAX_SAMPLES)]

    # 로봇 패치 (2개)
    robot_v = mpatches.FancyBboxPatch(
        (0, 0), 0.3, 0.2, boxstyle="round,pad=0.02",
        facecolor=cv, edgecolor="black", lw=1.5, alpha=0.9,
    )
    robot_c = mpatches.FancyBboxPatch(
        (0, 0), 0.3, 0.2, boxstyle="round,pad=0.02",
        facecolor=cc, edgecolor="black", lw=1.5, alpha=0.9,
    )
    ax_traj.add_patch(robot_v)
    ax_traj.add_patch(robot_c)
    (dir_v,) = ax_traj.plot([], [], color=cv, lw=2)
    (dir_c,) = ax_traj.plot([], [], color=cc, lw=2)

    ax_traj.legend(fontsize=8, ncol=2, loc="upper right")

    # 상단 우: 정보 텍스트
    ax_info = fig.add_subplot(gs[0, 2])
    ax_info.axis("off")
    info_text = ax_info.text(
        0.05, 0.95, "", transform=ax_info.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
    )

    # 하단: 실시간 메트릭
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_dist = fig.add_subplot(gs[1, 1])
    ax_speed = fig.add_subplot(gs[1, 2])

    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # 시뮬레이션 루프
    num_steps = int(sim_config.max_time / sim_config.dt)
    trace_vx, trace_vy = [], []
    trace_cx, trace_cy = [], []
    v_violations, c_violations = 0, 0
    v_collisions, c_collisions = 0, 0
    v_errors, c_errors = [], []
    v_solve_times, c_solve_times = [], []
    v_barrier_hist, c_barrier_hist = [], []
    v_dist_hist, c_dist_hist = [], []
    time_hist = []

    for step in range(num_steps):
        t = step * sim_config.dt
        state_v = sim_v.get_measurement()
        state_c = sim_c.get_measurement()

        ref_v = interp_v.get_reference(
            t, vanilla.params.N, vanilla.params.dt, current_theta=state_v[2])
        ref_c = interp_c.get_reference(
            t, cbf.params.N, cbf.params.dt, current_theta=state_c[2])

        ctrl_v, info_v = vanilla.compute_control(state_v, ref_v)
        ctrl_c, info_c = cbf.compute_control(state_c, ref_c)

        next_v = sim_v.step(ctrl_v)
        next_c = sim_c.step(ctrl_c)

        # 메트릭 수집
        if check_safety_violation(next_v, obstacles):
            v_violations += 1
        if check_safety_violation(next_c, obstacles):
            c_violations += 1
        if check_collision(next_v, obstacles):
            v_collisions += 1
        if check_collision(next_c, obstacles):
            c_collisions += 1

        err_v = np.linalg.norm(state_v[:2] - ref_v[0, :2])
        err_c = np.linalg.norm(state_c[:2] - ref_c[0, :2])
        v_errors.append(err_v)
        c_errors.append(err_c)
        v_solve_times.append(info_v["solve_time"])
        c_solve_times.append(info_c["solve_time"])
        time_hist.append(t)

        h_v = compute_min_barrier(state_v, obstacles)
        h_c = compute_min_barrier(state_c, obstacles)
        v_barrier_hist.append(h_v)
        c_barrier_hist.append(h_c)

        d_v = compute_min_surface_distance(state_v, obstacles)
        d_c = compute_min_surface_distance(state_c, obstacles)
        v_dist_hist.append(d_v)
        c_dist_hist.append(d_c)

        trace_vx.append(state_v[0])
        trace_vy.append(state_v[1])
        trace_cx.append(state_c[0])
        trace_cy.append(state_c[1])

        # 시각화 (매 2스텝)
        if step % 2 == 0:
            trace_v.set_data(trace_vx, trace_vy)
            trace_c.set_data(trace_cx, trace_cy)

            # 예측 궤적
            pred = info_v.get("predicted_trajectory")
            if pred is not None:
                pred_v.set_data(pred[:, 0], pred[:, 1])
            pred = info_c.get("predicted_trajectory")
            if pred is not None:
                pred_c.set_data(pred[:, 0], pred[:, 1])

            # 샘플 궤적
            for lines, info, color in [(sample_lines_v, info_v, cv),
                                        (sample_lines_c, info_c, cc)]:
                st = info.get("sample_trajectories")
                sw = info.get("sample_weights")
                if st is not None and sw is not None:
                    top_idx = np.argsort(sw)[-MAX_SAMPLES:]
                    max_w = max(np.max(sw), 1e-10)
                    for rank, idx in enumerate(top_idx):
                        alpha = float(np.clip(sw[idx] / max_w * 0.3, 0.02, 0.3))
                        lines[rank].set_data(st[idx, :, 0], st[idx, :, 1])
                        lines[rank].set_alpha(alpha)
                        lines[rank].set_color(color)

            # 로봇 위치
            for state, robot_patch, dir_line in [
                (state_v, robot_v, dir_v),
                (state_c, robot_c, dir_c),
            ]:
                x, y, theta = state[0], state[1], state[2]
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                cx = x - (0.15 * cos_t - 0.1 * sin_t)
                cy = y - (0.15 * sin_t + 0.1 * cos_t)
                robot_patch.set_xy((cx, cy))
                # FancyBboxPatch doesn't have .angle, use transform instead
                import matplotlib.transforms as mtransforms
                tr = mtransforms.Affine2D().rotate_around(x, y, theta) + ax_traj.transData
                robot_patch.set_transform(tr)
                dir_line.set_data([x, x + 0.24 * cos_t], [y, y + 0.24 * sin_t])

            # 정보 텍스트
            info_str = (
                f"━━ Step {step}/{num_steps}  t={t:.1f}s ━━\n\n"
                f"  Vanilla MPPI:\n"
                f"    pos=({state_v[0]:+.2f}, {state_v[1]:+.2f})\n"
                f"    v={ctrl_v[0]:+.3f}  w={ctrl_v[1]:+.3f}\n"
                f"    err={err_v:.3f}m  solve={info_v['solve_time']*1000:.1f}ms\n"
                f"    collisions={v_collisions}  violations={v_violations}\n\n"
                f"  CBF-MPPI:\n"
                f"    pos=({state_c[0]:+.2f}, {state_c[1]:+.2f})\n"
                f"    v={ctrl_c[0]:+.3f}  w={ctrl_c[1]:+.3f}\n"
                f"    err={err_c:.3f}m  solve={info_c['solve_time']*1000:.1f}ms\n"
                f"    collisions={c_collisions}  violations={c_violations}\n\n"
                f"  Safety:\n"
                f"    barrier(V)={h_v:+.3f}  dist(V)={d_v:.3f}m\n"
                f"    barrier(C)={h_c:+.3f}  dist(C)={d_c:.3f}m\n"
            )
            info_text.set_text(info_str)

            # 하단 그래프 업데이트
            ax_bar.cla()
            ax_bar.set_title("Barrier h(x)", fontsize=9)
            ax_bar.plot(time_hist, v_barrier_hist, color=cv, lw=0.8, label="Vanilla")
            ax_bar.plot(time_hist, c_barrier_hist, color=cc, lw=0.8, label="CBF")
            ax_bar.axhline(y=0, color="red", linestyle="--", alpha=0.4)
            ax_bar.legend(fontsize=7)
            ax_bar.grid(True, alpha=0.2)

            ax_dist.cla()
            ax_dist.set_title("Surface Distance [m]", fontsize=9)
            ax_dist.plot(time_hist, v_dist_hist, color=cv, lw=0.8, label="Vanilla")
            ax_dist.plot(time_hist, c_dist_hist, color=cc, lw=0.8, label="CBF")
            ax_dist.axhline(y=0, color="red", linestyle="--", alpha=0.4)
            ax_dist.axhline(y=0.3, color="orange", linestyle=":", alpha=0.4)
            ax_dist.legend(fontsize=7)
            ax_dist.grid(True, alpha=0.2)

            ax_speed.cla()
            ax_speed.set_title("Position Error [m]", fontsize=9)
            ax_speed.plot(time_hist, v_errors, color=cv, lw=0.8, label="Vanilla")
            ax_speed.plot(time_hist, c_errors, color=cc, lw=0.8, label="CBF")
            ax_speed.legend(fontsize=7)
            ax_speed.grid(True, alpha=0.2)

            fig.canvas.draw()
            fig.canvas.flush_events()

        # 종료 확인 (둘 다 끝나면)
        idx_v, dist_v = interp_v.find_closest_point(state_v[:2])
        idx_c, dist_c = interp_c.find_closest_point(state_c[:2])
        done_v = idx_v >= interp_v.num_points - 1 and dist_v < 0.1
        done_c = idx_c >= interp_c.num_points - 1 and dist_c < 0.1
        if done_v and done_c:
            break

    info_text.set_text(
        f"━━ COMPLETE ━━\n\n"
        f"Vanilla: {v_collisions} collisions, {v_violations} violations\n"
        f"CBF:     {c_collisions} collisions, {c_violations} violations\n\n"
        f"Close window to exit."
    )
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.ioff()
    plt.show()


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vanilla MPPI vs CBF-MPPI 비교 데모 + 벤치마크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
시나리오:
  head_on  — 경로 정면 단일 장애물 (가장 단순)
  narrow   — 좁은 통로 (양측 장애물)
  multi    — 사인 궤적 + 4개 장애물
  dense    — 직선 경로 + 6개 밀집 장애물

모드:
  (기본)     — 단일 시나리오 배치 실행 + 그래프
  --live     — 실시간 리플레이 (Vanilla vs CBF 동시)
  --benchmark — 전 시나리오 일괄 비교 리포트
""",
    )
    parser.add_argument(
        "--scenario", type=str, default="head_on",
        choices=SCENARIO_NAMES,
        help="시나리오 (기본: head_on)",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="전 시나리오 일괄 벤치마크 모드",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="비교 그래프 저장 경로",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="시각화 건너뛰기 (콘솔 요약만)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="실시간 리플레이 모드",
    )
    parser.add_argument(
        "--K", type=int, default=512,
        help="MPPI 샘플 수 (기본: 512)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 68)
    print("     Vanilla MPPI vs CBF-MPPI Safety Comparison Demo")
    print("=" * 68)

    robot_params = RobotParams(max_velocity=1.0, max_omega=1.5)
    sim_config = SimulationConfig(dt=0.05, max_time=20.0)

    # ── Benchmark 모드: 전 시나리오 ──
    if args.benchmark:
        scenarios = SCENARIO_NAMES
        print(f"\n  Mode: BENCHMARK ({len(scenarios)} scenarios)")
        print(f"  K={args.K}")

        all_results: Dict[str, Tuple[RunResult, RunResult]] = {}

        for si, scenario_name in enumerate(scenarios):
            scenario = get_scenario(scenario_name)
            trajectory = scenario["trajectory"]
            obstacles = scenario["obstacles"]
            label = scenario["label"]
            initial_state = trajectory[0].copy()

            print(f"\n  [{si+1}/{len(scenarios)}] {label} ({len(obstacles)} obstacles)")

            vanilla, cbf_ctrl, _ = create_controllers(
                robot_params, obstacles, K=args.K)

            # Vanilla
            print(f"    Running Vanilla MPPI ...")
            interp_v = TrajectoryInterpolator(trajectory, dt=sim_config.dt)
            v_result = simulate(
                vanilla, "Vanilla", interp_v,
                initial_state.copy(), sim_config, robot_params, obstacles,
            )
            print(f"      RMSE={v_result.position_rmse:.4f}m  "
                  f"collisions={v_result.collisions}  violations={v_result.safety_violations}")

            # CBF-MPPI
            print(f"    Running CBF-MPPI ...")
            interp_c = TrajectoryInterpolator(trajectory, dt=sim_config.dt)
            c_result = simulate(
                cbf_ctrl, "CBF-MPPI", interp_c,
                initial_state.copy(), sim_config, robot_params, obstacles,
            )
            print(f"      RMSE={c_result.position_rmse:.4f}m  "
                  f"collisions={c_result.collisions}  violations={c_result.safety_violations}")

            # 개별 시나리오 요약
            print_summary(v_result, c_result, label)

            all_results[label] = (v_result, c_result)

        # 종합 리포트
        print_benchmark_summary(all_results)

        # 시나리오별 그래프 (2x2 subplot)
        if not args.no_plot:
            import matplotlib.pyplot as plt
            fig, axes_grid = plt.subplots(2, 2, figsize=(18, 14))
            axes_flat = axes_grid.flatten()

            for idx, (label, (v_r, c_r)) in enumerate(all_results.items()):
                if idx >= 4:
                    break
                ax = axes_flat[idx]
                scenario = [s for s in SCENARIO_NAMES
                            if get_scenario(s)["label"] == label][0]
                obs = get_scenario(scenario)["obstacles"]
                traj = get_scenario(scenario)["trajectory"]

                ax.plot(traj[:, 0], traj[:, 1], "k--", lw=1, alpha=0.3, label="Ref")
                ax.plot(v_r.states[:, 0], v_r.states[:, 1],
                        color="tab:blue", lw=1.5, label="Vanilla")
                ax.plot(c_r.states[:, 0], c_r.states[:, 1],
                        color="tab:red", lw=1.5, label="CBF-MPPI")
                for o in obs:
                    circle = plt.Circle((o[0], o[1]), o[2], color="gray", alpha=0.6)
                    safety = plt.Circle((o[0], o[1]), o[2] + 0.5,
                                         color="orange", alpha=0.08, fill=True)
                    ax.add_patch(safety)
                    ax.add_patch(circle)
                ax.set_title(f"{label}\nV: col={v_r.collisions} rmse={v_r.position_rmse:.3f}  "
                             f"C: col={c_r.collisions} rmse={c_r.position_rmse:.3f}",
                             fontsize=9)
                ax.set_aspect("equal")
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.2)

            fig.suptitle("MPPI vs CBF-MPPI: All Scenarios", fontsize=14, fontweight="bold")
            plt.tight_layout()
            if args.save:
                plt.savefig(args.save, dpi=150, bbox_inches="tight")
                print(f"  Figure saved: {args.save}")
            plt.show()

        return

    # ── 단일 시나리오 모드 ──
    scenario = get_scenario(args.scenario)
    trajectory = scenario["trajectory"]
    obstacles = scenario["obstacles"]
    scenario_label = scenario["label"]

    print(f"\n  Scenario   : {scenario_label}")
    print(f"  Obstacles  : {len(obstacles)}")
    for i, obs in enumerate(obstacles):
        print(f"    [{i}] pos=({obs[0]:.1f}, {obs[1]:.1f}), r={obs[2]:.2f}")
    print(f"  K          : {args.K}")
    print(f"  Mode       : {'LIVE' if args.live else 'BATCH'}")

    initial_state = trajectory[0].copy()

    # Live 모드
    if args.live:
        live_replay(
            trajectory, obstacles, initial_state,
            sim_config, robot_params, scenario_label,
            K=min(args.K, 256),
        )
        return

    # Batch 모드
    vanilla, cbf_ctrl, cbf_params = create_controllers(
        robot_params, obstacles, K=args.K)

    # 파라미터 요약
    print(f"\n  ┌{'─' * 46}┐")
    print(f"  │{'CBF Parameters':^46s}│")
    print(f"  ├{'─' * 46}┤")
    print(f"  │  gamma={cbf_params.gamma}, margin={cbf_params.safety_margin}m{' ' * 22}│")
    print(f"  │  robot_radius={cbf_params.robot_radius}m, activation={cbf_params.activation_distance}m{' ' * 8}│")
    print(f"  │  cost_weight={cbf_params.cost_weight}, safety_filter=True{' ' * 6}│")
    print(f"  └{'─' * 46}┘")

    print("\n  Running Vanilla MPPI ...")
    vanilla_interp = TrajectoryInterpolator(trajectory, dt=sim_config.dt)
    vanilla_result = simulate(
        vanilla, "Vanilla MPPI", vanilla_interp,
        initial_state.copy(), sim_config, robot_params, obstacles,
    )
    print(f"    -> {len(vanilla_result.time_array)} steps, "
          f"collisions={vanilla_result.collisions}, "
          f"violations={vanilla_result.safety_violations}")

    print("  Running CBF-MPPI ...")
    cbf_interp = TrajectoryInterpolator(trajectory, dt=sim_config.dt)
    cbf_result = simulate(
        cbf_ctrl, "CBF-MPPI", cbf_interp,
        initial_state.copy(), sim_config, robot_params, obstacles,
    )
    print(f"    -> {len(cbf_result.time_array)} steps, "
          f"collisions={cbf_result.collisions}, "
          f"violations={cbf_result.safety_violations}")

    # ASCII 요약
    print_summary(vanilla_result, cbf_result, scenario_label)

    # 시각화
    if not args.no_plot:
        import matplotlib.pyplot as plt
        plot_comparison(
            vanilla_result, cbf_result, trajectory, obstacles,
            scenario_label, save_path=args.save,
        )
        plt.show()


if __name__ == "__main__":
    main()
