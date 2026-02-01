#!/usr/bin/env python3
"""MPC vs MPPI 컨트롤러 비교 데모.

전통 MPC (CasADi/IPOPT)와 MPPI (NumPy 샘플링) 컨트롤러를
동일 조건에서 실행하고 성능을 비교합니다.

실행:
    python examples/mpc_vs_mppi_demo.py
    python examples/mpc_vs_mppi_demo.py --trajectory circle
    python examples/mpc_vs_mppi_demo.py --trajectory sine --save comparison.png
    python examples/mpc_vs_mppi_demo.py --live  # dual-panel 실시간 리플레이
"""

import argparse
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mpc_controller import (
    DifferentialDriveModel,
    MPCController,
    MPCParams,
    MPPIController,
    MPPIParams,
    RobotParams,
    TrajectoryInterpolator,
    generate_circle_trajectory,
    generate_figure_eight_trajectory,
    generate_sinusoidal_trajectory,
)
from simulation.simulator import Simulator, SimulationConfig, SimulationResult


# ─────────────────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────────────────

@dataclass
class ComparisonResult:
    """단일 컨트롤러 시뮬레이션 결과."""

    name: str
    states: np.ndarray           # (T+1, 3)
    controls: np.ndarray         # (T, 2)
    references: np.ndarray       # (T, 3)
    tracking_errors: np.ndarray  # (T, 3)
    solve_times: np.ndarray      # (T,)
    costs: np.ndarray            # (T,)
    time_array: np.ndarray       # (T,)
    total_wall_time: float
    ess_history: Optional[np.ndarray] = None  # MPPI 전용
    info_history: Optional[List[dict]] = None  # --live 리플레이용

    @property
    def position_rmse(self) -> float:
        pos_err = self.tracking_errors[:, :2]
        return float(np.sqrt(np.mean(np.sum(pos_err ** 2, axis=1))))

    @property
    def heading_rmse(self) -> float:
        return float(np.sqrt(np.mean(self.tracking_errors[:, 2] ** 2)))

    @property
    def max_position_error(self) -> float:
        return float(np.max(np.linalg.norm(self.tracking_errors[:, :2], axis=1)))

    def to_sim_result(self) -> SimulationResult:
        """simulation.plot_comparison() 호환 SimulationResult 변환."""
        return SimulationResult(
            time=self.time_array,
            states=self.states[: len(self.time_array)],
            controls=self.controls,
            references=self.references,
            predicted_trajectories=[],
            tracking_errors=self.tracking_errors,
        )


# ─────────────────────────────────────────────────────────────
# 시뮬레이션 루프
# ─────────────────────────────────────────────────────────────

def simulate_controller(
    controller,
    name: str,
    interpolator: TrajectoryInterpolator,
    initial_state: np.ndarray,
    sim_config: SimulationConfig,
    robot_params: RobotParams,
    store_info: bool = False,
) -> ComparisonResult:
    """수동 시뮬레이션 루프로 컨트롤러 실행 + 메트릭 수집.

    Args:
        store_info: True이면 per-step info dict를 저장 (--live 리플레이용).
                    MPPI 샘플 궤적은 메모리가 크므로 예측 궤적만 저장.
    """

    sim = Simulator(robot_params, sim_config)
    sim.reset(initial_state)
    controller.reset()

    num_steps = int(sim_config.max_time / sim_config.dt)

    states = [initial_state.copy()]
    controls_list = []
    references_list = []
    errors_list = []
    solve_times_list = []
    costs_list = []
    time_list = []
    ess_list = []
    info_list = [] if store_info else None

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

        time_list.append(t)
        states.append(next_state.copy())
        controls_list.append(control.copy())
        references_list.append(ref[0].copy())
        errors_list.append(error)
        solve_times_list.append(info["solve_time"])
        costs_list.append(info["cost"])

        if "ess" in info:
            ess_list.append(info["ess"])

        if store_info:
            # 리플레이에 필요한 키만 경량 복사
            compact = {
                "predicted_trajectory": info["predicted_trajectory"].copy(),
                "cost": info["cost"],
                "solve_time": info["solve_time"],
            }
            if "ess" in info:
                compact["ess"] = info["ess"]
                compact["temperature"] = info.get("temperature", 0)
            if "solver_status" in info:
                compact["solver_status"] = info["solver_status"]
            info_list.append(compact)

        # 궤적 끝 도달 검사
        idx, dist = interpolator.find_closest_point(state[:2])
        if idx >= interpolator.num_points - 1 and dist < 0.1:
            break

    wall_time = time.perf_counter() - wall_start

    return ComparisonResult(
        name=name,
        states=np.array(states),
        controls=np.array(controls_list),
        references=np.array(references_list),
        tracking_errors=np.array(errors_list),
        solve_times=np.array(solve_times_list),
        costs=np.array(costs_list),
        time_array=np.array(time_list),
        total_wall_time=wall_time,
        ess_history=np.array(ess_list) if ess_list else None,
        info_history=info_list,
    )


# ─────────────────────────────────────────────────────────────
# ASCII 요약표
# ─────────────────────────────────────────────────────────────

def print_summary(mpc: ComparisonResult, mppi: ComparisonResult) -> None:
    """콘솔에 비교 요약표 출력."""

    avg_solve_mpc = np.mean(mpc.solve_times) * 1000
    avg_solve_mppi = np.mean(mppi.solve_times) * 1000
    max_solve_mpc = np.max(mpc.solve_times) * 1000
    max_solve_mppi = np.max(mppi.solve_times) * 1000

    lines = [
        "",
        "+" + "=" * 59 + "+",
        "|{:^59s}|".format("MPC vs MPPI 성능 비교"),
        "+" + "=" * 59 + "+",
        "| {:<28s}{:>14s}{:>14s} |".format("Metric", "MPC", "MPPI"),
        "+" + "-" * 59 + "+",
        "| {:<28s}{:>14.4f}{:>14.4f} |".format(
            "Position RMSE [m]", mpc.position_rmse, mppi.position_rmse
        ),
        "| {:<28s}{:>14.4f}{:>14.4f} |".format(
            "Max Position Error [m]", mpc.max_position_error, mppi.max_position_error
        ),
        "| {:<28s}{:>14.4f}{:>14.4f} |".format(
            "Heading RMSE [rad]", mpc.heading_rmse, mppi.heading_rmse
        ),
        "| {:<28s}{:>13.3f}{:>14.3f} |".format(
            "Avg Solve Time [ms]", avg_solve_mpc, avg_solve_mppi
        ),
        "| {:<28s}{:>13.3f}{:>14.3f} |".format(
            "Max Solve Time [ms]", max_solve_mpc, max_solve_mppi
        ),
        "| {:<28s}{:>13.3f}{:>14.3f} |".format(
            "Total Wall Time [s]", mpc.total_wall_time, mppi.total_wall_time
        ),
        "| {:<28s}{:>14d}{:>14d} |".format(
            "Simulation Steps", len(mpc.time_array), len(mppi.time_array)
        ),
    ]

    if mppi.ess_history is not None and len(mppi.ess_history) > 0:
        lines.append(
            "| {:<28s}{:>14s}{:>13.1f} |".format(
                "MPPI Avg ESS", "-", np.mean(mppi.ess_history)
            )
        )

    lines.append("+" + "-" * 59 + "+")

    # 승자 요약
    if mpc.position_rmse < mppi.position_rmse:
        acc_winner = "MPC"
    else:
        acc_winner = "MPPI"

    if avg_solve_mpc < avg_solve_mppi:
        speed_winner = "MPC"
    else:
        speed_winner = "MPPI"

    lines.append("| {:<57s} |".format(
        f"  Accuracy winner : {acc_winner}"
    ))
    lines.append("| {:<57s} |".format(
        f"  Speed winner    : {speed_winner}"
    ))
    lines.append("+" + "=" * 59 + "+")
    lines.append("")

    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────

def plot_figure1(mpc: ComparisonResult, mppi: ComparisonResult,
                 save_path: Optional[str] = None) -> plt.Figure:
    """Figure 1: simulation.plot_comparison() 재활용 (3패널)."""
    from simulation.visualizer import plot_comparison

    sim_results = [mpc.to_sim_result(), mppi.to_sim_result()]
    labels = [mpc.name, mppi.name]
    fig = plot_comparison(sim_results, labels,
                          title="MPC vs MPPI — Trajectory Comparison",
                          save_path=save_path)
    return fig


def plot_figure2(mpc: ComparisonResult, mppi: ComparisonResult,
                 save_path: Optional[str] = None) -> plt.Figure:
    """Figure 2: 커스텀 상세 비교 (2x3 = 6패널)."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    colors = {"MPC": "tab:blue", "MPPI": "tab:orange"}

    # ── (0,0) Solve Time 시간별 ──
    ax = axes[0, 0]
    ax.plot(mpc.time_array, mpc.solve_times * 1000,
            color=colors["MPC"], alpha=0.8, linewidth=1, label="MPC")
    ax.plot(mppi.time_array, mppi.solve_times * 1000,
            color=colors["MPPI"], alpha=0.8, linewidth=1, label="MPPI")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Solve Time [ms]")
    ax.set_title("Solve Time Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── (0,1) Solve Time 히스토그램 ──
    ax = axes[0, 1]
    bins = 40
    ax.hist(mpc.solve_times * 1000, bins=bins, alpha=0.6,
            color=colors["MPC"], label="MPC", edgecolor="white")
    ax.hist(mppi.solve_times * 1000, bins=bins, alpha=0.6,
            color=colors["MPPI"], label="MPPI", edgecolor="white")
    ax.set_xlabel("Solve Time [ms]")
    ax.set_ylabel("Count")
    ax.set_title("Solve Time Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── (0,2) Cost 시간별 ──
    ax = axes[0, 2]
    ax.plot(mpc.time_array, mpc.costs,
            color=colors["MPC"], alpha=0.8, linewidth=1, label="MPC")
    ax.plot(mppi.time_array, mppi.costs,
            color=colors["MPPI"], alpha=0.8, linewidth=1, label="MPPI")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cost")
    ax.set_title("Optimization Cost Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── (1,0) Linear Velocity ──
    ax = axes[1, 0]
    t_mpc = mpc.time_array[: len(mpc.controls)]
    t_mppi = mppi.time_array[: len(mppi.controls)]
    ax.plot(t_mpc, mpc.controls[:, 0],
            color=colors["MPC"], alpha=0.8, linewidth=1, label="MPC")
    ax.plot(t_mppi, mppi.controls[:, 0],
            color=colors["MPPI"], alpha=0.8, linewidth=1, label="MPPI")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("v [m/s]")
    ax.set_title("Linear Velocity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── (1,1) Angular Velocity ──
    ax = axes[1, 1]
    ax.plot(t_mpc, mpc.controls[:, 1],
            color=colors["MPC"], alpha=0.8, linewidth=1, label="MPC")
    ax.plot(t_mppi, mppi.controls[:, 1],
            color=colors["MPPI"], alpha=0.8, linewidth=1, label="MPPI")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("omega [rad/s]")
    ax.set_title("Angular Velocity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── (1,2) 요약 텍스트 ──
    ax = axes[1, 2]
    ax.axis("off")

    avg_st_mpc = np.mean(mpc.solve_times) * 1000
    avg_st_mppi = np.mean(mppi.solve_times) * 1000

    ess_line = ""
    if mppi.ess_history is not None and len(mppi.ess_history) > 0:
        ess_line = f"  MPPI Avg ESS      :     -    {np.mean(mppi.ess_history):>8.1f}\n"

    summary = (
        f"  {'Metric':<22s}{'MPC':>9s}{'MPPI':>9s}\n"
        f"  {'─' * 40}\n"
        f"  {'Pos RMSE [m]':<22s}{mpc.position_rmse:>9.4f}{mppi.position_rmse:>9.4f}\n"
        f"  {'Max Pos Err [m]':<22s}{mpc.max_position_error:>9.4f}{mppi.max_position_error:>9.4f}\n"
        f"  {'Heading RMSE [rad]':<22s}{mpc.heading_rmse:>9.4f}{mppi.heading_rmse:>9.4f}\n"
        f"  {'Avg Solve [ms]':<22s}{avg_st_mpc:>9.3f}{avg_st_mppi:>9.3f}\n"
        f"  {'Wall Time [s]':<22s}{mpc.total_wall_time:>9.3f}{mppi.total_wall_time:>9.3f}\n"
        f"{ess_line}"
    )

    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=11, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))
    ax.set_title("Summary Metrics")

    fig.suptitle("MPC vs MPPI — Detailed Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        base, ext = (save_path.rsplit(".", 1) if "." in save_path
                      else (save_path, "png"))
        detail_path = f"{base}_detail.{ext}"
        plt.savefig(detail_path, dpi=150, bbox_inches="tight")
        print(f"  Detail figure saved: {detail_path}")

    return fig


# ─────────────────────────────────────────────────────────────
# Live 비교 리플레이
# ─────────────────────────────────────────────────────────────

def live_comparison_replay(
    mpc: ComparisonResult,
    mppi: ComparisonResult,
    reference: np.ndarray,
    dt: float = 0.05,
    update_interval: int = 2,
) -> None:
    """시뮬레이션 완료 후 MPC/MPPI를 나란히 실시간 리플레이.

    ┌─────────────────┬─────────────────┐
    │  MPC 궤적       │  MPPI 궤적      │
    │  + 예측 라인    │  + 예측 라인    │
    │  + 로봇 패치    │  + 로봇 패치    │
    ├─────────────────┴─────────────────┤
    │  실시간 비교 메트릭 패널          │
    └───────────────────────────────────┘
    """
    if mpc.info_history is None or mppi.info_history is None:
        print("  [WARN] info_history 없음 — store_info=True로 시뮬레이션 필요")
        return

    plt.ion()

    fig = plt.figure(figsize=(18, 9), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    ax_mpc = fig.add_subplot(gs[0, 0])
    ax_mppi = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis("off")

    robot_length, robot_width = 0.3, 0.2

    # ── 축 공통 설정 ──
    x_margin, y_margin = 1.0, 1.0
    x_min = reference[:, 0].min() - x_margin
    x_max = reference[:, 0].max() + x_margin
    y_min = reference[:, 1].min() - y_margin
    y_max = reference[:, 1].max() + y_margin

    panels = {}
    for ax, label, color in [(ax_mpc, "MPC", "tab:blue"), (ax_mppi, "MPPI", "tab:orange")]:
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # 참조 궤적
        ax.plot(reference[:, 0], reference[:, 1],
                "k--", linewidth=1.5, alpha=0.4, label="Reference")

        # 실제 궤적 (점진적)
        (trace_line,) = ax.plot([], [], "-", color=color, linewidth=2, label="Actual")

        # 예측 궤적
        (pred_line,) = ax.plot([], [], "g-", alpha=0.6, linewidth=1.5, label="Prediction")

        # 로봇 패치
        robot_patch = patches.Rectangle(
            (0, 0), robot_length, robot_width,
            angle=0, fill=True, facecolor=color,
            edgecolor="black", linewidth=2, alpha=0.8,
        )
        ax.add_patch(robot_patch)

        # 방향 표시
        (dir_line,) = ax.plot([], [], "k-", linewidth=2)

        ax.legend(loc="upper right", fontsize=8)

        panels[label] = {
            "ax": ax, "trace_line": trace_line, "pred_line": pred_line,
            "robot_patch": robot_patch, "dir_line": dir_line,
        }

    # 정보 텍스트
    info_text = ax_info.text(
        0.5, 0.9, "", transform=ax_info.transAxes,
        fontsize=12, fontfamily="monospace",
        verticalalignment="top", horizontalalignment="center",
    )

    fig.suptitle("MPC vs MPPI — Live Comparison Replay",
                 fontsize=14, fontweight="bold")
    fig.canvas.draw()
    fig.canvas.flush_events()

    # ── 리플레이 루프 ──
    num_steps = min(len(mpc.time_array), len(mppi.time_array))

    trace_data = {"MPC": ([], []), "MPPI": ([], [])}

    for step in range(num_steps):
        for label, result in [("MPC", mpc), ("MPPI", mppi)]:
            if step >= len(result.time_array):
                continue

            p = panels[label]
            state = result.states[step]
            tx, ty = trace_data[label]
            tx.append(state[0])
            ty.append(state[1])

            if step % update_interval != 0:
                continue

            # 궤적 트레이스
            p["trace_line"].set_data(tx, ty)

            # 예측 궤적
            if result.info_history and step < len(result.info_history):
                pred = result.info_history[step].get("predicted_trajectory")
                if pred is not None:
                    p["pred_line"].set_data(pred[:, 0], pred[:, 1])

            # 로봇 위치
            x, y, theta = state
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            cx = x - (robot_length / 2 * cos_t - robot_width / 2 * sin_t)
            cy = y - (robot_length / 2 * sin_t + robot_width / 2 * cos_t)
            p["robot_patch"].set_xy((cx, cy))
            p["robot_patch"].angle = np.degrees(theta)

            dir_len = robot_length * 0.8
            p["dir_line"].set_data([x, x + dir_len * cos_t],
                                   [y, y + dir_len * sin_t])

        if step % update_interval != 0:
            continue

        # 정보 패널
        t = step * dt
        mpc_err = np.linalg.norm(mpc.tracking_errors[min(step, len(mpc.tracking_errors) - 1), :2])
        mppi_err = np.linalg.norm(mppi.tracking_errors[min(step, len(mppi.tracking_errors) - 1), :2])
        mpc_st = mpc.solve_times[min(step, len(mpc.solve_times) - 1)] * 1000
        mppi_st = mppi.solve_times[min(step, len(mppi.solve_times) - 1)] * 1000

        ess_str = ""
        if mppi.ess_history is not None and step < len(mppi.ess_history):
            ess_str = f"   MPPI ESS: {mppi.ess_history[step]:.0f}/512"

        info_str = (
            f"Time: {t:.2f}s  |  "
            f"Pos Error — MPC: {mpc_err:.4f}m  MPPI: {mppi_err:.4f}m  |  "
            f"Solve — MPC: {mpc_st:.2f}ms  MPPI: {mppi_st:.2f}ms"
            f"{ess_str}"
        )
        info_text.set_text(info_str)

        fig.canvas.draw()
        fig.canvas.flush_events()

    # 리플레이 완료 — 최종 상태 유지
    plt.ioff()
    plt.show()


# ─────────────────────────────────────────────────────────────
# 궤적 생성
# ─────────────────────────────────────────────────────────────

def generate_trajectory(traj_type: str) -> np.ndarray:
    """궤적 타입에 따른 참조 궤적 생성."""
    if traj_type == "circle":
        return generate_circle_trajectory(
            center=np.array([0.0, 0.0]),
            radius=2.0,
            num_points=400,
        )
    elif traj_type == "figure8":
        return generate_figure_eight_trajectory(
            center=np.array([0.0, 0.0]),
            scale=2.0,
            num_points=400,
        )
    else:  # sine
        return generate_sinusoidal_trajectory(
            start=np.array([0.0, 0.0]),
            length=10.0,
            amplitude=1.0,
            frequency=0.5,
            num_points=400,
        )


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MPC vs MPPI 비교 데모")
    parser.add_argument(
        "--trajectory", type=str, default="figure8",
        choices=["circle", "figure8", "sine"],
        help="궤적 타입 (기본: figure8)",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="비교 그래프 저장 경로",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="시각화 건너뛰기 (콘솔 요약만 출력)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="시뮬레이션 후 dual-panel 실시간 리플레이 (MPC/MPPI 나란히 비교)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("       MPC vs MPPI Controller Comparison Demo")
    print("=" * 60)

    # ── 공통 파라미터 ──
    robot_params = RobotParams(max_velocity=1.0, max_omega=1.5)
    sim_config = SimulationConfig(dt=0.05, max_time=20.0)

    # ── 궤적 생성 ──
    print(f"\n  Trajectory : {args.trajectory}")
    trajectory = generate_trajectory(args.trajectory)
    initial_state = trajectory[0].copy()

    # ── MPC 컨트롤러 ──
    mpc_params = MPCParams(N=20, dt=0.1)
    mpc_controller = MPCController(
        robot_params=robot_params,
        mpc_params=mpc_params,
    )
    mpc_interpolator = TrajectoryInterpolator(trajectory, dt=sim_config.dt)

    # ── MPPI 컨트롤러 ──
    mppi_params = MPPIParams(
        N=20, K=512, dt=0.05, lambda_=10.0,
        noise_sigma=np.array([0.3, 0.3]),
        Q=np.diag([10.0, 10.0, 1.0]),
        R=np.diag([0.01, 0.01]),
        Qf=np.diag([100.0, 100.0, 10.0]),
    )
    mppi_controller = MPPIController(
        robot_params=robot_params,
        mppi_params=mppi_params,
        seed=42,
    )
    mppi_interpolator = TrajectoryInterpolator(trajectory, dt=sim_config.dt)

    need_info = args.live

    # ── MPC 시뮬레이션 ──
    print("\n  Running MPC (CasADi/IPOPT) ...")
    mpc_result = simulate_controller(
        mpc_controller, "MPC", mpc_interpolator,
        initial_state.copy(), sim_config, robot_params,
        store_info=need_info,
    )
    print(f"    -> {len(mpc_result.time_array)} steps, "
          f"wall {mpc_result.total_wall_time:.2f}s")

    # ── MPPI 시뮬레이션 ──
    print("  Running MPPI (NumPy sampling) ...")
    mppi_result = simulate_controller(
        mppi_controller, "MPPI", mppi_interpolator,
        initial_state.copy(), sim_config, robot_params,
        store_info=need_info,
    )
    print(f"    -> {len(mppi_result.time_array)} steps, "
          f"wall {mppi_result.total_wall_time:.2f}s")

    # ── ASCII 요약 ──
    print_summary(mpc_result, mppi_result)

    # ── 시각화 ──
    if args.live:
        print("  Starting live comparison replay ...")
        live_comparison_replay(
            mpc_result, mppi_result, trajectory,
            dt=sim_config.dt, update_interval=2,
        )
    elif not args.no_plot:
        print("  Generating figures ...")
        plot_figure1(mpc_result, mppi_result, save_path=args.save)
        plot_figure2(mpc_result, mppi_result, save_path=args.save)
        plt.show()


if __name__ == "__main__":
    main()
