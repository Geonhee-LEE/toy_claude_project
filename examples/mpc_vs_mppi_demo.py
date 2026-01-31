#!/usr/bin/env python3
"""MPC vs MPPI 컨트롤러 비교 데모.

전통 MPC (CasADi/IPOPT)와 MPPI (NumPy 샘플링) 컨트롤러를
동일 조건에서 실행하고 성능을 비교합니다.

실행:
    python examples/mpc_vs_mppi_demo.py
    python examples/mpc_vs_mppi_demo.py --trajectory circle
    python examples/mpc_vs_mppi_demo.py --trajectory sine --save comparison.png
"""

import argparse
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

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
) -> ComparisonResult:
    """수동 시뮬레이션 루프로 컨트롤러 실행 + 메트릭 수집."""

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

    # ── MPC 시뮬레이션 ──
    print("\n  Running MPC (CasADi/IPOPT) ...")
    mpc_result = simulate_controller(
        mpc_controller, "MPC", mpc_interpolator,
        initial_state.copy(), sim_config, robot_params,
    )
    print(f"    -> {len(mpc_result.time_array)} steps, "
          f"wall {mpc_result.total_wall_time:.2f}s")

    # ── MPPI 시뮬레이션 ──
    print("  Running MPPI (NumPy sampling) ...")
    mppi_result = simulate_controller(
        mppi_controller, "MPPI", mppi_interpolator,
        initial_state.copy(), sim_config, robot_params,
    )
    print(f"    -> {len(mppi_result.time_array)} steps, "
          f"wall {mppi_result.total_wall_time:.2f}s")

    # ── ASCII 요약 ──
    print_summary(mpc_result, mppi_result)

    # ── 시각화 ──
    if not args.no_plot:
        print("  Generating figures ...")
        plot_figure1(mpc_result, mppi_result, save_path=args.save)
        plot_figure2(mpc_result, mppi_result, save_path=args.save)
        plt.show()


if __name__ == "__main__":
    main()
