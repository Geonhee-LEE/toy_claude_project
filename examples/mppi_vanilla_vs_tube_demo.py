#!/usr/bin/env python3
"""Vanilla MPPI vs Tube-MPPI 비교 데모.

외란 주입 하에서 Vanilla MPPI와 Tube-MPPI의 강건성을 비교합니다.
Tube-MPPI는 ancillary 피드백으로 모델 불확실성/외란에 대해 강건성을 확보합니다.

실행:
    python examples/mppi_vanilla_vs_tube_demo.py
    python examples/mppi_vanilla_vs_tube_demo.py --live
    python examples/mppi_vanilla_vs_tube_demo.py --noise 0.1
    python examples/mppi_vanilla_vs_tube_demo.py --trajectory circle
    python examples/mppi_vanilla_vs_tube_demo.py --no-plot --noise 0.1
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
    MPPIController,
    MPPIParams,
    RobotParams,
    TubeMPPIController,
    TrajectoryInterpolator,
    generate_circle_trajectory,
    generate_figure_eight_trajectory,
    generate_sinusoidal_trajectory,
)
from simulation.simulator import Simulator, SimulationConfig


# ─────────────────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────────────────

@dataclass
class RunResult:
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
    ess_history: np.ndarray      # (T,)
    temp_history: np.ndarray     # (T,)
    deviation_history: np.ndarray  # (T,) — tube: 명목-실제 편차
    info_history: Optional[List[dict]] = None

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

    @property
    def control_rate(self) -> float:
        if len(self.controls) < 2:
            return 0.0
        du = np.diff(self.controls, axis=0)
        return float(np.sqrt(np.mean(du ** 2)))


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
    add_noise: bool = True,
    store_info: bool = False,
) -> RunResult:
    """시뮬레이션 루프 실행 + 메트릭 수집."""
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
    deviation_list: List[float] = []
    info_list: Optional[List[dict]] = [] if store_info else None

    wall_start = time.perf_counter()

    for step in range(num_steps):
        t = step * sim_config.dt
        state = sim.get_measurement()

        ref = interpolator.get_reference(
            t, controller.params.N, controller.params.dt,
            current_theta=state[2],
        )

        control, info = controller.compute_control(state, ref)
        next_state = sim.step(control, add_noise=add_noise)
        error = sim.compute_tracking_error(state, ref[0])

        time_list.append(t)
        states.append(next_state.copy())
        controls_list.append(control.copy())
        references_list.append(ref[0].copy())
        errors_list.append(error)
        solve_times_list.append(info["solve_time"])
        costs_list.append(info["cost"])
        ess_list.append(info.get("ess", 0))
        temp_list.append(info.get("temperature", controller.params.lambda_))
        deviation_list.append(info.get("deviation", 0.0))

        if store_info:
            compact = {
                "predicted_trajectory": info["predicted_trajectory"].copy(),
                "sample_trajectories": info["sample_trajectories"],
                "sample_weights": info["sample_weights"],
                "best_trajectory": info["best_trajectory"],
                "cost": info["cost"],
                "solve_time": info["solve_time"],
                "ess": info.get("ess", 0),
                "temperature": info.get("temperature", 0),
            }
            # Tube 관련 정보
            if info.get("tube_enabled"):
                compact["nominal_trajectory"] = info["nominal_trajectory"].copy()
                compact["tube_boundary"] = info["tube_boundary"]
                compact["feedback_correction"] = info["feedback_correction"].copy()
            info_list.append(compact)

        idx, dist = interpolator.find_closest_point(state[:2])
        if idx >= interpolator.num_points - 1 and dist < 0.1:
            break

    wall_time = time.perf_counter() - wall_start

    return RunResult(
        name=name,
        states=np.array(states),
        controls=np.array(controls_list),
        references=np.array(references_list),
        tracking_errors=np.array(errors_list),
        solve_times=np.array(solve_times_list),
        costs=np.array(costs_list),
        time_array=np.array(time_list),
        total_wall_time=wall_time,
        ess_history=np.array(ess_list),
        temp_history=np.array(temp_list),
        deviation_history=np.array(deviation_list),
        info_history=info_list,
    )


# ─────────────────────────────────────────────────────────────
# ASCII 요약표
# ─────────────────────────────────────────────────────────────

def print_summary(vanilla: RunResult, tube: RunResult, noise_std: float) -> None:
    """콘솔 비교 요약표."""
    avg_st_v = np.mean(vanilla.solve_times) * 1000
    avg_st_t = np.mean(tube.solve_times) * 1000

    lines = [
        "",
        "+" + "=" * 62 + "+",
        "|{:^62s}|".format("Vanilla MPPI vs Tube-MPPI 성능 비교"),
        "|{:^62s}|".format(f"(process noise std = {noise_std:.3f})"),
        "+" + "=" * 62 + "+",
        "| {:<30s}{:>14s}{:>14s} |".format("Metric", "Vanilla", "Tube"),
        "+" + "-" * 62 + "+",
        "| {:<30s}{:>14.4f}{:>14.4f} |".format(
            "Position RMSE [m]", vanilla.position_rmse, tube.position_rmse
        ),
        "| {:<30s}{:>14.4f}{:>14.4f} |".format(
            "Max Position Error [m]", vanilla.max_position_error, tube.max_position_error
        ),
        "| {:<30s}{:>14.4f}{:>14.4f} |".format(
            "Heading RMSE [rad]", vanilla.heading_rmse, tube.heading_rmse
        ),
        "| {:<30s}{:>14.4f}{:>14.4f} |".format(
            "Control Rate RMS", vanilla.control_rate, tube.control_rate
        ),
        "| {:<30s}{:>13.3f}{:>14.3f} |".format(
            "Avg Solve Time [ms]", avg_st_v, avg_st_t
        ),
        "| {:<30s}{:>13.1f}{:>14.1f} |".format(
            "Avg ESS", np.mean(vanilla.ess_history), np.mean(tube.ess_history)
        ),
        "| {:<30s}{:>13.3f}{:>14.3f} |".format(
            "Total Wall Time [s]", vanilla.total_wall_time, tube.total_wall_time
        ),
        "| {:<30s}{:>14d}{:>14d} |".format(
            "Simulation Steps", len(vanilla.time_array), len(tube.time_array)
        ),
    ]

    # Tube 전용 메트릭
    if len(tube.deviation_history) > 0 and np.any(tube.deviation_history > 0):
        lines.append("+" + "-" * 62 + "+")
        lines.append("| {:<30s}{:>14s}{:>14.4f} |".format(
            "Avg Nominal Deviation [m]", "N/A", np.mean(tube.deviation_history)
        ))
        lines.append("| {:<30s}{:>14s}{:>14.4f} |".format(
            "Max Nominal Deviation [m]", "N/A", np.max(tube.deviation_history)
        ))

    lines.append("+" + "-" * 62 + "+")

    # 승자 판정
    acc = "Tube" if tube.position_rmse < vanilla.position_rmse else "Vanilla"
    smooth = "Tube" if tube.control_rate < vanilla.control_rate else "Vanilla"

    lines += [
        "| {:<60s} |".format(f"  Accuracy winner   : {acc}"),
        "| {:<60s} |".format(f"  Smoothness winner  : {smooth}"),
        "+" + "=" * 62 + "+",
        "",
    ]

    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────
# 정적 시각화 (2x3 패널)
# ─────────────────────────────────────────────────────────────

def plot_comparison_figure(
    vanilla: RunResult,
    tube: RunResult,
    reference: np.ndarray,
    noise_std: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """2x3 비교 그래프."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    cv, ct = "tab:blue", "tab:red"

    # ── (0,0) 궤적 비교 ──
    ax = axes[0, 0]
    ax.plot(reference[:, 0], reference[:, 1],
            "k--", linewidth=1.5, alpha=0.4, label="Reference")
    ax.plot(vanilla.states[:, 0], vanilla.states[:, 1],
            color=cv, linewidth=1.5, label="Vanilla")
    ax.plot(tube.states[:, 0], tube.states[:, 1],
            color=ct, linewidth=1.5, label="Tube")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Trajectory Comparison")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # ── (0,1) 위치 오차 ──
    ax = axes[0, 1]
    v_err = np.linalg.norm(vanilla.tracking_errors[:, :2], axis=1)
    t_err = np.linalg.norm(tube.tracking_errors[:, :2], axis=1)
    ax.plot(vanilla.time_array, v_err, color=cv, linewidth=1, label="Vanilla")
    ax.plot(tube.time_array, t_err, color=ct, linewidth=1, label="Tube")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [m]")
    ax.set_title("Position Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (0,2) 제어 변화율 ──
    ax = axes[0, 2]
    if len(vanilla.controls) > 1:
        du_v = np.linalg.norm(np.diff(vanilla.controls, axis=0), axis=1)
        du_t = np.linalg.norm(np.diff(tube.controls, axis=0), axis=1)
        ax.plot(vanilla.time_array[:len(du_v)], du_v,
                color=cv, alpha=0.7, linewidth=1, label="Vanilla")
        ax.plot(tube.time_array[:len(du_t)], du_t,
                color=ct, alpha=0.7, linewidth=1, label="Tube")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("|du|")
    ax.set_title("Control Rate (Smoothness)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (1,0) 선속도 비교 ──
    ax = axes[1, 0]
    ax.plot(vanilla.time_array[:len(vanilla.controls)], vanilla.controls[:, 0],
            color=cv, linewidth=1, label="Vanilla")
    ax.plot(tube.time_array[:len(tube.controls)], tube.controls[:, 0],
            color=ct, linewidth=1, label="Tube")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("v [m/s]")
    ax.set_title("Linear Velocity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (1,1) 명목-실제 편차 (Tube만) ──
    ax = axes[1, 1]
    ax.plot(tube.time_array, tube.deviation_history,
            color=ct, linewidth=1, label="Tube: nominal deviation")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Deviation [m]")
    ax.set_title("Nominal-Actual Deviation (Tube)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (1,2) ESS 비교 ──
    ax = axes[1, 2]
    ax.plot(vanilla.time_array, vanilla.ess_history,
            color=cv, alpha=0.7, linewidth=1, label="Vanilla")
    ax.plot(tube.time_array, tube.ess_history,
            color=ct, alpha=0.7, linewidth=1, label="Tube")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ESS")
    ax.set_title("Effective Sample Size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Vanilla MPPI vs Tube-MPPI (noise_std={noise_std:.3f})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────
# Live 비교 리플레이
# ─────────────────────────────────────────────────────────────

def live_comparison_replay(
    vanilla: RunResult,
    tube: RunResult,
    reference: np.ndarray,
    dt: float = 0.05,
    update_interval: int = 2,
) -> None:
    """Dual-panel 실시간 리플레이.

    Tube-MPPI 패널에는 tube 경계(반투명 노란 영역)와
    명목 궤적(주황)을 추가로 표시한다.
    """
    if vanilla.info_history is None or tube.info_history is None:
        print("  [WARN] info_history 없음 — store_info=True 필요")
        return

    plt.ion()

    fig = plt.figure(figsize=(18, 10), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    ax_vanilla = fig.add_subplot(gs[0, 0])
    ax_tube = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis("off")

    robot_length, robot_width = 0.3, 0.2
    max_samples = 20

    x_margin, y_margin = 1.0, 1.0
    x_min = reference[:, 0].min() - x_margin
    x_max = reference[:, 0].max() + x_margin
    y_min = reference[:, 1].min() - y_margin
    y_max = reference[:, 1].max() + y_margin

    panels = {}
    for ax, label, color in [
        (ax_vanilla, "Vanilla MPPI", "tab:blue"),
        (ax_tube, "Tube-MPPI", "tab:red"),
    ]:
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.plot(reference[:, 0], reference[:, 1],
                "k--", linewidth=1.5, alpha=0.4, label="Reference")

        (trace_line,) = ax.plot([], [], "-", color=color, linewidth=2, label="Actual")
        (pred_line,) = ax.plot([], [], "g-", alpha=0.6, linewidth=1.5, label="Prediction")

        sample_lines = []
        for _ in range(max_samples):
            (sl,) = ax.plot([], [], "-", color="steelblue", alpha=0.1, linewidth=0.5)
            sample_lines.append(sl)

        (best_line,) = ax.plot([], [], "m-", linewidth=1.5, label="Best Sample", alpha=0.7)

        robot_patch = patches.Rectangle(
            (0, 0), robot_length, robot_width,
            angle=0, fill=True, facecolor=color,
            edgecolor="black", linewidth=2, alpha=0.8,
        )
        ax.add_patch(robot_patch)
        (dir_line,) = ax.plot([], [], "k-", linewidth=2)

        extra = {}
        if label == "Tube-MPPI":
            # 명목 궤적 라인
            (nom_line,) = ax.plot([], [], "-", color="orange", linewidth=2,
                                  alpha=0.8, label="Nominal")
            # Tube 경계
            tube_fill = ax.fill_between([], [], [], color="gold", alpha=0.15,
                                         label="Tube boundary")
            extra["nom_line"] = nom_line
            extra["tube_fill"] = tube_fill

        ax.legend(loc="upper right", fontsize=7)

        panels[label] = {
            "ax": ax, "trace_line": trace_line, "pred_line": pred_line,
            "sample_lines": sample_lines, "best_line": best_line,
            "robot_patch": robot_patch, "dir_line": dir_line,
            **extra,
        }

    info_text = ax_info.text(
        0.5, 0.9, "", transform=ax_info.transAxes,
        fontsize=11, fontfamily="monospace",
        verticalalignment="top", horizontalalignment="center",
    )

    fig.suptitle("Vanilla MPPI vs Tube-MPPI — Live Comparison",
                 fontsize=14, fontweight="bold")
    fig.canvas.draw()
    fig.canvas.flush_events()

    num_steps = min(len(vanilla.time_array), len(tube.time_array))
    trace_data = {"Vanilla MPPI": ([], []), "Tube-MPPI": ([], [])}

    for step in range(num_steps):
        for label, result in [("Vanilla MPPI", vanilla), ("Tube-MPPI", tube)]:
            if step >= len(result.time_array):
                continue

            p = panels[label]
            state = result.states[step]
            tx, ty = trace_data[label]
            tx.append(state[0])
            ty.append(state[1])

            if step % update_interval != 0:
                continue

            p["trace_line"].set_data(tx, ty)

            if result.info_history and step < len(result.info_history):
                info = result.info_history[step]

                pred = info.get("predicted_trajectory")
                if pred is not None:
                    p["pred_line"].set_data(pred[:, 0], pred[:, 1])

                sample_traj = info.get("sample_trajectories")
                sample_weights = info.get("sample_weights")
                if sample_traj is not None and sample_weights is not None:
                    top_idx = np.argsort(sample_weights)[-max_samples:]
                    max_w = np.max(sample_weights)
                    for rank, idx in enumerate(top_idx):
                        alpha = float(np.clip(
                            sample_weights[idx] / max_w * 0.5, 0.03, 0.5
                        ))
                        p["sample_lines"][rank].set_data(
                            sample_traj[idx, :, 0],
                            sample_traj[idx, :, 1],
                        )
                        p["sample_lines"][rank].set_alpha(alpha)
                    for rank in range(len(top_idx), max_samples):
                        p["sample_lines"][rank].set_data([], [])

                best = info.get("best_trajectory")
                if best is not None:
                    p["best_line"].set_data(best[:, 0], best[:, 1])

                # Tube 전용 시각화
                if label == "Tube-MPPI":
                    nom_traj = info.get("nominal_trajectory")
                    if nom_traj is not None:
                        p["nom_line"].set_data(nom_traj[:, 0], nom_traj[:, 1])

                    boundary = info.get("tube_boundary")
                    if boundary is not None:
                        # 기존 fill_between 제거 후 재생성
                        p["tube_fill"].remove()
                        upper = boundary["upper"]
                        lower = boundary["lower"]
                        p["tube_fill"] = p["ax"].fill(
                            np.concatenate([upper[:, 0], lower[::-1, 0]]),
                            np.concatenate([upper[:, 1], lower[::-1, 1]]),
                            color="gold", alpha=0.15,
                        )[0]

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
        v_err = np.linalg.norm(
            vanilla.tracking_errors[min(step, len(vanilla.tracking_errors) - 1), :2]
        )
        t_err = np.linalg.norm(
            tube.tracking_errors[min(step, len(tube.tracking_errors) - 1), :2]
        )
        v_st = vanilla.solve_times[min(step, len(vanilla.solve_times) - 1)] * 1000
        t_st = tube.solve_times[min(step, len(tube.solve_times) - 1)] * 1000
        dev = tube.deviation_history[min(step, len(tube.deviation_history) - 1)]

        info_str = (
            f"Time: {t:.2f}s  |  "
            f"Pos Error — Vanilla: {v_err:.4f}m  Tube: {t_err:.4f}m  |  "
            f"Solve — V: {v_st:.1f}ms  T: {t_st:.1f}ms  |  "
            f"Deviation: {dev:.4f}m"
        )
        info_text.set_text(info_str)

        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()
    plt.show()


# ─────────────────────────────────────────────────────────────
# 궤적 생성
# ─────────────────────────────────────────────────────────────

def generate_trajectory(traj_type: str) -> np.ndarray:
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
    parser = argparse.ArgumentParser(
        description="Vanilla MPPI vs Tube-MPPI 비교 데모"
    )
    parser.add_argument(
        "--trajectory", type=str, default="figure8",
        choices=["circle", "figure8", "sine"],
        help="궤적 타입 (기본: figure8)",
    )
    parser.add_argument(
        "--noise", type=float, default=0.05,
        help="프로세스 노이즈 표준편차 (기본: 0.05)",
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
        help="시뮬레이션 후 dual-panel 실시간 리플레이",
    )
    args = parser.parse_args()

    print("\n" + "=" * 64)
    print("     Vanilla MPPI vs Tube-MPPI Comparison Demo")
    print("=" * 64)

    # ── 공통 파라미터 ──
    robot_params = RobotParams(max_velocity=1.0, max_omega=1.5)
    noise_std = np.array([args.noise, args.noise, args.noise * 0.5])
    sim_config = SimulationConfig(
        dt=0.05, max_time=20.0,
        process_noise_std=noise_std,
    )

    # ── 궤적 생성 ──
    print(f"\n  Trajectory     : {args.trajectory}")
    print(f"  Process noise  : {args.noise}")
    trajectory = generate_trajectory(args.trajectory)
    initial_state = trajectory[0].copy()

    # ── 공통 MPPI 파라미터 ──
    base_mppi = dict(
        N=20, K=512, dt=0.05, lambda_=10.0,
        noise_sigma=np.array([0.3, 0.3]),
        Q=np.diag([10.0, 10.0, 1.0]),
        R=np.diag([0.01, 0.01]),
        Qf=np.diag([100.0, 100.0, 10.0]),
    )

    # ── Vanilla MPPI ──
    vanilla_params = MPPIParams(**base_mppi)
    vanilla_controller = MPPIController(
        robot_params=robot_params,
        mppi_params=vanilla_params,
        seed=42,
    )

    # ── Tube-MPPI ──
    tube_params = MPPIParams(
        **base_mppi,
        tube_enabled=True,
        tube_disturbance_bound=args.noise,
    )
    tube_controller = TubeMPPIController(
        robot_params=robot_params,
        mppi_params=tube_params,
        seed=42,
    )

    need_info = args.live

    # ── 파라미터 요약 ──
    print(f"\n  ┌{'─' * 42}┐")
    print(f"  │{'Vanilla MPPI':^42s}│")
    print(f"  ├{'─' * 42}┤")
    print(f"  │  N=20, K=512, dt=0.05s                 │")
    print(f"  │  lambda=10.0, sigma=[0.3, 0.3]         │")
    print(f"  │  tube_enabled=False                     │")
    print(f"  └{'─' * 42}┘")
    print(f"  ┌{'─' * 42}┐")
    print(f"  │{'Tube-MPPI':^42s}│")
    print(f"  ├{'─' * 42}┤")
    print(f"  │  N=20, K=512, dt=0.05s                 │")
    print(f"  │  lambda=10.0, sigma=[0.3, 0.3]         │")
    print(f"  │  tube_enabled=True                      │")
    print(f"  │  disturbance_bound={args.noise:<22.3f}│")
    print(f"  │  K_fb=default, max_corr=default         │")
    print(f"  └{'─' * 42}┘")

    # ── Vanilla 시뮬레이션 ──
    np.random.seed(123)
    vanilla_interp = TrajectoryInterpolator(trajectory, dt=sim_config.dt)
    print("\n  Running Vanilla MPPI ...")
    vanilla_result = simulate(
        vanilla_controller, "Vanilla MPPI", vanilla_interp,
        initial_state.copy(), sim_config, robot_params,
        add_noise=True, store_info=need_info,
    )
    print(f"    -> {len(vanilla_result.time_array)} steps, "
          f"wall {vanilla_result.total_wall_time:.2f}s")

    # ── Tube 시뮬레이션 (동일 노이즈 시드) ──
    np.random.seed(123)
    tube_interp = TrajectoryInterpolator(trajectory, dt=sim_config.dt)
    print("  Running Tube-MPPI ...")
    tube_result = simulate(
        tube_controller, "Tube-MPPI", tube_interp,
        initial_state.copy(), sim_config, robot_params,
        add_noise=True, store_info=need_info,
    )
    print(f"    -> {len(tube_result.time_array)} steps, "
          f"wall {tube_result.total_wall_time:.2f}s")

    # ── ASCII 요약 ──
    print_summary(vanilla_result, tube_result, args.noise)

    # ── 시각화 ──
    if args.live:
        print("  Starting live comparison replay ...")
        live_comparison_replay(
            vanilla_result, tube_result, trajectory,
            dt=sim_config.dt, update_interval=2,
        )
    elif not args.no_plot:
        print("  Generating comparison figure ...")
        plot_comparison_figure(
            vanilla_result, tube_result, trajectory,
            noise_std=args.noise,
            save_path=args.save,
        )
        plt.show()


if __name__ == "__main__":
    main()
