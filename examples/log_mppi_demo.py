#!/usr/bin/env python3
"""Vanilla MPPI vs Log-MPPI 비교 데모.

극단적 비용 범위에서 Log-MPPI의 수치 안정성을 시각적으로 확인.

실행:
    python examples/log_mppi_demo.py
    python examples/log_mppi_demo.py --trajectory circle
    python examples/log_mppi_demo.py --live
    python examples/log_mppi_demo.py --save comparison.png
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

from mpc_controller import (
    DifferentialDriveModel,
    MPPIController,
    MPPIParams,
    RobotParams,
    TrajectoryInterpolator,
    generate_circle_trajectory,
    generate_figure_eight_trajectory,
    generate_sinusoidal_trajectory,
)
from mpc_controller.controllers.mppi.log_mppi import LogMPPIController
from simulation.simulator import Simulator, SimulationConfig


# ─────────────────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────────────────

@dataclass
class RunResult:
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
    store_info: bool = False,
) -> RunResult:
    sim = Simulator(robot_params, sim_config)
    sim.reset(initial_state)
    controller.reset()

    num_steps = int(sim_config.max_time / sim_config.dt)

    states = [initial_state.copy()]
    controls_list, refs_list, errors_list = [], [], []
    solve_times_list, costs_list, time_list = [], [], []
    ess_list, temp_list = [], []
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
        refs_list.append(ref[0].copy())
        errors_list.append(error)
        solve_times_list.append(info["solve_time"])
        costs_list.append(info["cost"])
        ess_list.append(info.get("ess", 0))
        temp_list.append(info.get("temperature", controller.params.lambda_))

        if store_info:
            info_list.append({
                "predicted_trajectory": info["predicted_trajectory"].copy(),
                "sample_trajectories": info["sample_trajectories"],
                "sample_weights": info["sample_weights"],
                "best_trajectory": info["best_trajectory"],
            })

        idx, dist = interpolator.find_closest_point(state[:2])
        if idx >= interpolator.num_points - 1 and dist < 0.1:
            break

    return RunResult(
        name=name,
        states=np.array(states),
        controls=np.array(controls_list),
        references=np.array(refs_list),
        tracking_errors=np.array(errors_list),
        solve_times=np.array(solve_times_list),
        costs=np.array(costs_list),
        time_array=np.array(time_list),
        total_wall_time=time.perf_counter() - wall_start,
        ess_history=np.array(ess_list),
        temp_history=np.array(temp_list),
        info_history=info_list,
    )


# ─────────────────────────────────────────────────────────────
# ASCII 요약
# ─────────────────────────────────────────────────────────────

def print_summary(vanilla: RunResult, log: RunResult) -> None:
    lines = [
        "",
        "+" + "=" * 62 + "+",
        "|{:^62s}|".format("Vanilla MPPI vs Log-MPPI 성능 비교"),
        "+" + "=" * 62 + "+",
        "| {:<30s}{:>14s}{:>14s} |".format("Metric", "Vanilla", "Log-MPPI"),
        "+" + "-" * 62 + "+",
        "| {:<30s}{:>14.4f}{:>14.4f} |".format(
            "Position RMSE [m]", vanilla.position_rmse, log.position_rmse),
        "| {:<30s}{:>14.4f}{:>14.4f} |".format(
            "Max Position Error [m]", vanilla.max_position_error, log.max_position_error),
        "| {:<30s}{:>14.4f}{:>14.4f} |".format(
            "Heading RMSE [rad]", vanilla.heading_rmse, log.heading_rmse),
        "| {:<30s}{:>14.4f}{:>14.4f} |".format(
            "Control Rate RMS", vanilla.control_rate, log.control_rate),
        "| {:<30s}{:>13.3f}{:>14.3f} |".format(
            "Avg Solve Time [ms]",
            np.mean(vanilla.solve_times) * 1000,
            np.mean(log.solve_times) * 1000),
        "| {:<30s}{:>13.1f}{:>14.1f} |".format(
            "Avg ESS", np.mean(vanilla.ess_history), np.mean(log.ess_history)),
        "| {:<30s}{:>13.3f}{:>14.3f} |".format(
            "Total Wall Time [s]", vanilla.total_wall_time, log.total_wall_time),
        "+" + "=" * 62 + "+",
        "",
    ]
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────
# 수치 안정성 비교 실험
# ─────────────────────────────────────────────────────────────

def numerical_stability_experiment() -> None:
    """극단적 비용 범위에서 Vanilla vs Log-MPPI 가중치 비교."""
    print("\n" + "=" * 64)
    print("  Numerical Stability Experiment: Extreme Cost Ranges")
    print("=" * 64)

    params = MPPIParams(N=10, K=64, dt=0.1, lambda_=1.0)
    vanilla = MPPIController(mppi_params=params, seed=42)
    log_ctrl = LogMPPIController(mppi_params=params, seed=42)

    scenarios = [
        ("Normal (1~10)", np.random.default_rng(0).uniform(1, 10, 64)),
        ("Large (1e6~1e8)", np.random.default_rng(0).uniform(1e6, 1e8, 64)),
        ("Extreme (1e12~1e15)", np.random.default_rng(0).uniform(1e12, 1e15, 64)),
        ("Mixed (1e-10~1e10)", np.concatenate([
            np.random.default_rng(0).uniform(1e-10, 1, 32),
            np.random.default_rng(0).uniform(1e8, 1e10, 32),
        ])),
    ]

    print(f"\n  {'Scenario':<25s} {'Vanilla OK?':>12s} {'Log-MPPI OK?':>14s} {'Max Diff':>12s}")
    print("  " + "-" * 63)

    for name, costs in scenarios:
        w_v = vanilla._compute_weights(costs)
        w_l = log_ctrl._compute_weights(costs)

        v_ok = np.all(np.isfinite(w_v)) and abs(np.sum(w_v) - 1.0) < 1e-6
        l_ok = np.all(np.isfinite(w_l)) and abs(np.sum(w_l) - 1.0) < 1e-6

        if v_ok and l_ok:
            diff = np.max(np.abs(w_v - w_l))
        else:
            diff = float("inf")

        print(f"  {name:<25s} {'OK' if v_ok else 'FAIL':>12s} "
              f"{'OK' if l_ok else 'FAIL':>14s} {diff:>12.2e}")

    print()


# ─────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────

def plot_comparison(
    vanilla: RunResult,
    log: RunResult,
    reference: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    cv, cl = "tab:blue", "tab:green"

    ax = axes[0, 0]
    ax.plot(reference[:, 0], reference[:, 1], "k--", lw=1.5, alpha=0.4, label="Ref")
    ax.plot(vanilla.states[:, 0], vanilla.states[:, 1], color=cv, lw=1.5, label="Vanilla")
    ax.plot(log.states[:, 0], log.states[:, 1], color=cl, lw=1.5, label="Log-MPPI")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title("Trajectory"); ax.legend(fontsize=8)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(vanilla.time_array,
            np.linalg.norm(vanilla.tracking_errors[:, :2], axis=1),
            color=cv, lw=1, label="Vanilla")
    ax.plot(log.time_array,
            np.linalg.norm(log.tracking_errors[:, :2], axis=1),
            color=cl, lw=1, label="Log-MPPI")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Pos Error [m]")
    ax.set_title("Position Error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(vanilla.time_array, vanilla.costs, color=cv, lw=1, alpha=0.7, label="Vanilla")
    ax.plot(log.time_array, log.costs, color=cl, lw=1, alpha=0.7, label="Log-MPPI")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Cost")
    ax.set_title("Cost"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(vanilla.time_array[:len(vanilla.controls)], vanilla.controls[:, 0],
            color=cv, lw=1, label="Vanilla")
    ax.plot(log.time_array[:len(log.controls)], log.controls[:, 0],
            color=cl, lw=1, label="Log-MPPI")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("v [m/s]")
    ax.set_title("Linear Velocity"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(vanilla.time_array, vanilla.ess_history, color=cv, lw=1, label="Vanilla")
    ax.plot(log.time_array, log.ess_history, color=cl, lw=1, label="Log-MPPI")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("ESS")
    ax.set_title("Effective Sample Size"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    if len(vanilla.controls) > 1:
        du_v = np.linalg.norm(np.diff(vanilla.controls, axis=0), axis=1)
        du_l = np.linalg.norm(np.diff(log.controls, axis=0), axis=1)
        ax.plot(vanilla.time_array[:len(du_v)], du_v, color=cv, lw=1, alpha=0.7, label="Vanilla")
        ax.plot(log.time_array[:len(du_l)], du_l, color=cl, lw=1, alpha=0.7, label="Log-MPPI")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("|du|")
    ax.set_title("Control Rate"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("Vanilla MPPI vs Log-MPPI", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────
# Live 리플레이
# ─────────────────────────────────────────────────────────────

def live_replay(
    vanilla: RunResult, log: RunResult, reference: np.ndarray,
    dt: float = 0.05, update_interval: int = 2,
) -> None:
    if vanilla.info_history is None or log.info_history is None:
        print("  [WARN] info_history 없음")
        return

    plt.ion()
    fig, (ax_v, ax_l) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, title, color in [(ax_v, "Vanilla MPPI", "tab:blue"),
                              (ax_l, "Log-MPPI", "tab:green")]:
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.plot(reference[:, 0], reference[:, 1], "k--", lw=1.5, alpha=0.4)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        margin = 1.0
        ax.set_xlim(reference[:, 0].min() - margin, reference[:, 0].max() + margin)
        ax.set_ylim(reference[:, 1].min() - margin, reference[:, 1].max() + margin)

    trace_v, trace_l = ([], []), ([], [])
    (line_v,) = ax_v.plot([], [], "tab:blue", lw=2)
    (line_l,) = ax_l.plot([], [], "tab:green", lw=2)

    fig.suptitle("Vanilla MPPI vs Log-MPPI — Live", fontsize=14, fontweight="bold")
    fig.canvas.draw()

    n_steps = min(len(vanilla.time_array), len(log.time_array))
    for step in range(n_steps):
        sv, sl = vanilla.states[step], log.states[step]
        trace_v[0].append(sv[0]); trace_v[1].append(sv[1])
        trace_l[0].append(sl[0]); trace_l[1].append(sl[1])

        if step % update_interval != 0:
            continue
        line_v.set_data(trace_v[0], trace_v[1])
        line_l.set_data(trace_l[0], trace_l[1])
        fig.canvas.draw(); fig.canvas.flush_events()

    plt.ioff()
    plt.show()


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────

def generate_trajectory(traj_type: str) -> np.ndarray:
    if traj_type == "circle":
        return generate_circle_trajectory(np.array([0.0, 0.0]), 2.0, 400)
    elif traj_type == "figure8":
        return generate_figure_eight_trajectory(np.array([0.0, 0.0]), 2.0, 400)
    else:
        return generate_sinusoidal_trajectory(np.array([0.0, 0.0]), 10.0, 1.0, 0.5, 400)


def main():
    parser = argparse.ArgumentParser(description="Vanilla vs Log-MPPI 비교 데모")
    parser.add_argument("--trajectory", type=str, default="figure8",
                        choices=["circle", "figure8", "sine"])
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 64)
    print("       Vanilla MPPI vs Log-MPPI Comparison Demo")
    print("=" * 64)

    # 수치 안정성 실험
    numerical_stability_experiment()

    # 궤적 추적 비교
    robot_params = RobotParams(max_velocity=1.0, max_omega=1.5)
    sim_config = SimulationConfig(dt=0.05, max_time=20.0)
    trajectory = generate_trajectory(args.trajectory)
    initial_state = trajectory[0].copy()

    params = MPPIParams(
        N=20, K=512, dt=0.05, lambda_=10.0,
        noise_sigma=np.array([0.3, 0.3]),
        Q=np.diag([10.0, 10.0, 1.0]),
        R=np.diag([0.01, 0.01]),
        Qf=np.diag([100.0, 100.0, 10.0]),
    )

    vanilla_ctrl = MPPIController(robot_params=robot_params, mppi_params=params, seed=42)
    log_ctrl = LogMPPIController(robot_params=robot_params, mppi_params=params, seed=42)

    need_info = args.live

    print(f"\n  Trajectory: {args.trajectory}")
    print("  Running Vanilla MPPI ...")
    v_interp = TrajectoryInterpolator(trajectory, dt=sim_config.dt)
    v_result = simulate(vanilla_ctrl, "Vanilla", v_interp,
                        initial_state.copy(), sim_config, robot_params, need_info)
    print(f"    -> {len(v_result.time_array)} steps, wall {v_result.total_wall_time:.2f}s")

    print("  Running Log-MPPI ...")
    l_interp = TrajectoryInterpolator(trajectory, dt=sim_config.dt)
    l_result = simulate(log_ctrl, "Log-MPPI", l_interp,
                        initial_state.copy(), sim_config, robot_params, need_info)
    print(f"    -> {len(l_result.time_array)} steps, wall {l_result.total_wall_time:.2f}s")

    print_summary(v_result, l_result)

    if args.live:
        live_replay(v_result, l_result, trajectory)
    elif not args.no_plot:
        plot_comparison(v_result, l_result, trajectory, save_path=args.save)
        plt.show()


if __name__ == "__main__":
    main()
