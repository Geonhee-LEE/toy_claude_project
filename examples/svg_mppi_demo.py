#!/usr/bin/env python3
"""SVG-MPPI 비교 데모 — Guide particle 기반 다중 모드 탐색.

SVMPC (full SVGD on all K) vs SVG-MPPI (SVGD on G guides only) vs Vanilla를
장애물 환경에서 비교합니다.

실행:
    python examples/svg_mppi_demo.py
    python examples/svg_mppi_demo.py --trajectory circle
    python examples/svg_mppi_demo.py --save svg_comparison.png
    python examples/svg_mppi_demo.py --no-plot
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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
from mpc_controller.controllers.mppi.stein_variational_mppi import (
    SteinVariationalMPPIController,
)
from mpc_controller.controllers.mppi.svg_mppi import SVGMPPIController
from simulation.simulator import Simulator, SimulationConfig


OBSTACLES = np.array([
    [1.5, 0.8, 0.35],
    [-0.5, -1.2, 0.3],
    [0.8, -0.5, 0.25],
])


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

    @property
    def position_rmse(self) -> float:
        pos_err = self.tracking_errors[:, :2]
        return float(np.sqrt(np.mean(np.sum(pos_err ** 2, axis=1))))

    @property
    def control_rate(self) -> float:
        if len(self.controls) < 2:
            return 0.0
        du = np.diff(self.controls, axis=0)
        return float(np.sqrt(np.mean(du ** 2)))


def simulate(
    controller, name: str,
    interpolator: TrajectoryInterpolator,
    initial_state: np.ndarray,
    sim_config: SimulationConfig,
    robot_params: RobotParams,
) -> RunResult:
    sim = Simulator(robot_params, sim_config)
    sim.reset(initial_state)
    controller.reset()

    num_steps = int(sim_config.max_time / sim_config.dt)
    states = [initial_state.copy()]
    controls_list, refs_list, errors_list = [], [], []
    solve_times_list, costs_list, time_list, ess_list = [], [], [], []

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
    )


def print_summary(results: List[RunResult]) -> None:
    col_w = 16
    header_cols = "".join(f"{r.name:>{col_w}s}" for r in results)
    sep = "+" + "=" * (28 + col_w * len(results) + 2) + "+"

    lines = [
        "", sep,
        "|{:^{w}s}|".format("SVG-MPPI vs SVMPC vs Vanilla", w=28 + col_w * len(results) + 2),
        sep,
        "| {:<28s}".format("Metric") + header_cols + " |",
        "+" + "-" * (28 + col_w * len(results) + 2) + "+",
    ]

    metrics = [
        ("Position RMSE [m]", lambda r: f"{r.position_rmse:>{col_w}.4f}"),
        ("Control Rate RMS", lambda r: f"{r.control_rate:>{col_w}.4f}"),
        ("Avg Solve [ms]", lambda r: f"{np.mean(r.solve_times)*1000:>{col_w}.3f}"),
        ("Avg ESS", lambda r: f"{np.mean(r.ess_history):>{col_w}.1f}"),
        ("Wall Time [s]", lambda r: f"{r.total_wall_time:>{col_w}.3f}"),
    ]
    for label, fmt in metrics:
        row = "| {:<28s}".format(label) + "".join(fmt(r) for r in results) + " |"
        lines.append(row)

    lines.append(sep)
    print("\n".join(lines))


def plot_comparison(
    results: List[RunResult], reference: np.ndarray,
    obstacles: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(reference[:, 0], reference[:, 1], "k--", lw=1.5, alpha=0.4, label="Ref")
    for obs in obstacles:
        circle = Circle((obs[0], obs[1]), obs[2], color="red", alpha=0.3)
        ax.add_patch(circle)
        safety = Circle((obs[0], obs[1]), obs[2] + 0.3, fill=False,
                        linestyle=":", color="red", alpha=0.5)
        ax.add_patch(safety)
    for i, r in enumerate(results):
        ax.plot(r.states[:, 0], r.states[:, 1], color=colors[i], lw=1.5, label=r.name)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title("Trajectory + Obstacles"); ax.legend(fontsize=8)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for i, r in enumerate(results):
        err = np.linalg.norm(r.tracking_errors[:, :2], axis=1)
        ax.plot(r.time_array, err, color=colors[i], lw=1, label=r.name)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Pos Error [m]")
    ax.set_title("Position Error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    for i, r in enumerate(results):
        ax.plot(r.time_array, r.ess_history, color=colors[i], lw=1, label=r.name)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("ESS")
    ax.set_title("ESS"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for i, r in enumerate(results):
        ax.plot(r.time_array[:len(r.controls)], r.controls[:, 0],
                color=colors[i], lw=1, label=r.name)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("v [m/s]")
    ax.set_title("Linear Velocity"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for i, r in enumerate(results):
        ax.plot(r.time_array[:len(r.controls)], r.controls[:, 1],
                color=colors[i], lw=1, label=r.name)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("omega [rad/s]")
    ax.set_title("Angular Velocity"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    for i, r in enumerate(results):
        ax.plot(r.time_array, r.costs, color=colors[i], lw=1, alpha=0.7, label=r.name)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Cost")
    ax.set_title("Cost"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("SVG-MPPI: Guide Particle Multi-mode Exploration",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def generate_trajectory(traj_type: str) -> np.ndarray:
    if traj_type == "circle":
        return generate_circle_trajectory(np.array([0.0, 0.0]), 2.0, 400)
    elif traj_type == "figure8":
        return generate_figure_eight_trajectory(np.array([0.0, 0.0]), 2.0, 400)
    else:
        return generate_sinusoidal_trajectory(np.array([0.0, 0.0]), 10.0, 1.0, 0.5, 400)


def main():
    parser = argparse.ArgumentParser(description="SVG-MPPI vs SVMPC 비교 데모")
    parser.add_argument("--trajectory", type=str, default="figure8",
                        choices=["circle", "figure8", "sine"])
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 64)
    print("    SVG-MPPI vs SVMPC vs Vanilla Comparison Demo")
    print("=" * 64)

    robot_params = RobotParams(max_velocity=1.0, max_omega=1.5)
    sim_config = SimulationConfig(dt=0.05, max_time=20.0)
    trajectory = generate_trajectory(args.trajectory)
    initial_state = trajectory[0].copy()

    configs = [
        ("Vanilla", MPPIParams(
            N=20, K=256, dt=0.05, lambda_=10.0,
            noise_sigma=np.array([0.3, 0.3]),
            Q=np.diag([10.0, 10.0, 1.0]),
            R=np.diag([0.01, 0.01]),
            Qf=np.diag([100.0, 100.0, 10.0]),
        ), MPPIController),
        ("SVMPC (L=2)", MPPIParams(
            N=20, K=256, dt=0.05, lambda_=10.0,
            noise_sigma=np.array([0.3, 0.3]),
            Q=np.diag([10.0, 10.0, 1.0]),
            R=np.diag([0.01, 0.01]),
            Qf=np.diag([100.0, 100.0, 10.0]),
            svgd_num_iterations=2,
            svgd_step_size=0.1,
        ), SteinVariationalMPPIController),
        ("SVG-MPPI", MPPIParams(
            N=20, K=256, dt=0.05, lambda_=10.0,
            noise_sigma=np.array([0.3, 0.3]),
            Q=np.diag([10.0, 10.0, 1.0]),
            R=np.diag([0.01, 0.01]),
            Qf=np.diag([100.0, 100.0, 10.0]),
            svg_num_guide_particles=16,
            svg_guide_iterations=3,
            svg_guide_step_size=0.2,
            svg_resample_std=0.1,
        ), SVGMPPIController),
    ]

    results = []
    for name, params, CtrlClass in configs:
        ctrl = CtrlClass(
            robot_params=robot_params, mppi_params=params,
            seed=42, obstacles=OBSTACLES,
        )
        interp = TrajectoryInterpolator(trajectory, dt=sim_config.dt)

        print(f"\n  Running {name} ...")
        result = simulate(ctrl, name, interp, initial_state.copy(), sim_config, robot_params)
        print(f"    -> {len(result.time_array)} steps, "
              f"RMSE={result.position_rmse:.4f}m, "
              f"Solve={np.mean(result.solve_times)*1000:.1f}ms")
        results.append(result)

    print_summary(results)

    if not args.no_plot:
        plot_comparison(results, trajectory, OBSTACLES, save_path=args.save)
        plt.show()


if __name__ == "__main__":
    main()
