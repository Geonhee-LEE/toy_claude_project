#!/usr/bin/env python3
"""Risk-Aware MPPI (CVaR) 데모 — alpha별 장애물 회피 비교.

alpha=0.3 (risk-averse), 0.5, 0.8, 1.0 (Vanilla)를
장애물 회피 시나리오에서 비교합니다.

실행:
    python examples/risk_aware_mppi_demo.py
    python examples/risk_aware_mppi_demo.py --trajectory circle
    python examples/risk_aware_mppi_demo.py --alpha 0.3 0.5 1.0
    python examples/risk_aware_mppi_demo.py --live
    python examples/risk_aware_mppi_demo.py --save comparison.png
    python examples/risk_aware_mppi_demo.py --no-obstacles
    python examples/risk_aware_mppi_demo.py --no-plot
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
from mpc_controller.controllers.mppi.risk_aware_mppi import RiskAwareMPPIController
from simulation.simulator import Simulator, SimulationConfig


# ─────────────────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    name: str
    alpha_value: float
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
    obstacle_distances: np.ndarray
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

    @property
    def min_obstacle_distance(self) -> float:
        if len(self.obstacle_distances) == 0:
            return float("inf")
        return float(np.min(self.obstacle_distances))


# ─────────────────────────────────────────────────────────────
# 시뮬레이션 루프
# ─────────────────────────────────────────────────────────────

def compute_obstacle_distance(state: np.ndarray, obstacles: np.ndarray) -> float:
    """현재 상태에서 장애물까지 최소 거리."""
    if obstacles is None or len(obstacles) == 0:
        return float("inf")
    dists = np.linalg.norm(state[:2] - obstacles[:, :2], axis=1) - obstacles[:, 2]
    return float(np.min(dists))


def simulate(
    controller,
    name: str,
    alpha_value: float,
    interpolator: TrajectoryInterpolator,
    initial_state: np.ndarray,
    sim_config: SimulationConfig,
    robot_params: RobotParams,
    obstacles: Optional[np.ndarray] = None,
    store_info: bool = False,
) -> RunResult:
    sim = Simulator(robot_params, sim_config)
    sim.reset(initial_state)
    controller.reset()

    num_steps = int(sim_config.max_time / sim_config.dt)
    states = [initial_state.copy()]
    controls_list, refs_list, errors_list = [], [], []
    solve_times_list, costs_list, time_list = [], [], []
    ess_list, temp_list, obs_dist_list = [], [], []
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
        obs_dist_list.append(compute_obstacle_distance(state, obstacles))

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
        name=name, alpha_value=alpha_value,
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
        obstacle_distances=np.array(obs_dist_list),
        info_history=info_list,
    )


# ─────────────────────────────────────────────────────────────
# ASCII 요약표
# ─────────────────────────────────────────────────────────────

def print_summary(results: List[RunResult]) -> None:
    n = len(results)
    col_w = 14

    header_cols = "".join(f"{r.name:>{col_w}s}" for r in results)
    sep = "+" + "=" * (30 + col_w * n + 2) + "+"

    lines = [
        "", sep,
        "|{:^{w}s}|".format("Risk-Aware MPPI (CVaR) 비교", w=30 + col_w * n + 2),
        sep,
        "| {:<30s}".format("Metric") + header_cols + " |",
        "+" + "-" * (30 + col_w * n + 2) + "+",
    ]

    metrics = [
        ("Position RMSE [m]", lambda r: f"{r.position_rmse:>{col_w}.4f}"),
        ("Max Pos Error [m]", lambda r: f"{r.max_position_error:>{col_w}.4f}"),
        ("Heading RMSE [rad]", lambda r: f"{r.heading_rmse:>{col_w}.4f}"),
        ("Control Rate RMS", lambda r: f"{r.control_rate:>{col_w}.4f}"),
        ("Avg Solve [ms]", lambda r: f"{np.mean(r.solve_times)*1000:>{col_w}.3f}"),
        ("Avg ESS", lambda r: f"{np.mean(r.ess_history):>{col_w}.1f}"),
        ("Min Obs Dist [m]", lambda r: f"{r.min_obstacle_distance:>{col_w}.4f}"),
        ("Wall Time [s]", lambda r: f"{r.total_wall_time:>{col_w}.3f}"),
    ]
    for label, fmt in metrics:
        row = "| {:<30s}".format(label) + "".join(fmt(r) for r in results) + " |"
        lines.append(row)

    lines.append(sep)
    lines.append("")
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────

ALPHA_COLORS = {
    0.3: "tab:red",
    0.5: "tab:orange",
    0.8: "tab:green",
    1.0: "tab:blue",
}


def plot_comparison(
    results: List[RunResult],
    reference: np.ndarray,
    obstacles: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 궤적 + 장애물
    ax = axes[0, 0]
    ax.plot(reference[:, 0], reference[:, 1], "k--", lw=1.5, alpha=0.4, label="Ref")
    if obstacles is not None:
        for obs in obstacles:
            circle = plt.Circle((obs[0], obs[1]), obs[2], color="red", alpha=0.3)
            ax.add_patch(circle)
            ax.plot(obs[0], obs[1], "rx", markersize=8)
    for r in results:
        c = ALPHA_COLORS.get(r.alpha_value, "tab:gray")
        ax.plot(r.states[:, 0], r.states[:, 1], color=c, lw=1.5, label=r.name)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title("Trajectory + Obstacles"); ax.legend(fontsize=7)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    # 위치 오차
    ax = axes[0, 1]
    for r in results:
        c = ALPHA_COLORS.get(r.alpha_value, "tab:gray")
        err = np.linalg.norm(r.tracking_errors[:, :2], axis=1)
        ax.plot(r.time_array, err, color=c, lw=1, label=r.name)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Pos Error [m]")
    ax.set_title("Position Error"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ESS
    ax = axes[0, 2]
    for r in results:
        c = ALPHA_COLORS.get(r.alpha_value, "tab:gray")
        ax.plot(r.time_array, r.ess_history, color=c, lw=1, label=r.name)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("ESS")
    ax.set_title("Effective Sample Size"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 장애물 거리
    ax = axes[1, 0]
    for r in results:
        c = ALPHA_COLORS.get(r.alpha_value, "tab:gray")
        ax.plot(r.time_array, r.obstacle_distances, color=c, lw=1, label=r.name)
    ax.axhline(y=0, color="red", ls="--", alpha=0.5, label="Collision")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Min Obs Dist [m]")
    ax.set_title("Obstacle Distance"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 제어 변화율
    ax = axes[1, 1]
    for r in results:
        c = ALPHA_COLORS.get(r.alpha_value, "tab:gray")
        if len(r.controls) > 1:
            du = np.linalg.norm(np.diff(r.controls, axis=0), axis=1)
            ax.plot(r.time_array[:len(du)], du, color=c, lw=1, alpha=0.7, label=r.name)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("|du|")
    ax.set_title("Control Rate"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 비용
    ax = axes[1, 2]
    for r in results:
        c = ALPHA_COLORS.get(r.alpha_value, "tab:gray")
        ax.plot(r.time_array, r.costs, color=c, lw=1, alpha=0.7, label=r.name)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Cost")
    ax.set_title("Cost"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle("Risk-Aware MPPI (CVaR): alpha comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────
# Live 리플레이
# ─────────────────────────────────────────────────────────────

def live_replay(
    results: List[RunResult],
    reference: np.ndarray,
    obstacles: Optional[np.ndarray] = None,
    dt: float = 0.05,
    update_interval: int = 2,
) -> None:
    n = len(results)
    if any(r.info_history is None for r in results):
        print("  [WARN] info_history 없음")
        return

    plt.ion()
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else list(axes)

    margin = 1.5
    x_min, x_max = reference[:, 0].min() - margin, reference[:, 0].max() + margin
    y_min, y_max = reference[:, 1].min() - margin, reference[:, 1].max() + margin

    traces, lines = [], []
    for i, r in enumerate(results):
        ax = axes[i]
        c = ALPHA_COLORS.get(r.alpha_value, "tab:gray")
        ax.set_title(r.name, fontsize=13, fontweight="bold", color=c)
        ax.plot(reference[:, 0], reference[:, 1], "k--", lw=1.5, alpha=0.4)
        if obstacles is not None:
            for obs in obstacles:
                circle = plt.Circle((obs[0], obs[1]), obs[2], color="red", alpha=0.3)
                ax.add_patch(circle)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        (line,) = ax.plot([], [], color=c, lw=2)
        traces.append(([], []))
        lines.append(line)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Risk-Aware MPPI Live — CVaR comparison", fontsize=14, fontweight="bold")
    fig.canvas.draw()

    n_steps = min(len(r.time_array) for r in results)
    for step in range(n_steps):
        for i, r in enumerate(results):
            s = r.states[step]
            traces[i][0].append(s[0])
            traces[i][1].append(s[1])
        if step % update_interval != 0:
            continue
        for i in range(n):
            lines[i].set_data(traces[i][0], traces[i][1])
        fig.canvas.draw()
        fig.canvas.flush_events()

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


def get_default_obstacles(traj_type: str) -> np.ndarray:
    """궤적 유형에 맞는 기본 장애물 배치.

    장애물은 경로 **근처**이지만 위가 아닌 곳에 배치하여
    safety 영역만 경로에 걸치도록 한다. (완전 차단 X, 부분 회피 유도)
    """
    if traj_type == "circle":
        # 경로에서 ~0.12m → safety=0.55m 침범, 부분 회피
        return np.array([
            [1.5, 1.5, 0.25],
            [-1.5, -1.5, 0.25],
        ])
    elif traj_type == "figure8":
        # 경로에서 0.3~0.4m → safety=0.55m 침범, 부분 회피
        return np.array([
            [1.5, -0.5, 0.25],
            [-0.5, 0.9, 0.25],
        ])
    else:
        return np.array([
            [3.0, 0.3, 0.25],
            [7.0, -0.2, 0.25],
        ])


def main():
    parser = argparse.ArgumentParser(description="Risk-Aware MPPI (CVaR) alpha 비교 데모")
    parser.add_argument("--trajectory", type=str, default="figure8",
                        choices=["circle", "figure8", "sine"])
    parser.add_argument("--alpha", type=float, nargs="*", default=None,
                        help="비교할 alpha 값 목록 (기본: 0.3 0.5 0.8 1.0)")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-obstacles", action="store_true")
    args = parser.parse_args()

    alpha_values = args.alpha if args.alpha else [0.3, 0.5, 0.8, 1.0]
    obstacles = None if args.no_obstacles else get_default_obstacles(args.trajectory)

    print("\n" + "=" * 64)
    print("    Risk-Aware MPPI (CVaR) Alpha Comparison Demo")
    print("=" * 64)
    print(f"  alpha values: {alpha_values}")
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Obstacles: {'None' if obstacles is None else f'{len(obstacles)} objects'}")

    robot_params = RobotParams(max_velocity=1.0, max_omega=1.5)
    sim_config = SimulationConfig(dt=0.05, max_time=20.0)
    trajectory = generate_trajectory(args.trajectory)
    initial_state = trajectory[0].copy()
    need_info = args.live

    results = []
    for alpha in alpha_values:
        # adaptive temp target_ess_ratio를 alpha에 비례 조정:
        # alpha=1.0 → target 0.5 (전체 512중 256)
        # alpha=0.5 → target 0.25 (활성 256중 128)
        # alpha=0.3 → target 0.15 (활성 154중 77)
        # 이를 통해 CVaR 절단 후 남은 샘플 내에서 적절한 ESS 목표 유지
        target_ess = 0.5 * alpha
        params = MPPIParams(
            N=20, K=512, dt=0.05, lambda_=10.0,
            noise_sigma=np.array([0.3, 0.3]),
            Q=np.diag([10.0, 10.0, 1.0]),
            R=np.diag([0.01, 0.01]),
            Qf=np.diag([100.0, 100.0, 10.0]),
            cvar_alpha=alpha,
            adaptive_temperature=True,
            adaptive_temp_config={
                "target_ess_ratio": target_ess,
                "adaptation_rate": 1.0,
                "lambda_min": 0.001,
                "lambda_max": 100.0,
            },
        )
        ctrl = RiskAwareMPPIController(
            robot_params=robot_params, mppi_params=params, seed=42,
            obstacles=obstacles,
        )
        interp = TrajectoryInterpolator(trajectory, dt=sim_config.dt)
        name = f"a={alpha}"

        print(f"\n  Running Risk-Aware MPPI ({name}) ...")
        result = simulate(
            ctrl, name, alpha, interp,
            initial_state.copy(), sim_config, robot_params,
            obstacles=obstacles, store_info=need_info,
        )
        print(f"    -> {len(result.time_array)} steps, "
              f"RMSE={result.position_rmse:.4f}m, "
              f"ESS={np.mean(result.ess_history):.1f}, "
              f"MinObsDist={result.min_obstacle_distance:.3f}m")
        results.append(result)

    print_summary(results)

    if args.live:
        live_replay(results, trajectory, obstacles=obstacles)
    elif not args.no_plot:
        plot_comparison(results, trajectory, obstacles=obstacles, save_path=args.save)
        if args.save is None:
            plt.show()


if __name__ == "__main__":
    main()
