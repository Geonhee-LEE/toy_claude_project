#!/usr/bin/env python3
"""MPPI 전체 변형 종합 벤치마크 — 9개 변형 동일 조건 비교.

모든 MPPI 변형을 동일한 궤적/장애물/파라미터로 시뮬레이션 후
성능 메트릭을 종합 비교한다.

실행:
    python examples/mppi_all_variants_benchmark.py
    python examples/mppi_all_variants_benchmark.py --live
    python examples/mppi_all_variants_benchmark.py --trajectory figure8
    python examples/mppi_all_variants_benchmark.py --save benchmark.png
    python examples/mppi_all_variants_benchmark.py --no-plot
    python examples/mppi_all_variants_benchmark.py --variants vanilla smooth spline
"""

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from mpc_controller import (
    DifferentialDriveModel,
    MPPIParams,
    RobotParams,
    TrajectoryInterpolator,
    generate_circle_trajectory,
    generate_figure_eight_trajectory,
    generate_sinusoidal_trajectory,
)
from mpc_controller.controllers.mppi.base_mppi import MPPIController
from mpc_controller.controllers.mppi.tube_mppi import TubeMPPIController
from mpc_controller.controllers.mppi.log_mppi import LogMPPIController
from mpc_controller.controllers.mppi.tsallis_mppi import TsallisMPPIController
from mpc_controller.controllers.mppi.risk_aware_mppi import RiskAwareMPPIController
from mpc_controller.controllers.mppi.stein_variational_mppi import (
    SteinVariationalMPPIController,
)
from mpc_controller.controllers.mppi.smooth_mppi import SmoothMPPIController
from mpc_controller.controllers.mppi.spline_mppi import SplineMPPIController
from mpc_controller.controllers.mppi.svg_mppi import SVGMPPIController
from simulation.simulator import Simulator, SimulationConfig


# ─────────────────────────────────────────────────────────────
# 변형별 색상 및 설정
# ─────────────────────────────────────────────────────────────

VARIANT_COLORS = {
    "Vanilla":   "#1f77b4",  # blue
    "Tube":      "#2ca02c",  # green
    "Log":       "#17becf",  # cyan
    "Tsallis":   "#9467bd",  # purple
    "CVaR":      "#d62728",  # red
    "SVMPC":     "#ff7f0e",  # orange
    "Smooth":    "#e377c2",  # pink
    "Spline":    "#8c564b",  # brown
    "SVG":       "#bcbd22",  # olive
}

ALL_VARIANT_NAMES = list(VARIANT_COLORS.keys())


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
        valid = self.obstacle_distances[np.isfinite(self.obstacle_distances)]
        return float(np.min(valid)) if len(valid) > 0 else float("inf")

    @property
    def avg_solve_ms(self) -> float:
        return float(np.mean(self.solve_times)) * 1000


# ─────────────────────────────────────────────────────────────
# 컨트롤러 팩토리
# ─────────────────────────────────────────────────────────────

def _base_params(K: int = 512) -> dict:
    """모든 변형에 공통 적용되는 기본 파라미터."""
    return dict(
        N=20, K=K, dt=0.05, lambda_=10.0,
        noise_sigma=np.array([0.3, 0.3]),
        Q=np.diag([10.0, 10.0, 1.0]),
        R=np.diag([0.01, 0.01]),
        Qf=np.diag([100.0, 100.0, 10.0]),
        adaptive_temperature=True,
        adaptive_temp_config={
            "target_ess_ratio": 0.5,
            "adaptation_rate": 1.0,
            "lambda_min": 0.001,
            "lambda_max": 100.0,
        },
    )


def create_controller(
    variant: str,
    robot_params: RobotParams,
    obstacles: Optional[np.ndarray] = None,
    K: int = 512,
    seed: int = 42,
):
    bp = _base_params(K)

    if variant == "Vanilla":
        return MPPIController(
            robot_params, MPPIParams(**bp), seed=seed, obstacles=obstacles,
        )
    elif variant == "Tube":
        bp["tube_enabled"] = True
        return TubeMPPIController(
            robot_params, MPPIParams(**bp), seed=seed, obstacles=obstacles,
        )
    elif variant == "Log":
        return LogMPPIController(
            robot_params, MPPIParams(**bp), seed=seed, obstacles=obstacles,
        )
    elif variant == "Tsallis":
        bp["tsallis_q"] = 1.1  # mild heavy-tail (1.2 too aggressive at low K)
        return TsallisMPPIController(
            robot_params, MPPIParams(**bp), seed=seed, obstacles=obstacles,
        )
    elif variant == "CVaR":
        bp["cvar_alpha"] = 0.7  # moderate risk-averse (0.5 too aggressive)
        return RiskAwareMPPIController(
            robot_params, MPPIParams(**bp), seed=seed, obstacles=obstacles,
        )
    elif variant == "SVMPC":
        bp["svgd_num_iterations"] = 3
        bp["svgd_step_size"] = 0.1
        return SteinVariationalMPPIController(
            robot_params, MPPIParams(**bp), seed=seed, obstacles=obstacles,
        )
    elif variant == "Smooth":
        bp["smooth_R_jerk"] = np.array([1.0, 1.0])
        bp["smooth_action_cost_weight"] = 2.0
        return SmoothMPPIController(
            robot_params, MPPIParams(**bp), seed=seed, obstacles=obstacles,
        )
    elif variant == "Spline":
        bp["spline_num_knots"] = 8
        bp["spline_degree"] = 3
        return SplineMPPIController(
            robot_params, MPPIParams(**bp), seed=seed, obstacles=obstacles,
        )
    elif variant == "SVG":
        bp["svgd_num_iterations"] = 3
        bp["svgd_step_size"] = 0.1
        bp["svg_num_guide_particles"] = 12
        bp["svg_guide_step_size"] = 0.15
        bp["svg_guide_iterations"] = 2
        bp["svg_resample_std"] = 0.15
        return SVGMPPIController(
            robot_params, MPPIParams(**bp), seed=seed, obstacles=obstacles,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ─────────────────────────────────────────────────────────────
# 시뮬레이션 루프
# ─────────────────────────────────────────────────────────────

def compute_obstacle_distance(state: np.ndarray, obstacles: np.ndarray) -> float:
    if obstacles is None or len(obstacles) == 0:
        return float("inf")
    dists = np.linalg.norm(state[:2] - obstacles[:, :2], axis=1) - obstacles[:, 2]
    return float(np.min(dists))


def simulate(
    controller,
    name: str,
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
        obstacle_distances=np.array(obs_dist_list),
        info_history=info_list,
    )


# ─────────────────────────────────────────────────────────────
# ASCII 종합 요약표
# ─────────────────────────────────────────────────────────────

def print_summary(results: List[RunResult]) -> None:
    n = len(results)
    col_w = 12

    # 이름 축약 (열 폭에 맞게)
    names = [r.name[:col_w] for r in results]
    header_cols = "".join(f"{nm:>{col_w}s}" for nm in names)

    title = "MPPI All Variants Benchmark"
    total_w = 28 + col_w * n + 2
    sep = "+" + "=" * total_w + "+"

    lines = [
        "", sep,
        "|{:^{w}s}|".format(title, w=total_w),
        sep,
        "| {:<28s}".format("Metric") + header_cols + " |",
        "+" + "-" * total_w + "+",
    ]

    metrics = [
        ("Pos RMSE [m]",      lambda r: f"{r.position_rmse:>{col_w}.4f}"),
        ("Max Pos Err [m]",   lambda r: f"{r.max_position_error:>{col_w}.4f}"),
        ("Heading RMSE [rad]",lambda r: f"{r.heading_rmse:>{col_w}.4f}"),
        ("Control Rate",      lambda r: f"{r.control_rate:>{col_w}.4f}"),
        ("Avg Solve [ms]",    lambda r: f"{r.avg_solve_ms:>{col_w}.2f}"),
        ("Avg Cost",          lambda r: f"{np.mean(r.costs):>{col_w}.2f}"),
        ("Avg ESS",           lambda r: f"{np.mean(r.ess_history):>{col_w}.1f}"),
        ("Min Obs Dist [m]",  lambda r: f"{r.min_obstacle_distance:>{col_w}.3f}"),
        ("Wall Time [s]",     lambda r: f"{r.total_wall_time:>{col_w}.2f}"),
        ("Steps",             lambda r: f"{len(r.time_array):>{col_w}d}"),
    ]

    for label, fmt in metrics:
        row = "| {:<28s}".format(label) + "".join(fmt(r) for r in results) + " |"
        lines.append(row)

    lines.append(sep)

    # 순위표 (Position RMSE)
    sorted_by_rmse = sorted(results, key=lambda r: r.position_rmse)
    lines.append("")
    lines.append("  Rankings (Position RMSE):")
    for rank, r in enumerate(sorted_by_rmse, 1):
        bar_len = int(r.position_rmse * 100)
        bar = "█" * min(bar_len, 40)
        lines.append(f"    {rank}. {r.name:<10s} {r.position_rmse:.4f}m  {bar}")

    # 순위표 (Control Smoothness)
    sorted_by_rate = sorted(results, key=lambda r: r.control_rate)
    lines.append("")
    lines.append("  Rankings (Control Smoothness — lower = smoother):")
    for rank, r in enumerate(sorted_by_rate, 1):
        bar_len = int(r.control_rate * 50)
        bar = "▓" * min(bar_len, 40)
        lines.append(f"    {rank}. {r.name:<10s} {r.control_rate:.4f}   {bar}")

    # 순위표 (Solve Speed)
    sorted_by_speed = sorted(results, key=lambda r: r.avg_solve_ms)
    lines.append("")
    lines.append("  Rankings (Solve Speed — lower = faster):")
    for rank, r in enumerate(sorted_by_speed, 1):
        bar_len = int(r.avg_solve_ms * 2)
        bar = "░" * min(bar_len, 40)
        lines.append(f"    {rank}. {r.name:<10s} {r.avg_solve_ms:.2f}ms  {bar}")

    lines.append("")
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────
# 정적 시각화 (6-panel)
# ─────────────────────────────────────────────────────────────

def plot_comparison(
    results: List[RunResult],
    reference: np.ndarray,
    obstacles: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> "plt.Figure":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # ── 1. 궤적 + 장애물 ──
    ax = axes[0, 0]
    ax.plot(reference[:, 0], reference[:, 1], "k--", lw=1, alpha=0.3, label="Ref")
    if obstacles is not None:
        for obs in obstacles:
            circle = plt.Circle((obs[0], obs[1]), obs[2], color="red", alpha=0.25)
            ax.add_patch(circle)
            ax.plot(obs[0], obs[1], "rx", markersize=6)
    for r in results:
        c = VARIANT_COLORS.get(r.name, "gray")
        ax.plot(r.states[:, 0], r.states[:, 1], color=c, lw=1.2,
                label=r.name, alpha=0.85)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title("Trajectories"); ax.legend(fontsize=6, ncol=2)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.2)

    # ── 2. 위치 오차 ──
    ax = axes[0, 1]
    for r in results:
        c = VARIANT_COLORS.get(r.name, "gray")
        err = np.linalg.norm(r.tracking_errors[:, :2], axis=1)
        ax.plot(r.time_array, err, color=c, lw=0.8, alpha=0.8, label=r.name)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Pos Error [m]")
    ax.set_title("Position Error"); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.2)

    # ── 3. 제어 변화율 (smoothness) ──
    ax = axes[0, 2]
    for r in results:
        c = VARIANT_COLORS.get(r.name, "gray")
        if len(r.controls) > 1:
            du = np.linalg.norm(np.diff(r.controls, axis=0), axis=1)
            ax.plot(r.time_array[:len(du)], du, color=c, lw=0.7,
                    alpha=0.7, label=r.name)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("|Δu|")
    ax.set_title("Control Rate (Smoothness)"); ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.2)

    # ── 4. 풀이 시간 ──
    ax = axes[1, 0]
    for r in results:
        c = VARIANT_COLORS.get(r.name, "gray")
        ax.plot(r.time_array, r.solve_times * 1000, color=c, lw=0.7,
                alpha=0.7, label=r.name)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Solve [ms]")
    ax.set_title("Solve Time"); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.2)

    # ── 5. ESS ──
    ax = axes[1, 1]
    for r in results:
        c = VARIANT_COLORS.get(r.name, "gray")
        ax.plot(r.time_array, r.ess_history, color=c, lw=0.7,
                alpha=0.7, label=r.name)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("ESS")
    ax.set_title("Effective Sample Size"); ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.2)

    # ── 6. 종합 바 차트 ──
    ax = axes[1, 2]
    names = [r.name for r in results]
    x = np.arange(len(names))
    width = 0.3

    rmse_vals = [r.position_rmse for r in results]
    rate_vals = [r.control_rate for r in results]
    speed_vals = [r.avg_solve_ms / 100 for r in results]  # scale for visibility

    colors = [VARIANT_COLORS.get(n, "gray") for n in names]

    bars1 = ax.bar(x - width, rmse_vals, width, label="RMSE [m]",
                   color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, rate_vals, width, label="Ctrl Rate",
                   color=colors, alpha=0.5, edgecolor="black", linewidth=0.5,
                   hatch="//")
    bars3 = ax.bar(x + width, speed_vals, width, label="Solve [×100ms]",
                   color=colors, alpha=0.3, edgecolor="black", linewidth=0.5,
                   hatch="xx")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_title("Summary Comparison")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("MPPI All Variants Benchmark", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────
# Live 모드 — 전체 변형 순차 실시간 시뮬레이션
# ─────────────────────────────────────────────────────────────

def live_benchmark(
    variants: List[str],
    trajectory: np.ndarray,
    initial_state: np.ndarray,
    sim_config: SimulationConfig,
    robot_params: RobotParams,
    obstacles: Optional[np.ndarray] = None,
) -> None:
    """전 변형을 순차적으로 실시간 시뮬레이션 + 결과 오버레이."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    K_live = 256  # live 모드 샘플 수 축소

    # ── 결과 누적 저장 ──
    all_results: List[RunResult] = []

    plt.ion()
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1])

    # 상단 좌: 궤적 뷰 (모든 변형 오버레이)
    ax_traj = fig.add_subplot(gs[0, 0:2])
    ax_traj.set_title("MPPI Live Benchmark", fontsize=13)
    ax_traj.set_xlabel("X [m]"); ax_traj.set_ylabel("Y [m]")
    ax_traj.grid(True, alpha=0.2); ax_traj.set_aspect("equal")

    # 참조 궤적
    ax_traj.plot(trajectory[:, 0], trajectory[:, 1],
                 "k--", lw=1.5, alpha=0.3, label="Reference")
    if obstacles is not None:
        for obs in obstacles:
            circle = plt.Circle((obs[0], obs[1]), obs[2], color="red", alpha=0.25)
            ax_traj.add_patch(circle)
            ax_traj.plot(obs[0], obs[1], "rx", markersize=8)

    x_margin, y_margin = 1.5, 1.5
    ax_traj.set_xlim(trajectory[:, 0].min() - x_margin,
                      trajectory[:, 0].max() + x_margin)
    ax_traj.set_ylim(trajectory[:, 1].min() - y_margin,
                      trajectory[:, 1].max() + y_margin)

    # 상단 우: 현재 변형 MPPI 상세 뷰
    ax_detail = fig.add_subplot(gs[0, 2])
    ax_detail.axis("off")
    ax_detail.set_title("Current Variant")
    info_text = ax_detail.text(
        0.05, 0.95, "", transform=ax_detail.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
    )

    # 하단: 성능 메트릭 테이블
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")
    ax_table.set_title("Performance Summary", fontsize=11)
    table_text = ax_table.text(
        0.02, 0.95, "Starting benchmark...", transform=ax_table.transAxes,
        fontsize=8, verticalalignment="top", fontfamily="monospace",
    )

    # 샘플 궤적 라인 (현재 변형용)
    MAX_SAMPLES = 20
    sample_lines = []
    for _ in range(MAX_SAMPLES):
        (line,) = ax_traj.plot([], [], "-", color="steelblue", alpha=0.08, lw=0.4)
        sample_lines.append(line)
    (pred_line,) = ax_traj.plot([], [], "c-", lw=2.5, alpha=0.9)
    (best_line,) = ax_traj.plot([], [], "m-", lw=1.5, alpha=0.7)

    # 로봇 패치
    robot_patch = patches.Rectangle(
        (0, 0), 0.3, 0.2, angle=0, fill=True,
        facecolor="red", edgecolor="black", lw=2, alpha=0.9,
    )
    ax_traj.add_patch(robot_patch)
    (dir_line,) = ax_traj.plot([], [], "k-", lw=2)

    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # ── 변형별 순차 실행 ──
    for vi, variant_name in enumerate(variants):
        color = VARIANT_COLORS.get(variant_name, "gray")

        controller = create_controller(
            variant_name, robot_params, obstacles, K=K_live, seed=42,
        )
        interp = TrajectoryInterpolator(trajectory, dt=sim_config.dt)
        sim = Simulator(robot_params, sim_config)
        sim.reset(initial_state.copy())
        controller.reset()

        num_steps = int(sim_config.max_time / sim_config.dt)
        trace_x, trace_y = [], []
        controls_list, errors_list = [], []
        solve_times_list, costs_list = [], []
        ess_list, obs_dist_list = [], []
        states = [initial_state.copy()]
        time_list = []

        # 이전 궤적 라인 (축적)
        (trace_line,) = ax_traj.plot([], [], color=color, lw=2.0,
                                      alpha=0.85, label=variant_name)
        ax_traj.legend(fontsize=7, ncol=3, loc="upper right")

        for step in range(num_steps):
            t = step * sim_config.dt
            state = sim.get_measurement()
            ref = interp.get_reference(
                t, controller.params.N, controller.params.dt,
                current_theta=state[2],
            )
            control, info = controller.compute_control(state, ref)
            next_state = sim.step(control)
            error = sim.compute_tracking_error(state, ref[0])

            trace_x.append(state[0])
            trace_y.append(state[1])
            time_list.append(t)
            states.append(next_state.copy())
            controls_list.append(control.copy())
            errors_list.append(error)
            solve_times_list.append(info["solve_time"])
            costs_list.append(info["cost"])
            ess_list.append(info.get("ess", 0))
            obs_dist_list.append(compute_obstacle_distance(state, obstacles))

            # 매 2스텝마다 시각화 업데이트
            if step % 2 == 0:
                trace_line.set_data(trace_x, trace_y)

                # 예측 궤적
                pred = info.get("predicted_trajectory")
                if pred is not None:
                    pred_line.set_data(pred[:, 0], pred[:, 1])
                    pred_line.set_color(color)

                # 샘플 궤적
                sample_traj = info.get("sample_trajectories")
                sample_weights = info.get("sample_weights")
                if sample_traj is not None and sample_weights is not None:
                    top_idx = np.argsort(sample_weights)[-MAX_SAMPLES:]
                    max_w = np.max(sample_weights)
                    for rank, idx in enumerate(top_idx):
                        alpha = float(np.clip(
                            sample_weights[idx] / max_w * 0.4, 0.02, 0.4,
                        ))
                        sample_lines[rank].set_data(
                            sample_traj[idx, :, 0], sample_traj[idx, :, 1],
                        )
                        sample_lines[rank].set_alpha(alpha)
                        sample_lines[rank].set_color(color)

                best_traj = info.get("best_trajectory")
                if best_traj is not None:
                    best_line.set_data(best_traj[:, 0], best_traj[:, 1])
                    best_line.set_color(color)

                # 로봇 위치
                x, y, theta = state
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                cx = x - (0.15 * cos_t - 0.1 * sin_t)
                cy = y - (0.15 * sin_t + 0.1 * cos_t)
                robot_patch.set_xy((cx, cy))
                robot_patch.angle = np.degrees(theta)
                robot_patch.set_facecolor(color)
                dir_line.set_data([x, x + 0.24 * cos_t],
                                  [y, y + 0.24 * sin_t])

                # 정보 텍스트
                pos_err = np.sqrt((state[0]-ref[0, 0])**2 + (state[1]-ref[0, 1])**2)
                ess = info.get("ess", 0)
                temp = info.get("temperature", 0)
                info_str = (
                    f"━━ {variant_name} ━━\n"
                    f"  [{vi+1}/{len(variants)}]\n\n"
                    f"Time:  {t:.2f}s\n"
                    f"Step:  {step}/{num_steps}\n\n"
                    f"State:\n"
                    f"  x={state[0]:+.3f}\n"
                    f"  y={state[1]:+.3f}\n"
                    f"  θ={np.degrees(state[2]):+.1f}°\n\n"
                    f"Control:\n"
                    f"  v={control[0]:+.3f} m/s\n"
                    f"  ω={control[1]:+.3f} rad/s\n\n"
                    f"Metrics:\n"
                    f"  pos_err={pos_err:.4f}m\n"
                    f"  cost={info['cost']:.2f}\n"
                    f"  solve={info['solve_time']*1000:.1f}ms\n"
                    f"  ESS={ess:.0f}/{K_live}\n"
                    f"  temp={temp:.1f}\n"
                )
                info_text.set_text(info_str)

                fig.canvas.draw()
                fig.canvas.flush_events()

            idx, dist = interp.find_closest_point(state[:2])
            if idx >= interp.num_points - 1 and dist < 0.1:
                break

        # 결과 저장
        result = RunResult(
            name=variant_name,
            states=np.array(states),
            controls=np.array(controls_list),
            references=np.array([]),
            tracking_errors=np.array(errors_list),
            solve_times=np.array(solve_times_list),
            costs=np.array(costs_list),
            time_array=np.array(time_list),
            total_wall_time=0,
            ess_history=np.array(ess_list),
            temp_history=np.array([]),
            obstacle_distances=np.array(obs_dist_list),
        )
        all_results.append(result)

        # 샘플 라인 클리어
        for sl in sample_lines:
            sl.set_data([], [])
        pred_line.set_data([], [])
        best_line.set_data([], [])

        # 하단 테이블 업데이트
        table_lines = [f"{'Variant':<10s} {'RMSE':>8s} {'CtrlRate':>10s} {'Solve':>9s} {'ESS':>7s} {'Steps':>7s}"]
        table_lines.append("-" * 55)
        for r in all_results:
            table_lines.append(
                f"{r.name:<10s} {r.position_rmse:>8.4f} {r.control_rate:>10.4f} "
                f"{r.avg_solve_ms:>8.2f}ms {np.mean(r.ess_history):>6.0f} {len(r.time_array):>6d}"
            )
        table_text.set_text("\n".join(table_lines))
        fig.canvas.draw()
        fig.canvas.flush_events()

    # 완료 표시
    info_text.set_text("━━ COMPLETE ━━\n\nAll variants done.\nClose window to exit.")
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.ioff()
    plt.show()


# ─────────────────────────────────────────────────────────────
# 궤적 생성
# ─────────────────────────────────────────────────────────────

def generate_trajectory(traj_type: str) -> np.ndarray:
    if traj_type == "circle":
        return generate_circle_trajectory(np.array([0.0, 0.0]), 2.0, 400)
    elif traj_type == "figure8":
        return generate_figure_eight_trajectory(np.array([0.0, 0.0]), 2.0, 400)
    else:
        return generate_sinusoidal_trajectory(
            np.array([0.0, 0.0]), 10.0, 1.0, 0.5, 400,
        )


def get_default_obstacles(traj_type: str) -> np.ndarray:
    if traj_type == "circle":
        return np.array([
            [1.5, 1.5, 0.25],
            [-1.5, -1.5, 0.25],
            [0.0, 2.2, 0.2],
        ])
    elif traj_type == "figure8":
        return np.array([
            [1.5, -0.5, 0.25],
            [-0.5, 0.9, 0.25],
            [-1.0, -0.8, 0.2],
        ])
    else:
        return np.array([
            [3.0, 0.3, 0.25],
            [7.0, -0.2, 0.25],
        ])


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MPPI 전체 변형 종합 벤치마크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
변형 목록:
  Vanilla   — 기본 MPPI (Williams et al., 2016)
  Tube      — Tube-MPPI (Williams et al., 2018)
  Log       — Log-space softmax (수치 안정)
  Tsallis   — q-exponential (q=1.1, 탐색 강화)
  CVaR      — Risk-Aware (α=0.7, 보수적)
  SVMPC     — Stein Variational (SVGD L=3)
  Smooth    — Δu input-lifting (jerk cost)
  Spline    — B-spline basis (P=8 knots)
  SVG       — Guide particle SVGD (G=16)
""",
    )
    parser.add_argument("--trajectory", type=str, default="figure8",
                        choices=["circle", "figure8", "sine"])
    parser.add_argument("--variants", type=str, nargs="*", default=None,
                        help=f"비교할 변형 (기본: 전체). 선택지: {ALL_VARIANT_NAMES}")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-obstacles", action="store_true")
    parser.add_argument("--K", type=int, default=512,
                        help="샘플 수 (기본: 512)")
    args = parser.parse_args()

    variants = args.variants if args.variants else ALL_VARIANT_NAMES
    # 유효성 검증
    for v in variants:
        if v not in ALL_VARIANT_NAMES:
            parser.error(f"Unknown variant: {v}. Choose from: {ALL_VARIANT_NAMES}")

    obstacles = None if args.no_obstacles else get_default_obstacles(args.trajectory)

    print()
    print("┌──────────────────────────────────────────────────────────────┐")
    print("│         MPPI All Variants Benchmark                         │")
    print("├──────────────────────────────────────────────────────────────┤")
    print(f"│  Trajectory:  {args.trajectory:<46s}│")
    print(f"│  Obstacles:   {'None' if obstacles is None else f'{len(obstacles)} objects':<46s}│")
    print(f"│  Variants:    {str(len(variants)) + ' / ' + str(len(ALL_VARIANT_NAMES)):<46s}│")
    print(f"│  Samples (K): {args.K:<46d}│")
    print(f"│  Mode:        {'LIVE' if args.live else 'BATCH':<46s}│")
    print("├──────────────────────────────────────────────────────────────┤")

    variant_desc = {
        "Vanilla": "기본 MPPI (baseline)",
        "Tube":    "Robust — 피드백 보정",
        "Log":     "log-softmax 수치 안정",
        "Tsallis": "q-exp (q=1.1) 탐색 강화",
        "CVaR":    "Risk-Aware (α=0.7) 보수적",
        "SVMPC":   "SVGD 커널 다양성 (L=3)",
        "Smooth":  "Δu input-lifting smooth",
        "Spline":  "B-spline basis (P=8)",
        "SVG":     "Guide SVGD (G=12) 다중모드",
    }
    for v in variants:
        desc = variant_desc.get(v, "")
        print(f"│  {v:<12s} {desc:<48s}│")
    print("└──────────────────────────────────────────────────────────────┘")
    print()

    robot_params = RobotParams(max_velocity=1.0, max_omega=1.5)
    sim_config = SimulationConfig(dt=0.05, max_time=20.0)
    trajectory = generate_trajectory(args.trajectory)
    initial_state = trajectory[0].copy()

    if args.live:
        live_benchmark(
            variants, trajectory, initial_state,
            sim_config, robot_params, obstacles,
        )
    else:
        results = []
        for vi, variant_name in enumerate(variants):
            controller = create_controller(
                variant_name, robot_params, obstacles, K=args.K, seed=42,
            )
            interp = TrajectoryInterpolator(trajectory, dt=sim_config.dt)

            print(f"  [{vi+1}/{len(variants)}] Running {variant_name} ...")
            result = simulate(
                controller, variant_name, interp,
                initial_state.copy(), sim_config, robot_params,
                obstacles=obstacles,
            )
            print(
                f"       RMSE={result.position_rmse:.4f}m  "
                f"Rate={result.control_rate:.4f}  "
                f"Solve={result.avg_solve_ms:.1f}ms  "
                f"ESS={np.mean(result.ess_history):.0f}  "
                f"Steps={len(result.time_array)}"
            )
            results.append(result)

        print_summary(results)

        if not args.no_plot:
            import matplotlib.pyplot as plt
            plot_comparison(
                results, trajectory,
                obstacles=obstacles, save_path=args.save,
            )
            if args.save is None:
                plt.show()


if __name__ == "__main__":
    main()
