"""8-panel matplotlib 대시보드 + ASCII 요약표 + Live 시뮬레이션.

Panel 구성 (batch):
  1. 궤적 비교 (Py/C++ XY)
  2. 위치 오차 vs Time
  3. 풀이 시간 vs Time
  4. 컴포넌트 분석 (bar chart)
  5. K 스케일링 (log-log)
  6. N 스케일링 (log-log)
  7. 전략별 비교 (grouped bar)
  8. 종합 Speedup (bar)

Live 모드:
  Python과 C++ MPPI를 동시에 실시간 시뮬레이션하며
  궤적, 샘플 분포, 제어 메트릭을 시각화.
"""

import time
from typing import Dict, List, Optional

import numpy as np

from .benchmark_runner import BenchmarkResults, RunResult, ComponentResult, ScalingResult


# ─────────────────────────────────────────────────────────────
# ASCII 요약
# ─────────────────────────────────────────────────────────────

def _print_pipeline_section(
    pipeline: list,
    ref_mode_label: str,
) -> None:
    """단일 ref_mode에 대한 파이프라인 결과 섹션 출력."""
    if not pipeline:
        return

    dd_results = [r for r in pipeline if r.model_type == "diff_drive"]
    py_by_wt = {r.weight_type: r for r in dd_results if r.backend == "python"}
    cpp_by_wt = {r.weight_type: r for r in dd_results if r.backend == "cpp"}

    for wt in ["vanilla", "log", "tsallis", "risk_aware"]:
        py_r = py_by_wt.get(wt)
        cpp_r = cpp_by_wt.get(wt)
        if py_r and cpp_r:
            py_ms = py_r.avg_solve_ms
            cpp_ms = cpp_r.avg_solve_ms
            speedup = py_ms / cpp_ms if cpp_ms > 0 else float("inf")
            rmse_match = abs(py_r.position_rmse - cpp_r.position_rmse) < 0.5
            rmse_str = f"{'✓' if rmse_match else '✗'} {cpp_r.position_rmse:.3f}"
            label = f"DD/{wt[:8]}"
            print(f"│ {label:<14s} │ {py_ms:>8.1f} │ {cpp_ms:>8.1f} │ {speedup:>6.1f}x │ {rmse_str:<11s} │")

    cpp_only = [r for r in pipeline if r.model_type != "diff_drive"]
    if cpp_only:
        print("├" + "─" * 16 + "┼" + "─" * 10 + "┼" + "─" * 10 + "┼" + "─" * 9 + "┼" + "─" * 13 + "┤")
        for r in cpp_only:
            model_short = r.model_type[:3].upper()
            label = f"{model_short}/{r.weight_type[:8]}"
            print(f"│ {label:<14s} │    ---   │ {r.avg_solve_ms:>8.1f} │   ---   │ {r.position_rmse:>8.3f}   │")


def print_summary(results: BenchmarkResults) -> None:
    """ASCII 테이블로 벤치마크 결과 요약 출력."""
    pipeline = results.pipeline
    component = results.component
    scaling = results.scaling

    # ref_mode 분류
    ref_modes = sorted(set(r.ref_mode for r in pipeline)) if pipeline else ["time"]
    has_both = len(ref_modes) > 1

    print()
    print("┌" + "═" * 62 + "┐")
    print("│     Python vs C++ MPPI Benchmark Summary" + " " * 21 + "│")

    for rm in ref_modes:
        subset = [r for r in pipeline if r.ref_mode == rm]
        if has_both:
            print("├" + "═" * 62 + "┤")
            tag = "LOOKAHEAD" if rm == "lookahead" else "TIME-BASED"
            print(f"│  ▶ {tag:<57s} │")
        print("├" + "─" * 16 + "┬" + "─" * 10 + "┬" + "─" * 10 + "┬" + "─" * 9 + "┬" + "─" * 13 + "┤")
        print("│ Config         │ Py (ms)  │ C++ (ms) │ Speedup │ Pos RMSE    │")
        print("├" + "─" * 16 + "┼" + "─" * 10 + "┼" + "─" * 10 + "┼" + "─" * 9 + "┼" + "─" * 13 + "┤")
        _print_pipeline_section(subset, rm)

    print("├" + "─" * 16 + "┴" + "─" * 10 + "┴" + "─" * 10 + "┴" + "─" * 9 + "┴" + "─" * 13 + "┤")

    # Component Breakdown
    if component:
        print("│ Component Breakdown (DiffDrive):                             │")

        comp_names = sorted(set(c.component for c in component))
        for comp in comp_names:
            py_c = next((c for c in component if c.component == comp and c.backend == "python"), None)
            cpp_c = next((c for c in component if c.component == comp and c.backend == "cpp"), None)
            if py_c and cpp_c:
                speedup = py_c.median_ms / cpp_c.median_ms if cpp_c.median_ms > 0 else float("inf")
                bar_len = min(int(speedup * 2), 20)
                bar = "█" * bar_len
                print(f"│   {comp:<10s}: Py={py_c.median_ms:>6.2f}ms  C++={cpp_c.median_ms:>6.2f}ms  {bar} {speedup:.1f}x │")

    print("└" + "═" * 62 + "┘")

    # Scaling 요약
    if scaling:
        print()
        print("  Scaling Summary:")
        for s in scaling:
            if s.py_times_ms and s.cpp_times_ms:
                avg_speedup = np.mean(
                    [p / c for p, c in zip(s.py_times_ms, s.cpp_times_ms) if c > 0]
                )
                print(f"    {s.variable} scaling: avg speedup = {avg_speedup:.1f}x "
                      f"({s.variable}={s.values})")
    print()


# ─────────────────────────────────────────────────────────────
# Matplotlib 대시보드
# ─────────────────────────────────────────────────────────────

def plot_dashboard(
    results: BenchmarkResults,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """8-panel matplotlib 대시보드."""
    try:
        import matplotlib
        if save_path and not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Python vs C++ MPPI Benchmark", fontsize=14, fontweight="bold")

    pipeline = results.pipeline
    component = results.component
    scaling = results.scaling

    # ── Panel 1: 궤적 비교 ──
    ax = axes[0, 0]
    ax.set_title("Trajectory Comparison")
    dd_results = [r for r in pipeline if r.model_type == "diff_drive"]
    colors = {"python": "#1f77b4", "cpp": "#d62728"}
    for r in dd_results:
        if r.weight_type == "vanilla" and len(r.states) > 0:
            c = colors[r.backend]
            ax.plot(r.states[:, 0], r.states[:, 1], color=c,
                    label=f"{r.backend}", linewidth=1.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: 위치 오차 vs Time ──
    ax = axes[0, 1]
    ax.set_title("Position Error (Vanilla)")
    for r in dd_results:
        if r.weight_type == "vanilla" and len(r.tracking_errors) > 0:
            c = colors[r.backend]
            pos_err = np.linalg.norm(r.tracking_errors[:, :2], axis=1)
            t = np.arange(len(pos_err)) * 0.05
            ax.plot(t, pos_err, color=c, label=r.backend, linewidth=1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error [m]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: 풀이 시간 vs Time ──
    ax = axes[0, 2]
    ax.set_title("Solve Time (Vanilla)")
    for r in dd_results:
        if r.weight_type == "vanilla" and len(r.solve_times) > 0:
            c = colors[r.backend]
            t = np.arange(len(r.solve_times)) * 0.05
            ax.plot(t, r.solve_times * 1000, color=c, label=r.backend, linewidth=1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Solve time [ms]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: 컴포넌트 분석 ──
    ax = axes[0, 3]
    ax.set_title("Component Breakdown")
    if component:
        comp_names = sorted(set(c.component for c in component))
        x = np.arange(len(comp_names))
        width = 0.35
        py_vals = [next((c.median_ms for c in component
                         if c.component == n and c.backend == "python"), 0) for n in comp_names]
        cpp_vals = [next((c.median_ms for c in component
                          if c.component == n and c.backend == "cpp"), 0) for n in comp_names]
        ax.bar(x - width / 2, py_vals, width, label="Python", color="#1f77b4")
        ax.bar(x + width / 2, cpp_vals, width, label="C++", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(comp_names, fontsize=8, rotation=30)
        ax.set_ylabel("Time [ms]")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 5: K 스케일링 ──
    ax = axes[1, 0]
    ax.set_title("K Scaling (Rollout)")
    k_data = next((s for s in scaling if s.variable == "K"), None)
    if k_data:
        ax.loglog(k_data.values, k_data.py_times_ms, "o-", color="#1f77b4", label="Python")
        ax.loglog(k_data.values, k_data.cpp_times_ms, "s-", color="#d62728", label="C++")
        ax.set_xlabel("K (samples)")
        ax.set_ylabel("Time [ms]")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # ── Panel 6: N 스케일링 ──
    ax = axes[1, 1]
    ax.set_title("N Scaling (Rollout)")
    n_data = next((s for s in scaling if s.variable == "N"), None)
    if n_data:
        ax.loglog(n_data.values, n_data.py_times_ms, "o-", color="#1f77b4", label="Python")
        ax.loglog(n_data.values, n_data.cpp_times_ms, "s-", color="#d62728", label="C++")
        ax.set_xlabel("N (horizon)")
        ax.set_ylabel("Time [ms]")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # ── Panel 7: 전략별 비교 ──
    ax = axes[1, 2]
    ax.set_title("Strategy Comparison (DD)")
    py_by_wt = {r.weight_type: r for r in dd_results if r.backend == "python"}
    cpp_by_wt = {r.weight_type: r for r in dd_results if r.backend == "cpp"}
    wts = [wt for wt in ["vanilla", "log", "tsallis", "risk_aware"]
           if wt in py_by_wt and wt in cpp_by_wt]
    if wts:
        x = np.arange(len(wts))
        width = 0.35
        py_ms = [py_by_wt[wt].avg_solve_ms for wt in wts]
        cpp_ms = [cpp_by_wt[wt].avg_solve_ms for wt in wts]
        ax.bar(x - width / 2, py_ms, width, label="Python", color="#1f77b4")
        ax.bar(x + width / 2, cpp_ms, width, label="C++", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(wts, fontsize=8, rotation=30)
        ax.set_ylabel("Avg solve [ms]")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 8: 종합 Speedup ──
    ax = axes[1, 3]
    ax.set_title("Overall Speedup")
    speedups = {}
    for wt in wts:
        py_ms_val = py_by_wt[wt].avg_solve_ms
        cpp_ms_val = cpp_by_wt[wt].avg_solve_ms
        if cpp_ms_val > 0:
            speedups[wt] = py_ms_val / cpp_ms_val
    if speedups:
        names = list(speedups.keys())
        vals = list(speedups.values())
        x = np.arange(len(names))
        bars = ax.bar(x, vals, color="#2ca02c")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, rotation=30)
        ax.set_ylabel("Speedup (x)")
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{v:.1f}x", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Dashboard saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Live 벤치마크 시뮬레이션
# ─────────────────────────────────────────────────────────────

def live_benchmark(
    scenario,
    K: int = 256,
    N: int = 20,
    weight_type: str = "vanilla",
    seed: int = 42,
    ref_mode: str = "time",
) -> None:
    """Python vs C++ MPPI 실시간 비교 시뮬레이션.

    좌측: Python MPPI, 우측: C++ MPPI를 동시에 실행하며
    궤적, 샘플 분포, 제어 메트릭을 실시간 업데이트.

    Parameters
    ----------
    ref_mode : str
        "time" — 시간 기반 인터폴레이션 (기본)
        "lookahead" — 위치 기반 lookahead 인터폴레이션
    """
    import logging
    logging.getLogger("mpc_controller").setLevel(logging.WARNING)
    for _name in ["mpc_controller", "mpc_controller.controllers",
                  "mpc_controller.controllers.mppi",
                  "mpc_controller.controllers.mppi.base_mppi"]:
        _lg = logging.getLogger(_name)
        _lg.setLevel(logging.WARNING)
        _lg.propagate = False

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    from mpc_controller import RobotParams, TrajectoryInterpolator
    from simulation.simulator import Simulator, SimulationConfig

    from .benchmark_runner import create_python_controller
    from .cpp_mppi_assembler import CppMPPIAssembler
    from .scenario import BenchmarkScenario, LookaheadInterpolator

    # ── 컨트롤러 생성 ──
    py_ctrl = create_python_controller(weight_type, K=K, N=N, seed=seed,
                                        obstacles=scenario.obstacles)
    cpp_ctrl = CppMPPIAssembler("diff_drive", weight_type, K=K, N=N, seed=seed,
                                 obstacles=scenario.obstacles)

    is_lookahead = (ref_mode == "lookahead")
    if is_lookahead:
        # Python/C++ 시뮬레이터 독립 프루닝 → 별도 인스턴스 필요
        py_interp = LookaheadInterpolator(scenario.trajectory, scenario.dt)
        cpp_interp = LookaheadInterpolator(scenario.trajectory, scenario.dt)
    else:
        interp = TrajectoryInterpolator(scenario.trajectory[:, :3], scenario.dt)
        py_interp = interp
        cpp_interp = interp
    robot_params = RobotParams()
    sim_config = SimulationConfig(dt=scenario.dt, max_time=scenario.sim_time)

    py_sim = Simulator(robot_params, sim_config)
    cpp_sim = Simulator(robot_params, sim_config)
    py_sim.reset(scenario.initial_state[:3].copy())
    cpp_sim.reset(scenario.initial_state[:3].copy())
    py_ctrl.reset()
    cpp_ctrl.reset()

    # ── 그래프 설정 ──
    plt.ion()
    fig = plt.figure(figsize=(22, 12))
    ref_label = "lookahead" if is_lookahead else "time-based"
    fig.suptitle(f"Python vs C++ MPPI Live Benchmark  [{weight_type}, {ref_label}]",
                 fontsize=14, fontweight="bold")

    gs = fig.add_gridspec(3, 4, height_ratios=[3, 1, 1],
                          hspace=0.35, wspace=0.3)

    # 상단 좌: Python 궤적
    ax_py = fig.add_subplot(gs[0, 0:2])
    ax_py.set_title("Python MPPI", fontsize=12, color="#1f77b4", fontweight="bold")
    ax_py.set_xlabel("x [m]"); ax_py.set_ylabel("y [m]")
    ax_py.grid(True, alpha=0.2); ax_py.set_aspect("equal")

    # 상단 우: C++ 궤적
    ax_cpp = fig.add_subplot(gs[0, 2:4])
    ax_cpp.set_title("C++ MPPI (pybind11)", fontsize=12, color="#d62728", fontweight="bold")
    ax_cpp.set_xlabel("x [m]"); ax_cpp.set_ylabel("y [m]")
    ax_cpp.grid(True, alpha=0.2); ax_cpp.set_aspect("equal")

    # 중단: 풀이 시간, 위치 오차
    ax_solve = fig.add_subplot(gs[1, 0:2])
    ax_solve.set_title("Solve Time [ms]", fontsize=10)
    ax_solve.set_xlabel("Step"); ax_solve.grid(True, alpha=0.2)

    ax_err = fig.add_subplot(gs[1, 2:4])
    ax_err.set_title("Position Error [m]", fontsize=10)
    ax_err.set_xlabel("Step"); ax_err.grid(True, alpha=0.2)

    # 하단: 정보 패널
    ax_info = fig.add_subplot(gs[2, :])
    ax_info.axis("off")
    info_text = ax_info.text(0.02, 0.95, "Starting...",
                              transform=ax_info.transAxes,
                              fontsize=10, verticalalignment="top",
                              fontfamily="monospace")

    # 참조 궤적 + 장애물 (양쪽에 동일하게)
    traj = scenario.trajectory[:, :3]
    margin = 1.5
    for ax in [ax_py, ax_cpp]:
        ax.plot(traj[:, 0], traj[:, 1], "k--", lw=1.5, alpha=0.3, label="Reference")
        if scenario.obstacles is not None:
            for obs in scenario.obstacles:
                circle = plt.Circle((obs[0], obs[1]), obs[2],
                                     color="red", alpha=0.2)
                ax.add_patch(circle)
                ax.plot(obs[0], obs[1], "rx", markersize=8)
        ax.set_xlim(traj[:, 0].min() - margin, traj[:, 0].max() + margin)
        ax.set_ylim(traj[:, 1].min() - margin, traj[:, 1].max() + margin)

    # 동적 라인 객체
    MAX_SAMPLES = 15

    def _make_lines(ax, color):
        sample_lines = []
        for _ in range(MAX_SAMPLES):
            (sl,) = ax.plot([], [], "-", color=color, alpha=0.05, lw=0.4)
            sample_lines.append(sl)
        (trace_line,) = ax.plot([], [], color=color, lw=2.5, alpha=0.9, label="Robot")
        (pred_line,) = ax.plot([], [], "c-", lw=2.0, alpha=0.8)
        (best_line,) = ax.plot([], [], "m-", lw=1.2, alpha=0.6)
        robot = patches.Rectangle((0, 0), 0.3, 0.2, angle=0, fill=True,
                                   facecolor=color, edgecolor="black", lw=2, alpha=0.9)
        ax.add_patch(robot)
        (dir_ln,) = ax.plot([], [], "k-", lw=2)
        ax.legend(fontsize=7, loc="upper right")
        return sample_lines, trace_line, pred_line, best_line, robot, dir_ln

    py_lines = _make_lines(ax_py, "#1f77b4")
    cpp_lines = _make_lines(ax_cpp, "#d62728")

    # 풀이 시간 / 오차 라인
    (py_solve_line,) = ax_solve.plot([], [], color="#1f77b4", lw=1, label="Python")
    (cpp_solve_line,) = ax_solve.plot([], [], color="#d62728", lw=1, label="C++")
    ax_solve.legend(fontsize=8)

    (py_err_line,) = ax_err.plot([], [], color="#1f77b4", lw=1, label="Python")
    (cpp_err_line,) = ax_err.plot([], [], color="#d62728", lw=1, label="C++")
    ax_err.legend(fontsize=8)

    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # ── 시뮬레이션 루프 ──
    num_steps = int(scenario.sim_time / scenario.dt)
    ctrl_N = N
    ctrl_dt = scenario.dt

    py_trace_x, py_trace_y = [], []
    cpp_trace_x, cpp_trace_y = [], []
    py_solve_times, cpp_solve_times = [], []
    py_errors, cpp_errors = [], []

    def _update_robot(state, robot_patch, dir_ln, color):
        x, y, theta = state[:3]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cx = x - (0.15 * cos_t - 0.1 * sin_t)
        cy = y - (0.15 * sin_t + 0.1 * cos_t)
        robot_patch.set_xy((cx, cy))
        robot_patch.angle = np.degrees(theta)
        robot_patch.set_facecolor(color)
        dir_ln.set_data([x, x + 0.24 * cos_t], [y, y + 0.24 * sin_t])

    def _update_samples(info, sample_lines, pred_line, best_line, color):
        sample_traj = info.get("sample_trajectories")
        sample_weights = info.get("sample_weights")
        if sample_traj is not None and sample_weights is not None:
            top_idx = np.argsort(sample_weights)[-MAX_SAMPLES:]
            max_w = np.max(sample_weights)
            for rank, idx in enumerate(top_idx):
                alpha = float(np.clip(sample_weights[idx] / max_w * 0.4, 0.02, 0.4))
                sample_lines[rank].set_data(
                    sample_traj[idx, :, 0], sample_traj[idx, :, 1])
                sample_lines[rank].set_alpha(alpha)
                sample_lines[rank].set_color(color)

        pred = info.get("predicted_trajectory")
        if pred is not None:
            pred_line.set_data(pred[:, 0], pred[:, 1])
            pred_line.set_color(color)

        best = info.get("best_trajectory")
        if best is not None:
            best_line.set_data(best[:, 0], best[:, 1])
            best_line.set_color(color)

    for step in range(num_steps):
        t = step * scenario.dt

        # ── Python 스텝 ──
        py_state = py_sim.get_measurement()
        if is_lookahead:
            py_ref = py_interp.get_reference(py_state, ctrl_N, ctrl_dt,
                                             current_theta=py_state[2])
        else:
            py_ref = py_interp.get_reference(t, ctrl_N, ctrl_dt,
                                             current_theta=py_state[2])
        py_u, py_info = py_ctrl.compute_control(py_state, py_ref)
        py_next = py_sim.step(py_u[:2])

        py_trace_x.append(py_state[0])
        py_trace_y.append(py_state[1])
        py_solve_times.append(py_info["solve_time"] * 1000)
        py_pos_err = np.sqrt((py_state[0] - py_ref[0, 0])**2 +
                              (py_state[1] - py_ref[0, 1])**2)
        py_errors.append(py_pos_err)

        # ── C++ 스텝 ──
        cpp_state = cpp_sim.get_measurement()
        if is_lookahead:
            cpp_ref = cpp_interp.get_reference(cpp_state, ctrl_N, ctrl_dt,
                                               current_theta=cpp_state[2])
        else:
            cpp_ref = cpp_interp.get_reference(t, ctrl_N, ctrl_dt,
                                               current_theta=cpp_state[2])
        cpp_u, cpp_info = cpp_ctrl.compute_control(cpp_state, cpp_ref)
        cpp_next = cpp_sim.step(cpp_u[:2])

        cpp_trace_x.append(cpp_state[0])
        cpp_trace_y.append(cpp_state[1])
        cpp_solve_times.append(cpp_info["solve_time"] * 1000)
        cpp_pos_err = np.sqrt((cpp_state[0] - cpp_ref[0, 0])**2 +
                               (cpp_state[1] - cpp_ref[0, 1])**2)
        cpp_errors.append(cpp_pos_err)

        # ── 시각화 업데이트 (매 2스텝) ──
        if step % 2 == 0:
            # 궤적 트레이스
            py_lines[1].set_data(py_trace_x, py_trace_y)
            cpp_lines[1].set_data(cpp_trace_x, cpp_trace_y)

            # 샘플 궤적
            _update_samples(py_info, py_lines[0], py_lines[2], py_lines[3], "#1f77b4")
            _update_samples(cpp_info, cpp_lines[0], cpp_lines[2], cpp_lines[3], "#d62728")

            # 로봇 위치
            _update_robot(py_state, py_lines[4], py_lines[5], "#1f77b4")
            _update_robot(cpp_state, cpp_lines[4], cpp_lines[5], "#d62728")

            # 풀이 시간 그래프
            steps_arr = list(range(len(py_solve_times)))
            py_solve_line.set_data(steps_arr, py_solve_times)
            cpp_solve_line.set_data(steps_arr, cpp_solve_times)
            ax_solve.set_xlim(0, max(len(steps_arr), 10))
            ax_solve.set_ylim(0, max(max(py_solve_times[-20:], default=1),
                                      max(cpp_solve_times[-20:], default=1)) * 1.5)

            # 오차 그래프
            py_err_line.set_data(steps_arr, py_errors)
            cpp_err_line.set_data(steps_arr, cpp_errors)
            ax_err.set_xlim(0, max(len(steps_arr), 10))
            ax_err.set_ylim(0, max(max(py_errors[-20:], default=1),
                                    max(cpp_errors[-20:], default=1)) * 1.2)

            # 정보 텍스트
            py_avg_ms = np.mean(py_solve_times) if py_solve_times else 0
            cpp_avg_ms = np.mean(cpp_solve_times) if cpp_solve_times else 0
            speedup = py_avg_ms / cpp_avg_ms if cpp_avg_ms > 0 else 0
            py_rmse = np.sqrt(np.mean(np.array(py_errors)**2)) if py_errors else 0
            cpp_rmse = np.sqrt(np.mean(np.array(cpp_errors)**2)) if cpp_errors else 0

            info_str = (
                f"Step {step:4d}/{num_steps}  │  Time {t:.2f}s  │  "
                f"Strategy: {weight_type}  │  K={K}  N={N}  │  Ref: {ref_label}\n"
                f"─────────────────────────────────────────────────────"
                f"──────────────────────────────────────\n"
                f"  Python:  v={py_u[0]:+.3f} m/s  ω={py_u[1]:+.3f} rad/s  │  "
                f"solve={py_info['solve_time']*1000:.2f}ms  "
                f"ESS={py_info.get('ess',0):.0f}/{K}  "
                f"pos_err={py_pos_err:.4f}m  "
                f"avg={py_avg_ms:.2f}ms\n"
                f"  C++:     v={cpp_u[0]:+.3f} m/s  ω={cpp_u[1]:+.3f} rad/s  │  "
                f"solve={cpp_info['solve_time']*1000:.2f}ms  "
                f"ESS={cpp_info.get('ess',0):.0f}/{K}  "
                f"pos_err={cpp_pos_err:.4f}m  "
                f"avg={cpp_avg_ms:.2f}ms\n"
                f"─────────────────────────────────────────────────────"
                f"──────────────────────────────────────\n"
                f"  Speedup: {speedup:.2f}x  │  "
                f"Py RMSE: {py_rmse:.4f}  │  C++ RMSE: {cpp_rmse:.4f}"
            )
            info_text.set_text(info_str)

            fig.canvas.draw()
            fig.canvas.flush_events()

        # 궤적 완료 검사
        py_idx, py_dist = py_interp.find_closest_point(py_state[:2])
        cpp_idx, cpp_dist = cpp_interp.find_closest_point(cpp_state[:2])
        if (py_idx >= py_interp.num_points - 1 and py_dist < 0.1 and
                cpp_idx >= cpp_interp.num_points - 1 and cpp_dist < 0.1):
            break

    # ── 완료 ──
    py_avg_ms = np.mean(py_solve_times)
    cpp_avg_ms = np.mean(cpp_solve_times)
    speedup = py_avg_ms / cpp_avg_ms if cpp_avg_ms > 0 else 0
    py_rmse = np.sqrt(np.mean(np.array(py_errors)**2))
    cpp_rmse = np.sqrt(np.mean(np.array(cpp_errors)**2))

    final_str = (
        f"━━━━━ COMPLETE ({step+1} steps) ━━━━━\n\n"
        f"  Python:  avg_solve = {py_avg_ms:.2f} ms  │  RMSE = {py_rmse:.4f} m\n"
        f"  C++:     avg_solve = {cpp_avg_ms:.2f} ms  │  RMSE = {cpp_rmse:.4f} m\n"
        f"  Speedup: {speedup:.2f}x\n\n"
        f"  Close window to exit."
    )
    info_text.set_text(final_str)
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.ioff()
    plt.show()
