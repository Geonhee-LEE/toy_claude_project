"""8-panel matplotlib 대시보드 + ASCII 요약표.

Panel 구성:
  1. 궤적 비교 (Py/C++ XY)
  2. 위치 오차 vs Time
  3. 풀이 시간 vs Time
  4. 컴포넌트 분석 (bar chart)
  5. K 스케일링 (log-log)
  6. N 스케일링 (log-log)
  7. 전략별 비교 (grouped bar)
  8. 종합 Speedup (bar)
"""

from typing import Dict, List, Optional

import numpy as np

from .benchmark_runner import BenchmarkResults, RunResult, ComponentResult, ScalingResult


# ─────────────────────────────────────────────────────────────
# ASCII 요약
# ─────────────────────────────────────────────────────────────

def print_summary(results: BenchmarkResults) -> None:
    """ASCII 테이블로 벤치마크 결과 요약 출력."""
    pipeline = results.pipeline
    component = results.component
    scaling = results.scaling

    print()
    print("┌" + "═" * 62 + "┐")
    print("│     Python vs C++ MPPI Benchmark Summary" + " " * 21 + "│")
    print("├" + "─" * 16 + "┬" + "─" * 10 + "┬" + "─" * 10 + "┬" + "─" * 9 + "┬" + "─" * 13 + "┤")
    print("│ Config         │ Py (ms)  │ C++ (ms) │ Speedup │ Pos RMSE    │")
    print("├" + "─" * 16 + "┼" + "─" * 10 + "┼" + "─" * 10 + "┼" + "─" * 9 + "┼" + "─" * 13 + "┤")

    # DiffDrive 비교 (Python vs C++)
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

    # C++ only (Swerve, NonCoaxial)
    cpp_only = [r for r in pipeline if r.model_type != "diff_drive"]
    if cpp_only:
        print("├" + "─" * 16 + "┼" + "─" * 10 + "┼" + "─" * 10 + "┼" + "─" * 9 + "┼" + "─" * 13 + "┤")
        for r in cpp_only:
            model_short = r.model_type[:3].upper()
            label = f"{model_short}/{r.weight_type[:8]}"
            print(f"│ {label:<14s} │    ---   │ {r.avg_solve_ms:>8.1f} │   ---   │ {r.position_rmse:>8.3f}   │")

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
