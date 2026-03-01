#!/usr/bin/env python3
"""Python vs C++ MPPI 벤치마크 CLI.

실행 예시:
    python examples/cpp_vs_python_benchmark/run_benchmark.py              # 전체
    python examples/cpp_vs_python_benchmark/run_benchmark.py --quick      # K=128,N=10,repeat=3
    python examples/cpp_vs_python_benchmark/run_benchmark.py --live       # 실시간 Py vs C++ 비교
    python examples/cpp_vs_python_benchmark/run_benchmark.py --compare-ref              # 실시간 Time vs Lookahead (vanilla)
    python examples/cpp_vs_python_benchmark/run_benchmark.py --compare-ref --weight all # 4종 policy 순회
    python examples/cpp_vs_python_benchmark/run_benchmark.py --pipeline   # 파이프라인만
    python examples/cpp_vs_python_benchmark/run_benchmark.py --component  # 컴포넌트만
    python examples/cpp_vs_python_benchmark/run_benchmark.py --scaling    # 스케일링만
    python examples/cpp_vs_python_benchmark/run_benchmark.py --save report.png --json results.json
    python examples/cpp_vs_python_benchmark/run_benchmark.py --ref-mode lookahead --live
    python examples/cpp_vs_python_benchmark/run_benchmark.py --ref-mode both --pipeline
"""

import argparse
import json
import logging
import os
import sys

# 프로젝트 루트를 sys.path에 추가 (직접 실행 지원)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# MPPI 로거 verbose 억제 (import 전에 설정)
for _name in ["mpc_controller", "mpc_controller.controllers",
              "mpc_controller.controllers.mppi",
              "mpc_controller.controllers.mppi.base_mppi"]:
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.WARNING)
    _lg.propagate = False

from examples.cpp_vs_python_benchmark.scenario import (
    circle_scenario, tight_turn_scenario, slalom_scenario, figure8_scenario,
)
from examples.cpp_vs_python_benchmark.benchmark_runner import (
    run_all_benchmarks, run_steering_compare,
)
from examples.cpp_vs_python_benchmark.visualization import (
    print_summary, plot_dashboard, live_benchmark, live_ref_comparison,
)


ALL_WEIGHT_TYPES = ["vanilla", "log", "tsallis", "risk_aware"]


def main():
    parser = argparse.ArgumentParser(
        description="Python vs C++ MPPI Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: K=128, N=10, repeat=3")
    parser.add_argument("--live", action="store_true",
                        help="Live mode: 실시간 Python vs C++ 비교 시뮬레이션")
    parser.add_argument("--compare-ref", action="store_true",
                        help="Live mode: 실시간 Time vs Lookahead 비교 시뮬레이션")
    parser.add_argument("--pipeline", action="store_true",
                        help="Run pipeline benchmark only")
    parser.add_argument("--component", action="store_true",
                        help="Run component benchmark only")
    parser.add_argument("--scaling", action="store_true",
                        help="Run scaling benchmark only")
    parser.add_argument("--K", type=int, default=512,
                        help="Number of samples (default: 512)")
    parser.add_argument("--N", type=int, default=20,
                        help="Horizon steps (default: 20)")
    parser.add_argument("--repeat", type=int, default=20,
                        help="Repeat count for microbenchmarks (default: 20)")
    parser.add_argument("--weight", type=str, default="vanilla",
                        choices=["vanilla", "log", "tsallis", "risk_aware", "all"],
                        help="Weight strategy (default: vanilla, 'all' for all 4)")
    parser.add_argument("--sim-time", type=float, default=30.0,
                        help="Simulation time in seconds (default: 30.0)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save dashboard plot to file")
    parser.add_argument("--json", type=str, default=None,
                        help="Save results as JSON")
    parser.add_argument("--ref-mode", type=str, default="time",
                        choices=["time", "lookahead", "both"],
                        help="Reference trajectory mode: time-based, lookahead, or both (default: time)")
    parser.add_argument("--max-steering-angle", type=float, default=90.0,
                        help="Max steering angle in degrees (default: 90)")
    parser.add_argument("--steering-compare", action="store_true",
                        help="Compare 90° vs 60° steering constraints")
    parser.add_argument("--scenario", type=str, default="circle",
                        choices=["circle", "tight", "slalom", "figure8"],
                        help="Scenario for steering compare (default: circle)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip matplotlib plots")
    args = parser.parse_args()

    # Quick mode 오버라이드
    if args.quick:
        args.K = 128
        args.N = 10
        args.repeat = 3

    scenario = circle_scenario(nx=3)
    scenario.sim_time = args.sim_time

    # ── Steering Compare 모드 (90° vs 60°) ──
    if args.steering_compare:
        scenario_map = {
            "circle": lambda: circle_scenario(nx=4),
            "tight": lambda: tight_turn_scenario(nx=4),
            "slalom": lambda: slalom_scenario(nx=4),
            "figure8": lambda: figure8_scenario(nx=4),
        }
        sc = scenario_map[args.scenario]()
        sc.sim_time = args.sim_time
        print()
        print("╔" + "═" * 58 + "╗")
        print("║   Steering Constraint Comparison (90° vs 60°)            ║")
        print(f"║   Scenario: {args.scenario:<12s}  K={args.K:<6d}  N={args.N:<4d}       ║")
        print("╚" + "═" * 58 + "╝")
        print()
        results = run_steering_compare(
            sc, K=args.K, N=args.N, show_plot=not args.no_plot,
        )
        return

    # ── Compare-ref 모드 (Time vs Lookahead) ──
    if args.compare_ref:
        weight_types = ALL_WEIGHT_TYPES if args.weight == "all" else [args.weight]
        print()
        print("╔" + "═" * 58 + "╗")
        print("║   Time vs Lookahead Live Comparison                    ║")
        print(f"║   K={args.K:<6d}  N={args.N:<4d}  sim={args.sim_time:.0f}s"
              + " " * (58 - 28 - len(f"{args.sim_time:.0f}")) + "║")
        wt_str = ", ".join(weight_types)
        print(f"║   Strategies: {wt_str:<43s}║")
        print("╚" + "═" * 58 + "╝")
        print()
        live_ref_comparison(
            scenario, K=args.K, N=args.N,
            weight_types=weight_types, sim_time=args.sim_time,
        )
        return

    # ── Live 모드 (Python vs C++) ──
    if args.live:
        ref_mode_live = "lookahead" if args.ref_mode == "both" else args.ref_mode
        print()
        print("╔" + "═" * 58 + "╗")
        print("║   Python vs C++ MPPI Live Benchmark                    ║")
        print(f"║   K={args.K:<6d}  N={args.N:<4d}  weight={args.weight:<16s}     ║")
        print(f"║   Mode: LIVE   ref={ref_mode_live:<12s}                   ║")
        print("╚" + "═" * 58 + "╝")
        print()
        live_benchmark(scenario, K=args.K, N=args.N,
                       weight_type=args.weight, ref_mode=ref_mode_live)
        return

    # ── Batch 모드 ──
    run_pip = args.pipeline or (not args.pipeline and not args.component and not args.scaling)
    run_comp = args.component or (not args.pipeline and not args.component and not args.scaling)
    run_scl = args.scaling or (not args.pipeline and not args.component and not args.scaling)

    K_values = [128, 256, 512] if args.quick else None
    N_values = [10, 20] if args.quick else None

    print()
    print("╔" + "═" * 58 + "╗")
    print("║   Python vs C++ MPPI Benchmark Suite                    ║")
    print(f"║   K={args.K:<6d}  N={args.N:<4d}  repeat={args.repeat:<4d}                    ║")
    print(f"║   Mode: BATCH   ref={args.ref_mode:<12s}                  ║")
    print("╚" + "═" * 58 + "╝")
    print()

    results = run_all_benchmarks(
        K=args.K,
        N=args.N,
        repeat=args.repeat,
        scenario=scenario,
        run_pipeline=run_pip,
        run_component=run_comp,
        run_scaling=run_scl,
        K_values=K_values,
        N_values=N_values,
        ref_mode=args.ref_mode,
    )

    # ASCII 요약
    print_summary(results)

    # JSON 저장
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"  → Results saved to {args.json}")

    # 플롯
    if not args.no_plot:
        plot_dashboard(results, save_path=args.save, show=(args.save is None))


if __name__ == "__main__":
    main()
