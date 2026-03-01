#!/usr/bin/env python3
"""Python vs C++ MPPI 벤치마크 CLI.

실행 예시:
    python examples/cpp_vs_python_benchmark/run_benchmark.py              # 전체
    python examples/cpp_vs_python_benchmark/run_benchmark.py --quick      # K=128,N=10,repeat=3
    python examples/cpp_vs_python_benchmark/run_benchmark.py --live       # 실시간 시뮬레이션
    python examples/cpp_vs_python_benchmark/run_benchmark.py --pipeline   # 파이프라인만
    python examples/cpp_vs_python_benchmark/run_benchmark.py --component  # 컴포넌트만
    python examples/cpp_vs_python_benchmark/run_benchmark.py --scaling    # 스케일링만
    python examples/cpp_vs_python_benchmark/run_benchmark.py --save report.png --json results.json
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

from examples.cpp_vs_python_benchmark.scenario import circle_scenario
from examples.cpp_vs_python_benchmark.benchmark_runner import run_all_benchmarks
from examples.cpp_vs_python_benchmark.visualization import (
    print_summary, plot_dashboard, live_benchmark,
)


def main():
    parser = argparse.ArgumentParser(
        description="Python vs C++ MPPI Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: K=128, N=10, repeat=3")
    parser.add_argument("--live", action="store_true",
                        help="Live mode: 실시간 Python vs C++ 비교 시뮬레이션")
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
                        choices=["vanilla", "log", "tsallis", "risk_aware"],
                        help="Weight strategy for live mode (default: vanilla)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save dashboard plot to file")
    parser.add_argument("--json", type=str, default=None,
                        help="Save results as JSON")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip matplotlib plots")
    args = parser.parse_args()

    # Quick mode 오버라이드
    if args.quick:
        args.K = 128
        args.N = 10
        args.repeat = 3

    scenario = circle_scenario(nx=3)

    # ── Live 모드 ──
    if args.live:
        print()
        print("╔" + "═" * 58 + "╗")
        print("║   Python vs C++ MPPI Live Benchmark                    ║")
        print(f"║   K={args.K:<6d}  N={args.N:<4d}  weight={args.weight:<16s}     ║")
        print(f"║   Mode: LIVE                                           ║")
        print("╚" + "═" * 58 + "╝")
        print()
        live_benchmark(scenario, K=args.K, N=args.N, weight_type=args.weight)
        return

    # ── Batch 모드 ──
    # 벤치마크 선택
    run_pip = args.pipeline or (not args.pipeline and not args.component and not args.scaling)
    run_comp = args.component or (not args.pipeline and not args.component and not args.scaling)
    run_scl = args.scaling or (not args.pipeline and not args.component and not args.scaling)

    # K/N 스케일링 값
    K_values = [128, 256, 512] if args.quick else None
    N_values = [10, 20] if args.quick else None

    print()
    print("╔" + "═" * 58 + "╗")
    print("║   Python vs C++ MPPI Benchmark Suite                    ║")
    print(f"║   K={args.K:<6d}  N={args.N:<4d}  repeat={args.repeat:<4d}                    ║")
    print(f"║   Mode: BATCH                                          ║")
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
