#!/usr/bin/env python3
"""GPU vs CPU MPPI 벤치마크.

Usage:
    python examples/gpu_benchmark.py
    python examples/gpu_benchmark.py --K 512,1024,2048,4096,8192 --N 30
    python examples/gpu_benchmark.py --N 30 --repeat 20

출력 예:
┌───────┬────────────┬───────────┬─────────┐
│ K     │ CPU (ms)   │ GPU (ms)  │ Speedup │
├───────┼────────────┼───────────┼─────────┤
│ 512   │ 8.2        │ 1.5       │ 5.5x    │
│ 1024  │ 16.1       │ 1.8       │ 8.9x    │
│ 4096  │ 64.8       │ 3.1       │ 20.9x   │
└───────┴────────────┴───────────┴─────────┘
"""

import argparse
import time
import sys

import numpy as np


def make_reference_trajectory(N, dt=0.05):
    """직선 참조 궤적 (전진)."""
    ref = np.zeros((N + 1, 3))
    for t in range(N + 1):
        ref[t, 0] = t * dt * 1.0
    return ref


def benchmark_cpu(K, N, dt, state, ref, repeat=20):
    """CPU MPPI 벤치마크."""
    from mpc_controller.controllers.mppi.base_mppi import MPPIController
    from mpc_controller.controllers.mppi.mppi_params import MPPIParams
    from mpc_controller.models.differential_drive import RobotParams

    params = MPPIParams(K=K, N=N, dt=dt, use_gpu=False)
    ctrl = MPPIController(RobotParams(), params, seed=42)

    # warmup
    ctrl.compute_control(state, ref)

    times = []
    for _ in range(repeat):
        ctrl.reset()
        start = time.perf_counter()
        ctrl.compute_control(state, ref)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return np.median(times)


def benchmark_gpu(K, N, dt, state, ref, repeat=20):
    """GPU MPPI 벤치마크."""
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
    except ImportError:
        return None

    from mpc_controller.controllers.mppi.base_mppi import MPPIController
    from mpc_controller.controllers.mppi.mppi_params import MPPIParams
    from mpc_controller.models.differential_drive import RobotParams

    params = MPPIParams(K=K, N=N, dt=dt, use_gpu=True)
    ctrl = MPPIController(RobotParams(), params, seed=42)

    if not ctrl._use_gpu:
        return None

    # warmup (JIT 컴파일)
    ctrl.compute_control(state, ref)
    ctrl.compute_control(state, ref)

    times = []
    for _ in range(repeat):
        ctrl.reset()
        start = time.perf_counter()
        ctrl.compute_control(state, ref)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return np.median(times)


def main():
    parser = argparse.ArgumentParser(description="GPU vs CPU MPPI Benchmark")
    parser.add_argument(
        "--K", type=str, default="256,512,1024,2048,4096",
        help="샘플 수 (쉼표 구분)",
    )
    parser.add_argument("--N", type=int, default=30, help="호라이즌 스텝")
    parser.add_argument("--dt", type=float, default=0.05, help="시간 간격")
    parser.add_argument("--repeat", type=int, default=20, help="반복 횟수")
    args = parser.parse_args()

    K_list = [int(k.strip()) for k in args.K.split(",")]
    N = args.N
    dt = args.dt

    state = np.array([0.0, 0.0, 0.0])
    ref = make_reference_trajectory(N, dt)

    # 백엔드 정보
    try:
        from mpc_controller.controllers.mppi.gpu_backend import get_backend_name
        backend = get_backend_name()
    except ImportError:
        backend = "numpy"

    print(f"\n{'='*58}")
    print(f"  MPPI GPU Benchmark  (backend: {backend})")
    print(f"  N={N}, dt={dt}, repeat={args.repeat}")
    print(f"{'='*58}\n")

    # 테이블 헤더
    print(f"┌{'─'*8}┬{'─'*13}┬{'─'*13}┬{'─'*10}┐")
    print(f"│ {'K':>6} │ {'CPU (ms)':>11} │ {'GPU (ms)':>11} │ {'Speedup':>8} │")
    print(f"├{'─'*8}┼{'─'*13}┼{'─'*13}┼{'─'*10}┤")

    for K in K_list:
        cpu_ms = benchmark_cpu(K, N, dt, state, ref, args.repeat)
        gpu_ms = benchmark_gpu(K, N, dt, state, ref, args.repeat)

        if gpu_ms is not None:
            speedup = cpu_ms / gpu_ms
            print(
                f"│ {K:>6} │ {cpu_ms:>11.2f} │ {gpu_ms:>11.2f} │ {speedup:>7.1f}x │"
            )
        else:
            print(
                f"│ {K:>6} │ {cpu_ms:>11.2f} │ {'N/A':>11} │ {'N/A':>8} │"
            )

    print(f"└{'─'*8}┴{'─'*13}┴{'─'*13}┴{'─'*10}┘")
    print()


if __name__ == "__main__":
    main()
