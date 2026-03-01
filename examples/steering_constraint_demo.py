#!/usr/bin/env python3
"""60° vs 90° 스티어링 제한 NonCoaxialSwerve 비교 데모.

시각화 레이아웃 (2×2):
┌─────────────────┐  ┌─────────────────┐
│  90° Steering   │  │  60° Steering   │
│  ○ ref path     │  │  ○ ref path     │
│  ● robot trail  │  │  ● robot trail  │
└─────────────────┘  └─────────────────┘
┌─────────────────┐  ┌─────────────────┐
│ δ(t) + limits   │  │ Position RMSE   │
│ --90° ---|--- +90│  │ bar chart       │
│ --60° --|-- +60° │  │ 90° vs 60°      │
└─────────────────┘  └─────────────────┘

실행 예시:
    python examples/steering_constraint_demo.py --scenario circle
    python examples/steering_constraint_demo.py --scenario tight
    python examples/steering_constraint_demo.py --scenario slalom
    python examples/steering_constraint_demo.py --scenario figure8
    python examples/steering_constraint_demo.py --angles 90 60 45
    python examples/steering_constraint_demo.py --live
    python examples/steering_constraint_demo.py --no-plot
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

# 프로젝트 루트를 sys.path에 추가 (직접 실행 지원)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mpc_controller import (
    LookaheadInterpolator,
    NonCoaxialSwerveDriveModel,
    NonCoaxialSwerveMPCController,
    NonCoaxialSwerveMPCParams,
    NonCoaxialSwerveParams,
)
from simulation import SimulationConfig, Simulator


def _suppress_mpc_logger():
    """MPC 로거 verbose 억제 (컨트롤러 생성 후 호출)."""
    lg = logging.getLogger("mpc_controller.controllers.non_coaxial_swerve_mpc")
    lg.handlers.clear()
    lg.setLevel(logging.ERROR)
    lg.propagate = False


# ─────────────────────────────────────────────────────────────
# 시나리오 생성
# ─────────────────────────────────────────────────────────────

def _circle_trajectory(radius=3.0, num_points=200):
    """원형 궤적 (3D: x, y, theta)."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    traj = np.zeros((num_points, 3))
    traj[:, 0] = radius * np.cos(angles)
    traj[:, 1] = radius * np.sin(angles)
    traj[:, 2] = np.unwrap(angles + np.pi / 2)
    initial = np.array([radius, 0.0, np.pi / 2, 0.0])
    return traj, initial, 15.0


def _tight_turn_trajectory(radius=1.5, num_points=200):
    """타이트 턴 (반경 1.5m)."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    traj = np.zeros((num_points, 3))
    traj[:, 0] = radius * np.cos(angles)
    traj[:, 1] = radius * np.sin(angles)
    traj[:, 2] = np.unwrap(angles + np.pi / 2)
    initial = np.array([radius, 0.0, np.pi / 2, 0.0])
    return traj, initial, 12.0


def _slalom_trajectory(gate_spacing=2.0, amplitude=1.5, num_gates=6, num_points=300):
    """S자 슬라럼."""
    total_length = gate_spacing * num_gates
    x = np.linspace(0, total_length, num_points)
    y = amplitude * np.sin(2 * np.pi * x / (2 * gate_spacing))
    dx = np.gradient(x)
    dy = np.gradient(y)
    traj = np.zeros((num_points, 3))
    traj[:, 0] = x
    traj[:, 1] = y
    traj[:, 2] = np.unwrap(np.arctan2(dy, dx))
    initial = np.array([0.0, 0.0, traj[0, 2], 0.0])
    return traj, initial, 15.0


def _figure8_trajectory(scale=3.0, num_points=300):
    """8자 궤적."""
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    traj = np.zeros((num_points, 3))
    traj[:, 0] = scale * np.sin(t)
    traj[:, 1] = scale * np.sin(t) * np.cos(t)
    dx = np.gradient(traj[:, 0])
    dy = np.gradient(traj[:, 1])
    traj[:, 2] = np.unwrap(np.arctan2(dy, dx))
    initial = np.array([0.0, 0.0, traj[0, 2], 0.0])
    return traj, initial, 20.0


SCENARIO_MAP = {
    "circle": _circle_trajectory,
    "tight": _tight_turn_trajectory,
    "slalom": _slalom_trajectory,
    "figure8": _figure8_trajectory,
}


# ─────────────────────────────────────────────────────────────
# 시뮬레이션 코어
# ─────────────────────────────────────────────────────────────

def run_single_sim(angle_deg, trajectory, initial_state, sim_time, dt=0.05):
    """단일 스티어링 각도로 NonCoaxialSwerve 시뮬레이션 실행.

    Returns
    -------
    dict with keys: states, controls, errors, deltas, solve_times, rmse
    """
    angle_rad = np.radians(angle_deg)

    robot_params = NonCoaxialSwerveParams(
        max_steering_angle=angle_rad,
    )
    mpc_params = NonCoaxialSwerveMPCParams(
        N=15,
        dt=0.1,
        Q=np.diag([10.0, 10.0, 5.0, 0.1]),
        R=np.diag([0.1, 0.1, 0.3]),
        Qf=np.diag([100.0, 100.0, 50.0, 0.1]),
        Rd=np.diag([0.5, 0.5, 1.0]),
    )

    controller = NonCoaxialSwerveMPCController(robot_params, mpc_params)
    _suppress_mpc_logger()
    traj_dt = sim_time / (len(trajectory) - 1)
    interpolator = LookaheadInterpolator(
        trajectory, dt=traj_dt, v_max=robot_params.max_speed,
    )

    sim_config = SimulationConfig(dt=dt, max_time=sim_time)
    sim = Simulator(robot_params, sim_config, model_type="non_coaxial_swerve")
    sim.reset(initial_state)

    num_steps = int(sim_time / dt)
    states = [initial_state.copy()]
    controls_list = []
    errors_list = []
    deltas_list = [initial_state[3]]
    solve_times_list = []

    for step in range(num_steps):
        state = sim.get_measurement()
        if len(state) < 4:
            state_ext = np.zeros(4)
            state_ext[:len(state)] = state
            state = state_ext

        ref = interpolator.get_reference(
            state, controller.params.N, controller.params.dt,
            current_theta=state[2],
        )

        control, info = controller.compute_control(state, ref)
        next_state = sim.step(control)
        if len(next_state) < 4:
            ns = np.zeros(4)
            ns[:len(next_state)] = next_state
            next_state = ns

        # 추적 오차 (위치)
        pos_err = np.sqrt((state[0] - ref[0, 0])**2 + (state[1] - ref[0, 1])**2)
        errors_list.append(pos_err)
        states.append(next_state.copy())
        controls_list.append(control.copy())
        deltas_list.append(next_state[3])
        solve_times_list.append(info.get("solve_time", 0))

        # 궤적 완료 검사
        idx, dist = interpolator.find_closest_point(state[:2])
        if idx >= interpolator.num_points - 2 and dist < 0.15:
            break

    states_arr = np.array(states)
    errors_arr = np.array(errors_list)
    rmse = float(np.sqrt(np.mean(errors_arr**2))) if len(errors_arr) > 0 else 0.0

    return {
        "angle_deg": angle_deg,
        "states": states_arr,
        "controls": np.array(controls_list) if controls_list else np.empty((0, 3)),
        "errors": errors_arr,
        "deltas": np.array(deltas_list),
        "solve_times": np.array(solve_times_list),
        "rmse": rmse,
        "avg_solve_ms": float(np.mean(solve_times_list)) * 1000 if solve_times_list else 0,
    }


# ─────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────

def plot_comparison(trajectory, results, scenario_name, save_path=None):
    """2×2 비교 플롯."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Steering Constraint Comparison — {scenario_name}", fontsize=14)

    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]

    # ── 상단: 경로 추적 (각 각도별 서브플롯) ──
    for i, res in enumerate(results[:2]):
        ax = axes[0, i]
        ax.plot(trajectory[:, 0], trajectory[:, 1], "k--", alpha=0.4, label="reference")
        c = colors[i]
        ax.plot(res["states"][:, 0], res["states"][:, 1], color=c, linewidth=1.5,
                label=f'{res["angle_deg"]:.0f}° actual')
        ax.set_title(f'{res["angle_deg"]:.0f}° Steering  (RMSE={res["rmse"]:.3f}m)')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── 좌하단: δ(t) 시계열 ──
    ax_delta = axes[1, 0]
    dt = 0.05
    for i, res in enumerate(results):
        t = np.arange(len(res["deltas"])) * dt
        ax_delta.plot(t, np.degrees(res["deltas"]), color=colors[i],
                      label=f'{res["angle_deg"]:.0f}°', linewidth=1.2)
        limit = res["angle_deg"]
        ax_delta.axhline(limit, color=colors[i], linestyle="--", alpha=0.4)
        ax_delta.axhline(-limit, color=colors[i], linestyle="--", alpha=0.4)

    ax_delta.set_xlabel("Time [s]")
    ax_delta.set_ylabel("Steering angle δ [deg]")
    ax_delta.set_title("Steering Angle History")
    ax_delta.legend(fontsize=8)
    ax_delta.grid(True, alpha=0.3)

    # ── 우하단: RMSE 비교 bar chart ──
    ax_bar = axes[1, 1]
    angles = [r["angle_deg"] for r in results]
    rmses = [r["rmse"] for r in results]
    bar_colors = colors[:len(results)]
    bars = ax_bar.bar([f'{a:.0f}°' for a in angles], rmses, color=bar_colors, alpha=0.8)
    for bar, rmse in zip(bars, rmses):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{rmse:.4f}", ha="center", va="bottom", fontsize=9)
    ax_bar.set_xlabel("Max Steering Angle")
    ax_bar.set_ylabel("Position RMSE [m]")
    ax_bar.set_title("Tracking Quality Comparison")
    ax_bar.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Plot saved to {save_path}")
    else:
        plt.show()


def print_ascii_summary(results):
    """ASCII 요약 테이블."""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        Steering Constraint Comparison Results           ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Angle   │ Pos RMSE (m)  │ Solve (ms)  │ Steps        ║")
    print("╠══════════╪═══════════════╪═════════════╪══════════════╣")
    for r in results:
        n_steps = len(r["errors"])
        print(f"║  {r['angle_deg']:5.0f}°  │ {r['rmse']:>12.4f}  │"
              f" {r['avg_solve_ms']:>10.2f}  │ {n_steps:>12d} ║")
    print("╚══════════╧═══════════════╧═════════════╧══════════════╝")

    # 비교 분석
    if len(results) >= 2:
        r0, r1 = results[0], results[1]
        ratio = r1["rmse"] / r0["rmse"] if r0["rmse"] > 0 else float("inf")
        print()
        print(f"  ► {r1['angle_deg']:.0f}° vs {r0['angle_deg']:.0f}° "
              f"RMSE ratio: {ratio:.2f}x")
        if ratio > 1:
            print(f"  ► {r1['angle_deg']:.0f}° 제한이 추적 품질을 "
                  f"{(ratio - 1) * 100:.1f}% 저하시킴")
        else:
            print(f"  ► {r1['angle_deg']:.0f}° 제한이 동등하거나 더 나은 추적 품질")
    print()


# ─────────────────────────────────────────────────────────────
# 실시간 애니메이션
# ─────────────────────────────────────────────────────────────

def run_live_comparison(trajectory, initial_state, sim_time,
                        angles_deg, dt=0.05):
    """matplotlib 실시간 비교 애니메이션."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(angles_deg), figsize=(7 * len(angles_deg), 6))
    if len(angles_deg) == 1:
        axes = [axes]

    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]

    # 컨트롤러/시뮬레이터 초기화
    sims = []
    for i, angle_deg in enumerate(angles_deg):
        angle_rad = np.radians(angle_deg)
        robot_params = NonCoaxialSwerveParams(max_steering_angle=angle_rad)
        mpc_params = NonCoaxialSwerveMPCParams(
            N=15, dt=0.1,
            Q=np.diag([10.0, 10.0, 5.0, 0.1]),
            R=np.diag([0.1, 0.1, 0.3]),
            Qf=np.diag([100.0, 100.0, 50.0, 0.1]),
            Rd=np.diag([0.5, 0.5, 1.0]),
        )
        ctrl = NonCoaxialSwerveMPCController(robot_params, mpc_params)
        _suppress_mpc_logger()
        traj_dt = sim_time / (len(trajectory) - 1)
        interp = LookaheadInterpolator(trajectory, dt=traj_dt, v_max=robot_params.max_speed)
        sim_cfg = SimulationConfig(dt=dt, max_time=sim_time)
        sim = Simulator(robot_params, sim_cfg, model_type="non_coaxial_swerve")
        sim.reset(initial_state)
        sims.append({
            "ctrl": ctrl, "interp": interp, "sim": sim,
            "trail_x": [initial_state[0]], "trail_y": [initial_state[1]],
            "angle_deg": angle_deg, "color": colors[i],
        })

    for ax in axes:
        ax.plot(trajectory[:, 0], trajectory[:, 1], "k--", alpha=0.3, label="ref")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.ion()
    plt.show()

    num_steps = int(sim_time / dt)
    for step in range(num_steps):
        for i, s in enumerate(sims):
            state = s["sim"].get_measurement()
            if len(state) < 4:
                state_ext = np.zeros(4)
                state_ext[:len(state)] = state
                state = state_ext

            ref = s["interp"].get_reference(
                state, s["ctrl"].params.N, s["ctrl"].params.dt,
                current_theta=state[2],
            )
            control, info = s["ctrl"].compute_control(state, ref)
            next_state = s["sim"].step(control)

            s["trail_x"].append(float(next_state[0]) if len(next_state) > 0 else 0)
            s["trail_y"].append(float(next_state[1]) if len(next_state) > 1 else 0)

        # 10스텝마다 업데이트
        if step % 10 == 0:
            for i, s in enumerate(sims):
                axes[i].clear()
                axes[i].plot(trajectory[:, 0], trajectory[:, 1], "k--", alpha=0.3)
                axes[i].plot(s["trail_x"], s["trail_y"],
                             color=s["color"], linewidth=1.5)
                axes[i].plot(s["trail_x"][-1], s["trail_y"][-1],
                             "o", color=s["color"], markersize=8)
                axes[i].set_title(f'{s["angle_deg"]:.0f}° Steering (t={step * dt:.1f}s)')
                axes[i].set_aspect("equal")
                axes[i].grid(True, alpha=0.3)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    plt.ioff()
    plt.show()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NonCoaxialSwerve Steering Constraint Demo (60° vs 90°)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenario", type=str, default="circle",
                        choices=list(SCENARIO_MAP.keys()),
                        help="Scenario (default: circle)")
    parser.add_argument("--angles", type=float, nargs="+", default=[90, 60],
                        help="Steering angles in degrees (default: 90 60)")
    parser.add_argument("--sim-time", type=float, default=None,
                        help="Override simulation time (seconds)")
    parser.add_argument("--live", action="store_true",
                        help="Real-time animation mode")
    parser.add_argument("--save", type=str, default=None,
                        help="Save plot to file")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip matplotlib plots (headless)")
    args = parser.parse_args()

    # 시나리오 생성
    trajectory, initial_state, default_sim_time = SCENARIO_MAP[args.scenario]()
    sim_time = args.sim_time or default_sim_time

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   NonCoaxialSwerve Steering Constraint Demo             ║")
    angles_str = ", ".join(f"{a:.0f}°" for a in args.angles)
    print(f"║   Scenario: {args.scenario:<12s}  Angles: {angles_str:<18s}  ║")
    print(f"║   Sim time: {sim_time:.1f}s                                      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # 실시간 모드
    if args.live:
        run_live_comparison(trajectory, initial_state, sim_time, args.angles)
        return

    # 배치 시뮬레이션
    results = []
    for angle in args.angles:
        print(f"  Running {angle:.0f}° simulation...", end=" ", flush=True)
        t0 = time.perf_counter()
        res = run_single_sim(angle, trajectory, initial_state, sim_time)
        elapsed = time.perf_counter() - t0
        print(f"done ({elapsed:.1f}s, RMSE={res['rmse']:.4f}m)")
        results.append(res)

    # ASCII 요약
    print_ascii_summary(results)

    # 플롯
    if not args.no_plot:
        plot_comparison(trajectory, results, args.scenario, save_path=args.save)


if __name__ == "__main__":
    main()
