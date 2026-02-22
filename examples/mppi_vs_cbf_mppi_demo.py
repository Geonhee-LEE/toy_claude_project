#!/usr/bin/env python3
"""Vanilla MPPI vs CBF-MPPI 비교 데모.

Vanilla MPPI (soft cost만)와 CBF-MPPI (Hybrid: soft CBF cost + QP safety filter)를
동일 조건에서 실행하고 안전성 + 성능을 비교합니다.

3가지 시나리오:
  1. Head-on: 경로 정면에 장애물
  2. Narrow passage: 좁은 통로 통과
  3. Multi-obstacle: 다중 장애물 필드

실행:
    python examples/mppi_vs_cbf_mppi_demo.py
    python examples/mppi_vs_cbf_mppi_demo.py --scenario narrow
    python examples/mppi_vs_cbf_mppi_demo.py --live
    python examples/mppi_vs_cbf_mppi_demo.py --save comparison.png
"""

import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from mpc_controller import (
    DifferentialDriveModel,
    MPPIController,
    MPPIParams,
    RobotParams,
    TrajectoryInterpolator,
    generate_sinusoidal_trajectory,
)
from mpc_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mpc_controller.controllers.mppi.mppi_params import CBFParams
from simulation.simulator import Simulator, SimulationConfig


# ─────────────────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    """단일 컨트롤러 시뮬레이션 결과."""

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
    safety_violations: int = 0
    min_barrier_values: np.ndarray = field(default_factory=lambda: np.array([]))
    info_history: Optional[List[dict]] = None

    @property
    def position_rmse(self) -> float:
        pos_err = self.tracking_errors[:, :2]
        return float(np.sqrt(np.mean(np.sum(pos_err ** 2, axis=1))))

    @property
    def heading_rmse(self) -> float:
        return float(np.sqrt(np.mean(self.tracking_errors[:, 2] ** 2)))

    @property
    def control_rate(self) -> float:
        if len(self.controls) < 2:
            return 0.0
        du = np.diff(self.controls, axis=0)
        return float(np.sqrt(np.mean(du ** 2)))


# ─────────────────────────────────────────────────────────────
# 장애물 시나리오
# ─────────────────────────────────────────────────────────────

def get_scenario(name: str) -> Dict:
    """시나리오별 장애물/궤적 생성."""
    if name == "head_on":
        trajectory = np.column_stack([
            np.linspace(0, 8, 300),
            np.zeros(300),
            np.zeros(300),
        ])
        obstacles = np.array([
            [4.0, 0.0, 0.3],
        ])
        return {"trajectory": trajectory, "obstacles": obstacles,
                "label": "Head-on Collision"}

    elif name == "narrow":
        trajectory = np.column_stack([
            np.linspace(0, 8, 300),
            np.zeros(300),
            np.zeros(300),
        ])
        obstacles = np.array([
            [4.0, 0.8, 0.3],
            [4.0, -0.8, 0.3],
        ])
        return {"trajectory": trajectory, "obstacles": obstacles,
                "label": "Narrow Passage"}

    else:  # multi
        trajectory = generate_sinusoidal_trajectory(
            start=np.array([0.0, 0.0]),
            length=10.0,
            amplitude=1.5,
            frequency=0.3,
            num_points=400,
        )
        obstacles = np.array([
            [2.0, 0.5, 0.3],
            [4.0, -0.3, 0.4],
            [6.0, 0.8, 0.3],
            [8.0, -0.5, 0.35],
        ])
        return {"trajectory": trajectory, "obstacles": obstacles,
                "label": "Multi-obstacle Field"}


# ─────────────────────────────────────────────────────────────
# 안전 위반 체크
# ─────────────────────────────────────────────────────────────

def check_safety_violation(
    state: np.ndarray,
    obstacles: np.ndarray,
    robot_radius: float = 0.2,
    safety_margin: float = 0.3,
) -> bool:
    """로봇이 장애물 안전 영역 내에 있는지 확인."""
    for obs in obstacles:
        dist = np.sqrt((state[0] - obs[0]) ** 2 + (state[1] - obs[1]) ** 2)
        d_safe = obs[2] + robot_radius + safety_margin
        if dist < d_safe:
            return True
    return False


def compute_min_barrier(
    state: np.ndarray,
    obstacles: np.ndarray,
    robot_radius: float = 0.2,
    safety_margin: float = 0.3,
) -> float:
    """모든 장애물에 대한 최소 barrier 값 계산."""
    min_h = float("inf")
    for obs in obstacles:
        dx = state[0] - obs[0]
        dy = state[1] - obs[1]
        d_safe = obs[2] + robot_radius + safety_margin
        h = dx ** 2 + dy ** 2 - d_safe ** 2
        min_h = min(min_h, h)
    return min_h


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
    obstacles: np.ndarray,
    store_info: bool = False,
) -> RunResult:
    """시뮬레이션 루프 + 안전 메트릭 수집."""
    sim = Simulator(robot_params, sim_config)
    sim.reset(initial_state)
    controller.reset()

    num_steps = int(sim_config.max_time / sim_config.dt)

    states = [initial_state.copy()]
    controls_list: List[np.ndarray] = []
    references_list: List[np.ndarray] = []
    errors_list: List[np.ndarray] = []
    solve_times_list: List[float] = []
    costs_list: List[float] = []
    time_list: List[float] = []
    ess_list: List[float] = []
    temp_list: List[float] = []
    barrier_list: List[float] = []
    info_list: Optional[List[dict]] = [] if store_info else None

    safety_violations = 0

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

        # 안전 위반 체크
        if check_safety_violation(next_state, obstacles):
            safety_violations += 1

        min_h = compute_min_barrier(state, obstacles)

        time_list.append(t)
        states.append(next_state.copy())
        controls_list.append(control.copy())
        references_list.append(ref[0].copy())
        errors_list.append(error)
        solve_times_list.append(info["solve_time"])
        costs_list.append(info["cost"])
        ess_list.append(info.get("ess", 0))
        temp_list.append(info.get("temperature", controller.params.lambda_))
        barrier_list.append(min_h)

        if store_info:
            info_list.append(info)

        idx, dist = interpolator.find_closest_point(state[:2])
        if idx >= interpolator.num_points - 1 and dist < 0.1:
            break

    wall_time = time.perf_counter()

    return RunResult(
        name=name,
        states=np.array(states),
        controls=np.array(controls_list),
        references=np.array(references_list),
        tracking_errors=np.array(errors_list),
        solve_times=np.array(solve_times_list),
        costs=np.array(costs_list),
        time_array=np.array(time_list),
        total_wall_time=0.0,
        ess_history=np.array(ess_list),
        temp_history=np.array(temp_list),
        safety_violations=safety_violations,
        min_barrier_values=np.array(barrier_list),
        info_history=info_list,
    )


# ─────────────────────────────────────────────────────────────
# ASCII 요약표
# ─────────────────────────────────────────────────────────────

def print_summary(vanilla: RunResult, cbf: RunResult, scenario_label: str) -> None:
    """콘솔 비교 요약표."""
    lines = [
        "",
        "+" + "=" * 66 + "+",
        "|{:^66s}|".format(f"Vanilla MPPI vs CBF-MPPI: {scenario_label}"),
        "+" + "=" * 66 + "+",
        "| {:<34s}{:>14s}{:>14s} |".format("Metric", "Vanilla", "CBF-MPPI"),
        "+" + "-" * 66 + "+",
        "| {:<34s}{:>14.4f}{:>14.4f} |".format(
            "Position RMSE [m]", vanilla.position_rmse, cbf.position_rmse
        ),
        "| {:<34s}{:>14.4f}{:>14.4f} |".format(
            "Heading RMSE [rad]", vanilla.heading_rmse, cbf.heading_rmse
        ),
        "| {:<34s}{:>14.4f}{:>14.4f} |".format(
            "Control Rate RMS", vanilla.control_rate, cbf.control_rate
        ),
        "| {:<34s}{:>14.3f}{:>14.3f} |".format(
            "Avg Solve Time [ms]",
            np.mean(vanilla.solve_times) * 1000,
            np.mean(cbf.solve_times) * 1000,
        ),
        "+" + "-" * 66 + "+",
        "|{:^66s}|".format("** Safety Metrics **"),
        "+" + "-" * 66 + "+",
        "| {:<34s}{:>14d}{:>14d} |".format(
            "Safety Violations", vanilla.safety_violations, cbf.safety_violations
        ),
        "| {:<34s}{:>14.4f}{:>14.4f} |".format(
            "Min Barrier Value",
            float(np.min(vanilla.min_barrier_values)) if len(vanilla.min_barrier_values) > 0 else float("inf"),
            float(np.min(cbf.min_barrier_values)) if len(cbf.min_barrier_values) > 0 else float("inf"),
        ),
        "| {:<34s}{:>14d}{:>14d} |".format(
            "Simulation Steps", len(vanilla.time_array), len(cbf.time_array)
        ),
        "+" + "-" * 66 + "+",
    ]

    # 판정
    safety_winner = "CBF-MPPI" if cbf.safety_violations <= vanilla.safety_violations else "Vanilla"
    tracking_winner = "CBF-MPPI" if cbf.position_rmse < vanilla.position_rmse else "Vanilla"

    lines += [
        "| {:<64s} |".format(f"  Safety winner    : {safety_winner}"),
        "| {:<64s} |".format(f"  Tracking winner  : {tracking_winner}"),
        "+" + "=" * 66 + "+",
        "",
    ]

    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────

def plot_comparison(
    vanilla: RunResult,
    cbf: RunResult,
    reference: np.ndarray,
    obstacles: np.ndarray,
    scenario_label: str,
    save_path: Optional[str] = None,
):
    """2x3 비교 그래프."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    cv, cc = "tab:blue", "tab:red"

    # (0,0) 궤적 비교
    ax = axes[0, 0]
    ax.plot(reference[:, 0], reference[:, 1],
            "k--", linewidth=1.5, alpha=0.4, label="Reference")
    ax.plot(vanilla.states[:, 0], vanilla.states[:, 1],
            color=cv, linewidth=1.5, label="Vanilla")
    ax.plot(cbf.states[:, 0], cbf.states[:, 1],
            color=cc, linewidth=1.5, label="CBF-MPPI")
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2],
                             color="gray", alpha=0.6)
        safety = plt.Circle((obs[0], obs[1]), obs[2] + 0.5,
                             color="gray", alpha=0.15, linestyle="--", fill=False)
        ax.add_patch(circle)
        ax.add_patch(safety)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Trajectory Comparison")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # (0,1) 위치 오차
    ax = axes[0, 1]
    v_err = np.linalg.norm(vanilla.tracking_errors[:, :2], axis=1)
    c_err = np.linalg.norm(cbf.tracking_errors[:, :2], axis=1)
    ax.plot(vanilla.time_array, v_err, color=cv, linewidth=1, label="Vanilla")
    ax.plot(cbf.time_array, c_err, color=cc, linewidth=1, label="CBF-MPPI")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [m]")
    ax.set_title("Position Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,2) Barrier 값
    ax = axes[0, 2]
    ax.plot(vanilla.time_array, vanilla.min_barrier_values,
            color=cv, linewidth=1, label="Vanilla")
    ax.plot(cbf.time_array, cbf.min_barrier_values,
            color=cc, linewidth=1, label="CBF-MPPI")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Safety boundary")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("min h(x)")
    ax.set_title("Minimum Barrier Value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) 선속도
    ax = axes[1, 0]
    ax.plot(vanilla.time_array[:len(vanilla.controls)], vanilla.controls[:, 0],
            color=cv, linewidth=1, label="Vanilla")
    ax.plot(cbf.time_array[:len(cbf.controls)], cbf.controls[:, 0],
            color=cc, linewidth=1, label="CBF-MPPI")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("v [m/s]")
    ax.set_title("Linear Velocity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) 각속도
    ax = axes[1, 1]
    ax.plot(vanilla.time_array[:len(vanilla.controls)], vanilla.controls[:, 1],
            color=cv, linewidth=1, label="Vanilla")
    ax.plot(cbf.time_array[:len(cbf.controls)], cbf.controls[:, 1],
            color=cc, linewidth=1, label="CBF-MPPI")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("omega [rad/s]")
    ax.set_title("Angular Velocity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,2) Solve time
    ax = axes[1, 2]
    ax.plot(vanilla.time_array, vanilla.solve_times * 1000,
            color=cv, linewidth=1, alpha=0.7, label="Vanilla")
    ax.plot(cbf.time_array, cbf.solve_times * 1000,
            color=cc, linewidth=1, alpha=0.7, label="CBF-MPPI")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Solve Time [ms]")
    ax.set_title("Computation Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Vanilla MPPI vs CBF-MPPI: {scenario_label}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vanilla MPPI vs CBF-MPPI 비교 데모"
    )
    parser.add_argument(
        "--scenario", type=str, default="head_on",
        choices=["head_on", "narrow", "multi"],
        help="시나리오 (기본: head_on)",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="비교 그래프 저장 경로",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="시각화 건너뛰기 (콘솔 요약만 출력)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="시뮬레이션 후 실시간 리플레이",
    )
    args = parser.parse_args()

    print("\n" + "=" * 68)
    print("     Vanilla MPPI vs CBF-MPPI Safety Comparison Demo")
    print("=" * 68)

    # 시나리오 로드
    scenario = get_scenario(args.scenario)
    trajectory = scenario["trajectory"]
    obstacles = scenario["obstacles"]
    scenario_label = scenario["label"]

    print(f"\n  Scenario   : {scenario_label}")
    print(f"  Obstacles  : {len(obstacles)}")
    for i, obs in enumerate(obstacles):
        print(f"    [{i}] pos=({obs[0]:.1f}, {obs[1]:.1f}), r={obs[2]:.2f}")

    # 공통 파라미터
    robot_params = RobotParams(max_velocity=1.0, max_omega=1.5)
    sim_config = SimulationConfig(dt=0.05, max_time=20.0)
    initial_state = trajectory[0].copy()

    mppi_params = MPPIParams(
        N=20, K=512, dt=0.05, lambda_=10.0,
        noise_sigma=np.array([0.3, 0.3]),
        Q=np.diag([10.0, 10.0, 1.0]),
        R=np.diag([0.01, 0.01]),
        Qf=np.diag([100.0, 100.0, 10.0]),
    )

    # Vanilla MPPI
    vanilla_controller = MPPIController(
        robot_params=robot_params,
        mppi_params=mppi_params,
        seed=42,
        obstacles=obstacles,
    )

    # CBF-MPPI
    cbf_params = CBFParams(
        enabled=True,
        gamma=1.0,
        safety_margin=0.3,
        robot_radius=0.2,
        activation_distance=3.0,
        cost_weight=500.0,
        use_safety_filter=True,
    )
    cbf_controller = CBFMPPIController(
        robot_params=robot_params,
        mppi_params=mppi_params,
        seed=42,
        obstacles=obstacles,
        cbf_params=cbf_params,
    )

    # 파라미터 요약
    print(f"\n  ┌{'─' * 46}┐")
    print(f"  │{'CBF Parameters':^46s}│")
    print(f"  ├{'─' * 46}┤")
    print(f"  │  gamma={cbf_params.gamma}, margin={cbf_params.safety_margin}m{' ' * 22}│")
    print(f"  │  robot_radius={cbf_params.robot_radius}m, activation={cbf_params.activation_distance}m{' ' * 8}│")
    print(f"  │  cost_weight={cbf_params.cost_weight}, safety_filter=True{' ' * 6}│")
    print(f"  └{'─' * 46}┘")

    # 시뮬레이션
    print("\n  Running Vanilla MPPI ...")
    vanilla_interp = TrajectoryInterpolator(trajectory, dt=sim_config.dt)
    wall_start = time.perf_counter()
    vanilla_result = simulate(
        vanilla_controller, "Vanilla MPPI", vanilla_interp,
        initial_state.copy(), sim_config, robot_params, obstacles,
        store_info=args.live,
    )
    vanilla_result.total_wall_time = time.perf_counter() - wall_start
    print(f"    -> {len(vanilla_result.time_array)} steps, "
          f"violations={vanilla_result.safety_violations}")

    print("  Running CBF-MPPI ...")
    cbf_interp = TrajectoryInterpolator(trajectory, dt=sim_config.dt)
    wall_start = time.perf_counter()
    cbf_result = simulate(
        cbf_controller, "CBF-MPPI", cbf_interp,
        initial_state.copy(), sim_config, robot_params, obstacles,
        store_info=args.live,
    )
    cbf_result.total_wall_time = time.perf_counter() - wall_start
    print(f"    -> {len(cbf_result.time_array)} steps, "
          f"violations={cbf_result.safety_violations}")

    # ASCII 요약
    print_summary(vanilla_result, cbf_result, scenario_label)

    # 시각화
    if not args.no_plot:
        print("  Generating comparison figure ...")
        import matplotlib.pyplot as plt
        plot_comparison(
            vanilla_result, cbf_result, trajectory, obstacles,
            scenario_label, save_path=args.save,
        )
        plt.show()


if __name__ == "__main__":
    main()
