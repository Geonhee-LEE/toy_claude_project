"""Vanilla MPPI 기본 데모 - 원형 궤적 추적.

MPPI 컨트롤러로 원형 궤적을 추적하며,
샘플 궤적과 가중 궤적을 시각화합니다.

실행:
    python examples/mppi_basic_demo.py
    python examples/mppi_basic_demo.py --live
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from mpc_controller import (
    DifferentialDriveModel,
    RobotParams,
    MPPIController,
    MPPIParams,
    generate_circle_trajectory,
    TrajectoryInterpolator,
)


def run_mppi_demo(live: bool = False):
    """MPPI 원형 궤적 추적 데모."""
    # ─── 파라미터 설정 ───
    robot_params = RobotParams(
        max_velocity=1.0,
        max_omega=1.5,
    )
    mppi_params = MPPIParams(
        N=20,
        K=512,
        dt=0.05,
        lambda_=10.0,
        noise_sigma=np.array([0.3, 0.3]),
        Q=np.diag([10.0, 10.0, 1.0]),
        R=np.diag([0.01, 0.01]),
        Qf=np.diag([100.0, 100.0, 10.0]),
    )

    # ─── 컨트롤러 & 모델 생성 ───
    controller = MPPIController(
        robot_params=robot_params,
        mppi_params=mppi_params,
        seed=42,
    )
    model = DifferentialDriveModel(robot_params)

    # ─── 원형 궤적 생성 ───
    radius = 2.0
    trajectory = generate_circle_trajectory(
        center=np.array([0.0, 0.0]),
        radius=radius,
        num_points=400,
    )
    interpolator = TrajectoryInterpolator(trajectory, dt=mppi_params.dt)

    # ─── 실시간 시각화 설정 ───
    visualizer = None
    if live:
        from simulation import MPPILiveVisualizer

        visualizer = MPPILiveVisualizer(
            reference_trajectory=trajectory,
            title="MPPI Circle Tracking (Live)",
            update_interval=2,
        )

    # ─── 시뮬레이션 ───
    state = np.array([radius, 0.0, np.pi / 2])  # 원 위 시작
    dt_sim = mppi_params.dt
    total_time = 10.0  # 10초
    num_steps = int(total_time / dt_sim)

    states_history = [state.copy()]
    controls_history = []
    costs_history = []
    ess_history = []

    print("=" * 60)
    print("  MPPI 원형 궤적 추적 데모")
    print("=" * 60)
    print(f"  샘플 수: {mppi_params.K}")
    print(f"  호라이즌: {mppi_params.N} (dt={mppi_params.dt}s)")
    print(f"  온도: {mppi_params.lambda_}")
    print(f"  목표 반지름: {radius}m")
    print(f"  실시간 시각화: {'활성' if live else '비활성'}")
    print("=" * 60)

    for i in range(num_steps):
        t = i * dt_sim

        # 참조 궤적
        ref = interpolator.get_reference(
            t, mppi_params.N, mppi_params.dt,
            current_theta=state[2],
        )

        # MPPI 제어 계산
        control, info = controller.compute_control(state, ref)

        # 실시간 시각화 업데이트
        if visualizer is not None:
            visualizer.update(
                state=state,
                control=control,
                reference=ref[0],
                prediction=info.get("predicted_trajectory"),
                info=info,
                time=t,
            )

        # 상태 전파
        state = model.forward_simulate(state, control, dt_sim)

        # 기록
        states_history.append(state.copy())
        controls_history.append(control.copy())
        costs_history.append(info["cost"])
        ess_history.append(info["ess"])

    # 실시간 시각화 종료
    if visualizer is not None:
        visualizer.wait_for_close()
        return

    states_history = np.array(states_history)
    controls_history = np.array(controls_history)

    # ─── RMSE 계산 ───
    dist_from_center = np.sqrt(
        states_history[:, 0] ** 2 + states_history[:, 1] ** 2
    )
    position_errors = np.abs(dist_from_center - radius)
    rmse = np.sqrt(np.mean(position_errors ** 2))

    print(f"\n  Position RMSE: {rmse:.4f} m")
    print(f"  {'PASS' if rmse < 0.2 else 'FAIL'} (목표 < 0.2m)")
    print(f"  평균 ESS: {np.mean(ess_history):.1f}/{mppi_params.K}")
    print("=" * 60)

    # ─── 시각화 ───
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 궤적 플롯
    ax = axes[0, 0]
    theta_ref = np.linspace(0, 2 * np.pi, 200)
    ax.plot(radius * np.cos(theta_ref), radius * np.sin(theta_ref),
            "g--", linewidth=1.5, label="Reference", alpha=0.7)
    ax.plot(states_history[:, 0], states_history[:, 1],
            "b-", linewidth=1.5, label="MPPI Tracking")
    ax.plot(states_history[0, 0], states_history[0, 1],
            "ko", markersize=8, label="Start")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("MPPI Circle Tracking")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 2. 위치 오차
    ax = axes[0, 1]
    time_arr = np.arange(len(position_errors)) * dt_sim
    ax.plot(time_arr, position_errors, "r-", linewidth=1.0)
    ax.axhline(y=0.2, color="k", linestyle="--", alpha=0.5, label="RMSE target")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [m]")
    ax.set_title(f"Position Error (RMSE={rmse:.4f}m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 제어 입력
    ax = axes[1, 0]
    time_ctrl = np.arange(len(controls_history)) * dt_sim
    ax.plot(time_ctrl, controls_history[:, 0], "b-", label="v [m/s]")
    ax.plot(time_ctrl, controls_history[:, 1], "r-", label="omega [rad/s]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Control Input")
    ax.set_title("Control Inputs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 비용 & ESS
    ax = axes[1, 1]
    ax2 = ax.twinx()
    l1 = ax.plot(time_ctrl, costs_history, "b-", alpha=0.7, label="Min Cost")
    l2 = ax2.plot(time_ctrl, ess_history, "r-", alpha=0.7, label="ESS")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cost", color="b")
    ax2.set_ylabel("ESS", color="r")
    ax.set_title("Cost & Effective Sample Size")
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mppi_circle_tracking_demo.png", dpi=150, bbox_inches="tight")
    print(f"  그래프 저장: mppi_circle_tracking_demo.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPPI 원형 궤적 추적 데모")
    parser.add_argument(
        "--live",
        action="store_true",
        help="실시간 시각화 활성화 (샘플 궤적, 가중 평균, 최적 샘플 표시)",
    )
    args = parser.parse_args()

    run_mppi_demo(live=args.live)
