#!/usr/bin/env python3
"""
Unified MPC Path Tracking Demo.

통합 데모 스크립트: 로봇 모델과 환경을 자유롭게 선택 가능

사용 예시:
    # Differential drive + empty environment
    python examples/unified_demo.py --model differential --environment empty --trajectory circle

    # Swerve drive (holonomic) + obstacle field
    python examples/unified_demo.py --model swerve --environment obstacles --trajectory lateral --live

    # Non-coaxial swerve drive (±90° steering limit)
    python examples/unified_demo.py --model non_coaxial_swerve --trajectory circle

    # Differential drive in parking lot
    python examples/unified_demo.py --model differential --environment parking --trajectory figure8
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from mpc_controller import (
    # Differential drive
    MPCController,
    MPCParams,
    RobotParams,
    # Swerve drive (holonomic)
    SwerveMPCController,
    SwerveMPCParams,
    SwerveParams,
    # Non-coaxial swerve drive (with steering limits)
    NonCoaxialSwerveMPCController,
    NonCoaxialSwerveMPCParams,
    NonCoaxialSwerveParams,
    # Trajectory
    TrajectoryInterpolator,
    generate_circle_trajectory,
    generate_figure_eight_trajectory,
    generate_sinusoidal_trajectory,
    generate_line_trajectory,
)
from simulation import (
    SimulationConfig,
    run_simulation,
    plot_trajectory,
    LiveVisualizer,
    get_environment,
)


def generate_lateral_trajectory(
    start: np.ndarray,
    length: float = 5.0,
    num_points: int = 100,
) -> np.ndarray:
    """Generate a lateral movement trajectory (swerve drive only)."""
    trajectory = np.zeros((num_points, 3))
    trajectory[:, 0] = start[0]
    trajectory[:, 1] = np.linspace(start[1], start[1] + length, num_points)
    trajectory[:, 2] = 0.0
    return trajectory


def generate_holonomic_trajectory(
    center: np.ndarray,
    radius: float = 2.0,
    num_points: int = 100,
) -> np.ndarray:
    """Generate a circle trajectory where robot always faces center (swerve only)."""
    trajectory = np.zeros((num_points, 3))
    angles = np.linspace(0, 2 * np.pi, num_points)

    for i, angle in enumerate(angles):
        trajectory[i, 0] = center[0] + radius * np.cos(angle)
        trajectory[i, 1] = center[1] + radius * np.sin(angle)
        trajectory[i, 2] = angle + np.pi

    return trajectory


def print_banner(model: str, environment: str, trajectory: str):
    """Print ASCII art banner with configuration."""
    model_desc = {
        "differential": "differential (unicycle)",
        "swerve": "swerve (holonomic)",
        "non_coaxial_swerve": "non_coaxial_swerve (±90° steering)",
    }
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           MPC Path Tracking - Unified Demo                    ║
╠═══════════════════════════════════════════════════════════════╣""")
    print(f"║  Model:       {model_desc.get(model, model):<48}║")
    print(f"║  Environment: {environment:<48}║")
    print(f"║  Trajectory:  {trajectory:<48}║")
    print("""╚═══════════════════════════════════════════════════════════════╝
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Unified MPC Path Tracking Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model differential --trajectory circle
  %(prog)s --model swerve --trajectory lateral --environment obstacles
  %(prog)s --model swerve --trajectory holonomic --live
        """,
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="differential",
        choices=["differential", "swerve", "non_coaxial_swerve"],
        help="Robot model type (default: differential)",
    )

    # Environment selection
    parser.add_argument(
        "--environment",
        type=str,
        default="empty",
        choices=["empty", "obstacles", "corridor", "parking", "maze"],
        help="Simulation environment (default: empty)",
    )

    # Trajectory selection
    parser.add_argument(
        "--trajectory",
        type=str,
        default="circle",
        choices=["circle", "figure8", "sine", "line", "lateral", "holonomic"],
        help="Trajectory type (lateral/holonomic only for swerve)",
    )

    # Simulation options
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Add process and measurement noise",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable real-time visualization",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save result figure",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=30.0,
        help="Maximum simulation time (default: 30.0)",
    )

    args = parser.parse_args()

    # Validate trajectory for model type
    swerve_only_trajectories = ["lateral", "holonomic"]
    if args.model == "differential" and args.trajectory in swerve_only_trajectories:
        print(f"Warning: '{args.trajectory}' trajectory is swerve-only. Switching to 'circle'.")
        args.trajectory = "circle"

    # Non-coaxial swerve cannot do lateral/holonomic (requires >90° steering)
    if args.model == "non_coaxial_swerve" and args.trajectory in ["lateral", "holonomic"]:
        print(f"Warning: '{args.trajectory}' requires unlimited steering. Switching to 'circle'.")
        args.trajectory = "circle"

    # Print configuration banner
    print_banner(args.model, args.environment, args.trajectory)

    # Get environment
    env = get_environment(args.environment)
    print(f"Environment loaded: {env.name()}")
    print(f"  - Bounds: {env.bounds}")
    print(f"  - Obstacles: {len(env.obstacles)}")

    # Simulation config
    sim_config = SimulationConfig(
        dt=0.05,
        max_time=args.max_time,
    )

    num_points = 300

    # Setup model-specific parameters
    if args.model == "swerve":
        robot_params = SwerveParams(
            length=0.6,
            width=0.5,
            max_vx=1.5,
            max_vy=1.5,
            max_omega=2.0,
        )
        mpc_params = SwerveMPCParams(
            N=20,
            dt=0.1,
            Q=np.diag([10.0, 10.0, 1.0]),
            R=np.diag([0.1, 0.1, 0.1]),
            Qf=np.diag([100.0, 100.0, 10.0]),
            Rd=np.diag([0.5, 0.5, 0.5]),
        )
        controller = SwerveMPCController(robot_params, mpc_params)
        model_type = "swerve"
        initial_state_dim = 3
        print(f"\nSwerve Drive Robot (Holonomic):")
        print(f"  - Max velocities: vx={robot_params.max_vx}, vy={robot_params.max_vy}, omega={robot_params.max_omega}")

    elif args.model == "non_coaxial_swerve":
        robot_params = NonCoaxialSwerveParams(
            length=0.6,
            width=0.5,
            max_speed=1.5,
            max_omega=2.0,
            max_steering_angle=np.pi / 2,  # ±90°
            max_steering_rate=2.0,
        )
        mpc_params = NonCoaxialSwerveMPCParams(
            N=20,
            dt=0.1,
            Q=np.diag([10.0, 10.0, 1.0, 0.1]),  # [x, y, theta, delta]
            R=np.diag([0.1, 0.1, 0.5]),  # [v, omega, delta_dot]
            Qf=np.diag([100.0, 100.0, 10.0, 0.1]),
            Rd=np.diag([0.5, 0.5, 1.0]),
        )
        controller = NonCoaxialSwerveMPCController(robot_params, mpc_params)
        model_type = "non_coaxial_swerve"
        initial_state_dim = 4  # [x, y, theta, delta]
        print(f"\nNon-Coaxial Swerve Drive Robot:")
        print(f"  - Max speed: {robot_params.max_speed} m/s")
        print(f"  - Max omega: {robot_params.max_omega} rad/s")
        print(f"  - Steering limit: ±{np.degrees(robot_params.max_steering_angle):.0f}°")
        print(f"  - Steering rate: {robot_params.max_steering_rate} rad/s")

    else:
        robot_params = RobotParams(
            wheel_base=0.5,
            max_velocity=1.0,
            max_omega=1.5,
        )
        mpc_params = MPCParams(
            N=20,
            dt=0.1,
            Q=np.diag([10.0, 10.0, 1.0]),
            R=np.diag([0.1, 0.1]),
            Qf=np.diag([100.0, 100.0, 10.0]),
            Rd=np.diag([0.5, 0.5]),
        )
        controller = MPCController(robot_params, mpc_params)
        model_type = "differential"
        initial_state_dim = 3
        print(f"\nDifferential Drive Robot:")
        print(f"  - Max velocities: v={robot_params.max_velocity}, omega={robot_params.max_omega}")

    # Generate trajectory based on argument
    if args.trajectory == "circle":
        trajectory = generate_circle_trajectory(
            center=np.array([0.0, 0.0]),
            radius=2.0,
            num_points=num_points,
        )
        initial_state_3d = np.array([2.0, 0.0, np.pi / 2])
        title = f"{args.model.replace('_', ' ').title()}: Circle"

    elif args.trajectory == "figure8":
        trajectory = generate_figure_eight_trajectory(
            center=np.array([0.0, 0.0]),
            scale=2.0,
            num_points=num_points,
        )
        initial_state_3d = np.array([0.0, 0.0, np.pi / 4])
        title = f"{args.model.replace('_', ' ').title()}: Figure-8"

    elif args.trajectory == "sine":
        trajectory = generate_sinusoidal_trajectory(
            start=np.array([0.0, 0.0]),
            length=10.0,
            amplitude=1.5,
            frequency=2.0,
            num_points=num_points,
        )
        initial_state_3d = np.array([0.0, 0.0, 0.0])
        title = f"{args.model.replace('_', ' ').title()}: Sinusoidal"

    elif args.trajectory == "line":
        trajectory = generate_line_trajectory(
            start=np.array([0.0, 0.0]),
            end=np.array([8.0, 4.0]),
            num_points=num_points,
        )
        initial_state_3d = np.array([0.0, 0.0, 0.0])
        title = f"{args.model.replace('_', ' ').title()}: Straight Line"

    elif args.trajectory == "lateral":
        trajectory = generate_lateral_trajectory(
            start=np.array([0.0, -2.0]),
            length=4.0,
            num_points=num_points,
        )
        initial_state_3d = np.array([0.0, -2.0, 0.0])
        title = "Swerve Drive: Lateral (Crab Walk)"

    elif args.trajectory == "holonomic":
        trajectory = generate_holonomic_trajectory(
            center=np.array([0.0, 0.0]),
            radius=2.0,
            num_points=num_points,
        )
        initial_state_3d = np.array([2.0, 0.0, np.pi])
        title = "Swerve Drive: Holonomic Circle"

    # Adjust initial state for non-coaxial swerve (add delta=0)
    if args.model == "non_coaxial_swerve":
        initial_state = np.append(initial_state_3d, 0.0)  # [x, y, theta, delta=0]
    else:
        initial_state = initial_state_3d

    # Add environment name to title
    if args.environment != "empty":
        title += f" [{env.name()}]"

    # Create trajectory interpolator
    traj_dt = sim_config.max_time / (num_points - 1)
    interpolator = TrajectoryInterpolator(trajectory, traj_dt)

    # Setup live visualization
    visualizer = None
    if args.live:
        visualizer = LiveVisualizer(
            reference_trajectory=trajectory,
            title=title + " (Live)",
            update_interval=2,
            environment=env,
        )

    # Run simulation
    print(f"\nRunning simulation...")
    print(f"  - Trajectory: {args.trajectory}")
    print(f"  - Noise: {'enabled' if args.noise else 'disabled'}")
    print(f"  - Live visualization: {'enabled' if args.live else 'disabled'}")

    result = run_simulation(
        controller=controller,
        trajectory_interpolator=interpolator,
        initial_state=initial_state,
        config=sim_config,
        robot_params=robot_params,
        add_noise=args.noise,
        visualizer=visualizer,
        model_type=model_type,
    )

    if visualizer is not None:
        visualizer.close()

    # Print results
    print("""
┌─────────────────────────────────────────┐
│              Results                    │
├─────────────────────────────────────────┤""")
    print(f"│  Position RMSE:    {result.position_rmse:>10.4f} m       │")
    print(f"│  Heading RMSE:     {np.degrees(result.heading_rmse):>10.2f}°       │")
    print(f"│  Max Position Err: {result.max_position_error:>10.4f} m       │")
    print(f"│  Simulation Time:  {result.time[-1]:>10.2f} s       │")
    print("└─────────────────────────────────────────┘")

    # Model-specific analysis
    if args.model == "swerve":
        controls = result.controls
        print("\nSwerve Drive Control Analysis:")
        print(f"  - Mean |vx|: {np.mean(np.abs(controls[:, 0])):.4f} m/s")
        print(f"  - Mean |vy|: {np.mean(np.abs(controls[:, 1])):.4f} m/s")
        print(f"  - Mean |omega|: {np.mean(np.abs(controls[:, 2])):.4f} rad/s")
        lateral_ratio = np.sum(np.abs(controls[:, 1])) / (np.sum(np.abs(controls[:, 0])) + 1e-6)
        print(f"  - Lateral motion ratio: {lateral_ratio:.2f}")

    elif args.model == "non_coaxial_swerve":
        controls = result.controls
        states = result.states
        print("\nNon-Coaxial Swerve Drive Analysis:")
        print(f"  - Mean |v|: {np.mean(np.abs(controls[:, 0])):.4f} m/s")
        print(f"  - Mean |omega|: {np.mean(np.abs(controls[:, 1])):.4f} rad/s")
        print(f"  - Mean |delta_dot|: {np.mean(np.abs(controls[:, 2])):.4f} rad/s")
        if states.shape[1] >= 4:
            steering_angles = states[:, 3]
            print(f"  - Steering angle range: [{np.degrees(np.min(steering_angles)):.1f}°, {np.degrees(np.max(steering_angles)):.1f}°]")
            print(f"  - Mean |steering|: {np.degrees(np.mean(np.abs(steering_angles))):.1f}°")

    # Check collisions with environment
    collision_count = 0
    for state in result.states:
        if env.is_collision(state, robot_radius=0.2):
            collision_count += 1

    if collision_count > 0:
        print(f"\n⚠️  Collision detected: {collision_count} timesteps in collision")
    else:
        print("\n✓ No collisions detected")

    # Plot results with environment
    fig = plot_trajectory(
        result,
        title=title,
        show_predictions=True,
        prediction_interval=20,
        save_path=args.save,
        environment=env,
    )

    plt.show()


if __name__ == "__main__":
    main()
