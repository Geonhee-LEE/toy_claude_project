#!/usr/bin/env python3
"""
Swerve Drive Path Tracking Demo with MPC Controller.

This example demonstrates:
1. Swerve drive omnidirectional movement capabilities
2. MPC-based path tracking with swerve drive
3. Comparison with differential drive behavior
4. Environment-based simulation
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from mpc_controller import (
    SwerveMPCController,
    SwerveMPCParams,
    SwerveParams,
    TrajectoryInterpolator,
    generate_circle_trajectory,
    generate_figure_eight_trajectory,
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
    """Generate a lateral movement trajectory (facing forward, moving sideways)."""
    trajectory = np.zeros((num_points, 3))
    trajectory[:, 0] = start[0]  # x stays constant
    trajectory[:, 1] = np.linspace(start[1], start[1] + length, num_points)  # y increases
    trajectory[:, 2] = 0.0  # Heading stays at 0 (facing +x)
    return trajectory


def generate_holonomic_trajectory(
    center: np.ndarray,
    radius: float = 2.0,
    num_points: int = 100,
) -> np.ndarray:
    """Generate a circle trajectory where robot always faces center (holonomic)."""
    trajectory = np.zeros((num_points, 3))
    angles = np.linspace(0, 2 * np.pi, num_points)

    for i, angle in enumerate(angles):
        # Position on circle
        trajectory[i, 0] = center[0] + radius * np.cos(angle)
        trajectory[i, 1] = center[1] + radius * np.sin(angle)
        # Heading: always face toward center
        trajectory[i, 2] = angle + np.pi  # Face inward

    return trajectory


def main():
    parser = argparse.ArgumentParser(description="Swerve Drive MPC Path Tracking Demo")
    parser.add_argument(
        "--trajectory",
        type=str,
        default="lateral",
        choices=["lateral", "holonomic", "figure8", "circle"],
        help="Trajectory type",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="empty",
        choices=["empty", "obstacles", "corridor", "parking", "maze"],
        help="Environment type",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Add process and measurement noise",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save result figure",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable real-time visualization during simulation",
    )
    args = parser.parse_args()

    # Swerve drive robot parameters
    robot_params = SwerveParams(
        length=0.6,
        width=0.5,
        max_vx=1.5,
        max_vy=1.5,
        max_omega=2.0,
    )

    # MPC parameters for swerve drive
    mpc_params = SwerveMPCParams(
        N=20,
        dt=0.1,
        Q=np.diag([10.0, 10.0, 1.0]),  # State weights [x, y, theta]
        R=np.diag([0.1, 0.1, 0.1]),  # Control weights [vx, vy, omega]
        Qf=np.diag([100.0, 100.0, 10.0]),  # Terminal weights
        Rd=np.diag([0.5, 0.5, 0.5]),  # Control rate weights
    )

    # Simulation config
    sim_config = SimulationConfig(
        dt=0.05,
        max_time=30.0,
    )

    # Get environment
    env = get_environment(args.environment)
    print(f"Environment: {env.name()}")

    # Generate trajectory based on argument
    num_points = 300

    if args.trajectory == "lateral":
        trajectory = generate_lateral_trajectory(
            start=np.array([0.0, -2.0]),
            length=4.0,
            num_points=num_points,
        )
        initial_state = np.array([0.0, -2.0, 0.0])
        title = "Swerve Drive: Lateral Movement (crab walk)"

    elif args.trajectory == "holonomic":
        trajectory = generate_holonomic_trajectory(
            center=np.array([0.0, 0.0]),
            radius=2.0,
            num_points=num_points,
        )
        initial_state = np.array([2.0, 0.0, np.pi])
        title = "Swerve Drive: Holonomic Circle (facing center)"

    elif args.trajectory == "figure8":
        trajectory = generate_figure_eight_trajectory(
            center=np.array([0.0, 0.0]),
            scale=2.0,
            num_points=num_points,
        )
        initial_state = np.array([0.0, 0.0, np.pi / 4])
        title = "Swerve Drive: Figure-8 Trajectory"

    else:  # circle
        trajectory = generate_circle_trajectory(
            center=np.array([0.0, 0.0]),
            radius=2.0,
            num_points=num_points,
        )
        initial_state = np.array([2.0, 0.0, np.pi / 2])
        title = "Swerve Drive: Circle Trajectory"

    # Create trajectory interpolator
    traj_dt = sim_config.max_time / (num_points - 1)
    interpolator = TrajectoryInterpolator(trajectory, traj_dt)

    # Create swerve drive controller
    controller = SwerveMPCController(robot_params, mpc_params)

    # Setup live visualization if requested
    visualizer = None
    if args.live:
        visualizer = LiveVisualizer(
            reference_trajectory=trajectory,
            title=title + " (Live)",
            update_interval=2,
        )

    # Run simulation
    print(f"\nRunning simulation: {args.trajectory} trajectory")
    print(f"  - Robot type: Swerve Drive")
    print(f"  - Max velocities: vx={robot_params.max_vx}, vy={robot_params.max_vy}, omega={robot_params.max_omega}")
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
        model_type="swerve",
    )

    # Close live visualizer if used
    if visualizer is not None:
        visualizer.close()

    # Print results
    print("\nResults:")
    print(f"  - Position RMSE: {result.position_rmse:.4f} m")
    print(f"  - Heading RMSE: {np.degrees(result.heading_rmse):.2f} deg")
    print(f"  - Max position error: {result.max_position_error:.4f} m")
    print(f"  - Simulation time: {result.time[-1]:.2f} s")

    # Analyze swerve-specific metrics
    controls = result.controls
    print("\nSwerve Drive Control Analysis:")
    print(f"  - Mean |vx|: {np.mean(np.abs(controls[:, 0])):.4f} m/s")
    print(f"  - Mean |vy|: {np.mean(np.abs(controls[:, 1])):.4f} m/s")
    print(f"  - Mean |omega|: {np.mean(np.abs(controls[:, 2])):.4f} rad/s")
    print(f"  - Lateral motion ratio: {np.sum(np.abs(controls[:, 1])) / (np.sum(np.abs(controls[:, 0])) + 1e-6):.2f}")

    # Plot results
    fig = plot_trajectory(
        result,
        title=title,
        show_predictions=True,
        prediction_interval=20,
        save_path=args.save,
    )

    plt.show()


if __name__ == "__main__":
    main()
