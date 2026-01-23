#!/usr/bin/env python3
"""
Path Tracking Demo with MPC Controller.

This example demonstrates:
1. Different trajectory types (circle, figure-8, sinusoidal)
2. MPC-based path tracking
3. Result visualization
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from mpc_controller import (
    MPCController,
    MPCParams,
    RobotParams,
    TrajectoryInterpolator,
    generate_circle_trajectory,
    generate_figure_eight_trajectory,
    generate_sinusoidal_trajectory,
)
from simulation import (
    SimulationConfig,
    run_simulation,
    plot_trajectory,
    LiveVisualizer,
)


def main():
    parser = argparse.ArgumentParser(description="MPC Path Tracking Demo")
    parser.add_argument(
        "--trajectory",
        type=str,
        default="figure8",
        choices=["circle", "figure8", "sine"],
        help="Trajectory type",
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

    # Robot parameters
    robot_params = RobotParams(
        wheel_base=0.5,
        max_velocity=1.0,
        max_omega=1.5,
    )

    # MPC parameters
    mpc_params = MPCParams(
        N=20,
        dt=0.1,
        Q=np.diag([10.0, 10.0, 1.0]),  # State weights
        R=np.diag([0.1, 0.1]),  # Control weights
        Qf=np.diag([100.0, 100.0, 10.0]),  # Terminal weights
        Rd=np.diag([0.5, 0.5]),  # Control rate weights
    )

    # Simulation config
    sim_config = SimulationConfig(
        dt=0.05,
        max_time=30.0,
    )

    # Generate trajectory based on argument
    num_points = 300
    
    if args.trajectory == "circle":
        trajectory = generate_circle_trajectory(
            center=np.array([0.0, 0.0]),
            radius=2.0,
            num_points=num_points,
        )
        initial_state = np.array([2.0, 0.0, np.pi / 2])
        title = "Circle Trajectory Tracking"
        
    elif args.trajectory == "figure8":
        trajectory = generate_figure_eight_trajectory(
            center=np.array([0.0, 0.0]),
            scale=2.0,
            num_points=num_points,
        )
        initial_state = np.array([0.0, 0.0, np.pi / 4])
        title = "Figure-8 Trajectory Tracking"
        
    else:  # sine
        trajectory = generate_sinusoidal_trajectory(
            start=np.array([0.0, 0.0]),
            length=10.0,
            amplitude=1.5,
            frequency=2.0,
            num_points=num_points,
        )
        initial_state = np.array([0.0, 0.0, 0.0])
        title = "Sinusoidal Trajectory Tracking"

    # Create trajectory interpolator
    traj_dt = sim_config.max_time / (num_points - 1)
    interpolator = TrajectoryInterpolator(trajectory, traj_dt)

    # Create controller
    controller = MPCController(robot_params, mpc_params)

    # Setup live visualization if requested
    visualizer = None
    if args.live:
        visualizer = LiveVisualizer(
            reference_trajectory=trajectory,
            title=title + " (Live)",
            update_interval=2,  # Update every 2 steps for smoother display
        )

    # Run simulation
    print(f"Running simulation: {args.trajectory} trajectory")
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
