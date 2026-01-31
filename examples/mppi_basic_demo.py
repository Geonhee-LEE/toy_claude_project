#!/usr/bin/env python3
"""MPPI Basic Demo.

Demonstrates basic MPPI (Model Predictive Path Integral) control.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from mpc_controller.controllers.mppi.base_mppi import MPPIController
from mpc_controller.controllers.mppi.mppi_params import MPPIParams
from mpc_controller.models.differential_drive import RobotParams
from mpc_controller.utils.trajectory import (
    TrajectoryInterpolator,
    generate_circle_trajectory,
)
from simulation.simulator import SimulationConfig, Simulator
from simulation.visualizer import LiveVisualizer


class MPPILiveVisualizer(LiveVisualizer):
    """
    Live visualizer specialized for MPPI controller.
    
    Visualizes MPPI-specific elements:
    - Sample trajectories with weight-based transparency
    - Weighted average trajectory
    - Best sample trajectory
    - MPPI status information (ESS, temperature, cost)
    """
    
    def __init__(self, trajectory=None, update_interval=0.1):
        super().__init__(trajectory, update_interval)
        self.mppi_info_text = None
        
    def setup_plot(self):
        """Setup MPPI-specific plot elements."""
        super().setup_plot()
        
        # Add text area for MPPI info
        self.mppi_info_text = self.ax.text(
            0.02, 0.98, "", 
            transform=self.ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )
    
    def update_visualization(self, state, control, reference, prediction, info, time_val):
        """Update MPPI-specific visualization elements."""
        # Clear previous MPPI-specific elements
        for line in self.ax.lines[:]:
            if hasattr(line, '_mppi_element'):
                line.remove()
        
        # Update basic elements (robot, prediction, etc.)
        super().update_visualization(state, control, reference, prediction, info, time_val)
        
        # Extract MPPI-specific information
        sample_trajectories = info.get('sample_trajectories', None)
        weights = info.get('weights', None)
        weighted_avg_trajectory = info.get('weighted_avg_trajectory', None)
        best_sample_idx = info.get('best_sample_idx', None)
        ess = info.get('effective_sample_size', 0)
        temperature = info.get('temperature', 0)
        cost = info.get('cost', 0)
        
        # Visualize sample trajectories with weight-based transparency
        if sample_trajectories is not None and weights is not None:
            # Normalize weights for transparency
            max_weight = np.max(weights)
            normalized_weights = weights / max_weight if max_weight > 0 else weights
            
            # Plot top samples (e.g., top 20% by weight)
            num_samples = len(sample_trajectories)
            top_indices = np.argsort(weights)[-max(1, num_samples // 5):]
            
            for idx in top_indices:
                traj = sample_trajectories[idx]
                weight = normalized_weights[idx]
                alpha = 0.2 + 0.6 * weight  # Alpha between 0.2 and 0.8
                
                line, = self.ax.plot(
                    traj[:, 0], traj[:, 1],
                    color='gray', alpha=alpha, linewidth=1,
                    zorder=1
                )
                line._mppi_element = True
        
        # Visualize weighted average trajectory (cyan)
        if weighted_avg_trajectory is not None:
            line, = self.ax.plot(
                weighted_avg_trajectory[:, 0], weighted_avg_trajectory[:, 1],
                color='cyan', linewidth=2, alpha=0.8,
                label='Weighted Average', zorder=3
            )
            line._mppi_element = True
        
        # Visualize best sample trajectory (magenta)
        if sample_trajectories is not None and best_sample_idx is not None:
            best_traj = sample_trajectories[best_sample_idx]
            line, = self.ax.plot(
                best_traj[:, 0], best_traj[:, 1],
                color='magenta', linewidth=2, alpha=0.9,
                label='Best Sample', zorder=4
            )
            line._mppi_element = True
        
        # Update MPPI status info
        if self.mppi_info_text is not None:
            info_str = f"MPPI Status:\n"
            info_str += f"ESS: {ess:.1f}\n"
            info_str += f"Temperature: {temperature:.3f}\n"
            info_str += f"Cost: {cost:.2f}\n"
            info_str += f"Time: {time_val:.1f}s"
            self.mppi_info_text.set_text(info_str)
        
        # Update legend if we have MPPI elements
        if sample_trajectories is not None:
            handles, labels = self.ax.get_legend_handles_labels()
            # Filter out duplicate labels
            by_label = dict(zip(labels, handles))
            self.ax.legend(by_label.values(), by_label.keys(), loc='upper right')


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def run_mppi_demo(live_visualization: bool = False) -> None:
    """Run MPPI demonstration."""
    # Robot parameters
    robot_params = RobotParams(
        wheel_base=0.5,
        max_velocity=1.0,
        max_omega=1.5
    )
    
    # MPPI parameters
    mppi_params = MPPIParams(
        num_samples=200,
        horizon=20,
        dt=0.1,
        lambda_=1.0,
        sigma=np.array([0.5, 1.0]),  # [v, omega] noise std
        gamma_mean=0.0,
        gamma_sigma=1.0,
        a_mean=0.8,
        a_sigma=0.2,
    )
    
    # Create MPPI controller
    mppi = MPPIController(robot_params, mppi_params)
    
    # Generate reference trajectory (circle)
    center = np.array([0.0, 0.0])
    radius = 2.0
    trajectory = generate_circle_trajectory(
        center=center,
        radius=radius,
        num_points=200,
        start_angle=0.0,
        end_angle=2 * np.pi
    )
    
    # Create trajectory interpolator
    traj_interpolator = TrajectoryInterpolator(trajectory, dt=0.05)
    
    # Initial state
    initial_state = np.array([radius, 0.0, np.pi/2])
    
    # Simulation configuration
    sim_config = SimulationConfig(
        dt=0.05,
        max_time=20.0,
        process_noise_std=np.array([0.01, 0.01, 0.005]),
        measurement_noise_std=np.array([0.02, 0.02, 0.01])
    )
    
    # Create simulator
    sim = Simulator(robot_params, sim_config)
    sim.reset(initial_state)
    
    # Setup visualization
    visualizer = None
    if live_visualization:
        visualizer = MPPILiveVisualizer(
            trajectory=trajectory,
            update_interval=0.1
        )
        visualizer.start()
    
    print("\n=== MPPI Controller Demo ===")
    print(f"Samples: {mppi_params.num_samples}")
    print(f"Horizon: {mppi_params.horizon}")
    print(f"Lambda: {mppi_params.lambda_}")
    print(f"Sigma: {mppi_params.sigma}")
    print(f"Live visualization: {live_visualization}")
    print("\nRunning simulation...")
    
    # Simulation loop
    times = []
    states = []
    controls = []
    references = []
    tracking_errors = []
    
    num_steps = int(sim_config.max_time / sim_config.dt)
    
    for step in range(num_steps):
        t = step * sim_config.dt
        
        # Get current state
        current_state = sim.get_measurement(add_noise=False)
        
        # Get reference trajectory
        ref_traj = traj_interpolator.get_reference(
            t, mppi_params.horizon, mppi_params.dt,
            current_theta=current_state[2]
        )
        
        # Compute control with MPPI
        start_time = time.perf_counter()
        control, info = mppi.compute_control(current_state, ref_traj)
        solve_time = time.perf_counter() - start_time
        
        # Step simulation
        next_state = sim.step(control, add_noise=False)
        
        # Compute tracking error
        error = sim.compute_tracking_error(current_state, ref_traj[0])
        
        # Log data
        times.append(t)
        states.append(current_state.copy())
        controls.append(control.copy())
        references.append(ref_traj[0].copy())
        tracking_errors.append(error.copy())
        
        # Update live visualization
        if visualizer is not None:
            visualizer.update(
                state=current_state,
                control=control,
                reference=ref_traj[0],
                prediction=info.get('predicted_trajectory', np.array([])),
                info=info,
                time=t
            )
        
        # Print progress
        if step % 50 == 0:
            pos_error = np.linalg.norm(error[:2])
            print(f"Step {step:3d}, Time: {t:5.1f}s, "
                  f"Pos Error: {pos_error:.3f}m, "
                  f"ESS: {info.get('effective_sample_size', 0):.1f}, "
                  f"Solve: {solve_time*1000:.1f}ms")
        
        # Early termination if close to trajectory end
        _, dist = traj_interpolator.find_closest_point(current_state[:2])
        if dist < 0.1 and t > sim_config.max_time * 0.8:
            print(f"\nReached trajectory end at t={t:.1f}s")
            break
    
    print("\nSimulation completed!")
    
    # Convert to numpy arrays
    times = np.array(times)
    states = np.array(states)
    controls = np.array(controls)
    references = np.array(references)
    tracking_errors = np.array(tracking_errors)
    
    # Calculate performance metrics
    pos_errors = np.linalg.norm(tracking_errors[:, :2], axis=1)
    pos_rmse = np.sqrt(np.mean(pos_errors**2))
    pos_max = np.max(pos_errors)
    heading_rmse = np.sqrt(np.mean(tracking_errors[:, 2]**2))
    
    print(f"\n=== Performance Metrics ===")
    print(f"Position RMSE: {pos_rmse:.3f} m")
    print(f"Position Max Error: {pos_max:.3f} m")
    print(f"Heading RMSE: {np.degrees(heading_rmse):.1f} deg")
    print(f"Average solve time: {np.mean([info.get('solve_time', 0) for info in [{}]*len(times)])*1000:.1f} ms")
    
    # Keep live visualization open
    if live_visualization and visualizer is not None:
        print("\nLive visualization is active. Close the plot window to exit.")
        visualizer.wait()
    else:
        # Static result visualization
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('MPPI Controller Results', fontsize=16)
        
        # Trajectory tracking
        ax = axes[0, 0]
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'k--', linewidth=2, label='Reference')
        ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, label='Actual')
        ax.plot(states[0, 0], states[0, 1], 'go', markersize=8, label='Start')
        ax.plot(states[-1, 0], states[-1, 1], 'ro', markersize=8, label='End')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Trajectory Tracking')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')
        
        # Control inputs
        ax = axes[0, 1]
        ax.plot(times, controls[:, 0], 'b-', label='Linear velocity')
        ax.plot(times, controls[:, 1], 'r-', label='Angular velocity')
        ax.axhline(y=robot_params.max_velocity, color='b', linestyle='--', alpha=0.5)
        ax.axhline(y=-robot_params.max_velocity, color='b', linestyle='--', alpha=0.5)
        ax.axhline(y=robot_params.max_omega, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=-robot_params.max_omega, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Control Input')
        ax.set_title('Control Inputs')
        ax.legend()
        ax.grid(True)
        
        # Position errors
        ax = axes[1, 0]
        ax.plot(times, pos_errors, 'r-', linewidth=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position Error [m]')
        ax.set_title('Position Tracking Error')
        ax.grid(True)
        
        # Heading errors
        ax = axes[1, 1]
        ax.plot(times, np.degrees(tracking_errors[:, 2]), 'g-', linewidth=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Heading Error [deg]')
        ax.set_title('Heading Tracking Error')
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()


def main() -> None:
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="MPPI Basic Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mppi_basic_demo.py                    # Run with static result plots
  python mppi_basic_demo.py --live            # Run with live visualization
        """
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable real-time visualization with MPPI-specific elements"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        run_mppi_demo(live_visualization=args.live)
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logging.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
