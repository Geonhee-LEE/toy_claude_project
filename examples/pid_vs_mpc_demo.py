#!/usr/bin/env python3
"""
PID vs MPC Controller Comparison Demo.

This example demonstrates:
1. Side-by-side comparison of PID and MPC controllers
2. Performance metrics (tracking error, computation time)
3. Visualization of both controllers' behavior
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List

from mpc_controller.controllers import MPCController, MPCParams, PIDController, PIDGains
from mpc_controller.models import RobotParams
from mpc_controller.utils import (
    generate_circle_trajectory,
    generate_figure_eight_trajectory,
    generate_sinusoidal_trajectory,
    normalize_angle,
)


@dataclass
class SimulationResult:
    """Simulation result data."""
    name: str
    states: np.ndarray
    controls: np.ndarray
    errors: np.ndarray
    solve_times: np.ndarray
    total_time: float


def simulate_controller(
    controller,
    controller_name: str,
    reference_trajectory: np.ndarray,
    initial_state: np.ndarray,
    dt: float = 0.05,
    max_steps: int = 500,
) -> SimulationResult:
    """
    Run simulation with a controller.

    Args:
        controller: Controller instance (MPC or PID)
        controller_name: Name for display
        reference_trajectory: Reference trajectory (N, 3)
        initial_state: Initial robot state [x, y, theta]
        dt: Simulation time step
        max_steps: Maximum simulation steps

    Returns:
        SimulationResult with trajectory data
    """
    robot_params = RobotParams()

    # Storage
    states = [initial_state.copy()]
    controls = []
    errors = []
    solve_times = []

    current_state = initial_state.copy()
    controller.reset()

    start_time = time.perf_counter()

    for step in range(max_steps):
        # Get reference for MPC horizon
        # Find closest point on trajectory
        distances = np.sqrt(
            (reference_trajectory[:, 0] - current_state[0])**2 +
            (reference_trajectory[:, 1] - current_state[1])**2
        )
        closest_idx = np.argmin(distances)

        # Extract reference segment
        ref_length = min(21, len(reference_trajectory) - closest_idx)
        if ref_length < 21:
            # Pad with last point
            ref_segment = np.zeros((21, 3))
            ref_segment[:ref_length] = reference_trajectory[closest_idx:closest_idx + ref_length]
            ref_segment[ref_length:] = reference_trajectory[-1]
        else:
            ref_segment = reference_trajectory[closest_idx:closest_idx + 21]

        # Compute control
        control, info = controller.compute_control(current_state, ref_segment)
        solve_times.append(info['solve_time'])

        # Compute tracking error
        error = np.sqrt(
            (current_state[0] - reference_trajectory[closest_idx, 0])**2 +
            (current_state[1] - reference_trajectory[closest_idx, 1])**2
        )
        errors.append(error)

        # Apply control (simple Euler integration)
        v, omega = control
        current_state[0] += v * np.cos(current_state[2]) * dt
        current_state[1] += v * np.sin(current_state[2]) * dt
        current_state[2] += omega * dt
        current_state[2] = normalize_angle(current_state[2])

        states.append(current_state.copy())
        controls.append(control)

        # Check if reached goal
        goal_dist = np.sqrt(
            (current_state[0] - reference_trajectory[-1, 0])**2 +
            (current_state[1] - reference_trajectory[-1, 1])**2
        )
        if goal_dist < 0.1 and closest_idx > len(reference_trajectory) - 10:
            break

    total_time = time.perf_counter() - start_time

    return SimulationResult(
        name=controller_name,
        states=np.array(states),
        controls=np.array(controls),
        errors=np.array(errors),
        solve_times=np.array(solve_times),
        total_time=total_time,
    )


def plot_comparison(
    reference: np.ndarray,
    pid_result: SimulationResult,
    mpc_result: SimulationResult,
    save_path: str = None,
) -> None:
    """Plot comparison of PID and MPC results."""

    fig = plt.figure(figsize=(16, 10))

    # 1. Trajectory comparison
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(reference[:, 0], reference[:, 1], 'k--', linewidth=2, label='Reference', alpha=0.7)
    ax1.plot(pid_result.states[:, 0], pid_result.states[:, 1], 'b-', linewidth=1.5, label='PID')
    ax1.plot(mpc_result.states[:, 0], mpc_result.states[:, 1], 'r-', linewidth=1.5, label='MPC')
    ax1.scatter([reference[0, 0]], [reference[0, 1]], c='green', s=100, marker='o', zorder=5, label='Start')
    ax1.scatter([reference[-1, 0]], [reference[-1, 1]], c='red', s=100, marker='*', zorder=5, label='Goal')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('Trajectory Comparison')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    # 2. Tracking error over time
    ax2 = fig.add_subplot(2, 3, 2)
    time_pid = np.arange(len(pid_result.errors)) * 0.05
    time_mpc = np.arange(len(mpc_result.errors)) * 0.05
    ax2.plot(time_pid, pid_result.errors, 'b-', label=f'PID (avg: {np.mean(pid_result.errors):.3f}m)')
    ax2.plot(time_mpc, mpc_result.errors, 'r-', label=f'MPC (avg: {np.mean(mpc_result.errors):.3f}m)')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Tracking Error [m]')
    ax2.set_title('Tracking Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Solve time comparison
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(time_pid, pid_result.solve_times * 1000, 'b-', alpha=0.7, label='PID')
    ax3.plot(time_mpc, mpc_result.solve_times * 1000, 'r-', alpha=0.7, label='MPC')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Solve Time [ms]')
    ax3.set_title('Computation Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # 4. Linear velocity
    ax4 = fig.add_subplot(2, 3, 4)
    if len(pid_result.controls) > 0:
        ax4.plot(time_pid[:len(pid_result.controls)], pid_result.controls[:, 0], 'b-', label='PID')
    if len(mpc_result.controls) > 0:
        ax4.plot(time_mpc[:len(mpc_result.controls)], mpc_result.controls[:, 0], 'r-', label='MPC')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Linear Velocity [m/s]')
    ax4.set_title('Linear Velocity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Angular velocity
    ax5 = fig.add_subplot(2, 3, 5)
    if len(pid_result.controls) > 0:
        ax5.plot(time_pid[:len(pid_result.controls)], pid_result.controls[:, 1], 'b-', label='PID')
    if len(mpc_result.controls) > 0:
        ax5.plot(time_mpc[:len(mpc_result.controls)], mpc_result.controls[:, 1], 'r-', label='MPC')
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Angular Velocity [rad/s]')
    ax5.set_title('Angular Velocity')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Summary statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = f"""
╔══════════════════════════════════════════════════════════╗
║                  Performance Summary                      ║
╠══════════════════════════════════════════════════════════╣
║                      PID          MPC                     ║
╠══════════════════════════════════════════════════════════╣
║  Avg Tracking Error   {np.mean(pid_result.errors):6.4f} m    {np.mean(mpc_result.errors):6.4f} m          ║
║  Max Tracking Error   {np.max(pid_result.errors):6.4f} m    {np.max(mpc_result.errors):6.4f} m          ║
║  Avg Solve Time       {np.mean(pid_result.solve_times)*1000:6.3f} ms   {np.mean(mpc_result.solve_times)*1000:6.3f} ms         ║
║  Total Sim Time       {pid_result.total_time:6.3f} s    {mpc_result.total_time:6.3f} s          ║
║  Steps to Goal        {len(pid_result.states):6d}       {len(mpc_result.states):6d}              ║
╠══════════════════════════════════════════════════════════╣
║  Speed Ratio: PID is {np.mean(mpc_result.solve_times)/np.mean(pid_result.solve_times):5.1f}x faster                       ║
╚══════════════════════════════════════════════════════════╝
"""
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
             fontsize=10, fontfamily='monospace', verticalalignment='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def print_results(pid_result: SimulationResult, mpc_result: SimulationResult) -> None:
    """Print comparison results to console."""
    print("\n" + "="*60)
    print("           PID vs MPC Comparison Results")
    print("="*60)
    print(f"\n{'Metric':<25} {'PID':>15} {'MPC':>15}")
    print("-"*55)
    print(f"{'Avg Tracking Error [m]':<25} {np.mean(pid_result.errors):>15.4f} {np.mean(mpc_result.errors):>15.4f}")
    print(f"{'Max Tracking Error [m]':<25} {np.max(pid_result.errors):>15.4f} {np.max(mpc_result.errors):>15.4f}")
    print(f"{'Avg Solve Time [ms]':<25} {np.mean(pid_result.solve_times)*1000:>15.3f} {np.mean(mpc_result.solve_times)*1000:>15.3f}")
    print(f"{'Max Solve Time [ms]':<25} {np.max(pid_result.solve_times)*1000:>15.3f} {np.max(mpc_result.solve_times)*1000:>15.3f}")
    print(f"{'Total Simulation [s]':<25} {pid_result.total_time:>15.3f} {mpc_result.total_time:>15.3f}")
    print(f"{'Steps to Complete':<25} {len(pid_result.states):>15d} {len(mpc_result.states):>15d}")
    print("-"*55)

    speed_ratio = np.mean(mpc_result.solve_times) / np.mean(pid_result.solve_times)
    print(f"\n{'Speed Comparison:':<25} PID is {speed_ratio:.1f}x faster than MPC")

    if np.mean(pid_result.errors) < np.mean(mpc_result.errors):
        print(f"{'Accuracy Comparison:':<25} PID has lower average error")
    else:
        print(f"{'Accuracy Comparison:':<25} MPC has lower average error")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="PID vs MPC Comparison Demo")
    parser.add_argument(
        "--trajectory",
        type=str,
        default="figure8",
        choices=["circle", "figure8", "sine"],
        help="Trajectory type (default: figure8)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save result figure",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (just print results)",
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("         PID vs MPC Controller Comparison Demo")
    print("="*60)

    # Generate reference trajectory
    print(f"\nGenerating {args.trajectory} trajectory...")
    if args.trajectory == "circle":
        reference = generate_circle_trajectory(
            center=np.array([0.0, 0.0]),
            radius=2.0,
            num_points=200
        )
    elif args.trajectory == "figure8":
        reference = generate_figure_eight_trajectory(
            center=np.array([0.0, 0.0]),
            scale=2.0,
            num_points=200
        )
    else:  # sine
        reference = generate_sinusoidal_trajectory(
            start=np.array([0.0, 0.0]),
            length=10.0,
            amplitude=1.0,
            frequency=0.5,
            num_points=200
        )

    # Initial state
    initial_state = reference[0].copy()
    initial_state[2] = reference[1, 2]  # Face towards first waypoint

    # Create controllers
    print("\nInitializing controllers...")

    pid = PIDController(
        pid_gains=PIDGains(
            kp_linear=1.5,
            kd_linear=0.2,
            kp_angular=2.5,
            kd_angular=0.3,
            lookahead_distance=0.3,
        )
    )

    mpc = MPCController(
        mpc_params=MPCParams(
            N=20,
            dt=0.1,
        )
    )

    # Run simulations
    print("\nRunning PID simulation...")
    pid_result = simulate_controller(pid, "PID", reference, initial_state.copy())
    print(f"  Completed in {pid_result.total_time:.3f}s ({len(pid_result.states)} steps)")

    print("\nRunning MPC simulation...")
    mpc_result = simulate_controller(mpc, "MPC", reference, initial_state.copy())
    print(f"  Completed in {mpc_result.total_time:.3f}s ({len(mpc_result.states)} steps)")

    # Print results
    print_results(pid_result, mpc_result)

    # Plot comparison
    if not args.no_plot:
        print("Generating comparison plot...")
        plot_comparison(reference, pid_result, mpc_result, args.save)


if __name__ == "__main__":
    main()
