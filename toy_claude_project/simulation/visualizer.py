"""Visualization utilities for simulation results."""

from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np

from simulation.simulator import SimulationResult


def plot_trajectory(
    result: SimulationResult,
    title: str = "Path Tracking Result",
    show_predictions: bool = False,
    prediction_interval: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the simulation trajectory result.

    Args:
        result: Simulation result
        title: Plot title
        show_predictions: Whether to show MPC predictions
        prediction_interval: Show prediction every N steps
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- Trajectory plot ---
    ax = axes[0, 0]
    
    # Reference trajectory
    ax.plot(
        result.references[:, 0],
        result.references[:, 1],
        "b--",
        linewidth=2,
        label="Reference",
        alpha=0.7,
    )
    
    # Actual trajectory
    ax.plot(
        result.states[:, 0],
        result.states[:, 1],
        "r-",
        linewidth=2,
        label="Actual",
    )
    
    # MPC predictions
    if show_predictions:
        for i in range(0, len(result.predicted_trajectories), prediction_interval):
            pred = result.predicted_trajectories[i]
            ax.plot(pred[:, 0], pred[:, 1], "g-", alpha=0.3, linewidth=1)
    
    # Start and end markers
    ax.plot(result.states[0, 0], result.states[0, 1], "go", markersize=10, label="Start")
    ax.plot(result.states[-1, 0], result.states[-1, 1], "ro", markersize=10, label="End")
    
    # Robot heading arrows (sparse)
    arrow_interval = max(1, len(result.states) // 20)
    for i in range(0, len(result.states), arrow_interval):
        x, y, theta = result.states[i]
        dx = 0.1 * np.cos(theta)
        dy = 0.1 * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=0.05, head_length=0.02, fc="red", ec="red")
    
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Trajectory")
    ax.legend()
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    
    # --- Tracking error plot ---
    ax = axes[0, 1]
    
    position_error = np.linalg.norm(result.tracking_errors[:, :2], axis=1)
    ax.plot(result.time, position_error, "b-", linewidth=2, label="Position error")
    ax.plot(result.time, np.abs(result.tracking_errors[:, 2]), "r-", linewidth=2, label="|Heading error|")
    
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error")
    ax.set_title(f"Tracking Error (Position RMSE: {result.position_rmse:.4f} m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Control input plot ---
    ax = axes[1, 0]
    
    ax.plot(result.time, result.controls[:, 0], "b-", linewidth=2, label="v [m/s]")
    ax.plot(result.time, result.controls[:, 1], "r-", linewidth=2, label="ω [rad/s]")
    
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Control")
    ax.set_title("Control Inputs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- State plot ---
    ax = axes[1, 1]
    
    ax.plot(result.time, result.states[:, 0], "b-", linewidth=2, label="x [m]")
    ax.plot(result.time, result.states[:, 1], "r-", linewidth=2, label="y [m]")
    ax.plot(result.time, result.states[:, 2], "g-", linewidth=2, label="θ [rad]")
    
    # Reference states
    ax.plot(result.time, result.references[:, 0], "b--", alpha=0.5)
    ax.plot(result.time, result.references[:, 1], "r--", alpha=0.5)
    ax.plot(result.time, result.references[:, 2], "g--", alpha=0.5)
    
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("State")
    ax.set_title("States (solid: actual, dashed: reference)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def create_animation(
    result: SimulationResult,
    title: str = "MPC Path Tracking",
    interval: int = 50,
    robot_length: float = 0.3,
    robot_width: float = 0.2,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """
    Create animation of the simulation.

    Args:
        result: Simulation result
        title: Animation title
        interval: Frame interval in ms
        robot_length: Robot body length for visualization
        robot_width: Robot body width for visualization
        save_path: Path to save animation (mp4)

    Returns:
        Matplotlib animation
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Reference trajectory
    ax.plot(
        result.references[:, 0],
        result.references[:, 1],
        "b--",
        linewidth=2,
        label="Reference",
        alpha=0.7,
    )
    
    # Trajectory trace (will be updated)
    (trace_line,) = ax.plot([], [], "r-", linewidth=2, label="Actual")
    
    # MPC prediction (will be updated)
    (prediction_line,) = ax.plot([], [], "g-", alpha=0.5, linewidth=1, label="MPC Prediction")
    
    # Robot representation
    robot_patch = patches.Rectangle(
        (0, 0),
        robot_length,
        robot_width,
        angle=0,
        fill=True,
        facecolor="red",
        edgecolor="black",
        linewidth=2,
        alpha=0.8,
    )
    ax.add_patch(robot_patch)
    
    # Direction indicator
    (direction_line,) = ax.plot([], [], "k-", linewidth=2)
    
    # Time text
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=12, verticalalignment="top")
    
    # Set axis limits
    x_min = min(result.states[:, 0].min(), result.references[:, 0].min()) - 1
    x_max = max(result.states[:, 0].max(), result.references[:, 0].max()) + 1
    y_min = min(result.states[:, 1].min(), result.references[:, 1].min()) - 1
    y_max = max(result.states[:, 1].max(), result.references[:, 1].max()) + 1
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    def init():
        trace_line.set_data([], [])
        prediction_line.set_data([], [])
        direction_line.set_data([], [])
        time_text.set_text("")
        return trace_line, prediction_line, robot_patch, direction_line, time_text
    
    def update(frame):
        # Update trace
        trace_line.set_data(result.states[:frame+1, 0], result.states[:frame+1, 1])
        
        # Update MPC prediction
        if frame < len(result.predicted_trajectories):
            pred = result.predicted_trajectories[frame]
            prediction_line.set_data(pred[:, 0], pred[:, 1])
        
        # Update robot position and orientation
        x, y, theta = result.states[frame]
        
        # Robot center to corner offset
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        corner_x = x - (robot_length / 2 * cos_t - robot_width / 2 * sin_t)
        corner_y = y - (robot_length / 2 * sin_t + robot_width / 2 * cos_t)
        
        robot_patch.set_xy((corner_x, corner_y))
        robot_patch.angle = np.degrees(theta)
        
        # Direction indicator
        dir_len = robot_length * 0.8
        direction_line.set_data(
            [x, x + dir_len * cos_t],
            [y, y + dir_len * sin_t],
        )
        
        # Update time text
        time_text.set_text(f"Time: {result.time[frame]:.2f} s")
        
        return trace_line, prediction_line, robot_patch, direction_line, time_text
    
    anim = FuncAnimation(
        fig,
        update,
        frames=len(result.states),
        init_func=init,
        interval=interval,
        blit=True,
    )
    
    if save_path:
        anim.save(save_path, writer="ffmpeg", fps=int(1000 / interval))
    
    return anim


def plot_comparison(
    results: List[SimulationResult],
    labels: List[str],
    title: str = "Controller Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare multiple simulation results.

    Args:
        results: List of simulation results
        labels: Labels for each result
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Trajectory comparison
    ax = axes[0]
    ax.plot(
        results[0].references[:, 0],
        results[0].references[:, 1],
        "k--",
        linewidth=2,
        label="Reference",
        alpha=0.5,
    )
    for result, label, color in zip(results, labels, colors):
        ax.plot(result.states[:, 0], result.states[:, 1], color=color, linewidth=2, label=label)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Trajectories")
    ax.legend()
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    
    # Position error comparison
    ax = axes[1]
    for result, label, color in zip(results, labels, colors):
        pos_error = np.linalg.norm(result.tracking_errors[:, :2], axis=1)
        ax.plot(result.time, pos_error, color=color, linewidth=2, label=f"{label} (RMSE: {result.position_rmse:.4f})")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [m]")
    ax.set_title("Position Tracking Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Metrics bar chart
    ax = axes[2]
    x = np.arange(len(results))
    width = 0.35
    
    rmse_vals = [r.position_rmse for r in results]
    max_err_vals = [r.max_position_error for r in results]
    
    bars1 = ax.bar(x - width/2, rmse_vals, width, label="RMSE", color="steelblue")
    bars2 = ax.bar(x + width/2, max_err_vals, width, label="Max Error", color="coral")
    
    ax.set_ylabel("Error [m]")
    ax.set_title("Error Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig
