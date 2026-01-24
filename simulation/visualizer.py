"""Visualization utilities for simulation results."""

from typing import Callable, List, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np

from simulation.simulator import SimulationResult

if TYPE_CHECKING:
    from simulation.environments import Environment


def draw_environment(ax: plt.Axes, environment: "Environment") -> None:
    """
    Draw environment obstacles on the given axes.

    Args:
        ax: Matplotlib axes
        environment: Environment instance with obstacles
    """
    from simulation.environments import CircleObstacle, RectangleObstacle, WallObstacle

    for obs in environment.obstacles:
        if isinstance(obs, CircleObstacle):
            circle = patches.Circle(
                obs.center,
                obs.radius,
                fill=True,
                facecolor="gray",
                edgecolor="black",
                linewidth=1,
                alpha=0.7,
            )
            ax.add_patch(circle)

        elif isinstance(obs, RectangleObstacle):
            rect = patches.Rectangle(
                (obs.center[0] - obs.width / 2, obs.center[1] - obs.height / 2),
                obs.width,
                obs.height,
                fill=True,
                facecolor="gray",
                edgecolor="black",
                linewidth=1,
                alpha=0.7,
            )
            ax.add_patch(rect)

        elif isinstance(obs, WallObstacle):
            ax.plot(
                [obs.start[0], obs.end[0]],
                [obs.start[1], obs.end[1]],
                "k-",
                linewidth=max(2, obs.thickness * 20),
                solid_capstyle="round",
            )


class LiveVisualizer:
    """Real-time visualization for MPC simulation."""

    def __init__(
        self,
        reference_trajectory: np.ndarray,
        title: str = "MPC Path Tracking (Live)",
        robot_length: float = 0.3,
        robot_width: float = 0.2,
        update_interval: int = 1,
        environment: Optional["Environment"] = None,
    ):
        """
        Initialize live visualizer.

        Args:
            reference_trajectory: Reference trajectory points (N, 3)
            title: Plot title
            robot_length: Robot body length for visualization
            robot_width: Robot body width for visualization
            update_interval: Update display every N steps
            environment: Optional environment to display obstacles
        """
        self.reference_trajectory = reference_trajectory
        self.robot_length = robot_length
        self.robot_width = robot_width
        self.update_interval = update_interval
        self.step_count = 0
        self.environment = environment

        # Enable interactive mode
        plt.ion()

        # Create figure with subplots
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- Main trajectory plot ---
        self.ax_traj = self.axes[0]
        self.ax_traj.set_title(title)
        self.ax_traj.set_xlabel("X [m]")
        self.ax_traj.set_ylabel("Y [m]")
        self.ax_traj.grid(True, alpha=0.3)
        self.ax_traj.axis("equal")

        # Draw environment obstacles
        if environment is not None:
            draw_environment(self.ax_traj, environment)

        # Reference trajectory
        self.ax_traj.plot(
            reference_trajectory[:, 0],
            reference_trajectory[:, 1],
            "b--",
            linewidth=2,
            label="Reference",
            alpha=0.7,
        )

        # Actual trajectory trace
        (self.trace_line,) = self.ax_traj.plot(
            [], [], "r-", linewidth=2, label="Actual"
        )

        # MPC prediction line
        (self.prediction_line,) = self.ax_traj.plot(
            [], [], "g-", alpha=0.6, linewidth=1.5, label="MPC Prediction"
        )

        # Robot patch
        self.robot_patch = patches.Rectangle(
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
        self.ax_traj.add_patch(self.robot_patch)

        # Direction indicator
        (self.direction_line,) = self.ax_traj.plot([], [], "k-", linewidth=2)

        # Set axis limits
        x_margin = 1.0
        y_margin = 1.0
        self.ax_traj.set_xlim(
            reference_trajectory[:, 0].min() - x_margin,
            reference_trajectory[:, 0].max() + x_margin,
        )
        self.ax_traj.set_ylim(
            reference_trajectory[:, 1].min() - y_margin,
            reference_trajectory[:, 1].max() + y_margin,
        )
        self.ax_traj.legend(loc="upper right")

        # --- Info panel ---
        self.ax_info = self.axes[1]
        self.ax_info.axis("off")
        self.ax_info.set_title("Simulation Status")

        self.info_text = self.ax_info.text(
            0.1, 0.9, "", transform=self.ax_info.transAxes,
            fontsize=12, verticalalignment="top", fontfamily="monospace"
        )

        # Data storage for trace
        self.trace_x = []
        self.trace_y = []

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(
        self,
        state: np.ndarray,
        control: np.ndarray,
        reference: np.ndarray,
        prediction: Optional[np.ndarray] = None,
        info: Optional[dict] = None,
        time: float = 0.0,
    ) -> None:
        """
        Update visualization with current state.

        Args:
            state: Current robot state [x, y, theta]
            control: Current control input [v, omega]
            reference: Current reference state [x, y, theta]
            prediction: MPC predicted trajectory (N+1, 3)
            info: Additional info dict from controller
            time: Current simulation time
        """
        self.step_count += 1

        # Store trace
        self.trace_x.append(state[0])
        self.trace_y.append(state[1])

        # Only update display at specified interval
        if self.step_count % self.update_interval != 0:
            return

        # Update trace
        self.trace_line.set_data(self.trace_x, self.trace_y)

        # Update MPC prediction
        if prediction is not None:
            self.prediction_line.set_data(prediction[:, 0], prediction[:, 1])

        # Update robot position
        x, y, theta = state
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        corner_x = x - (self.robot_length / 2 * cos_t - self.robot_width / 2 * sin_t)
        corner_y = y - (self.robot_length / 2 * sin_t + self.robot_width / 2 * cos_t)

        self.robot_patch.set_xy((corner_x, corner_y))
        self.robot_patch.angle = np.degrees(theta)

        # Direction indicator
        dir_len = self.robot_length * 0.8
        self.direction_line.set_data(
            [x, x + dir_len * cos_t],
            [y, y + dir_len * sin_t],
        )

        # Update info text
        pos_error = np.sqrt((state[0] - reference[0])**2 + (state[1] - reference[1])**2)
        heading_error = np.arctan2(np.sin(state[2] - reference[2]), np.cos(state[2] - reference[2]))

        info_str = (
            f"Time: {time:.2f} s\n"
            f"\n"
            f"State:\n"
            f"  x: {state[0]:+.3f} m\n"
            f"  y: {state[1]:+.3f} m\n"
            f"  θ: {np.degrees(state[2]):+.1f}°\n"
            f"\n"
            f"Control:\n"
            f"  v: {control[0]:+.3f} m/s\n"
            f"  ω: {control[1]:+.3f} rad/s\n"
            f"\n"
            f"Tracking Error:\n"
            f"  position: {pos_error:.4f} m\n"
            f"  heading: {np.degrees(heading_error):+.1f}°\n"
        )

        if info is not None:
            info_str += (
                f"\n"
                f"MPC Info:\n"
                f"  cost: {info.get('cost', 0):.4f}\n"
                f"  solve_time: {info.get('solve_time', 0)*1000:.2f} ms\n"
                f"  status: {info.get('solver_status', 'N/A')}\n"
            )

            # Soft constraint visualization
            soft_info = info.get("soft_constraints", {})
            if soft_info:
                has_violations = soft_info.get("has_violations", False)
                info_str += (
                    f"\n"
                    f"Soft Constraints:\n"
                    f"  violations: {'YES' if has_violations else 'NO'}\n"
                )
                if has_violations:
                    info_str += (
                        f"  max_vel_viol: {soft_info.get('max_velocity_violation', 0):.4f}\n"
                        f"  max_acc_viol: {soft_info.get('max_acceleration_violation', 0):.4f}\n"
                    )
                    # Change robot color to indicate violation
                    self.robot_patch.set_facecolor("orange")
                else:
                    self.robot_patch.set_facecolor("red")

        self.info_text.set_text(info_str)

        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        """Close the visualization window."""
        plt.ioff()

    def wait_for_close(self) -> None:
        """Keep window open until user closes it."""
        plt.ioff()
        plt.show()


def plot_trajectory(
    result: SimulationResult,
    title: str = "Path Tracking Result",
    show_predictions: bool = False,
    prediction_interval: int = 10,
    save_path: Optional[str] = None,
    environment: Optional["Environment"] = None,
) -> plt.Figure:
    """
    Plot the simulation trajectory result.

    Args:
        result: Simulation result
        title: Plot title
        show_predictions: Whether to show MPC predictions
        prediction_interval: Show prediction every N steps
        save_path: Path to save figure
        environment: Optional environment to display obstacles

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Trajectory plot ---
    ax = axes[0, 0]

    # Draw environment obstacles first (background)
    if environment is not None:
        draw_environment(ax, environment)

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
    environment: Optional["Environment"] = None,
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
        environment: Optional environment to display obstacles

    Returns:
        Matplotlib animation
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw environment obstacles first (background)
    if environment is not None:
        draw_environment(ax, environment)

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
