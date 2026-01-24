"""2D mobile robot simulator."""

from dataclasses import dataclass, field
from typing import List, Tuple, Union

import numpy as np

from mpc_controller.models.differential_drive import DifferentialDriveModel, RobotParams
from mpc_controller.models.swerve_drive import SwerveDriveModel, SwerveParams
from mpc_controller.models.non_coaxial_swerve import (
    NonCoaxialSwerveDriveModel,
    NonCoaxialSwerveParams,
)


@dataclass
class SimulationConfig:
    """Simulation configuration."""

    dt: float = 0.05  # Simulation time step [s]
    max_time: float = 30.0  # Maximum simulation time [s]
    
    # Noise parameters
    process_noise_std: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.01, 0.005]))
    measurement_noise_std: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.02, 0.01]))
    
    # Control delay (steps)
    control_delay: int = 0


@dataclass
class SimulationResult:
    """Container for simulation results."""

    time: np.ndarray
    states: np.ndarray  # (T, 3)
    controls: np.ndarray  # (T, 2)
    references: np.ndarray  # (T, 3)
    predicted_trajectories: List[np.ndarray]  # MPC predictions at each step
    tracking_errors: np.ndarray  # (T, 3)
    
    @property
    def position_rmse(self) -> float:
        """Root mean squared position error."""
        pos_errors = self.tracking_errors[:, :2]
        return np.sqrt(np.mean(np.sum(pos_errors**2, axis=1)))
    
    @property
    def heading_rmse(self) -> float:
        """Root mean squared heading error."""
        return np.sqrt(np.mean(self.tracking_errors[:, 2]**2))
    
    @property
    def max_position_error(self) -> float:
        """Maximum position error."""
        pos_errors = np.linalg.norm(self.tracking_errors[:, :2], axis=1)
        return np.max(pos_errors)


class Simulator:
    """
    2D mobile robot simulator.

    Supports both differential drive and swerve drive models.
    """

    def __init__(
        self,
        robot_params: Union[RobotParams, SwerveParams, NonCoaxialSwerveParams, None] = None,
        config: SimulationConfig | None = None,
        model_type: str = "differential",
    ):
        """
        Initialize simulator.

        Args:
            robot_params: Robot parameters
            config: Simulation configuration
            model_type: "differential", "swerve", or "non_coaxial_swerve"
        """
        self.model_type = model_type
        if model_type == "swerve":
            self.robot = SwerveDriveModel(robot_params or SwerveParams())
        elif model_type == "non_coaxial_swerve":
            self.robot = NonCoaxialSwerveDriveModel(robot_params or NonCoaxialSwerveParams())
        else:
            self.robot = DifferentialDriveModel(robot_params or RobotParams())
        self.config = config or SimulationConfig()
        self.reset()

    def reset(self, initial_state: np.ndarray | None = None) -> None:
        """Reset simulator state."""
        if initial_state is None:
            self.state = np.zeros(3)
        else:
            self.state = initial_state.copy()
        
        self.time = 0.0
        self.control_buffer = []  # For control delay

    def step(
        self,
        control: np.ndarray,
        add_noise: bool = False,
    ) -> np.ndarray:
        """
        Execute one simulation step.

        Args:
            control: Control input [v, omega]
            add_noise: Whether to add process noise

        Returns:
            New state [x, y, theta]
        """
        # Apply control delay
        control_dim = self.robot.CONTROL_DIM
        if self.config.control_delay > 0:
            self.control_buffer.append(control.copy())
            if len(self.control_buffer) > self.config.control_delay:
                control = self.control_buffer.pop(0)
            else:
                control = np.zeros(control_dim)

        # Apply control limits
        u_lb, u_ub = self.robot.get_control_bounds()
        control = np.clip(control, u_lb, u_ub)

        # Forward simulate
        self.state = self.robot.forward_simulate(self.state, control, self.config.dt)

        # Add process noise
        if add_noise:
            noise = np.random.normal(0, self.config.process_noise_std)
            self.state += noise
            self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))

        self.time += self.config.dt
        return self.state.copy()

    def get_measurement(self, add_noise: bool = False) -> np.ndarray:
        """
        Get current state measurement.

        Args:
            add_noise: Whether to add measurement noise

        Returns:
            Measured state [x, y, theta]
        """
        measurement = self.state.copy()
        
        if add_noise:
            noise = np.random.normal(0, self.config.measurement_noise_std)
            measurement += noise
            measurement[2] = np.arctan2(np.sin(measurement[2]), np.cos(measurement[2]))
        
        return measurement

    def compute_tracking_error(
        self,
        state: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """
        Compute tracking error with proper angle wrapping.

        Args:
            state: Current state [x, y, theta]
            reference: Reference state [x, y, theta]

        Returns:
            Error [ex, ey, etheta]
        """
        error = state - reference
        error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
        return error


def run_simulation(
    controller,
    trajectory_interpolator,
    initial_state: np.ndarray,
    config: SimulationConfig | None = None,
    robot_params: Union[RobotParams, SwerveParams, NonCoaxialSwerveParams, None] = None,
    add_noise: bool = False,
    visualizer=None,
    model_type: str = "differential",
    environment=None,
) -> SimulationResult:
    """
    Run a full simulation with MPC controller.

    Args:
        controller: MPC controller instance
        trajectory_interpolator: Trajectory interpolator
        initial_state: Initial robot state
        config: Simulation configuration
        robot_params: Robot parameters
        add_noise: Whether to add noise
        visualizer: Optional LiveVisualizer instance for real-time visualization
        model_type: "differential", "swerve", or "non_coaxial_swerve"
        environment: Optional environment for collision checking

    Returns:
        SimulationResult containing all logged data
    """
    config = config or SimulationConfig()
    sim = Simulator(robot_params, config, model_type=model_type)
    sim.reset(initial_state)

    # Pre-allocate arrays
    num_steps = int(config.max_time / config.dt)
    times = []
    states = []
    controls = []
    references = []
    predictions = []
    errors = []

    controller.reset()
    
    for step in range(num_steps):
        t = step * config.dt
        
        # Get current measurement
        current_state = sim.get_measurement(add_noise)
        
        # Get reference trajectory for MPC horizon
        ref_traj = trajectory_interpolator.get_reference(
            t,
            controller.params.N,
            controller.params.dt,
        )
        
        # Compute control
        control, info = controller.compute_control(current_state, ref_traj)
        
        # Step simulation
        next_state = sim.step(control, add_noise)
        
        # Log data
        times.append(t)
        states.append(current_state.copy())
        controls.append(control.copy())
        references.append(ref_traj[0].copy())
        predictions.append(info["predicted_trajectory"].copy())
        errors.append(sim.compute_tracking_error(current_state, ref_traj[0]))

        # Update live visualization if provided
        if visualizer is not None:
            visualizer.update(
                state=current_state,
                control=control,
                reference=ref_traj[0],
                prediction=info["predicted_trajectory"],
                info=info,
                time=t,
            )
        
        # Check if reached end of trajectory
        _, dist = trajectory_interpolator.find_closest_point(current_state[:2])
        idx, _ = trajectory_interpolator.find_closest_point(current_state[:2])
        if idx >= trajectory_interpolator.num_points - 1 and dist < 0.1:
            break

    return SimulationResult(
        time=np.array(times),
        states=np.array(states),
        controls=np.array(controls),
        references=np.array(references),
        predicted_trajectories=predictions,
        tracking_errors=np.array(errors),
    )
