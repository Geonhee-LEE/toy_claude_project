"""MPC Controller package for mobile robot control."""

from mpc_controller.models.differential_drive import DifferentialDriveModel, RobotParams
from mpc_controller.controllers.mpc import MPCController, MPCParams
from mpc_controller.utils.trajectory import (
    TrajectoryInterpolator,
    generate_line_trajectory,
    generate_circle_trajectory,
    generate_figure_eight_trajectory,
    generate_sinusoidal_trajectory,
)

__all__ = [
    "DifferentialDriveModel",
    "RobotParams",
    "MPCController",
    "MPCParams",
    "TrajectoryInterpolator",
    "generate_line_trajectory",
    "generate_circle_trajectory",
    "generate_figure_eight_trajectory",
    "generate_sinusoidal_trajectory",
]
