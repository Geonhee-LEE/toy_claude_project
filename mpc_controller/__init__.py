"""MPC Controller package for mobile robot control."""

from mpc_controller.models.differential_drive import DifferentialDriveModel, RobotParams
from mpc_controller.models.swerve_drive import SwerveDriveModel, SwerveParams
from mpc_controller.models.non_coaxial_swerve import (
    NonCoaxialSwerveDriveModel,
    NonCoaxialSwerveParams,
)
from mpc_controller.controllers.mpc import MPCController, MPCParams
from mpc_controller.controllers.mppi import MPPIController, MPPIParams, TubeMPPIController
from mpc_controller.controllers.swerve_mpc import SwerveMPCController, SwerveMPCParams
from mpc_controller.controllers.non_coaxial_swerve_mpc import (
    NonCoaxialSwerveMPCController,
    NonCoaxialSwerveMPCParams,
)
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
    "SwerveDriveModel",
    "SwerveParams",
    "NonCoaxialSwerveDriveModel",
    "NonCoaxialSwerveParams",
    "MPCController",
    "MPCParams",
    "MPPIController",
    "TubeMPPIController",
    "MPPIParams",
    "SwerveMPCController",
    "SwerveMPCParams",
    "NonCoaxialSwerveMPCController",
    "NonCoaxialSwerveMPCParams",
    "TrajectoryInterpolator",
    "generate_line_trajectory",
    "generate_circle_trajectory",
    "generate_figure_eight_trajectory",
    "generate_sinusoidal_trajectory",
]
