"""Robot kinematic models and cost functions."""

from mpc_controller.models.differential_drive import DifferentialDriveModel, RobotParams
from mpc_controller.models.swerve_drive import SwerveDriveModel, SwerveParams
from mpc_controller.models.non_coaxial_swerve import (
    NonCoaxialSwerveDriveModel,
    NonCoaxialSwerveParams,
)
from mpc_controller.models.cost_functions import (
    CostFunction,
    PositionCost,
    OrientationCost,
    ControlEffortCost,
    ControlSmoothnessCost,
    ObstacleAvoidanceCost,
    TerminalCost,
    CompositeCost,
)
from mpc_controller.models.soft_constraints import (
    ConstraintType,
    PenaltyType,
    SoftConstraintParams,
    SoftConstraint,
    VelocitySoftConstraint,
    AccelerationSoftConstraint,
    ObstacleSoftConstraint,
    PositionSoftConstraint,
    SoftConstraintManager,
    ConstraintViolation,
    SoftConstraintResult,
)

__all__ = [
    # Kinematic models
    "DifferentialDriveModel",
    "RobotParams",
    "SwerveDriveModel",
    "SwerveParams",
    "NonCoaxialSwerveDriveModel",
    "NonCoaxialSwerveParams",
    # Cost functions
    "CostFunction",
    "PositionCost",
    "OrientationCost",
    "ControlEffortCost",
    "ControlSmoothnessCost",
    "ObstacleAvoidanceCost",
    "TerminalCost",
    "CompositeCost",
    # Soft constraints
    "ConstraintType",
    "PenaltyType",
    "SoftConstraintParams",
    "SoftConstraint",
    "VelocitySoftConstraint",
    "AccelerationSoftConstraint",
    "ObstacleSoftConstraint",
    "PositionSoftConstraint",
    "SoftConstraintManager",
    "ConstraintViolation",
    "SoftConstraintResult",
]
