"""
mpc_controller_ros2.mppi — Python bindings for C++ MPPI controller core.

Usage:
    from mpc_controller_ros2.mppi import MPPIParams, DiffDriveModel, ...
"""

from mpc_controller_ros2.mppi.mppi_py import (
    # Parameters
    MPPIParams,
    # Motion models
    MotionModel,
    DiffDriveModel,
    SwerveDriveModel,
    NonCoaxialSwerveModel,
    create_motion_model,
    # Dynamics
    BatchDynamicsWrapper,
    # Samplers
    GaussianSampler,
    ColoredNoiseSampler,
    # Cost functions
    CostBreakdown,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ControlRateCost,
    PreferForwardCost,
    ObstacleCost,
    VelocityTrackingCost,
    CompositeMPPICost,
    # Weight computation
    WeightComputation,
    VanillaMPPIWeights,
    LogMPPIWeights,
    TsallisMPPIWeights,
    RiskAwareMPPIWeights,
    # Adaptive temperature
    AdaptiveTemperature,
    AdaptiveInfo,
    # Tube-MPPI
    TubeMPPI,
    TubeMPPIInfo,
    AncillaryController,
    # CBF
    CircleBarrier,
    BarrierFunctionSet,
    CBFSafetyFilter,
    CBFFilterInfo,
    # Savitzky-Golay
    SavitzkyGolayFilter,
    # Utility functions
    normalize_angle,
    normalize_angle_batch,
    softmax_weights,
    log_sum_exp,
    compute_ess,
    euclidean_distance_2d,
    rowwise_min,
    colwise_min,
    q_exponential,
)

__all__ = [
    "MPPIParams",
    "MotionModel", "DiffDriveModel", "SwerveDriveModel",
    "NonCoaxialSwerveModel", "create_motion_model",
    "BatchDynamicsWrapper",
    "GaussianSampler", "ColoredNoiseSampler",
    "CostBreakdown", "StateTrackingCost", "TerminalCost",
    "ControlEffortCost", "ControlRateCost", "PreferForwardCost",
    "ObstacleCost", "VelocityTrackingCost", "CompositeMPPICost",
    "WeightComputation", "VanillaMPPIWeights", "LogMPPIWeights",
    "TsallisMPPIWeights", "RiskAwareMPPIWeights",
    "AdaptiveTemperature", "AdaptiveInfo",
    "TubeMPPI", "TubeMPPIInfo", "AncillaryController",
    "CircleBarrier", "BarrierFunctionSet", "CBFSafetyFilter", "CBFFilterInfo",
    "SavitzkyGolayFilter",
    "normalize_angle", "normalize_angle_batch", "softmax_weights",
    "log_sum_exp", "compute_ess", "euclidean_distance_2d",
    "rowwise_min", "colwise_min", "q_exponential",
]
