"""Utility functions."""

from mpc_controller.utils.trajectory import (
    TrajectoryInterpolator,
    generate_line_trajectory,
    generate_circle_trajectory,
    generate_figure_eight_trajectory,
    generate_sinusoidal_trajectory,
    normalize_angle,
    unwrap_angles,
    angle_difference,
)
from mpc_controller.utils.math_utils import (
    normalize_angle_array,
    euclidean_distance,
    rotation_matrix_2d,
    transform_to_local,
    transform_to_global,
    clamp,
    interpolate_angle,
    compute_curvature,
    moving_average,
    low_pass_filter,
    PIDController,
)
from mpc_controller.utils.config import (
    ConfigManager,
    ProjectConfig,
    RobotConfig,
    MPCConfig,
    SimulationConfig,
    ObstacleConfig,
    get_config,
    load_config,
)
from mpc_controller.utils.logger import (
    setup_logger,
    get_logger,
    ColoredFormatter,
    debug,
    info,
    warning,
    error,
    critical,
)

__all__ = [
    # Trajectory
    "TrajectoryInterpolator",
    "generate_line_trajectory",
    "generate_circle_trajectory",
    "generate_figure_eight_trajectory",
    "generate_sinusoidal_trajectory",
    "normalize_angle",
    "unwrap_angles",
    "angle_difference",
    # Math utils
    "normalize_angle_array",
    "euclidean_distance",
    "rotation_matrix_2d",
    "transform_to_local",
    "transform_to_global",
    "clamp",
    "interpolate_angle",
    "compute_curvature",
    "moving_average",
    "low_pass_filter",
    "PIDController",
    # Config
    "ConfigManager",
    "ProjectConfig",
    "RobotConfig",
    "MPCConfig",
    "SimulationConfig",
    "ObstacleConfig",
    "get_config",
    "load_config",
    # Logger
    "setup_logger",
    "get_logger",
    "ColoredFormatter",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
]
