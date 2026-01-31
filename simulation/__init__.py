"""2D Simulation package."""

from simulation.simulator import (
    Simulator,
    SimulationConfig,
    SimulationResult,
    run_simulation,
)
from simulation.visualizer import (
    plot_trajectory,
    create_animation,
    plot_comparison,
    LiveVisualizer,
    draw_environment,
)
from simulation.mppi_live_visualizer import MPPILiveVisualizer
from simulation.environments import (
    Obstacle,
    CircleObstacle,
    RectangleObstacle,
    WallObstacle,
    Environment,
    EmptyEnvironment,
    ObstacleFieldEnvironment,
    CorridorEnvironment,
    ParkingLotEnvironment,
    MazeEnvironment,
    get_environment,
)

__all__ = [
    "Simulator",
    "SimulationConfig",
    "SimulationResult",
    "run_simulation",
    "plot_trajectory",
    "create_animation",
    "plot_comparison",
    "LiveVisualizer",
    "draw_environment",
    "MPPILiveVisualizer",
    # Environments
    "Obstacle",
    "CircleObstacle",
    "RectangleObstacle",
    "WallObstacle",
    "Environment",
    "EmptyEnvironment",
    "ObstacleFieldEnvironment",
    "CorridorEnvironment",
    "ParkingLotEnvironment",
    "MazeEnvironment",
    "get_environment",
]
