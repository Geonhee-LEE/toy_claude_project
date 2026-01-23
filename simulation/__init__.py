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
)
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
