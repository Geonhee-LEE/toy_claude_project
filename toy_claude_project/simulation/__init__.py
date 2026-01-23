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
)

__all__ = [
    "Simulator",
    "SimulationConfig",
    "SimulationResult",
    "run_simulation",
    "plot_trajectory",
    "create_animation",
    "plot_comparison",
]
