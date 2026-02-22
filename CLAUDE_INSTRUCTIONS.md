# Claude Development Instructions

This document provides guidelines for Claude when implementing features in this project.

## Project Overview

Mobile robot MPC controller with:
- Differential drive, Swerve, Non-coaxial Swerve kinematic models
- CasADi-based nonlinear MPC
- MPPI sampling-based control (9 variants + GPU acceleration)
- MPPI-CBF safety integration (Control Barrier Function)
- ROS2 nav2 plugin (8 C++ controllers)
- 2D simulation environment
- Visualization tools

## Code Style

### Python

- Use type hints for all function parameters and return values
- Follow PEP 8 with line length of 100 characters
- Use `numpy` style docstrings
- Prefer dataclasses for configuration/parameter containers

### Naming Conventions

- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Architecture Patterns

1. **Models** (`mpc_controller/models/`):
   - Define robot kinematic/dynamic models
   - Provide both CasADi symbolic and NumPy numerical implementations
   - Include parameter dataclasses

2. **Controllers** (`mpc_controller/controllers/`):
   - Implement control algorithms
   - Use CasADi for optimization
   - Support warm starting
   - Return info dicts with diagnostic data

3. **Simulation** (`simulation/`):
   - Keep simulator independent of controller
   - Support noise injection
   - Log all relevant data in result containers

## Feature Implementation Guidelines

### Adding a New Robot Model

```python
# mpc_controller/models/new_model.py
from dataclasses import dataclass
import casadi as ca
import numpy as np

@dataclass
class NewModelParams:
    """Model parameters."""
    param1: float = 1.0

class NewModel:
    STATE_DIM = ...
    CONTROL_DIM = ...
    
    def __init__(self, params: NewModelParams | None = None):
        self.params = params or NewModelParams()
        self._setup_casadi_model()
    
    def _setup_casadi_model(self) -> None:
        # CasADi symbolic model
        pass
    
    def forward_simulate(self, state, control, dt) -> np.ndarray:
        # NumPy implementation
        pass
```

### Adding a New Controller Feature

1. Add parameters to `MPCParams` dataclass
2. Modify `_setup_optimizer()` to include new constraints/costs
3. Update `compute_control()` if new outputs needed
4. Add tests in `tests/test_mpc.py`

### Adding Obstacle Avoidance

Example structure for obstacle avoidance:

```python
@dataclass
class Obstacle:
    center: np.ndarray
    radius: float

class MPCControllerWithObstacles(MPCController):
    def __init__(self, ..., obstacles: list[Obstacle] = None):
        self.obstacles = obstacles or []
        super().__init__(...)
    
    def _setup_optimizer(self):
        # Add obstacle avoidance constraints
        for obs in self.obstacles:
            # Soft constraint: ||p - p_obs|| >= r_obs
            pass
```

## Testing Requirements

- All new features must have corresponding tests
- Test both success cases and edge cases
- Use `pytest.fixture` for common setup
- Aim for >80% code coverage

## Common Tasks via GitHub Issues

### Example Issue: Add PID Baseline Controller

```markdown
Title: Add PID baseline controller

## Task
Implement a simple PID controller as a baseline for comparison.

## Requirements
- Same interface as MPCController
- Separate gains for position and heading
- Include in comparison visualization

## Files to modify/create
- mpc_controller/controllers/pid.py (new)
- mpc_controller/controllers/__init__.py
- tests/test_pid.py (new)
- examples/compare_controllers.py (new)
```

### Example Issue: Add Obstacle Avoidance

```markdown
Title: Implement obstacle avoidance

## Task
Add circular obstacle avoidance to MPC controller.

## Requirements
- Define Obstacle dataclass
- Add soft constraints with slack variables
- Visualize obstacle regions
- Test with multiple obstacles

## Acceptance Criteria
- Robot avoids all obstacles
- Tracking performance degrades gracefully
- No solver failures
```

## Continuous Integration

All PRs must pass:
1. `ruff check` - linting
2. `mypy` - type checking  
3. `pytest` - unit tests

## Performance Considerations

- MPC solve time should be < 50ms for real-time control
- MPPI solve time: ~20ms (CPU, K=1024), ~2-4ms (GPU, K=4096)
- Use warm starting for consecutive solves
- GPU acceleration: `MPPIParams(use_gpu=True)` for JAX JIT backend
- Profile with `cProfile` for optimization bottlenecks
