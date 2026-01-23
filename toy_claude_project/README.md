# Toy Claude Project

Mobile Robot MPC Controller with Claude-Driven Development Workflow

## Overview

This project demonstrates:
1. **MPC-based mobile robot control** - Path tracking with Model Predictive Control
2. **Claude-driven development** - Automated development workflow via GitHub Issues

## Features

- Differential drive robot model
- CasADi-based MPC controller
- 2D simulation with visualization
- Automated CI/CD with Claude integration

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run demo
python examples/path_tracking_demo.py
```

## Project Structure

```
├── mpc_controller/       # Core MPC implementation
│   ├── models/           # Robot kinematic models
│   ├── controllers/      # MPC controller
│   └── utils/            # Trajectory utilities
├── simulation/           # 2D simulator & visualizer
├── tests/                # Unit tests
├── examples/             # Demo scripts
└── .github/workflows/    # CI/CD & Claude automation
```

## Development Workflow

### Via GitHub Issues (Mobile-friendly)

1. Create an issue with label `claude-task`
2. Describe what you want in the issue body
3. Claude automatically creates a PR with the implementation
4. Review and merge

### Issue Template Example

```markdown
Title: Add obstacle avoidance to MPC

## Task
Implement obstacle avoidance constraints in the MPC controller.

## Requirements
- Support circular obstacles
- Soft constraints with slack variables
- Visualization of obstacle regions
```

## Dependencies

- Python >= 3.10
- CasADi >= 3.6
- NumPy
- Matplotlib

## License

MIT
