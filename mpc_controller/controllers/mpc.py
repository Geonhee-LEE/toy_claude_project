"""Model Predictive Controller for path tracking."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import casadi as ca
import numpy as np

from mpc_controller.models.differential_drive import DifferentialDriveModel, RobotParams

# Module-level logger
logger = logging.getLogger(__name__)


@dataclass
class MPCParams:
    """MPC tuning parameters."""

    # Horizon
    N: int = 20  # Prediction horizon
    dt: float = 0.1  # Time step [s]

    # State weights [x, y, theta]
    Q: np.ndarray | None = None

    # Control weights [v, omega]
    R: np.ndarray | None = None

    # Terminal state weights
    Qf: np.ndarray | None = None

    # Control rate weights (smoothness)
    Rd: np.ndarray | None = None

    def __post_init__(self):
        if self.Q is None:
            self.Q = np.diag([10.0, 10.0, 1.0])
        if self.R is None:
            self.R = np.diag([0.1, 0.1])
        if self.Qf is None:
            self.Qf = np.diag([100.0, 100.0, 10.0])
        if self.Rd is None:
            self.Rd = np.diag([0.5, 0.5])


class MPCController:
    """
    Nonlinear MPC controller for path tracking.

    Uses CasADi + IPOPT for optimization.
    """

    def __init__(
        self,
        robot_params: RobotParams | None = None,
        mpc_params: MPCParams | None = None,
        log_file: str | Path | None = None,
    ):
        self.robot = DifferentialDriveModel(robot_params or RobotParams())
        self.params = mpc_params or MPCParams()
        self._iteration_count = 0
        self._setup_logging(log_file)
        self._setup_optimizer()

    def _setup_logging(self, log_file: str | Path | None = None) -> None:
        """Setup logging configuration."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        if log_file is not None:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            logger.info(f"MPC logging to file: {log_file}")

    def _setup_optimizer(self) -> None:
        """Setup the NLP optimization problem."""
        N = self.params.N
        nx = DifferentialDriveModel.STATE_DIM
        nu = DifferentialDriveModel.CONTROL_DIM

        # Decision variables
        X = ca.SX.sym("X", nx, N + 1)  # States over horizon
        U = ca.SX.sym("U", nu, N)  # Controls over horizon

        # Parameters: initial state + reference trajectory
        P = ca.SX.sym("P", nx + nx * (N + 1))  # x0 + ref_traj

        # Get discrete dynamics
        f_discrete = self.robot.discrete_dynamics(self.params.dt)

        # Build cost function and constraints
        cost = 0
        constraints = []
        lb_constraints = []
        ub_constraints = []

        # Initial state constraint
        constraints.append(X[:, 0] - P[:nx])
        lb_constraints.extend([0.0] * nx)
        ub_constraints.extend([0.0] * nx)

        # Extract weight matrices
        Q = ca.DM(self.params.Q)
        R = ca.DM(self.params.R)
        Qf = ca.DM(self.params.Qf)
        Rd = ca.DM(self.params.Rd)

        for k in range(N):
            # Reference state at step k
            ref_k = P[nx + k * nx : nx + (k + 1) * nx]

            # State error
            state_error = X[:, k] - ref_k

            # Handle angle wrapping for theta error
            state_error[2] = ca.atan2(ca.sin(state_error[2]), ca.cos(state_error[2]))

            # Stage cost
            cost += ca.mtimes([state_error.T, Q, state_error])
            cost += ca.mtimes([U[:, k].T, R, U[:, k]])

            # Control rate cost (smoothness)
            if k > 0:
                du = U[:, k] - U[:, k - 1]
                cost += ca.mtimes([du.T, Rd, du])

            # Dynamics constraint
            x_next = f_discrete(X[:, k], U[:, k])
            constraints.append(X[:, k + 1] - x_next)
            lb_constraints.extend([0.0] * nx)
            ub_constraints.extend([0.0] * nx)

        # Terminal cost
        ref_N = P[nx + N * nx : nx + (N + 1) * nx]
        terminal_error = X[:, N] - ref_N
        terminal_error[2] = ca.atan2(ca.sin(terminal_error[2]), ca.cos(terminal_error[2]))
        cost += ca.mtimes([terminal_error.T, Qf, terminal_error])

        # Control bounds
        u_lb, u_ub = self.robot.get_control_bounds()

        # Stack decision variables
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        constraints = ca.vertcat(*constraints)

        # Variable bounds
        n_vars = opt_vars.shape[0]
        lbx = np.full(n_vars, -np.inf)
        ubx = np.full(n_vars, np.inf)

        # Control bounds
        for k in range(N):
            idx_u = nx * (N + 1) + k * nu
            lbx[idx_u : idx_u + nu] = u_lb
            ubx[idx_u : idx_u + nu] = u_ub

        # Create NLP
        nlp = {
            "x": opt_vars,
            "f": cost,
            "g": constraints,
            "p": P,
        }

        # Solver options
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 100,
            "ipopt.tol": 1e-4,
            "print_time": 0,
        }

        self.solver = ca.nlpsol("mpc_solver", "ipopt", nlp, opts)
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = np.array(lb_constraints)
        self.ubg = np.array(ub_constraints)

        # Store dimensions
        self.nx = nx
        self.nu = nu
        self.N = N

        # Warm start
        self.prev_solution = None

    def compute_control(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        Compute optimal control input.

        Args:
            current_state: Current robot state [x, y, theta]
            reference_trajectory: Reference trajectory, shape (N+1, 3)

        Returns:
            Tuple of:
                - Optimal control [v, omega]
                - Info dict with predicted trajectory, solve time, etc.
        """
        # Build parameter vector
        p = np.concatenate([current_state, reference_trajectory.flatten()])

        # Initial guess (warm start)
        if self.prev_solution is not None:
            x0 = self.prev_solution
        else:
            # Initialize with straight-line trajectory
            x0 = np.zeros(self.nx * (self.N + 1) + self.nu * self.N)
            for k in range(self.N + 1):
                x0[k * self.nx : (k + 1) * self.nx] = current_state

        # Solve NLP
        solve_start = time.perf_counter()
        solution = self.solver(
            x0=x0,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            p=p,
        )
        solve_time = time.perf_counter() - solve_start

        # Extract solution
        opt_vars = np.array(solution["x"]).flatten()
        self.prev_solution = opt_vars

        # Extract predicted states and controls
        X_opt = opt_vars[: self.nx * (self.N + 1)].reshape(self.N + 1, self.nx)
        U_opt = opt_vars[self.nx * (self.N + 1) :].reshape(self.N, self.nu)

        # Return first control input
        u_opt = U_opt[0]
        cost = float(solution["f"])
        solver_status = self.solver.stats()["return_status"]

        # Log MPC iteration metrics
        self._iteration_count += 1
        logger.info(
            f"MPC iteration {self._iteration_count}: "
            f"solve_time={solve_time*1000:.2f}ms, "
            f"cost={cost:.4f}, "
            f"status={solver_status}"
        )

        info = {
            "predicted_trajectory": X_opt,
            "predicted_controls": U_opt,
            "cost": cost,
            "solve_time": solve_time,
            "solver_status": solver_status,
        }

        return u_opt, info

    def reset(self) -> None:
        """Reset warm start and iteration count."""
        self.prev_solution = None
        self._iteration_count = 0
        logger.debug("MPC controller reset")
