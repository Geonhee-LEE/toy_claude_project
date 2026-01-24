"""Model Predictive Controller for non-coaxial swerve drive path tracking."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import casadi as ca
import numpy as np

from mpc_controller.models.non_coaxial_swerve import (
    NonCoaxialSwerveDriveModel,
    NonCoaxialSwerveParams,
)

# Module-level logger
logger = logging.getLogger(__name__)


@dataclass
class NonCoaxialSwerveMPCParams:
    """MPC tuning parameters for non-coaxial swerve drive."""

    # Horizon
    N: int = 20  # Prediction horizon
    dt: float = 0.1  # Time step [s]

    # State weights [x, y, theta, delta]
    Q: np.ndarray | None = None

    # Control weights [v, omega, delta_dot]
    R: np.ndarray | None = None

    # Terminal state weights
    Qf: np.ndarray | None = None

    # Control rate weights (smoothness)
    Rd: np.ndarray | None = None

    # Steering target weight (penalize deviation from desired steering)
    Q_steering: float = 1.0

    def __post_init__(self):
        if self.Q is None:
            # [x, y, theta, delta] - lower weight on delta to allow flexibility
            self.Q = np.diag([10.0, 10.0, 1.0, 0.1])
        if self.R is None:
            # [v, omega, delta_dot]
            self.R = np.diag([0.1, 0.1, 0.5])  # Higher weight on steering rate
        if self.Qf is None:
            self.Qf = np.diag([100.0, 100.0, 10.0, 0.1])
        if self.Rd is None:
            self.Rd = np.diag([0.5, 0.5, 1.0])  # Smooth steering changes


class NonCoaxialSwerveMPCController:
    """
    Nonlinear MPC controller for non-coaxial swerve drive path tracking.

    스티어링 각도 제한 (±90°)이 있는 swerve drive를 위한 MPC 컨트롤러.
    스티어링 각도가 상태에 포함되어 연속적인 스티어링 제어가 가능.

    ┌─────────────────────────────────────────────────────────────┐
    │                    Control Flow                             │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  Reference    ┌─────────┐    Optimal    ┌─────────┐        │
    │  Trajectory ──┤   MPC   ├──► Control ──►│  Robot  │        │
    │  [x,y,θ]      └────┬────┘   [v,ω,δ̇]     └────┬────┘        │
    │                    │                          │             │
    │                    └──── State Feedback ◄─────┘             │
    │                         [x,y,θ,δ]                           │
    │                                                             │
    │  Constraints:                                               │
    │    - |δ| ≤ 90°  (steering angle limit)                     │
    │    - |δ̇| ≤ max_rate  (steering rate limit)                 │
    │    - |v| ≤ max_speed                                        │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        robot_params: NonCoaxialSwerveParams | None = None,
        mpc_params: NonCoaxialSwerveMPCParams | None = None,
        log_file: str | Path | None = None,
    ):
        self.robot = NonCoaxialSwerveDriveModel(robot_params or NonCoaxialSwerveParams())
        self.params = mpc_params or NonCoaxialSwerveMPCParams()
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
            logger.info(f"Non-coaxial Swerve MPC logging to file: {log_file}")

    def _setup_optimizer(self) -> None:
        """Setup the NLP optimization problem."""
        N = self.params.N
        nx = NonCoaxialSwerveDriveModel.STATE_DIM  # 4
        nu = NonCoaxialSwerveDriveModel.CONTROL_DIM  # 3

        # Decision variables
        X = ca.SX.sym("X", nx, N + 1)  # States over horizon
        U = ca.SX.sym("U", nu, N)  # Controls over horizon

        # Parameters: initial state + reference trajectory (only x, y, theta)
        # Reference is 3D (x, y, theta), not 4D
        n_ref = 3
        P = ca.SX.sym("P", nx + n_ref * (N + 1))

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

        # State bounds for steering angle
        max_delta = self.robot.params.max_steering_angle

        for k in range(N):
            # Reference state at step k (only x, y, theta)
            ref_k = P[nx + k * n_ref : nx + (k + 1) * n_ref]

            # State error (for x, y, theta)
            state_error = ca.vertcat(
                X[0, k] - ref_k[0],  # x error
                X[1, k] - ref_k[1],  # y error
                ca.atan2(ca.sin(X[2, k] - ref_k[2]), ca.cos(X[2, k] - ref_k[2])),  # theta error
                X[3, k],  # delta (penalize deviation from 0 or desired)
            )

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

            # Steering angle constraint (state constraint)
            constraints.append(X[3, k])
            lb_constraints.append(-max_delta)
            ub_constraints.append(max_delta)

        # Terminal steering constraint
        constraints.append(X[3, N])
        lb_constraints.append(-max_delta)
        ub_constraints.append(max_delta)

        # Terminal cost
        ref_N = P[nx + N * n_ref : nx + (N + 1) * n_ref]
        terminal_error = ca.vertcat(
            X[0, N] - ref_N[0],
            X[1, N] - ref_N[1],
            ca.atan2(ca.sin(X[2, N] - ref_N[2]), ca.cos(X[2, N] - ref_N[2])),
            X[3, N],
        )
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

        # State bounds (steering angle)
        for k in range(N + 1):
            idx_delta = k * nx + 3
            lbx[idx_delta] = -max_delta
            ubx[idx_delta] = max_delta

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

        self.solver = ca.nlpsol("non_coaxial_swerve_mpc_solver", "ipopt", nlp, opts)
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = np.array(lb_constraints)
        self.ubg = np.array(ub_constraints)

        # Store dimensions
        self.nx = nx
        self.nu = nu
        self.n_ref = n_ref
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
            current_state: Current robot state [x, y, theta, delta]
            reference_trajectory: Reference trajectory, shape (N+1, 3) for [x, y, theta]

        Returns:
            Tuple of:
                - Optimal control [v, omega, delta_dot]
                - Info dict with predicted trajectory, solve time, etc.
        """
        # Ensure reference is 3D (x, y, theta)
        if reference_trajectory.shape[1] == 4:
            reference_trajectory = reference_trajectory[:, :3]

        state_for_mpc = current_state.copy()

        # Adjust reference trajectory theta to be continuous with current state
        # This prevents issues when crossing the ±π boundary
        ref_adjusted = reference_trajectory.copy()
        current_theta = state_for_mpc[2]  # theta is at index 2

        for i in range(len(ref_adjusted)):
            ref_theta = ref_adjusted[i, 2]
            diff = np.arctan2(np.sin(ref_theta - current_theta), np.cos(ref_theta - current_theta))
            ref_adjusted[i, 2] = current_theta + diff
            current_theta = ref_adjusted[i, 2]

        # Build parameter vector
        p = np.concatenate([state_for_mpc, ref_adjusted.flatten()])

        # Initial guess (warm start)
        if self.prev_solution is not None:
            x0 = self.prev_solution
        else:
            # Initialize with current state
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
            f"Non-coaxial Swerve MPC iteration {self._iteration_count}: "
            f"solve_time={solve_time*1000:.2f}ms, "
            f"cost={cost:.4f}, "
            f"delta={np.degrees(current_state[3]):.1f}°, "
            f"status={solver_status}"
        )

        info = {
            "predicted_trajectory": X_opt,
            "predicted_controls": U_opt,
            "cost": cost,
            "solve_time": solve_time,
            "solver_status": solver_status,
            "steering_angle": current_state[3],
        }

        return u_opt, info

    def reset(self) -> None:
        """Reset warm start and iteration count."""
        self.prev_solution = None
        self._iteration_count = 0
        logger.debug("Non-coaxial Swerve MPC controller reset")
