"""Model Predictive Controller for path tracking."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import casadi as ca
import numpy as np

from mpc_controller.models.differential_drive import DifferentialDriveModel, RobotParams
from mpc_controller.models.soft_constraints import (
    SoftConstraintManager,
    SoftConstraintParams,
    SoftConstraintResult,
    VelocitySoftConstraint,
    AccelerationSoftConstraint,
)

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

    # Soft constraint parameters
    soft_constraints: SoftConstraintParams | None = None

    # Acceleration limits (for soft constraints)
    a_max: float = 1.0      # Max linear acceleration [m/s^2]
    alpha_max: float = 2.0  # Max angular acceleration [rad/s^2]

    def __post_init__(self):
        if self.Q is None:
            self.Q = np.diag([10.0, 10.0, 1.0])
        if self.R is None:
            self.R = np.diag([0.1, 0.1])
        if self.Qf is None:
            self.Qf = np.diag([100.0, 100.0, 10.0])
        if self.Rd is None:
            self.Rd = np.diag([0.5, 0.5])
        if self.soft_constraints is None:
            self.soft_constraints = SoftConstraintParams()


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
        enable_soft_constraints: bool = True,
    ):
        self.robot = DifferentialDriveModel(robot_params or RobotParams())
        self.params = mpc_params or MPCParams()
        self._iteration_count = 0
        self._enable_soft_constraints = enable_soft_constraints
        self._soft_constraint_result: Optional[SoftConstraintResult] = None
        self._setup_logging(log_file)
        self._setup_soft_constraints()
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

    def _setup_soft_constraints(self) -> None:
        """Setup soft constraint manager."""
        if not self._enable_soft_constraints:
            self.soft_constraint_manager = None
            return

        soft_params = self.params.soft_constraints
        self.soft_constraint_manager = SoftConstraintManager(soft_params)

        # Get control bounds for soft constraints
        u_lb, u_ub = self.robot.get_control_bounds()
        v_max = u_ub[0]
        omega_max = u_ub[1]

        # Add velocity soft constraints
        self.soft_constraint_manager.add_velocity_constraint(
            v_max=v_max,
            omega_max=omega_max,
        )

        # Add acceleration soft constraints
        self.soft_constraint_manager.add_acceleration_constraint(
            a_max=self.params.a_max,
            alpha_max=self.params.alpha_max,
            dt=self.params.dt,
        )

        logger.info(
            f"Soft constraints enabled: "
            f"velocity={soft_params.enable_velocity_soft}, "
            f"acceleration={soft_params.enable_acceleration_soft}"
        )

    def _setup_optimizer(self) -> None:
        """Setup the NLP optimization problem."""
        N = self.params.N
        nx = DifferentialDriveModel.STATE_DIM
        nu = DifferentialDriveModel.CONTROL_DIM

        # Decision variables
        X = ca.SX.sym("X", nx, N + 1)  # States over horizon
        U = ca.SX.sym("U", nu, N)  # Controls over horizon

        # Slack variables for soft constraints
        use_soft = (
            self._enable_soft_constraints
            and self.soft_constraint_manager is not None
            and self.soft_constraint_manager.enabled
        )

        if use_soft:
            # Velocity slack: [slack_v, slack_omega] for each timestep
            S_vel = ca.SX.sym("S_vel", 2, N)
            # Acceleration slack: [slack_a, slack_alpha] for k > 0
            S_acc = ca.SX.sym("S_acc", 2, N - 1)
            self.n_slack = 2 * N + 2 * (N - 1)
        else:
            S_vel = None
            S_acc = None
            self.n_slack = 0

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

        # Control bounds
        u_lb, u_ub = self.robot.get_control_bounds()
        v_max = u_ub[0]
        omega_max = u_ub[1]

        # Get soft constraint weights
        if use_soft:
            soft_params = self.params.soft_constraints
            vel_weight = soft_params.velocity_weight
            acc_weight = soft_params.acceleration_weight

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

            # Soft velocity constraints
            if use_soft and soft_params.enable_velocity_soft:
                v_k = U[0, k]
                omega_k = U[1, k]
                slack_v = S_vel[0, k]
                slack_omega = S_vel[1, k]

                # |v| <= v_max + slack_v
                constraints.append(v_k - v_max - slack_v)
                constraints.append(-v_k - v_max - slack_v)
                lb_constraints.extend([-ca.inf, -ca.inf])
                ub_constraints.extend([0.0, 0.0])

                # |omega| <= omega_max + slack_omega
                constraints.append(omega_k - omega_max - slack_omega)
                constraints.append(-omega_k - omega_max - slack_omega)
                lb_constraints.extend([-ca.inf, -ca.inf])
                ub_constraints.extend([0.0, 0.0])

                # Add penalty to cost
                cost += vel_weight * (slack_v ** 2 + slack_omega ** 2)

            # Soft acceleration constraints (for k > 0)
            if use_soft and soft_params.enable_acceleration_soft and k > 0:
                v_curr = U[0, k]
                v_prev = U[0, k - 1]
                omega_curr = U[1, k]
                omega_prev = U[1, k - 1]
                dt = self.params.dt

                a = (v_curr - v_prev) / dt
                alpha = (omega_curr - omega_prev) / dt

                slack_a = S_acc[0, k - 1]
                slack_alpha = S_acc[1, k - 1]

                a_max = self.params.a_max
                alpha_max = self.params.alpha_max

                # |a| <= a_max + slack_a
                constraints.append(a - a_max - slack_a)
                constraints.append(-a - a_max - slack_a)
                lb_constraints.extend([-ca.inf, -ca.inf])
                ub_constraints.extend([0.0, 0.0])

                # |alpha| <= alpha_max + slack_alpha
                constraints.append(alpha - alpha_max - slack_alpha)
                constraints.append(-alpha - alpha_max - slack_alpha)
                lb_constraints.extend([-ca.inf, -ca.inf])
                ub_constraints.extend([0.0, 0.0])

                # Add penalty to cost
                cost += acc_weight * (slack_a ** 2 + slack_alpha ** 2)

        # Terminal cost
        ref_N = P[nx + N * nx : nx + (N + 1) * nx]
        terminal_error = X[:, N] - ref_N
        terminal_error[2] = ca.atan2(ca.sin(terminal_error[2]), ca.cos(terminal_error[2]))
        cost += ca.mtimes([terminal_error.T, Qf, terminal_error])

        # Stack decision variables
        if use_soft:
            opt_vars = ca.vertcat(
                ca.reshape(X, -1, 1),
                ca.reshape(U, -1, 1),
                ca.reshape(S_vel, -1, 1),
                ca.reshape(S_acc, -1, 1),
            )
        else:
            opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        constraints = ca.vertcat(*constraints)

        # Variable bounds
        n_vars = opt_vars.shape[0]
        lbx = np.full(n_vars, -np.inf)
        ubx = np.full(n_vars, np.inf)

        # Control bounds (hard constraints remain)
        for k in range(N):
            idx_u = nx * (N + 1) + k * nu
            # Use slightly relaxed bounds when soft constraints are enabled
            if use_soft:
                # Allow some slack in hard bounds (the soft constraints will penalize)
                lbx[idx_u : idx_u + nu] = u_lb * 1.5
                ubx[idx_u : idx_u + nu] = u_ub * 1.5
            else:
                lbx[idx_u : idx_u + nu] = u_lb
                ubx[idx_u : idx_u + nu] = u_ub

        # Slack variable bounds: s >= 0
        if use_soft:
            idx_slack_start = nx * (N + 1) + nu * N
            lbx[idx_slack_start:] = 0.0  # s >= 0
            ubx[idx_slack_start:] = np.inf  # s <= inf

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
        self.use_soft = use_soft

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
        state_for_mpc = current_state.copy()

        # Adjust reference trajectory theta to be continuous with current state
        # This prevents issues when crossing the ±π boundary
        ref_adjusted = reference_trajectory.copy()
        current_theta = state_for_mpc[2]

        for i in range(len(ref_adjusted)):
            # Compute shortest angular difference from current theta to reference
            ref_theta = ref_adjusted[i, 2]
            diff = np.arctan2(np.sin(ref_theta - current_theta), np.cos(ref_theta - current_theta))
            ref_adjusted[i, 2] = current_theta + diff
            # Use this adjusted theta as the new "current" for continuity
            current_theta = ref_adjusted[i, 2]

        # Build parameter vector
        p = np.concatenate([state_for_mpc, ref_adjusted.flatten()])

        # Calculate total decision variable size
        n_state_vars = self.nx * (self.N + 1)
        n_control_vars = self.nu * self.N
        n_total = n_state_vars + n_control_vars + self.n_slack

        # Initial guess (warm start)
        if self.prev_solution is not None and len(self.prev_solution) == n_total:
            x0 = self.prev_solution
        else:
            # Initialize with straight-line trajectory
            x0 = np.zeros(n_total)
            for k in range(self.N + 1):
                x0[k * self.nx : (k + 1) * self.nx] = current_state
            # Initialize slack variables to zero
            if self.n_slack > 0:
                x0[n_state_vars + n_control_vars:] = 0.0

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
        X_opt = opt_vars[:n_state_vars].reshape(self.N + 1, self.nx)
        U_opt = opt_vars[n_state_vars:n_state_vars + n_control_vars].reshape(self.N, self.nu)

        # Extract and process soft constraint results
        soft_constraint_info = {}
        if self.use_soft and self.n_slack > 0:
            slack_values = opt_vars[n_state_vars + n_control_vars:]

            # Parse slack values
            n_vel_slack = 2 * self.N
            n_acc_slack = 2 * (self.N - 1)

            vel_slacks = slack_values[:n_vel_slack].reshape(2, self.N)
            acc_slacks = slack_values[n_vel_slack:n_vel_slack + n_acc_slack].reshape(2, self.N - 1)

            # Calculate total violation
            total_vel_violation = np.sum(vel_slacks)
            total_acc_violation = np.sum(acc_slacks)
            max_vel_violation = np.max(vel_slacks) if vel_slacks.size > 0 else 0.0
            max_acc_violation = np.max(acc_slacks) if acc_slacks.size > 0 else 0.0

            soft_constraint_info = {
                "velocity_slack": vel_slacks,
                "acceleration_slack": acc_slacks,
                "total_velocity_violation": total_vel_violation,
                "total_acceleration_violation": total_acc_violation,
                "max_velocity_violation": max_vel_violation,
                "max_acceleration_violation": max_acc_violation,
                "has_violations": (total_vel_violation + total_acc_violation) > 1e-6,
            }

            # Store for visualization
            self._soft_constraint_result = soft_constraint_info

            # Log violations if any
            if soft_constraint_info["has_violations"]:
                logger.warning(
                    f"Soft constraint violations: "
                    f"vel_max={max_vel_violation:.4f}, "
                    f"acc_max={max_acc_violation:.4f}"
                )

        # Return first control input
        u_opt = U_opt[0]
        cost = float(solution["f"])
        solver_status = self.solver.stats()["return_status"]

        # Log MPC iteration metrics
        self._iteration_count += 1
        log_msg = (
            f"MPC iteration {self._iteration_count}: "
            f"solve_time={solve_time*1000:.2f}ms, "
            f"cost={cost:.4f}, "
            f"status={solver_status}"
        )
        if self.use_soft and soft_constraint_info.get("has_violations", False):
            log_msg += f", violations=True"
        logger.info(log_msg)

        info = {
            "predicted_trajectory": X_opt,
            "predicted_controls": U_opt,
            "cost": cost,
            "solve_time": solve_time,
            "solver_status": solver_status,
            "soft_constraints": soft_constraint_info,
        }

        return u_opt, info

    def get_soft_constraint_result(self) -> Optional[dict]:
        """
        Get the latest soft constraint result for visualization.

        Returns:
            Dictionary with soft constraint violation info, or None if not available
        """
        return self._soft_constraint_result

    def reset(self) -> None:
        """Reset warm start and iteration count."""
        self.prev_solution = None
        self._iteration_count = 0
        logger.debug("MPC controller reset")
