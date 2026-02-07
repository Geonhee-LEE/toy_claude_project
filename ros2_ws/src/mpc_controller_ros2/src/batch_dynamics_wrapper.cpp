#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

BatchDynamicsWrapper::BatchDynamicsWrapper(const MPPIParams& params)
: params_(params)
{
}

Eigen::MatrixXd BatchDynamicsWrapper::dynamicsBatch(
  const Eigen::MatrixXd& states,
  const Eigen::MatrixXd& controls
) const
{
  // states: M x 3 [x, y, theta]
  // controls: M x 2 [v, omega]
  // return: M x 3 [x_dot, y_dot, theta_dot]

  int M = states.rows();
  Eigen::MatrixXd state_dot(M, 3);

  // Differential drive kinematics
  // x_dot = v * cos(theta)
  // y_dot = v * sin(theta)
  // theta_dot = omega

  Eigen::VectorXd theta = states.col(2);
  Eigen::VectorXd v = controls.col(0);
  Eigen::VectorXd omega = controls.col(1);

  state_dot.col(0) = v.array() * theta.array().cos();  // x_dot
  state_dot.col(1) = v.array() * theta.array().sin();  // y_dot
  state_dot.col(2) = omega;                             // theta_dot

  return state_dot;
}

Eigen::MatrixXd BatchDynamicsWrapper::propagateBatch(
  const Eigen::MatrixXd& states,
  const Eigen::MatrixXd& controls,
  double dt
) const
{
  // RK4 integration
  Eigen::MatrixXd k1 = dynamicsBatch(states, controls);
  Eigen::MatrixXd k2 = dynamicsBatch(states + dt / 2.0 * k1, controls);
  Eigen::MatrixXd k3 = dynamicsBatch(states + dt / 2.0 * k2, controls);
  Eigen::MatrixXd k4 = dynamicsBatch(states + dt * k3, controls);

  Eigen::MatrixXd states_next = states + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

  // Normalize angles
  states_next.col(2) = normalizeAngleBatch(states_next.col(2));

  return states_next;
}

std::vector<Eigen::MatrixXd> BatchDynamicsWrapper::rolloutBatch(
  const Eigen::Vector3d& x0,
  const std::vector<Eigen::MatrixXd>& control_sequences,
  double dt
) const
{
  int K = control_sequences.size();
  int N = control_sequences[0].rows();

  std::vector<Eigen::MatrixXd> trajectories;
  trajectories.reserve(K);

  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd traj(N + 1, 3);
    traj.row(0) = x0.transpose();

    for (int t = 0; t < N; ++t) {
      Eigen::MatrixXd state = traj.row(t);
      Eigen::MatrixXd control = control_sequences[k].row(t);
      Eigen::MatrixXd next_state = propagateBatch(state, control, dt);
      traj.row(t + 1) = next_state;
    }

    trajectories.push_back(traj);
  }

  return trajectories;
}

Eigen::MatrixXd BatchDynamicsWrapper::clipControls(const Eigen::MatrixXd& controls) const
{
  Eigen::MatrixXd clipped = controls;

  // Clip v
  clipped.col(0) = clipped.col(0).cwiseMax(params_.v_min).cwiseMin(params_.v_max);

  // Clip omega
  clipped.col(1) = clipped.col(1).cwiseMax(params_.omega_min).cwiseMin(params_.omega_max);

  return clipped;
}

}  // namespace mpc_controller_ros2
