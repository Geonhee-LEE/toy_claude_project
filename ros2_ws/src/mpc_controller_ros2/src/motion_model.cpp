#include "mpc_controller_ros2/motion_model.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

Eigen::MatrixXd MotionModel::propagateBatch(
  const Eigen::MatrixXd& states,
  const Eigen::MatrixXd& controls,
  double dt) const
{
  // RK4 integration
  Eigen::MatrixXd k1 = dynamicsBatch(states, controls);
  Eigen::MatrixXd k2 = dynamicsBatch(states + dt / 2.0 * k1, controls);
  Eigen::MatrixXd k3 = dynamicsBatch(states + dt / 2.0 * k2, controls);
  Eigen::MatrixXd k4 = dynamicsBatch(states + dt * k3, controls);

  Eigen::MatrixXd states_next = states + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

  // Normalize states (angle wrapping etc.)
  normalizeStates(states_next);

  return states_next;
}

std::vector<Eigen::MatrixXd> MotionModel::rolloutBatch(
  const Eigen::VectorXd& x0,
  const std::vector<Eigen::MatrixXd>& control_sequences,
  double dt) const
{
  int K = control_sequences.size();
  int N = control_sequences[0].rows();
  int nx = stateDim();

  std::vector<Eigen::MatrixXd> trajectories;
  trajectories.reserve(K);

  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd traj(N + 1, nx);
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

}  // namespace mpc_controller_ros2
