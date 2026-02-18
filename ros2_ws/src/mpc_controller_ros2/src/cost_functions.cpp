#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

// StateTrackingCost
StateTrackingCost::StateTrackingCost(const Eigen::Matrix3d& Q) : Q_(Q) {}

Eigen::VectorXd StateTrackingCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  int K = trajectories.size();
  int N = reference.rows() - 1;
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  for (int k = 0; k < K; ++k) {
    for (int t = 0; t < N; ++t) {
      Eigen::Vector3d state = trajectories[k].row(t).transpose();
      Eigen::Vector3d ref = reference.row(t).transpose();
      Eigen::Vector3d error = state - ref;

      // Normalize angle error
      error(2) = normalizeAngle(error(2));

      costs(k) += error.transpose() * Q_ * error;
    }
  }

  return costs;
}

// TerminalCost
TerminalCost::TerminalCost(const Eigen::Matrix3d& Qf) : Qf_(Qf) {}

Eigen::VectorXd TerminalCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  int K = trajectories.size();
  int N = reference.rows() - 1;
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  for (int k = 0; k < K; ++k) {
    Eigen::Vector3d terminal_state = trajectories[k].row(N).transpose();
    Eigen::Vector3d terminal_ref = reference.row(N).transpose();
    Eigen::Vector3d error = terminal_state - terminal_ref;

    error(2) = normalizeAngle(error(2));

    costs(k) = error.transpose() * Qf_ * error;
  }

  return costs;
}

// ControlEffortCost
ControlEffortCost::ControlEffortCost(const Eigen::Matrix2d& R) : R_(R) {}

Eigen::VectorXd ControlEffortCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  int K = controls.size();
  int N = controls[0].rows();
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  for (int k = 0; k < K; ++k) {
    for (int t = 0; t < N; ++t) {
      Eigen::Vector2d u = controls[k].row(t).transpose();
      costs(k) += u.transpose() * R_ * u;
    }
  }

  return costs;
}

// ControlRateCost
ControlRateCost::ControlRateCost(const Eigen::Matrix2d& R_rate) : R_rate_(R_rate) {}

Eigen::VectorXd ControlRateCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  int K = controls.size();
  int N = controls[0].rows();
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  for (int k = 0; k < K; ++k) {
    for (int t = 0; t < N - 1; ++t) {
      Eigen::Vector2d u_curr = controls[k].row(t).transpose();
      Eigen::Vector2d u_next = controls[k].row(t + 1).transpose();
      Eigen::Vector2d du = u_next - u_curr;
      costs(k) += du.transpose() * R_rate_ * du;
    }
  }

  return costs;
}

// PreferForwardCost
PreferForwardCost::PreferForwardCost(double weight) : weight_(weight) {}

Eigen::VectorXd PreferForwardCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  (void)trajectories;
  (void)reference;

  int K = controls.size();
  int N = controls[0].rows();
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  for (int k = 0; k < K; ++k) {
    for (int t = 0; t < N; ++t) {
      double v = controls[k](t, 0);
      if (v < 0.0) {
        // 후진 시 이차 페널티: weight * v²
        costs(k) += weight_ * v * v;
      }
    }
  }

  return costs;
}

// ObstacleCost
ObstacleCost::ObstacleCost(double weight, double safety_distance)
: weight_(weight), safety_distance_(safety_distance)
{
}

void ObstacleCost::setObstacles(const std::vector<Eigen::Vector3d>& obstacles)
{
  obstacles_ = obstacles;
}

Eigen::VectorXd ObstacleCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  int K = trajectories.size();
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  if (obstacles_.empty()) {
    return costs;
  }

  for (int k = 0; k < K; ++k) {
    const auto& traj = trajectories[k];
    int N = traj.rows() - 1;

    for (int t = 0; t <= N; ++t) {
      Eigen::Vector2d pos = traj.row(t).head<2>().transpose();

      for (const auto& obs : obstacles_) {
        Eigen::Vector2d obs_pos = obs.head<2>();
        double dist = (pos - obs_pos).norm();
        double penetration = safety_distance_ - dist;

        if (penetration > 0) {
          costs(k) += weight_ * penetration * penetration;
        }
      }
    }
  }

  return costs;
}

// CompositeMPPICost
void CompositeMPPICost::addCost(std::unique_ptr<MPPICostFunction> cost)
{
  costs_.push_back(std::move(cost));
}

Eigen::VectorXd CompositeMPPICost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& reference
) const
{
  if (costs_.empty()) {
    return Eigen::VectorXd::Zero(trajectories.size());
  }

  Eigen::VectorXd total_costs = costs_[0]->compute(trajectories, controls, reference);

  for (size_t i = 1; i < costs_.size(); ++i) {
    total_costs += costs_[i]->compute(trajectories, controls, reference);
  }

  return total_costs;
}

}  // namespace mpc_controller_ros2
