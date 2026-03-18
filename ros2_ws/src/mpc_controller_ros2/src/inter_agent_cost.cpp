#include "mpc_controller_ros2/inter_agent_cost.hpp"
#include <algorithm>
#include <cmath>

namespace mpc_controller_ros2
{

InterAgentCost::InterAgentCost(
  AgentTrajectoryManager* manager,
  double weight, double safety_distance, double robot_radius)
  : manager_(manager),
    weight_(weight),
    safety_distance_(safety_distance),
    robot_radius_(robot_radius)
{
}

Eigen::VectorXd InterAgentCost::compute(
  const std::vector<Eigen::MatrixXd>& trajectories,
  const std::vector<Eigen::MatrixXd>& controls,
  const Eigen::MatrixXd& /*reference*/) const
{
  int K = static_cast<int>(trajectories.size());
  Eigen::VectorXd costs = Eigen::VectorXd::Zero(K);

  if (!manager_) return costs;

  auto others = manager_->getOtherAgents();
  if (others.empty()) return costs;

  for (int k = 0; k < K; ++k) {
    int N = static_cast<int>(trajectories[k].rows());
    for (int t = 0; t < N; ++t) {
      Eigen::Vector2d pos_k = trajectories[k].row(t).head(2);

      for (const auto& agent : others) {
        int t_idx = std::min(t, static_cast<int>(agent.trajectory.rows()) - 1);
        if (t_idx < 0) continue;
        Eigen::Vector2d pos_agent = agent.trajectory.row(t_idx).head(2);

        double dist = (pos_k - pos_agent).norm();
        double d_safe = robot_radius_ + agent.radius + safety_distance_;

        if (dist < d_safe) {
          double violation = d_safe - dist;
          costs(k) += weight_ * violation * violation;
        }
      }
    }
  }

  return costs;
}

}  // namespace mpc_controller_ros2
