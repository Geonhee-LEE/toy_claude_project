#include "mpc_controller_ros2/agent_trajectory_manager.hpp"

namespace mpc_controller_ros2
{

AgentTrajectoryManager::AgentTrajectoryManager(int own_id, double timeout)
  : own_id_(own_id), timeout_(timeout)
{
}

void AgentTrajectoryManager::updateOwnTrajectory(
  const Eigen::MatrixXd& trajectory,
  const Eigen::Vector2d& velocity, double dt)
{
  std::lock_guard<std::mutex> lock(mutex_);
  AgentPrediction pred;
  pred.agent_id = own_id_;
  pred.trajectory = trajectory;
  pred.velocity = velocity;
  pred.dt = dt;
  pred.timestamp = 0.0;  // Own trajectory: always fresh
  predictions_[own_id_] = pred;
}

void AgentTrajectoryManager::updateAgentTrajectory(
  int agent_id, const AgentPrediction& pred)
{
  if (agent_id == own_id_) return;
  std::lock_guard<std::mutex> lock(mutex_);
  predictions_[agent_id] = pred;
}

std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector2d>>
AgentTrajectoryManager::toObstaclesWithVelocity(double /*current_time*/) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<Eigen::Vector3d> obstacles;
  std::vector<Eigen::Vector2d> velocities;

  for (const auto& [id, pred] : predictions_) {
    if (id == own_id_) continue;
    if (pred.trajectory.rows() < 1) continue;

    // Current position (row 0) as obstacle [x, y, radius]
    Eigen::Vector3d obs;
    obs(0) = pred.trajectory(0, 0);
    obs(1) = pred.trajectory(0, 1);
    obs(2) = pred.radius;
    obstacles.push_back(obs);
    velocities.push_back(pred.velocity);
  }

  return {obstacles, velocities};
}

std::vector<AgentPrediction> AgentTrajectoryManager::getOtherAgents() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<AgentPrediction> others;
  for (const auto& [id, pred] : predictions_) {
    if (id == own_id_) continue;
    others.push_back(pred);
  }
  return others;
}

void AgentTrajectoryManager::pruneStale(double current_time)
{
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = predictions_.begin(); it != predictions_.end(); ) {
    if (it->first != own_id_ &&
        (current_time - it->second.timestamp) > timeout_) {
      it = predictions_.erase(it);
    } else {
      ++it;
    }
  }
}

void AgentTrajectoryManager::reset()
{
  std::lock_guard<std::mutex> lock(mutex_);
  predictions_.clear();
}

size_t AgentTrajectoryManager::numAgents() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  size_t count = 0;
  for (const auto& [id, pred] : predictions_) {
    if (id != own_id_) ++count;
  }
  return count;
}

}  // namespace mpc_controller_ros2
