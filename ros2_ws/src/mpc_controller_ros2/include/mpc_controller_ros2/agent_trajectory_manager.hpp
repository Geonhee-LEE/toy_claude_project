#ifndef MPC_CONTROLLER_ROS2__AGENT_TRAJECTORY_MANAGER_HPP_
#define MPC_CONTROLLER_ROS2__AGENT_TRAJECTORY_MANAGER_HPP_

#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace mpc_controller_ros2
{

struct AgentPrediction {
  int agent_id{-1};
  Eigen::MatrixXd trajectory;   // (N+1) x nx
  Eigen::Vector2d velocity{0.0, 0.0};  // 현재 속도 [vx, vy]
  double radius{0.2};
  double timestamp{0.0};
  double dt{0.1};
};

class AgentTrajectoryManager {
public:
  explicit AgentTrajectoryManager(int own_id, double timeout = 2.0);
  void updateOwnTrajectory(const Eigen::MatrixXd& trajectory,
                            const Eigen::Vector2d& velocity, double dt);
  void updateAgentTrajectory(int agent_id, const AgentPrediction& pred);

  // Convert other agents to obstacles with velocity (for BarrierFunctionSet)
  std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector2d>>
  toObstaclesWithVelocity(double current_time) const;

  std::vector<AgentPrediction> getOtherAgents() const;
  void pruneStale(double current_time);
  void reset();

  int ownId() const { return own_id_; }
  size_t numAgents() const;

private:
  int own_id_;
  double timeout_;
  std::unordered_map<int, AgentPrediction> predictions_;
  mutable std::mutex mutex_;
};

}  // namespace mpc_controller_ros2
#endif
