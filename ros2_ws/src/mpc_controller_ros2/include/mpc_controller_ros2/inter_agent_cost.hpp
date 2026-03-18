#ifndef MPC_CONTROLLER_ROS2__INTER_AGENT_COST_HPP_
#define MPC_CONTROLLER_ROS2__INTER_AGENT_COST_HPP_

#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/agent_trajectory_manager.hpp"

namespace mpc_controller_ros2
{

class InterAgentCost : public MPPICostFunction {
public:
  InterAgentCost(AgentTrajectoryManager* manager,
                 double weight, double safety_distance,
                 double robot_radius);
  std::string name() const override { return "inter_agent"; }
  Eigen::VectorXd compute(
    const std::vector<Eigen::MatrixXd>& trajectories,
    const std::vector<Eigen::MatrixXd>& controls,
    const Eigen::MatrixXd& reference) const override;
private:
  AgentTrajectoryManager* manager_;
  double weight_;
  double safety_distance_;
  double robot_radius_;
};

}  // namespace mpc_controller_ros2
#endif
