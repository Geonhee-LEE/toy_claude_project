#ifndef MPC_CONTROLLER_ROS2__MULTI_AGENT_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__MULTI_AGENT_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/agent_trajectory_manager.hpp"
#include "mpc_controller_ros2/inter_agent_cost.hpp"
#include <nav_msgs/msg/path.hpp>

namespace mpc_controller_ros2
{

class MultiAgentMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  MultiAgentMPPIControllerPlugin() = default;
  ~MultiAgentMPPIControllerPlugin() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros
  ) override;

protected:
  std::pair<Eigen::VectorXd, MPPIInfo> computeControl(
    const Eigen::VectorXd& current_state,
    const Eigen::MatrixXd& reference_trajectory
  ) override;

private:
  std::unique_ptr<AgentTrajectoryManager> agent_manager_;

  // Multi-agent parameters (declared locally, not in MPPIParams)
  bool multi_agent_enabled_{false};
  int multi_agent_id_{0};
  int multi_agent_count_{3};
  double inter_agent_cost_weight_{500.0};
  double inter_agent_safety_dist_{0.3};
  double inter_agent_robot_radius_{0.2};
  double inter_agent_timeout_{2.0};
  std::string multi_agent_topic_prefix_{"/agent"};

  // ROS2 pub/sub for trajectory sharing
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr traj_pub_;
  std::vector<rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr> traj_subs_;

  void onAgentTrajectory(const nav_msgs::msg::Path::SharedPtr msg);
  void publishOwnTrajectory(const MPPIInfo& info);
};

}  // namespace mpc_controller_ros2
#endif
