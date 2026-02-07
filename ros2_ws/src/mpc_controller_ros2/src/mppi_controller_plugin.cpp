#include "mpc_controller_ros2/mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::MPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void MPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros
)
{
  node_ = parent.lock();
  plugin_name_ = name;
  tf_buffer_ = tf;
  costmap_ros_ = costmap_ros;

  RCLCPP_INFO(node_->get_logger(), "Configuring MPPI controller: %s", plugin_name_.c_str());

  // Declare and load parameters
  declareParameters();
  loadParameters();

  // Initialize components
  dynamics_ = std::make_unique<BatchDynamicsWrapper>(params_);
  sampler_ = std::make_unique<GaussianSampler>(params_.noise_sigma);

  // Initialize cost function
  cost_function_ = std::make_unique<CompositeMPPICost>();
  cost_function_->addCost(std::make_unique<StateTrackingCost>(params_.Q));
  cost_function_->addCost(std::make_unique<TerminalCost>(params_.Qf));
  cost_function_->addCost(std::make_unique<ControlEffortCost>(params_.R));
  cost_function_->addCost(std::make_unique<ControlRateCost>(params_.R_rate));
  cost_function_->addCost(
    std::make_unique<ObstacleCost>(params_.obstacle_weight, params_.safety_distance)
  );

  // Initialize control sequence
  control_sequence_ = Eigen::MatrixXd::Zero(params_.N, 2);

  // Create marker publisher
  marker_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
    plugin_name_ + "/mppi_markers", 10
  );

  RCLCPP_INFO(node_->get_logger(), "MPPI controller configured successfully");
}

void MPPIControllerPlugin::cleanup()
{
  RCLCPP_INFO(node_->get_logger(), "Cleaning up MPPI controller");
  marker_pub_.reset();
}

void MPPIControllerPlugin::activate()
{
  RCLCPP_INFO(node_->get_logger(), "Activating MPPI controller");
  // Publisher activation handled by lifecycle node
}

void MPPIControllerPlugin::deactivate()
{
  RCLCPP_INFO(node_->get_logger(), "Deactivating MPPI controller");
  // Publisher deactivation handled by lifecycle node
}

geometry_msgs::msg::TwistStamped MPPIControllerPlugin::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped& pose,
  const geometry_msgs::msg::Twist& velocity,
  nav2_core::GoalChecker* goal_checker
)
{
  // Stub implementation
  geometry_msgs::msg::TwistStamped cmd_vel;
  cmd_vel.header.stamp = node_->now();
  cmd_vel.header.frame_id = pose.header.frame_id;
  cmd_vel.twist.linear.x = 0.0;
  cmd_vel.twist.angular.z = 0.0;

  RCLCPP_INFO_THROTTLE(
    node_->get_logger(), *node_->get_clock(), 1000,
    "MPPI controller stub: returning zero velocity"
  );

  return cmd_vel;
}

void MPPIControllerPlugin::setPlan(const nav_msgs::msg::Path& path)
{
  global_plan_ = path;
  RCLCPP_INFO(
    node_->get_logger(),
    "Received new plan with %zu poses", path.poses.size()
  );
}

void MPPIControllerPlugin::setSpeedLimit(const double& speed_limit, const bool& percentage)
{
  speed_limit_ = speed_limit;
  speed_limit_valid_ = true;
  RCLCPP_INFO(
    node_->get_logger(),
    "Speed limit set to %.2f (percentage: %s)",
    speed_limit, percentage ? "true" : "false"
  );
}

std::pair<Eigen::Vector2d, MPPIInfo> MPPIControllerPlugin::computeControl(
  const Eigen::Vector3d& current_state,
  const Eigen::MatrixXd& reference_trajectory
)
{
  // Stub implementation
  MPPIInfo info;
  Eigen::Vector2d control = Eigen::Vector2d::Zero();
  return {control, info};
}

Eigen::Vector3d MPPIControllerPlugin::poseToState(const geometry_msgs::msg::PoseStamped& pose)
{
  Eigen::Vector3d state;
  state(0) = pose.pose.position.x;
  state(1) = pose.pose.position.y;
  state(2) = quaternionToYaw(pose.pose.orientation);
  return state;
}

Eigen::MatrixXd MPPIControllerPlugin::pathToReferenceTrajectory(const nav_msgs::msg::Path& path)
{
  // Stub implementation
  return Eigen::MatrixXd::Zero(params_.N + 1, 3);
}

std::vector<Eigen::Vector3d> MPPIControllerPlugin::extractObstaclesFromCostmap()
{
  // Stub implementation
  return {};
}

void MPPIControllerPlugin::publishVisualization(
  const MPPIInfo& info,
  const Eigen::Vector3d& current_state
)
{
  // Stub implementation
}

void MPPIControllerPlugin::declareParameters()
{
  // Stub implementation
}

void MPPIControllerPlugin::loadParameters()
{
  // Stub implementation - using defaults
}

}  // namespace mpc_controller_ros2
