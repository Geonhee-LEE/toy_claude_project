#include "mpc_controller_ros2/multi_agent_mppi_controller_plugin.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::MultiAgentMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void MultiAgentMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  auto node = parent.lock();

  // Declare and load multi-agent parameters
  node->declare_parameter(name + ".multi_agent_enabled", false);
  node->declare_parameter(name + ".multi_agent_id", 0);
  node->declare_parameter(name + ".multi_agent_count", 3);
  node->declare_parameter(name + ".inter_agent_cost_weight", 500.0);
  node->declare_parameter(name + ".inter_agent_safety_dist", 0.3);
  node->declare_parameter(name + ".inter_agent_robot_radius", 0.2);
  node->declare_parameter(name + ".inter_agent_timeout", 2.0);
  node->declare_parameter(name + ".multi_agent_topic_prefix", std::string("/agent"));

  node->get_parameter(name + ".multi_agent_enabled", multi_agent_enabled_);
  node->get_parameter(name + ".multi_agent_id", multi_agent_id_);
  node->get_parameter(name + ".multi_agent_count", multi_agent_count_);
  node->get_parameter(name + ".inter_agent_cost_weight", inter_agent_cost_weight_);
  node->get_parameter(name + ".inter_agent_safety_dist", inter_agent_safety_dist_);
  node->get_parameter(name + ".inter_agent_robot_radius", inter_agent_robot_radius_);
  node->get_parameter(name + ".inter_agent_timeout", inter_agent_timeout_);
  node->get_parameter(name + ".multi_agent_topic_prefix", multi_agent_topic_prefix_);

  if (!multi_agent_enabled_) return;

  agent_manager_ = std::make_unique<AgentTrajectoryManager>(
    multi_agent_id_, inter_agent_timeout_);

  // Add inter-agent cost
  auto inter_cost = std::make_unique<InterAgentCost>(
    agent_manager_.get(),
    inter_agent_cost_weight_,
    inter_agent_safety_dist_,
    inter_agent_robot_radius_);
  cost_function_->addCost(std::move(inter_cost));

  // Trajectory publisher
  std::string pub_topic = multi_agent_topic_prefix_ +
    "_" + std::to_string(multi_agent_id_) + "/predicted_path";
  traj_pub_ = node->create_publisher<nav_msgs::msg::Path>(pub_topic, 10);

  // Subscribe to all other agents' trajectories
  for (int i = 0; i < multi_agent_count_; ++i) {
    if (i == multi_agent_id_) continue;
    std::string sub_topic = multi_agent_topic_prefix_ +
      "_" + std::to_string(i) + "/predicted_path";
    auto sub = node->create_subscription<nav_msgs::msg::Path>(
      sub_topic, 10,
      [this](const nav_msgs::msg::Path::SharedPtr msg) {
        this->onAgentTrajectory(msg);
      });
    traj_subs_.push_back(sub);
  }

  RCLCPP_INFO(node->get_logger(),
    "Multi-Agent MPPI configured: agent_id=%d, count=%d, weight=%.0f, safety=%.2f",
    multi_agent_id_, multi_agent_count_,
    inter_agent_cost_weight_, inter_agent_safety_dist_);
}

void MultiAgentMPPIControllerPlugin::onAgentTrajectory(
  const nav_msgs::msg::Path::SharedPtr msg)
{
  if (!agent_manager_ || msg->poses.empty()) return;

  // Extract agent_id from frame_id (format: "agent_N")
  int agent_id = -1;
  std::string frame = msg->header.frame_id;
  auto pos = frame.find("agent_");
  if (pos != std::string::npos) {
    try { agent_id = std::stoi(frame.substr(pos + 6)); }
    catch (...) { return; }
  }
  if (agent_id < 0 || agent_id == multi_agent_id_) return;

  int N = static_cast<int>(msg->poses.size());
  int nx = 3;  // x, y, theta
  Eigen::MatrixXd traj(N, nx);
  for (int t = 0; t < N; ++t) {
    traj(t, 0) = msg->poses[t].pose.position.x;
    traj(t, 1) = msg->poses[t].pose.position.y;
    double yaw = std::atan2(
      2.0 * (msg->poses[t].pose.orientation.w * msg->poses[t].pose.orientation.z),
      1.0 - 2.0 * msg->poses[t].pose.orientation.z * msg->poses[t].pose.orientation.z);
    traj(t, 2) = yaw;
  }

  // Estimate velocity from first two points
  Eigen::Vector2d vel(0.0, 0.0);
  if (N >= 2 && params_.dt > 0) {
    vel(0) = (traj(1, 0) - traj(0, 0)) / params_.dt;
    vel(1) = (traj(1, 1) - traj(0, 1)) / params_.dt;
  }

  AgentPrediction pred;
  pred.agent_id = agent_id;
  pred.trajectory = traj;
  pred.velocity = vel;
  pred.radius = inter_agent_robot_radius_;
  pred.timestamp = rclcpp::Time(msg->header.stamp).seconds();
  pred.dt = params_.dt;

  agent_manager_->updateAgentTrajectory(agent_id, pred);
}

void MultiAgentMPPIControllerPlugin::publishOwnTrajectory(const MPPIInfo& info)
{
  if (!traj_pub_) return;

  nav_msgs::msg::Path path;
  path.header.stamp = node_->now();
  path.header.frame_id = "agent_" + std::to_string(multi_agent_id_);

  const auto& traj = info.weighted_avg_trajectory;
  for (int t = 0; t < traj.rows(); ++t) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header = path.header;
    pose.pose.position.x = traj(t, 0);
    pose.pose.position.y = traj(t, 1);
    if (traj.cols() > 2) {
      double yaw = traj(t, 2);
      pose.pose.orientation.z = std::sin(yaw / 2.0);
      pose.pose.orientation.w = std::cos(yaw / 2.0);
    }
    path.poses.push_back(pose);
  }

  traj_pub_->publish(path);
}

std::pair<Eigen::VectorXd, MPPIInfo> MultiAgentMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  if (!multi_agent_enabled_ || !agent_manager_) {
    return MPPIControllerPlugin::computeControl(current_state, reference_trajectory);
  }

  // Prune stale agent predictions
  double now = node_->now().seconds();
  agent_manager_->pruneStale(now);

  // Add other agents as CBF obstacles (if CBF enabled)
  if (params_.cbf_enabled) {
    auto [obs, vels] = agent_manager_->toObstaclesWithVelocity(now);
    if (!obs.empty()) {
      // Merge with existing obstacles via setObstaclesWithVelocity
      barrier_set_.setObstaclesWithVelocity(obs, vels);
    }
  }

  // Run parent MPPI (InterAgentCost is already in cost_function_)
  auto [u_opt, info] = MPPIControllerPlugin::computeControl(
    current_state, reference_trajectory);

  // Publish own predicted trajectory
  publishOwnTrajectory(info);

  // Update own trajectory in manager
  Eigen::Vector2d own_vel(0.0, 0.0);
  if (current_state.size() >= 3) {
    own_vel(0) = u_opt(0) * std::cos(current_state(2));
    own_vel(1) = u_opt(0) * std::sin(current_state(2));
  }
  agent_manager_->updateOwnTrajectory(
    info.weighted_avg_trajectory, own_vel, params_.dt);

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
