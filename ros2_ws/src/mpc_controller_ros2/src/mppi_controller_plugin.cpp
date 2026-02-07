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
  int N = params_.N;
  int K = params_.K;
  int nu = 2;  // [v, omega]

  // 1. Shift previous control sequence (warm start)
  for (int t = 0; t < N - 1; ++t) {
    control_sequence_.row(t) = control_sequence_.row(t + 1);
  }
  control_sequence_.row(N - 1).setZero();

  // 2. Sample noise
  auto noise_samples = sampler_->sample(K, N, nu);

  // 3. Add noise to control sequence and clip
  std::vector<Eigen::MatrixXd> perturbed_controls;
  perturbed_controls.reserve(K);

  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd perturbed = control_sequence_ + noise_samples[k];
    perturbed = dynamics_->clipControls(perturbed);
    perturbed_controls.push_back(perturbed);
  }

  // 4. Batch rollout
  auto trajectories = dynamics_->rolloutBatch(
    current_state,
    perturbed_controls,
    params_.dt
  );

  // 5. Compute costs
  Eigen::VectorXd costs = cost_function_->compute(
    trajectories,
    perturbed_controls,
    reference_trajectory
  );

  // 6. Compute softmax weights
  Eigen::VectorXd weights = softmaxWeights(costs, params_.lambda);

  // 7. Update control sequence with weighted average of noise
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_noise += weights(k) * noise_samples[k];
  }
  control_sequence_ += weighted_noise;

  // Clip updated control sequence
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // 8. Extract optimal control (first timestep)
  Eigen::Vector2d u_opt = control_sequence_.row(0).transpose();

  // Compute weighted average trajectory
  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, 3);
  for (int k = 0; k < K; ++k) {
    weighted_traj += weights(k) * trajectories[k];
  }

  // Find best sample
  int best_idx;
  double min_cost = costs.minCoeff(&best_idx);

  // Compute ESS
  double ess = computeESS(weights);

  // Build info struct
  MPPIInfo info;
  info.sample_trajectories = trajectories;
  info.sample_weights = weights;
  info.best_trajectory = trajectories[best_idx];
  info.temperature = params_.lambda;
  info.ess = ess;
  info.costs = costs;

  RCLCPP_DEBUG(
    node_->get_logger(),
    "MPPI: min_cost=%.4f, mean_cost=%.4f, ESS=%.1f/%d",
    min_cost, costs.mean(), ess, K
  );

  return {u_opt, info};
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
  std::string prefix = plugin_name_ + ".";

  // MPPI parameters
  node_->declare_parameter(prefix + "N", params_.N);
  node_->declare_parameter(prefix + "dt", params_.dt);
  node_->declare_parameter(prefix + "K", params_.K);
  node_->declare_parameter(prefix + "lambda", params_.lambda);

  // Noise parameters
  node_->declare_parameter(prefix + "noise_sigma_v", params_.noise_sigma(0));
  node_->declare_parameter(prefix + "noise_sigma_omega", params_.noise_sigma(1));

  // Control limits
  node_->declare_parameter(prefix + "v_max", params_.v_max);
  node_->declare_parameter(prefix + "v_min", params_.v_min);
  node_->declare_parameter(prefix + "omega_max", params_.omega_max);
  node_->declare_parameter(prefix + "omega_min", params_.omega_min);

  // Cost weights - Q
  node_->declare_parameter(prefix + "Q_x", params_.Q(0, 0));
  node_->declare_parameter(prefix + "Q_y", params_.Q(1, 1));
  node_->declare_parameter(prefix + "Q_theta", params_.Q(2, 2));

  // Cost weights - Qf
  node_->declare_parameter(prefix + "Qf_x", params_.Qf(0, 0));
  node_->declare_parameter(prefix + "Qf_y", params_.Qf(1, 1));
  node_->declare_parameter(prefix + "Qf_theta", params_.Qf(2, 2));

  // Cost weights - R
  node_->declare_parameter(prefix + "R_v", params_.R(0, 0));
  node_->declare_parameter(prefix + "R_omega", params_.R(1, 1));

  // Cost weights - R_rate
  node_->declare_parameter(prefix + "R_rate_v", params_.R_rate(0, 0));
  node_->declare_parameter(prefix + "R_rate_omega", params_.R_rate(1, 1));

  // Obstacle avoidance
  node_->declare_parameter(prefix + "obstacle_weight", params_.obstacle_weight);
  node_->declare_parameter(prefix + "safety_distance", params_.safety_distance);

  RCLCPP_INFO(node_->get_logger(), "MPPI parameters declared");
}

void MPPIControllerPlugin::loadParameters()
{
  std::string prefix = plugin_name_ + ".";

  // MPPI parameters
  params_.N = node_->get_parameter(prefix + "N").as_int();
  params_.dt = node_->get_parameter(prefix + "dt").as_double();
  params_.K = node_->get_parameter(prefix + "K").as_int();
  params_.lambda = node_->get_parameter(prefix + "lambda").as_double();

  // Noise parameters
  params_.noise_sigma(0) = node_->get_parameter(prefix + "noise_sigma_v").as_double();
  params_.noise_sigma(1) = node_->get_parameter(prefix + "noise_sigma_omega").as_double();

  // Control limits
  params_.v_max = node_->get_parameter(prefix + "v_max").as_double();
  params_.v_min = node_->get_parameter(prefix + "v_min").as_double();
  params_.omega_max = node_->get_parameter(prefix + "omega_max").as_double();
  params_.omega_min = node_->get_parameter(prefix + "omega_min").as_double();

  // Cost weights - Q
  params_.Q(0, 0) = node_->get_parameter(prefix + "Q_x").as_double();
  params_.Q(1, 1) = node_->get_parameter(prefix + "Q_y").as_double();
  params_.Q(2, 2) = node_->get_parameter(prefix + "Q_theta").as_double();

  // Cost weights - Qf
  params_.Qf(0, 0) = node_->get_parameter(prefix + "Qf_x").as_double();
  params_.Qf(1, 1) = node_->get_parameter(prefix + "Qf_y").as_double();
  params_.Qf(2, 2) = node_->get_parameter(prefix + "Qf_theta").as_double();

  // Cost weights - R
  params_.R(0, 0) = node_->get_parameter(prefix + "R_v").as_double();
  params_.R(1, 1) = node_->get_parameter(prefix + "R_omega").as_double();

  // Cost weights - R_rate
  params_.R_rate(0, 0) = node_->get_parameter(prefix + "R_rate_v").as_double();
  params_.R_rate(1, 1) = node_->get_parameter(prefix + "R_rate_omega").as_double();

  // Obstacle avoidance
  params_.obstacle_weight = node_->get_parameter(prefix + "obstacle_weight").as_double();
  params_.safety_distance = node_->get_parameter(prefix + "safety_distance").as_double();

  RCLCPP_INFO(
    node_->get_logger(),
    "MPPI parameters loaded: N=%d, K=%d, dt=%.3f, lambda=%.1f",
    params_.N, params_.K, params_.dt, params_.lambda
  );
}

}  // namespace mpc_controller_ros2
