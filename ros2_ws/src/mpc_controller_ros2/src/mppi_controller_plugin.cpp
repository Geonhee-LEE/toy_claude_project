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

  // Register parameter callback for dynamic reconfiguration
  param_callback_handle_ = node_->add_on_set_parameters_callback(
    std::bind(&MPPIControllerPlugin::onSetParametersCallback, this, std::placeholders::_1)
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
  (void)velocity;  // Unused parameter
  (void)goal_checker;  // Unused parameter

  geometry_msgs::msg::TwistStamped cmd_vel;
  cmd_vel.header.stamp = node_->now();
  cmd_vel.header.frame_id = pose.header.frame_id;

  // Check if plan is available
  if (global_plan_.poses.empty()) {
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 1000,
      "No plan available, returning zero velocity"
    );
    cmd_vel.twist.linear.x = 0.0;
    cmd_vel.twist.angular.z = 0.0;
    return cmd_vel;
  }

  try {
    // 1. Convert current pose to state
    Eigen::Vector3d current_state = poseToState(pose);

    // 2. Convert path to reference trajectory
    Eigen::MatrixXd reference_trajectory = pathToReferenceTrajectory(global_plan_);

    // 3. Extract obstacles from costmap
    std::vector<Eigen::Vector3d> obstacles = extractObstaclesFromCostmap();

    // Update obstacle cost function
    if (cost_function_) {
      // Rebuild cost function with updated obstacles
      cost_function_ = std::make_unique<CompositeMPPICost>();
      cost_function_->addCost(std::make_unique<StateTrackingCost>(params_.Q));
      cost_function_->addCost(std::make_unique<TerminalCost>(params_.Qf));
      cost_function_->addCost(std::make_unique<ControlEffortCost>(params_.R));
      cost_function_->addCost(std::make_unique<ControlRateCost>(params_.R_rate));

      if (!obstacles.empty()) {
        auto obstacle_cost = std::make_unique<ObstacleCost>(
          params_.obstacle_weight,
          params_.safety_distance
        );
        obstacle_cost->setObstacles(obstacles);
        cost_function_->addCost(std::move(obstacle_cost));
      }
    }

    // 4. Compute optimal control
    auto [u_opt, info] = computeControl(current_state, reference_trajectory);

    // 5. Apply speed limit if set
    double v_cmd = u_opt(0);
    double omega_cmd = u_opt(1);

    if (speed_limit_valid_) {
      v_cmd = std::clamp(v_cmd, -speed_limit_, speed_limit_);
    }

    // 6. Build Twist message
    cmd_vel.twist.linear.x = v_cmd;
    cmd_vel.twist.linear.y = 0.0;
    cmd_vel.twist.linear.z = 0.0;
    cmd_vel.twist.angular.x = 0.0;
    cmd_vel.twist.angular.y = 0.0;
    cmd_vel.twist.angular.z = omega_cmd;

    // 7. Publish visualization
    publishVisualization(info, current_state);

    RCLCPP_DEBUG(
      node_->get_logger(),
      "MPPI: v=%.3f, omega=%.3f, min_cost=%.4f, ESS=%.1f/%d",
      v_cmd, omega_cmd,
      info.costs.minCoeff(), info.ess, params_.K
    );

  } catch (const std::exception& e) {
    RCLCPP_ERROR(
      node_->get_logger(),
      "Exception in computeVelocityCommands: %s", e.what()
    );
    cmd_vel.twist.linear.x = 0.0;
    cmd_vel.twist.angular.z = 0.0;
  }

  return cmd_vel;
}

void MPPIControllerPlugin::setPlan(const nav_msgs::msg::Path& path)
{
  global_plan_ = path;

  // Reset control sequence for new plan (warm start from zero)
  control_sequence_ = Eigen::MatrixXd::Zero(params_.N, 2);

  RCLCPP_INFO(
    node_->get_logger(),
    "Received new plan with %zu poses, reset control sequence",
    path.poses.size()
  );

  if (path.poses.empty()) {
    RCLCPP_WARN(node_->get_logger(), "Received empty plan");
  }
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
  if (path.poses.empty()) {
    RCLCPP_WARN(node_->get_logger(), "Empty path provided");
    return Eigen::MatrixXd::Zero(params_.N + 1, 3);
  }

  int path_size = path.poses.size();
  Eigen::MatrixXd reference = Eigen::MatrixXd::Zero(params_.N + 1, 3);

  // If path is shorter than horizon, repeat last pose
  if (path_size <= params_.N + 1) {
    for (int t = 0; t < path_size; ++t) {
      reference(t, 0) = path.poses[t].pose.position.x;
      reference(t, 1) = path.poses[t].pose.position.y;
      reference(t, 2) = quaternionToYaw(path.poses[t].pose.orientation);
    }
    // Fill remaining with last pose
    for (int t = path_size; t <= params_.N; ++t) {
      reference.row(t) = reference.row(path_size - 1);
    }
  } else {
    // Interpolate path to match horizon length
    for (int t = 0; t <= params_.N; ++t) {
      double interp_idx = static_cast<double>(t) / params_.N * (path_size - 1);
      int idx_lower = static_cast<int>(std::floor(interp_idx));
      int idx_upper = std::min(idx_lower + 1, path_size - 1);
      double alpha = interp_idx - idx_lower;

      // Linear interpolation
      reference(t, 0) = (1.0 - alpha) * path.poses[idx_lower].pose.position.x +
                        alpha * path.poses[idx_upper].pose.position.x;
      reference(t, 1) = (1.0 - alpha) * path.poses[idx_lower].pose.position.y +
                        alpha * path.poses[idx_upper].pose.position.y;

      // Angle interpolation (using shorter path on circle)
      double theta_lower = quaternionToYaw(path.poses[idx_lower].pose.orientation);
      double theta_upper = quaternionToYaw(path.poses[idx_upper].pose.orientation);
      double theta_diff = normalizeAngle(theta_upper - theta_lower);
      reference(t, 2) = normalizeAngle(theta_lower + alpha * theta_diff);
    }
  }

  return reference;
}

std::vector<Eigen::Vector3d> MPPIControllerPlugin::extractObstaclesFromCostmap()
{
  std::vector<Eigen::Vector3d> obstacles;

  if (!costmap_ros_) {
    return obstacles;
  }

  auto costmap = costmap_ros_->getCostmap();
  unsigned int size_x = costmap->getSizeInCellsX();
  unsigned int size_y = costmap->getSizeInCellsY();
  double resolution = costmap->getResolution();
  double origin_x = costmap->getOriginX();
  double origin_y = costmap->getOriginY();

  // Sample obstacles from costmap (grid-based detection)
  // To avoid too many obstacles, we sample every N cells
  int sample_step = 3;

  for (unsigned int i = 0; i < size_x; i += sample_step) {
    for (unsigned int j = 0; j < size_y; j += sample_step) {
      unsigned char cost = costmap->getCost(i, j);

      // Check if cell is occupied (LETHAL_OBSTACLE = 254)
      if (cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
        // Convert cell coordinates to world coordinates
        double wx = origin_x + (i + 0.5) * resolution;
        double wy = origin_y + (j + 0.5) * resolution;

        // Create obstacle: [x, y, radius]
        // Use resolution as approximate radius
        Eigen::Vector3d obs(wx, wy, resolution * 0.5);
        obstacles.push_back(obs);
      }
    }
  }

  RCLCPP_DEBUG(
    node_->get_logger(),
    "Extracted %zu obstacles from costmap",
    obstacles.size()
  );

  return obstacles;
}

void MPPIControllerPlugin::publishVisualization(
  const MPPIInfo& info,
  const Eigen::Vector3d& current_state
)
{
  if (!marker_pub_ || marker_pub_->get_subscription_count() == 0) {
    return;  // No subscribers
  }

  visualization_msgs::msg::MarkerArray marker_array;
  auto stamp = node_->now();
  std::string frame_id = "map";  // TODO: get from costmap

  // 1. Best trajectory (red)
  if (info.best_trajectory.size() > 0) {
    visualization_msgs::msg::Marker best_marker;
    best_marker.header.stamp = stamp;
    best_marker.header.frame_id = frame_id;
    best_marker.ns = "mppi_best_trajectory";
    best_marker.id = 0;
    best_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    best_marker.action = visualization_msgs::msg::Marker::ADD;
    best_marker.scale.x = 0.05;
    best_marker.color.r = 1.0;
    best_marker.color.g = 0.0;
    best_marker.color.b = 0.0;
    best_marker.color.a = 1.0;

    for (int t = 0; t < info.best_trajectory.rows(); ++t) {
      geometry_msgs::msg::Point p;
      p.x = info.best_trajectory(t, 0);
      p.y = info.best_trajectory(t, 1);
      p.z = 0.0;
      best_marker.points.push_back(p);
    }
    marker_array.markers.push_back(best_marker);
  }

  // 2. Sample trajectories (gray, semi-transparent)
  // Only visualize a subset to avoid overwhelming RVIZ
  int sample_vis_stride = std::max(1, static_cast<int>(info.sample_trajectories.size() / 20));

  for (size_t k = 0; k < info.sample_trajectories.size(); k += sample_vis_stride) {
    visualization_msgs::msg::Marker sample_marker;
    sample_marker.header.stamp = stamp;
    sample_marker.header.frame_id = frame_id;
    sample_marker.ns = "mppi_samples";
    sample_marker.id = k;
    sample_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    sample_marker.action = visualization_msgs::msg::Marker::ADD;
    sample_marker.scale.x = 0.01;
    sample_marker.color.r = 0.5;
    sample_marker.color.g = 0.5;
    sample_marker.color.b = 0.5;
    sample_marker.color.a = 0.3;

    const auto& traj = info.sample_trajectories[k];
    for (int t = 0; t < traj.rows(); ++t) {
      geometry_msgs::msg::Point p;
      p.x = traj(t, 0);
      p.y = traj(t, 1);
      p.z = 0.0;
      sample_marker.points.push_back(p);
    }
    marker_array.markers.push_back(sample_marker);
  }

  // 3. Current position (green sphere)
  visualization_msgs::msg::Marker current_marker;
  current_marker.header.stamp = stamp;
  current_marker.header.frame_id = frame_id;
  current_marker.ns = "mppi_current";
  current_marker.id = 0;
  current_marker.type = visualization_msgs::msg::Marker::SPHERE;
  current_marker.action = visualization_msgs::msg::Marker::ADD;
  current_marker.pose.position.x = current_state(0);
  current_marker.pose.position.y = current_state(1);
  current_marker.pose.position.z = 0.0;
  current_marker.pose.orientation.w = 1.0;
  current_marker.scale.x = 0.2;
  current_marker.scale.y = 0.2;
  current_marker.scale.z = 0.2;
  current_marker.color.r = 0.0;
  current_marker.color.g = 1.0;
  current_marker.color.b = 0.0;
  current_marker.color.a = 1.0;
  marker_array.markers.push_back(current_marker);

  marker_pub_->publish(marker_array);
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

rcl_interfaces::msg::SetParametersResult MPPIControllerPlugin::onSetParametersCallback(
  const std::vector<rclcpp::Parameter>& parameters
)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  std::string prefix = plugin_name_ + ".";
  bool need_recreate_sampler = false;
  bool need_recreate_cost_function = false;

  for (const auto& param : parameters) {
    std::string param_name = param.get_name();

    // Skip parameters not belonging to this plugin
    if (param_name.find(prefix) != 0) {
      continue;
    }

    // Remove prefix for easier matching
    std::string short_name = param_name.substr(prefix.length());

    // Handle parameters that require restart (N, K, dt)
    if (short_name == "N" || short_name == "K" || short_name == "dt") {
      result.successful = false;
      result.reason = "Parameter '" + short_name + "' requires controller restart (affects memory allocation)";
      RCLCPP_WARN(
        node_->get_logger(),
        "Cannot change parameter '%s' at runtime - requires restart",
        short_name.c_str()
      );
      return result;
    }

    // Runtime changeable parameters
    try {
      // Temperature parameter (lambda)
      if (short_name == "lambda") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "lambda must be >= 0.0";
          return result;
        }
        params_.lambda = value;
        RCLCPP_INFO(node_->get_logger(), "Updated lambda: %.2f", value);
      }
      // Noise parameters
      else if (short_name == "noise_sigma_v") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "noise_sigma_v must be >= 0.0";
          return result;
        }
        params_.noise_sigma(0) = value;
        need_recreate_sampler = true;
        RCLCPP_INFO(node_->get_logger(), "Updated noise_sigma_v: %.3f", value);
      }
      else if (short_name == "noise_sigma_omega") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "noise_sigma_omega must be >= 0.0";
          return result;
        }
        params_.noise_sigma(1) = value;
        need_recreate_sampler = true;
        RCLCPP_INFO(node_->get_logger(), "Updated noise_sigma_omega: %.3f", value);
      }
      // Control limits
      else if (short_name == "v_max") {
        double value = param.as_double();
        if (value <= params_.v_min) {
          result.successful = false;
          result.reason = "v_max must be > v_min";
          return result;
        }
        params_.v_max = value;
        RCLCPP_INFO(node_->get_logger(), "Updated v_max: %.3f", value);
      }
      else if (short_name == "v_min") {
        double value = param.as_double();
        if (value >= params_.v_max) {
          result.successful = false;
          result.reason = "v_min must be < v_max";
          return result;
        }
        params_.v_min = value;
        RCLCPP_INFO(node_->get_logger(), "Updated v_min: %.3f", value);
      }
      else if (short_name == "omega_max") {
        double value = param.as_double();
        if (value <= params_.omega_min) {
          result.successful = false;
          result.reason = "omega_max must be > omega_min";
          return result;
        }
        params_.omega_max = value;
        RCLCPP_INFO(node_->get_logger(), "Updated omega_max: %.3f", value);
      }
      else if (short_name == "omega_min") {
        double value = param.as_double();
        if (value >= params_.omega_max) {
          result.successful = false;
          result.reason = "omega_min must be < omega_max";
          return result;
        }
        params_.omega_min = value;
        RCLCPP_INFO(node_->get_logger(), "Updated omega_min: %.3f", value);
      }
      // Cost weights - Q
      else if (short_name == "Q_x") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "Q_x must be >= 0.0";
          return result;
        }
        params_.Q(0, 0) = value;
        need_recreate_cost_function = true;
        RCLCPP_INFO(node_->get_logger(), "Updated Q_x: %.3f", value);
      }
      else if (short_name == "Q_y") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "Q_y must be >= 0.0";
          return result;
        }
        params_.Q(1, 1) = value;
        need_recreate_cost_function = true;
        RCLCPP_INFO(node_->get_logger(), "Updated Q_y: %.3f", value);
      }
      else if (short_name == "Q_theta") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "Q_theta must be >= 0.0";
          return result;
        }
        params_.Q(2, 2) = value;
        need_recreate_cost_function = true;
        RCLCPP_INFO(node_->get_logger(), "Updated Q_theta: %.3f", value);
      }
      // Cost weights - Qf
      else if (short_name == "Qf_x") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "Qf_x must be >= 0.0";
          return result;
        }
        params_.Qf(0, 0) = value;
        need_recreate_cost_function = true;
        RCLCPP_INFO(node_->get_logger(), "Updated Qf_x: %.3f", value);
      }
      else if (short_name == "Qf_y") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "Qf_y must be >= 0.0";
          return result;
        }
        params_.Qf(1, 1) = value;
        need_recreate_cost_function = true;
        RCLCPP_INFO(node_->get_logger(), "Updated Qf_y: %.3f", value);
      }
      else if (short_name == "Qf_theta") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "Qf_theta must be >= 0.0";
          return result;
        }
        params_.Qf(2, 2) = value;
        need_recreate_cost_function = true;
        RCLCPP_INFO(node_->get_logger(), "Updated Qf_theta: %.3f", value);
      }
      // Cost weights - R
      else if (short_name == "R_v") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "R_v must be >= 0.0";
          return result;
        }
        params_.R(0, 0) = value;
        need_recreate_cost_function = true;
        RCLCPP_INFO(node_->get_logger(), "Updated R_v: %.3f", value);
      }
      else if (short_name == "R_omega") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "R_omega must be >= 0.0";
          return result;
        }
        params_.R(1, 1) = value;
        need_recreate_cost_function = true;
        RCLCPP_INFO(node_->get_logger(), "Updated R_omega: %.3f", value);
      }
      // Cost weights - R_rate
      else if (short_name == "R_rate_v") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "R_rate_v must be >= 0.0";
          return result;
        }
        params_.R_rate(0, 0) = value;
        need_recreate_cost_function = true;
        RCLCPP_INFO(node_->get_logger(), "Updated R_rate_v: %.3f", value);
      }
      else if (short_name == "R_rate_omega") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "R_rate_omega must be >= 0.0";
          return result;
        }
        params_.R_rate(1, 1) = value;
        need_recreate_cost_function = true;
        RCLCPP_INFO(node_->get_logger(), "Updated R_rate_omega: %.3f", value);
      }
      // Obstacle avoidance
      else if (short_name == "obstacle_weight") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "obstacle_weight must be >= 0.0";
          return result;
        }
        params_.obstacle_weight = value;
        need_recreate_cost_function = true;
        RCLCPP_INFO(node_->get_logger(), "Updated obstacle_weight: %.3f", value);
      }
      else if (short_name == "safety_distance") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "safety_distance must be >= 0.0";
          return result;
        }
        params_.safety_distance = value;
        need_recreate_cost_function = true;
        RCLCPP_INFO(node_->get_logger(), "Updated safety_distance: %.3f", value);
      }

    } catch (const rclcpp::ParameterTypeException& e) {
      result.successful = false;
      result.reason = "Type mismatch for parameter: " + short_name;
      RCLCPP_ERROR(node_->get_logger(), "Parameter type error: %s", e.what());
      return result;
    }
  }

  // Recreate components if necessary
  if (need_recreate_sampler) {
    sampler_ = std::make_unique<GaussianSampler>(params_.noise_sigma);
    RCLCPP_INFO(node_->get_logger(), "Recreated sampler with new noise parameters");
  }

  if (need_recreate_cost_function) {
    // Note: Cost function is recreated every control cycle with obstacles,
    // so no explicit recreation needed here. The new parameters will be
    // used in the next control computation.
    RCLCPP_INFO(node_->get_logger(), "Cost function will use new weights in next iteration");
  }

  return result;
}

}  // namespace mpc_controller_ros2
