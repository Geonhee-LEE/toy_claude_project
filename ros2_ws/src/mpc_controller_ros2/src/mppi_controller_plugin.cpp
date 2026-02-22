#include "mpc_controller_ros2/mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <chrono>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::MPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

MPPIControllerPlugin::MPPIControllerPlugin()
  : weight_computation_(std::make_unique<VanillaMPPIWeights>())
{
}

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

  // 샘플러 선택: Colored Noise vs Gaussian
  if (params_.colored_noise) {
    sampler_ = std::make_unique<ColoredNoiseSampler>(
      params_.noise_sigma,
      params_.noise_beta
    );
    RCLCPP_INFO(node_->get_logger(), "Using ColoredNoiseSampler (beta=%.2f)", params_.noise_beta);
  } else {
    sampler_ = std::make_unique<GaussianSampler>(params_.noise_sigma);
    RCLCPP_INFO(node_->get_logger(), "Using GaussianSampler");
  }

  // Adaptive Temperature 초기화
  if (params_.adaptive_temperature) {
    adaptive_temp_ = std::make_unique<AdaptiveTemperature>(
      params_.lambda,
      params_.target_ess_ratio,
      params_.adaptation_rate,
      params_.lambda_min,
      params_.lambda_max
    );
    RCLCPP_INFO(node_->get_logger(), "Adaptive Temperature enabled (target ESS ratio=%.2f)",
      params_.target_ess_ratio);
  }

  // Tube-MPPI 초기화
  if (params_.tube_enabled) {
    tube_mppi_ = std::make_unique<TubeMPPI>(params_);
    RCLCPP_INFO(node_->get_logger(), "Tube-MPPI enabled (tube_width=%.2f)", params_.tube_width);
  }

  // Initialize cost function (1회만 생성)
  cost_function_ = std::make_unique<CompositeMPPICost>();
  cost_function_->addCost(std::make_unique<StateTrackingCost>(params_.Q));
  cost_function_->addCost(std::make_unique<TerminalCost>(params_.Qf));
  cost_function_->addCost(std::make_unique<ControlEffortCost>(params_.R));
  cost_function_->addCost(std::make_unique<ControlRateCost>(params_.R_rate));
  cost_function_->addCost(
    std::make_unique<PreferForwardCost>(params_.prefer_forward_weight, params_.prefer_forward_linear_ratio)
  );

  // CostmapObstacleCost 추가 (비소유 포인터 저장)
  if (params_.use_costmap_cost) {
    auto costmap_cost = std::make_unique<CostmapObstacleCost>(
      params_.obstacle_weight, params_.costmap_lethal_cost, params_.costmap_critical_cost);
    costmap_obstacle_cost_ptr_ = costmap_cost.get();
    cost_function_->addCost(std::move(costmap_cost));
    RCLCPP_INFO(node_->get_logger(), "CostmapObstacleCost enabled (lethal=%.0f, critical=%.0f)",
      params_.costmap_lethal_cost, params_.costmap_critical_cost);
  } else {
    // 기존 ObstacleCost 사용
    cost_function_->addCost(
      std::make_unique<ObstacleCost>(params_.obstacle_weight, params_.safety_distance)
    );
  }

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

  RCLCPP_INFO(
    node_->get_logger(), "MPPI controller configured successfully (weight strategy: %s)",
    weight_computation_->name().c_str()
  );
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
  (void)goal_checker;  // Unused parameter

  // 현재 속도 저장 (동적 lookahead, goal approach에 활용)
  current_velocity_(0) = velocity.linear.x;
  current_velocity_(1) = velocity.angular.z;

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
    // 1. pose를 plan 프레임("map")으로 변환 — 프레임 일관성 보장
    //    nav2 controller_server는 pose를 costmap 프레임("odom")으로 전달하지만,
    //    global_plan_은 planner 프레임("map")이므로 변환 필수
    std::string plan_frame = global_plan_.header.frame_id;
    if (plan_frame.empty()) { plan_frame = "map"; }

    geometry_msgs::msg::PoseStamped pose_in_plan_frame;
    Eigen::Vector3d odom_state = poseToState(pose);  // odom 프레임 (후방 체크용)

    if (pose.header.frame_id != plan_frame) {
      try {
        pose_in_plan_frame = tf_buffer_->transform(
          pose, plan_frame, tf2::durationFromSec(0.1));
      } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(
          node_->get_logger(), *node_->get_clock(), 1000,
          "Failed to transform pose to %s: %s", plan_frame.c_str(), ex.what());
        // 폴백: 변환 없이 원본 pose 사용
        pose_in_plan_frame = pose;
      }
    } else {
      pose_in_plan_frame = pose;
    }

    Eigen::Vector3d current_state = poseToState(pose_in_plan_frame);

    // 2. Prune plan (이미 지나간 waypoint 제거) — map 프레임에서 수행
    prunePlan(current_state);

    // 2.5. pruned_plan_ 남은 경로 거리 계산 (goal approach용)
    goal_dist_ = 0.0;
    for (size_t i = 1; i < pruned_plan_.poses.size(); ++i) {
      double dx = pruned_plan_.poses[i].pose.position.x -
                  pruned_plan_.poses[i - 1].pose.position.x;
      double dy = pruned_plan_.poses[i].pose.position.y -
                  pruned_plan_.poses[i - 1].pose.position.y;
      goal_dist_ += std::sqrt(dx * dx + dy * dy);
    }

    // 3. Convert pruned path to reference trajectory (lookahead 기반)
    Eigen::MatrixXd reference_trajectory = pathToReferenceTrajectory(pruned_plan_, current_state);

    // 4. Update costmap TF (cost_function_ 재생성 없이 TF만 갱신)
    if (params_.use_costmap_cost) {
      updateCostmapObstacles();
    }

    // 4.5. Goal approach: 목표 근처에서 noise sigma 스케일링 (정밀 제어)
    double goal_scale = 1.0;
    if (goal_dist_ < params_.goal_slowdown_dist && params_.goal_slowdown_dist > 1e-6) {
      goal_scale = std::clamp(goal_dist_ / params_.goal_slowdown_dist, 0.1, 1.0);
      Eigen::Vector2d scaled_sigma = params_.noise_sigma * goal_scale;
      if (params_.colored_noise) {
        sampler_ = std::make_unique<ColoredNoiseSampler>(scaled_sigma, params_.noise_beta);
      } else {
        sampler_ = std::make_unique<GaussianSampler>(scaled_sigma);
      }
    } else if (goal_scale < 1.0) {
      // goal에서 멀어지면 원래 sigma 복원
      if (params_.colored_noise) {
        sampler_ = std::make_unique<ColoredNoiseSampler>(params_.noise_sigma, params_.noise_beta);
      } else {
        sampler_ = std::make_unique<GaussianSampler>(params_.noise_sigma);
      }
    }

    // 5. Compute optimal control (measure computation time)
    auto start_time = std::chrono::high_resolution_clock::now();
    auto [u_opt, info] = computeControl(current_state, reference_trajectory);
    auto end_time = std::chrono::high_resolution_clock::now();
    double computation_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // 5.5. Tube-MPPI 피드백 보정 (활성화된 경우)
    Eigen::Vector2d final_control = u_opt;
    if (params_.tube_enabled && tube_mppi_) {
      auto [corrected_control, tube_info] = tube_mppi_->computeCorrectedControl(
        u_opt,
        info.weighted_avg_trajectory,
        current_state
      );
      final_control = corrected_control;
      info.tube_info = tube_info;
      info.tube_mppi_used = true;

      RCLCPP_DEBUG(
        node_->get_logger(),
        "Tube-MPPI: e_fwd=%.3f, e_lat=%.3f, e_ang=%.3f, dv=%.3f, dw=%.3f",
        tube_info.body_error(0), tube_info.body_error(1), tube_info.body_error(2),
        tube_info.feedback_correction(0), tube_info.feedback_correction(1)
      );

      // Tube 시각화
      if (params_.visualize_tube) {
        publishTubeVisualization(tube_info, info.weighted_avg_trajectory);
      }
    }

    // 6. Apply speed limit if set
    double v_cmd = final_control(0);
    double omega_cmd = final_control(1);

    // Goal approach 감속: 거리 비례로 v_max 스케일링
    if (goal_dist_ < params_.goal_slowdown_dist && params_.goal_slowdown_dist > 1e-6) {
      double v_scale = std::clamp(goal_dist_ / params_.goal_slowdown_dist, 0.1, 1.0);
      double effective_v_max = params_.v_max * v_scale;
      v_cmd = std::clamp(v_cmd, params_.v_min, effective_v_max);
    }

    if (speed_limit_valid_) {
      v_cmd = std::clamp(v_cmd, -speed_limit_, speed_limit_);
    }

    // 6.5. 후방 안전 검사: odom 프레임 좌표로 costmap 조회
    if (v_cmd < 0.0 && costmap_ros_) {
      auto costmap = costmap_ros_->getCostmap();
      if (costmap) {
        double theta = odom_state(2);
        double rear_x = odom_state(0) - params_.safety_distance * std::cos(theta);
        double rear_y = odom_state(1) - params_.safety_distance * std::sin(theta);

        unsigned int mx, my;
        if (costmap->worldToMap(rear_x, rear_y, mx, my)) {
          unsigned char rear_cost = costmap->getCost(mx, my);
          if (rear_cost >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
            RCLCPP_WARN_THROTTLE(
              node_->get_logger(), *node_->get_clock(), 500,
              "Rear obstacle detected (cost=%d), blocking backward motion", rear_cost);
            v_cmd = 0.0;
          }
        }
      }
    }

    // 7. Build Twist message
    cmd_vel.twist.linear.x = v_cmd;
    cmd_vel.twist.linear.y = 0.0;
    cmd_vel.twist.linear.z = 0.0;
    cmd_vel.twist.angular.x = 0.0;
    cmd_vel.twist.angular.y = 0.0;
    cmd_vel.twist.angular.z = omega_cmd;

    // 8. Publish visualization (plan 프레임에서)
    publishVisualization(info, current_state, reference_trajectory, info.weighted_avg_trajectory, computation_time_ms);

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
  prune_start_idx_ = 0;

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
  // 마지막 스텝: 이전 값 유지 (급감속 방지)
  control_sequence_.row(N - 1) = control_sequence_.row(N - 2);

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

  // 6. Compute weights via strategy (Adaptive Temperature 적용)
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);

    RCLCPP_DEBUG(
      node_->get_logger(),
      "Adaptive Temp: ESS=%.1f, λ=%.2f→%.2f",
      ess, params_.lambda, current_lambda
    );
  }
  Eigen::VectorXd weights = weight_computation_->compute(costs, current_lambda);

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
  info.weighted_avg_trajectory = weighted_traj;
  info.temperature = (params_.adaptive_temperature && adaptive_temp_) ?
    adaptive_temp_->getLambda() : params_.lambda;
  info.ess = ess;
  info.costs = costs;

  // M2 확장 정보
  info.colored_noise_used = params_.colored_noise;
  info.adaptive_temp_used = params_.adaptive_temperature;
  info.tube_mppi_used = params_.tube_enabled;

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

void MPPIControllerPlugin::prunePlan(const Eigen::Vector3d& current_state)
{
  if (global_plan_.poses.empty()) {
    pruned_plan_ = global_plan_;
    return;
  }

  // prune_start_idx_부터 점진적 탐색 (amortized O(1))
  double min_dist_sq = std::numeric_limits<double>::max();
  size_t closest_idx = prune_start_idx_;
  size_t search_end = std::min(
    prune_start_idx_ + 50, global_plan_.poses.size());

  for (size_t i = prune_start_idx_; i < search_end; ++i) {
    double dx = global_plan_.poses[i].pose.position.x - current_state(0);
    double dy = global_plan_.poses[i].pose.position.y - current_state(1);
    double dist_sq = dx * dx + dy * dy;
    if (dist_sq < min_dist_sq) {
      min_dist_sq = dist_sq;
      closest_idx = i;
    }
  }

  prune_start_idx_ = closest_idx;

  // closest_idx부터 잘라서 pruned_plan_ 생성
  pruned_plan_.header = global_plan_.header;
  pruned_plan_.poses.assign(
    global_plan_.poses.begin() + closest_idx,
    global_plan_.poses.end()
  );
}

Eigen::MatrixXd MPPIControllerPlugin::pathToReferenceTrajectory(
  const nav_msgs::msg::Path& path, const Eigen::Vector3d& /*current_state*/)
{
  if (path.poses.empty()) {
    RCLCPP_WARN(node_->get_logger(), "Empty path provided");
    return Eigen::MatrixXd::Zero(params_.N + 1, 3);
  }

  int path_size = path.poses.size();
  Eigen::MatrixXd reference = Eigen::MatrixXd::Zero(params_.N + 1, 3);

  // 동적 lookahead: 현재 속도 비례 (정지 시 min_lookahead, 고속 시 v_max*N*dt)
  double max_lookahead = params_.v_max * params_.N * params_.dt;
  double lookahead = params_.lookahead_dist;
  if (lookahead <= 0.0) {
    double speed = std::abs(current_velocity_(0));
    lookahead = std::clamp(
      speed * params_.N * params_.dt,
      params_.min_lookahead,
      max_lookahead);
  }

  // 경로를 따라 누적 arc-length 계산
  std::vector<double> arc_lengths(path_size, 0.0);
  for (int i = 1; i < path_size; ++i) {
    double dx = path.poses[i].pose.position.x - path.poses[i - 1].pose.position.x;
    double dy = path.poses[i].pose.position.y - path.poses[i - 1].pose.position.y;
    arc_lengths[i] = arc_lengths[i - 1] + std::sqrt(dx * dx + dy * dy);
  }

  double total_path_length = arc_lengths.back();
  double effective_lookahead = std::min(lookahead, total_path_length);
  double step_distance = effective_lookahead / params_.N;

  // arc-length stepping으로 참조 궤적 생성
  int path_idx = 0;
  for (int t = 0; t <= params_.N; ++t) {
    double target_arc = t * step_distance;

    // 목표 arc-length에 해당하는 경로 구간 탐색
    while (path_idx < path_size - 1 && arc_lengths[path_idx + 1] < target_arc) {
      ++path_idx;
    }

    if (path_idx >= path_size - 1) {
      // 경로 끝 도달: 마지막 점 반복 + goal orientation 사용
      reference(t, 0) = path.poses[path_size - 1].pose.position.x;
      reference(t, 1) = path.poses[path_size - 1].pose.position.y;
      reference(t, 2) = quaternionToYaw(path.poses[path_size - 1].pose.orientation);
    } else {
      // 구간 내 선형 보간
      double seg_len = arc_lengths[path_idx + 1] - arc_lengths[path_idx];
      double alpha = (seg_len > 1e-6) ?
        (target_arc - arc_lengths[path_idx]) / seg_len : 0.0;

      reference(t, 0) = (1.0 - alpha) * path.poses[path_idx].pose.position.x +
                        alpha * path.poses[path_idx + 1].pose.position.x;
      reference(t, 1) = (1.0 - alpha) * path.poses[path_idx].pose.position.y +
                        alpha * path.poses[path_idx + 1].pose.position.y;

      // heading: 경로 접선 방향 (atan2) 사용 — waypoint orientation 대신
      double dx = path.poses[path_idx + 1].pose.position.x -
                  path.poses[path_idx].pose.position.x;
      double dy = path.poses[path_idx + 1].pose.position.y -
                  path.poses[path_idx].pose.position.y;
      double seg_heading = std::atan2(dy, dx);

      // 경로 끝 근처: 접선 heading → goal heading 블렌딩
      double remaining_arc = total_path_length - target_arc;
      double blend_dist = std::min(effective_lookahead * 0.3, 0.5);  // 블렌딩 구간
      if (remaining_arc < blend_dist && blend_dist > 1e-6) {
        double goal_heading = quaternionToYaw(
          path.poses[path_size - 1].pose.orientation);
        double blend_alpha = 1.0 - (remaining_arc / blend_dist);
        double heading_diff = normalizeAngle(goal_heading - seg_heading);
        reference(t, 2) = normalizeAngle(seg_heading + blend_alpha * heading_diff);
      } else {
        reference(t, 2) = seg_heading;
      }
    }
  }

  return reference;
}

void MPPIControllerPlugin::updateCostmapObstacles()
{
  if (!costmap_obstacle_cost_ptr_ || !costmap_ros_) {
    return;
  }

  auto costmap = costmap_ros_->getCostmap();
  if (!costmap) {
    return;
  }

  // costmap 포인터 갱신
  costmap_obstacle_cost_ptr_->setCostmap(costmap);

  // map→odom TF 조회
  try {
    auto transform = tf_buffer_->lookupTransform(
      "odom", "map", tf2::TimePointZero,
      tf2::durationFromSec(0.1));

    double tx = transform.transform.translation.x;
    double ty = transform.transform.translation.y;

    // quaternion → yaw
    double qz = transform.transform.rotation.z;
    double qw = transform.transform.rotation.w;
    double yaw = 2.0 * std::atan2(qz, qw);

    costmap_obstacle_cost_ptr_->setMapToOdomTransform(
      tx, ty, std::cos(yaw), std::sin(yaw), true);
  } catch (const tf2::TransformException& ex) {
    RCLCPP_DEBUG_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 2000,
      "map→odom TF not available (%s), using identity", ex.what());
    costmap_obstacle_cost_ptr_->setMapToOdomTransform(0.0, 0.0, 1.0, 0.0, false);
  }
}

void MPPIControllerPlugin::publishVisualization(
  const MPPIInfo& info,
  const Eigen::Vector3d& current_state,
  const Eigen::MatrixXd& reference_trajectory,
  const Eigen::MatrixXd& weighted_avg_trajectory,
  double computation_time_ms
)
{
  if (!marker_pub_ || marker_pub_->get_subscription_count() == 0) {
    return;  // No subscribers
  }

  visualization_msgs::msg::MarkerArray marker_array;
  auto stamp = node_->now();
  std::string frame_id = global_plan_.header.frame_id.empty() ?
    "map" : global_plan_.header.frame_id;

  // 1. Reference trajectory (yellow dashed line)
  if (params_.visualize_reference && reference_trajectory.rows() > 0) {
    visualization_msgs::msg::Marker ref_marker;
    ref_marker.header.stamp = stamp;
    ref_marker.header.frame_id = frame_id;
    ref_marker.ns = "mppi_reference";
    ref_marker.id = 0;
    ref_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    ref_marker.action = visualization_msgs::msg::Marker::ADD;
    ref_marker.scale.x = 0.03;
    ref_marker.color.r = 1.0;
    ref_marker.color.g = 1.0;
    ref_marker.color.b = 0.0;
    ref_marker.color.a = 0.6;

    for (int t = 0; t < reference_trajectory.rows(); ++t) {
      geometry_msgs::msg::Point p;
      p.x = reference_trajectory(t, 0);
      p.y = reference_trajectory(t, 1);
      p.z = 0.0;
      ref_marker.points.push_back(p);
    }
    marker_array.markers.push_back(ref_marker);
  }

  // 2. Sample trajectories (gray, semi-transparent)
  if (params_.visualize_samples && !info.sample_trajectories.empty()) {
    int sample_vis_stride = std::max(
      1,
      static_cast<int>(info.sample_trajectories.size()) / params_.max_visualized_samples
    );

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
  }

  // 3. Weighted average trajectory (blue)
  if (params_.visualize_weighted_avg && weighted_avg_trajectory.rows() > 0) {
    visualization_msgs::msg::Marker weighted_marker;
    weighted_marker.header.stamp = stamp;
    weighted_marker.header.frame_id = frame_id;
    weighted_marker.ns = "mppi_weighted_avg";
    weighted_marker.id = 0;
    weighted_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    weighted_marker.action = visualization_msgs::msg::Marker::ADD;
    weighted_marker.scale.x = 0.06;
    weighted_marker.color.r = 0.0;
    weighted_marker.color.g = 0.5;
    weighted_marker.color.b = 1.0;
    weighted_marker.color.a = 0.9;

    for (int t = 0; t < weighted_avg_trajectory.rows(); ++t) {
      geometry_msgs::msg::Point p;
      p.x = weighted_avg_trajectory(t, 0);
      p.y = weighted_avg_trajectory(t, 1);
      p.z = 0.0;
      weighted_marker.points.push_back(p);
    }
    marker_array.markers.push_back(weighted_marker);
  }

  // 4. Best trajectory (red)
  if (params_.visualize_best && info.best_trajectory.size() > 0) {
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

  // 5. Control sequence arrows
  if (params_.visualize_control_sequence && control_sequence_.rows() > 0) {
    // Visualize control arrows at intervals
    int arrow_stride = std::max(1, static_cast<int>(control_sequence_.rows()) / 5);

    for (int t = 0; t < control_sequence_.rows(); t += arrow_stride) {
      if (t < weighted_avg_trajectory.rows()) {
        visualization_msgs::msg::Marker arrow_marker;
        arrow_marker.header.stamp = stamp;
        arrow_marker.header.frame_id = frame_id;
        arrow_marker.ns = "mppi_control_arrows";
        arrow_marker.id = t;
        arrow_marker.type = visualization_msgs::msg::Marker::ARROW;
        arrow_marker.action = visualization_msgs::msg::Marker::ADD;

        // Arrow from current position
        geometry_msgs::msg::Point start, end;
        start.x = weighted_avg_trajectory(t, 0);
        start.y = weighted_avg_trajectory(t, 1);
        start.z = 0.0;

        // Arrow direction based on velocity command
        double theta = weighted_avg_trajectory(t, 2);
        double v = control_sequence_(t, 0);
        double arrow_length = 0.3 * std::abs(v);  // Scale by velocity

        end.x = start.x + arrow_length * std::cos(theta);
        end.y = start.y + arrow_length * std::sin(theta);
        end.z = 0.0;

        arrow_marker.points.push_back(start);
        arrow_marker.points.push_back(end);

        arrow_marker.scale.x = 0.05;  // Shaft diameter
        arrow_marker.scale.y = 0.1;   // Head diameter
        arrow_marker.scale.z = 0.1;   // Head length

        arrow_marker.color.r = 0.0;
        arrow_marker.color.g = 1.0;
        arrow_marker.color.b = 0.5;
        arrow_marker.color.a = 0.8;

        marker_array.markers.push_back(arrow_marker);
      }
    }
  }

  // 6. Current position (green sphere)
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

  // 7. Text information (ESS, cost, lambda, computation time)
  if (params_.visualize_text_info) {
    visualization_msgs::msg::Marker text_marker;
    text_marker.header.stamp = stamp;
    text_marker.header.frame_id = frame_id;
    text_marker.ns = "mppi_text_info";
    text_marker.id = 0;
    text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::msg::Marker::ADD;

    // Position above current state
    text_marker.pose.position.x = current_state(0);
    text_marker.pose.position.y = current_state(1);
    text_marker.pose.position.z = 1.0;
    text_marker.pose.orientation.w = 1.0;

    // Format text
    char text_buffer[256];
    snprintf(
      text_buffer,
      sizeof(text_buffer),
      "ESS: %.1f/%d\nλ: %.1f\nCost: %.2f/%.2f/%.2f\nTime: %.1f ms",
      info.ess,
      params_.K,
      info.temperature,
      info.costs.minCoeff(),
      info.costs.mean(),
      info.costs.maxCoeff(),
      computation_time_ms
    );
    text_marker.text = text_buffer;

    text_marker.scale.z = 0.2;  // Text height
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    text_marker.color.a = 1.0;

    marker_array.markers.push_back(text_marker);
  }

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

  // Forward preference
  node_->declare_parameter(prefix + "prefer_forward_weight", params_.prefer_forward_weight);
  node_->declare_parameter(prefix + "prefer_forward_linear_ratio", params_.prefer_forward_linear_ratio);

  // Costmap obstacle cost
  node_->declare_parameter(prefix + "use_costmap_cost", params_.use_costmap_cost);
  node_->declare_parameter(prefix + "costmap_lethal_cost", params_.costmap_lethal_cost);
  node_->declare_parameter(prefix + "costmap_critical_cost", params_.costmap_critical_cost);
  node_->declare_parameter(prefix + "lookahead_dist", params_.lookahead_dist);
  node_->declare_parameter(prefix + "min_lookahead", params_.min_lookahead);
  node_->declare_parameter(prefix + "goal_slowdown_dist", params_.goal_slowdown_dist);

  // Phase 1: Colored Noise
  node_->declare_parameter(prefix + "colored_noise", params_.colored_noise);
  node_->declare_parameter(prefix + "noise_beta", params_.noise_beta);

  // Phase 2: Adaptive Temperature
  node_->declare_parameter(prefix + "adaptive_temperature", params_.adaptive_temperature);
  node_->declare_parameter(prefix + "target_ess_ratio", params_.target_ess_ratio);
  node_->declare_parameter(prefix + "adaptation_rate", params_.adaptation_rate);
  node_->declare_parameter(prefix + "lambda_min", params_.lambda_min);
  node_->declare_parameter(prefix + "lambda_max", params_.lambda_max);

  // Phase 3: Tube-MPPI
  node_->declare_parameter(prefix + "tube_enabled", params_.tube_enabled);
  node_->declare_parameter(prefix + "tube_width", params_.tube_width);
  node_->declare_parameter(prefix + "k_forward", params_.k_forward);
  node_->declare_parameter(prefix + "k_lateral", params_.k_lateral);
  node_->declare_parameter(prefix + "k_angle", params_.k_angle);

  // SOTA 변형 파라미터 (Tsallis, Risk-Aware)
  node_->declare_parameter(prefix + "tsallis_q", params_.tsallis_q);
  node_->declare_parameter(prefix + "cvar_alpha", params_.cvar_alpha);

  // SVMPC (Stein Variational MPC)
  node_->declare_parameter(prefix + "svgd_num_iterations", params_.svgd_num_iterations);
  node_->declare_parameter(prefix + "svgd_step_size", params_.svgd_step_size);
  node_->declare_parameter(prefix + "svgd_bandwidth", params_.svgd_bandwidth);

  // M3.5 Smooth-MPPI
  node_->declare_parameter(prefix + "smooth_R_jerk_v", params_.smooth_R_jerk_v);
  node_->declare_parameter(prefix + "smooth_R_jerk_omega", params_.smooth_R_jerk_omega);
  node_->declare_parameter(prefix + "smooth_action_cost_weight", params_.smooth_action_cost_weight);

  // M3.5 Spline-MPPI
  node_->declare_parameter(prefix + "spline_num_knots", params_.spline_num_knots);
  node_->declare_parameter(prefix + "spline_degree", params_.spline_degree);

  // M3.5 SVG-MPPI
  node_->declare_parameter(prefix + "svg_num_guide_particles", params_.svg_num_guide_particles);
  node_->declare_parameter(prefix + "svg_guide_iterations", params_.svg_guide_iterations);
  node_->declare_parameter(prefix + "svg_guide_step_size", params_.svg_guide_step_size);
  node_->declare_parameter(prefix + "svg_resample_std", params_.svg_resample_std);

  // Visualization
  node_->declare_parameter(prefix + "visualize_samples", params_.visualize_samples);
  node_->declare_parameter(prefix + "visualize_best", params_.visualize_best);
  node_->declare_parameter(prefix + "visualize_weighted_avg", params_.visualize_weighted_avg);
  node_->declare_parameter(prefix + "visualize_reference", params_.visualize_reference);
  node_->declare_parameter(prefix + "visualize_text_info", params_.visualize_text_info);
  node_->declare_parameter(prefix + "visualize_control_sequence", params_.visualize_control_sequence);
  node_->declare_parameter(prefix + "visualize_tube", params_.visualize_tube);
  node_->declare_parameter(prefix + "max_visualized_samples", params_.max_visualized_samples);

  RCLCPP_INFO(node_->get_logger(), "MPPI parameters declared (M2+SOTA features included)");
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

  // Forward preference
  params_.prefer_forward_weight = node_->get_parameter(prefix + "prefer_forward_weight").as_double();
  params_.prefer_forward_linear_ratio = node_->get_parameter(prefix + "prefer_forward_linear_ratio").as_double();

  // Costmap obstacle cost
  params_.use_costmap_cost = node_->get_parameter(prefix + "use_costmap_cost").as_bool();
  params_.costmap_lethal_cost = node_->get_parameter(prefix + "costmap_lethal_cost").as_double();
  params_.costmap_critical_cost = node_->get_parameter(prefix + "costmap_critical_cost").as_double();
  params_.lookahead_dist = node_->get_parameter(prefix + "lookahead_dist").as_double();
  params_.min_lookahead = node_->get_parameter(prefix + "min_lookahead").as_double();
  params_.goal_slowdown_dist = node_->get_parameter(prefix + "goal_slowdown_dist").as_double();

  // Phase 1: Colored Noise
  params_.colored_noise = node_->get_parameter(prefix + "colored_noise").as_bool();
  params_.noise_beta = node_->get_parameter(prefix + "noise_beta").as_double();

  // Phase 2: Adaptive Temperature
  params_.adaptive_temperature = node_->get_parameter(prefix + "adaptive_temperature").as_bool();
  params_.target_ess_ratio = node_->get_parameter(prefix + "target_ess_ratio").as_double();
  params_.adaptation_rate = node_->get_parameter(prefix + "adaptation_rate").as_double();
  params_.lambda_min = node_->get_parameter(prefix + "lambda_min").as_double();
  params_.lambda_max = node_->get_parameter(prefix + "lambda_max").as_double();

  // Phase 3: Tube-MPPI
  params_.tube_enabled = node_->get_parameter(prefix + "tube_enabled").as_bool();
  params_.tube_width = node_->get_parameter(prefix + "tube_width").as_double();
  params_.k_forward = node_->get_parameter(prefix + "k_forward").as_double();
  params_.k_lateral = node_->get_parameter(prefix + "k_lateral").as_double();
  params_.k_angle = node_->get_parameter(prefix + "k_angle").as_double();

  // SOTA 변형 파라미터 (Tsallis, Risk-Aware)
  params_.tsallis_q = node_->get_parameter(prefix + "tsallis_q").as_double();
  params_.cvar_alpha = node_->get_parameter(prefix + "cvar_alpha").as_double();

  // SVMPC (Stein Variational MPC)
  params_.svgd_num_iterations = node_->get_parameter(prefix + "svgd_num_iterations").as_int();
  params_.svgd_step_size = node_->get_parameter(prefix + "svgd_step_size").as_double();
  params_.svgd_bandwidth = node_->get_parameter(prefix + "svgd_bandwidth").as_double();

  // M3.5 Smooth-MPPI
  params_.smooth_R_jerk_v = node_->get_parameter(prefix + "smooth_R_jerk_v").as_double();
  params_.smooth_R_jerk_omega = node_->get_parameter(prefix + "smooth_R_jerk_omega").as_double();
  params_.smooth_action_cost_weight = node_->get_parameter(prefix + "smooth_action_cost_weight").as_double();

  // M3.5 Spline-MPPI
  params_.spline_num_knots = node_->get_parameter(prefix + "spline_num_knots").as_int();
  params_.spline_degree = node_->get_parameter(prefix + "spline_degree").as_int();

  // M3.5 SVG-MPPI
  params_.svg_num_guide_particles = node_->get_parameter(prefix + "svg_num_guide_particles").as_int();
  params_.svg_guide_iterations = node_->get_parameter(prefix + "svg_guide_iterations").as_int();
  params_.svg_guide_step_size = node_->get_parameter(prefix + "svg_guide_step_size").as_double();
  params_.svg_resample_std = node_->get_parameter(prefix + "svg_resample_std").as_double();

  // Visualization
  params_.visualize_samples = node_->get_parameter(prefix + "visualize_samples").as_bool();
  params_.visualize_best = node_->get_parameter(prefix + "visualize_best").as_bool();
  params_.visualize_weighted_avg = node_->get_parameter(prefix + "visualize_weighted_avg").as_bool();
  params_.visualize_reference = node_->get_parameter(prefix + "visualize_reference").as_bool();
  params_.visualize_text_info = node_->get_parameter(prefix + "visualize_text_info").as_bool();
  params_.visualize_control_sequence = node_->get_parameter(prefix + "visualize_control_sequence").as_bool();
  params_.visualize_tube = node_->get_parameter(prefix + "visualize_tube").as_bool();
  params_.max_visualized_samples = node_->get_parameter(prefix + "max_visualized_samples").as_int();

  RCLCPP_INFO(
    node_->get_logger(),
    "MPPI parameters loaded: N=%d, K=%d, dt=%.3f, lambda=%.1f",
    params_.N, params_.K, params_.dt, params_.lambda
  );
  RCLCPP_INFO(
    node_->get_logger(),
    "M2 features: colored_noise=%s, adaptive_temp=%s, tube_mppi=%s",
    params_.colored_noise ? "ON" : "OFF",
    params_.adaptive_temperature ? "ON" : "OFF",
    params_.tube_enabled ? "ON" : "OFF"
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
      // Forward preference
      else if (short_name == "prefer_forward_weight") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "prefer_forward_weight must be >= 0.0";
          return result;
        }
        params_.prefer_forward_weight = value;
        need_recreate_cost_function = true;
        RCLCPP_INFO(node_->get_logger(), "Updated prefer_forward_weight: %.3f", value);
      }
      else if (short_name == "prefer_forward_linear_ratio") {
        double value = param.as_double();
        if (value < 0.0 || value > 1.0) {
          result.successful = false;
          result.reason = "prefer_forward_linear_ratio must be in [0.0, 1.0]";
          return result;
        }
        params_.prefer_forward_linear_ratio = value;
        need_recreate_cost_function = true;
        RCLCPP_INFO(node_->get_logger(), "Updated prefer_forward_linear_ratio: %.3f", value);
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

void MPPIControllerPlugin::publishTubeVisualization(
  const TubeMPPIInfo& tube_info,
  const Eigen::MatrixXd& nominal_trajectory
)
{
  if (!marker_pub_ || marker_pub_->get_subscription_count() == 0) {
    return;
  }

  visualization_msgs::msg::MarkerArray marker_array;
  auto stamp = node_->now();
  std::string frame_id = global_plan_.header.frame_id.empty() ?
    "map" : global_plan_.header.frame_id;

  // Tube 경계 계산
  auto boundaries = tube_mppi_->computeTubeBoundary(nominal_trajectory);

  if (boundaries.empty()) {
    return;
  }

  // 좌측 경계선
  visualization_msgs::msg::Marker left_marker;
  left_marker.header.stamp = stamp;
  left_marker.header.frame_id = frame_id;
  left_marker.ns = "mppi_tube_left";
  left_marker.id = 0;
  left_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  left_marker.action = visualization_msgs::msg::Marker::ADD;
  left_marker.scale.x = 0.02;
  left_marker.color.r = 1.0;
  left_marker.color.g = 0.5;
  left_marker.color.b = 0.0;
  left_marker.color.a = 0.6;

  // 우측 경계선
  visualization_msgs::msg::Marker right_marker = left_marker;
  right_marker.ns = "mppi_tube_right";

  for (const auto& pair : boundaries) {
    geometry_msgs::msg::Point left_pt, right_pt;
    left_pt.x = pair.first(0);
    left_pt.y = pair.first(1);
    left_pt.z = 0.0;
    right_pt.x = pair.second(0);
    right_pt.y = pair.second(1);
    right_pt.z = 0.0;

    left_marker.points.push_back(left_pt);
    right_marker.points.push_back(right_pt);
  }

  marker_array.markers.push_back(left_marker);
  marker_array.markers.push_back(right_marker);

  // Nominal 상태 마커 (Tube-MPPI가 추적하는 이상 위치)
  visualization_msgs::msg::Marker nominal_marker;
  nominal_marker.header.stamp = stamp;
  nominal_marker.header.frame_id = frame_id;
  nominal_marker.ns = "mppi_nominal_state";
  nominal_marker.id = 0;
  nominal_marker.type = visualization_msgs::msg::Marker::SPHERE;
  nominal_marker.action = visualization_msgs::msg::Marker::ADD;
  nominal_marker.pose.position.x = tube_info.nominal_state(0);
  nominal_marker.pose.position.y = tube_info.nominal_state(1);
  nominal_marker.pose.position.z = 0.05;
  nominal_marker.pose.orientation.w = 1.0;
  nominal_marker.scale.x = 0.15;
  nominal_marker.scale.y = 0.15;
  nominal_marker.scale.z = 0.15;
  nominal_marker.color.r = 1.0;
  nominal_marker.color.g = 0.8;
  nominal_marker.color.b = 0.0;
  nominal_marker.color.a = 0.9;
  marker_array.markers.push_back(nominal_marker);

  // 피드백 보정 벡터 화살표
  if (tube_info.feedback_correction.norm() > 0.01) {
    visualization_msgs::msg::Marker fb_arrow;
    fb_arrow.header.stamp = stamp;
    fb_arrow.header.frame_id = frame_id;
    fb_arrow.ns = "mppi_feedback_correction";
    fb_arrow.id = 0;
    fb_arrow.type = visualization_msgs::msg::Marker::ARROW;
    fb_arrow.action = visualization_msgs::msg::Marker::ADD;

    geometry_msgs::msg::Point start, end;
    start.x = tube_info.nominal_state(0);
    start.y = tube_info.nominal_state(1);
    start.z = 0.1;

    // 피드백 보정 방향 표시
    double theta = tube_info.nominal_state(2);
    double correction_length = 0.5 * tube_info.feedback_correction(0);
    end.x = start.x + correction_length * std::cos(theta);
    end.y = start.y + correction_length * std::sin(theta);
    end.z = 0.1;

    fb_arrow.points.push_back(start);
    fb_arrow.points.push_back(end);
    fb_arrow.scale.x = 0.04;
    fb_arrow.scale.y = 0.08;
    fb_arrow.scale.z = 0.08;
    fb_arrow.color.r = 1.0;
    fb_arrow.color.g = 0.0;
    fb_arrow.color.b = 1.0;
    fb_arrow.color.a = 0.8;
    marker_array.markers.push_back(fb_arrow);
  }

  marker_pub_->publish(marker_array);
}

}  // namespace mpc_controller_ros2
