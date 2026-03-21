#include "mpc_controller_ros2/mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"
#include "mpc_controller_ros2/ackermann_model.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include "mpc_controller_ros2/conformal_predictor.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <chrono>
#include <set>
#include <omp.h>

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

  // Initialize components (MotionModel 기반 + Residual/Ensemble Dynamics 래핑)
  std::unique_ptr<MotionModel> model;
  if (params_.ensemble_enabled && !params_.ensemble_weights_dir.empty()) {
    model = MotionModelFactory::createWithEnsemble(params_.motion_model, params_);
    RCLCPP_INFO(node_->get_logger(),
      "Ensemble Dynamics enabled (M=%d, alpha=%.2f, dir=%s)",
      params_.ensemble_size, params_.ensemble_alpha, params_.ensemble_weights_dir.c_str());
  } else if (params_.residual_enabled && !params_.residual_weights_path.empty()) {
    model = MotionModelFactory::createWithResidual(params_.motion_model, params_);
    RCLCPP_INFO(node_->get_logger(),
      "Residual Dynamics enabled (alpha=%.2f, path=%s)",
      params_.residual_alpha, params_.residual_weights_path.c_str());
  } else {
    model = MotionModelFactory::create(params_.motion_model, params_);
  }
  dynamics_ = std::make_unique<BatchDynamicsWrapper>(params_, std::move(model));

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
    std::make_unique<PreferForwardCost>(params_.prefer_forward_weight,
      params_.prefer_forward_linear_ratio, params_.prefer_forward_velocity_incentive)
  );

  // CostmapObstacleCost 추가 (비소유 포인터 저장)
  if (params_.use_costmap_cost) {
    auto costmap_cost = std::make_unique<CostmapObstacleCost>(
      params_.obstacle_weight, params_.costmap_lethal_cost, params_.costmap_critical_cost,
      params_.costmap_eval_stride);
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

  // CBF 초기화
  if (params_.cbf_enabled) {
    barrier_set_ = BarrierFunctionSet(
      params_.cbf_robot_radius, params_.cbf_safety_margin,
      params_.cbf_activation_distance);

    // CBFCost를 비용 함수에 추가 (soft 유도, horizon_discount 지원)
    cost_function_->addCost(std::make_unique<CBFCost>(
      &barrier_set_, params_.cbf_cost_weight, params_.cbf_gamma, params_.dt,
      params_.cbf_horizon_discount));

    // CBF Safety Filter 초기화 (hard 보장)
    if (params_.cbf_use_safety_filter) {
      int nu_dim = dynamics_->model().controlDim();
      Eigen::VectorXd u_min(nu_dim), u_max(nu_dim);
      bool is_nc = (params_.motion_model == "non_coaxial_swerve");
      bool is_ackermann = (params_.motion_model == "ackermann");
      if (is_nc) {
        // Non-Coaxial: [v, omega, delta_dot]
        u_min << params_.v_min, params_.omega_min, -params_.max_steering_rate;
        u_max << params_.v_max, params_.omega_max,  params_.max_steering_rate;
      } else if (is_ackermann) {
        // Ackermann: [v, delta_dot]
        u_min << params_.v_min, -params_.max_steering_rate;
        u_max << params_.v_max,  params_.max_steering_rate;
      } else if (nu_dim >= 3) {
        // Swerve: [vx, vy, omega]
        double effective_vy_max = (params_.vy_max > 0) ? params_.vy_max : params_.v_max;
        u_min << params_.v_min, -effective_vy_max, params_.omega_min;
        u_max << params_.v_max,  effective_vy_max, params_.omega_max;
      } else {
        // DiffDrive: [v, omega]
        u_min << params_.v_min, params_.omega_min;
        u_max << params_.v_max, params_.omega_max;
      }
      cbf_safety_filter_ = std::make_unique<CBFSafetyFilter>(
        &barrier_set_, params_.cbf_gamma, params_.dt, u_min, u_max);
    }

    // BarrierRateCost (BR-MPPI) 추가 — weight > 0 시 활성화
    if (params_.barrier_rate_cost_weight > 0.0) {
      cost_function_->addCost(std::make_unique<BarrierRateCost>(
        &barrier_set_, params_.barrier_rate_cost_weight, params_.dt));
      RCLCPP_INFO(node_->get_logger(),
        "BarrierRateCost enabled (weight=%.1f)", params_.barrier_rate_cost_weight);
    }

    RCLCPP_INFO(node_->get_logger(),
      "CBF enabled (gamma=%.2f, safety_margin=%.2f, cost_weight=%.0f, safety_filter=%s)",
      params_.cbf_gamma, params_.cbf_safety_margin, params_.cbf_cost_weight,
      params_.cbf_use_safety_filter ? "ON" : "OFF");

    // Conformal Predictor 초기화 (동적 안전 마진)
    if (params_.conformal_enabled) {
      ConformalPredictor::Params cp_params;
      cp_params.coverage_probability = params_.conformal_coverage;
      cp_params.window_size = params_.conformal_window_size;
      cp_params.initial_margin = params_.conformal_initial_margin;
      cp_params.min_margin = params_.conformal_min_margin;
      cp_params.max_margin = params_.conformal_max_margin;
      cp_params.decay_rate = params_.conformal_decay_rate;
      conformal_predictor_ = std::make_unique<ConformalPredictor>(cp_params);
      prev_predicted_state_valid_ = false;
      RCLCPP_INFO(node_->get_logger(),
        "ConformalPredictor enabled (coverage=%.2f, window=%d)",
        params_.conformal_coverage, params_.conformal_window_size);
    }
  }

  // Dynamic Obstacle Tracker 초기화
  if (params_.dynamic_obstacle_tracking_enabled) {
    obstacle_tracker_ = std::make_unique<DynamicObstacleTracker>(
      params_.obstacle_cluster_distance,
      params_.obstacle_min_cluster_size,
      params_.obstacle_velocity_ema_alpha,
      params_.obstacle_max_association_distance,
      params_.obstacle_track_timeout);
    RCLCPP_INFO(node_->get_logger(),
      "DynamicObstacleTracker enabled (cluster_dist=%.2f, ema_alpha=%.2f)",
      params_.obstacle_cluster_distance, params_.obstacle_velocity_ema_alpha);
  }

  // VelocityTrackingCost 등록 (경로 방향 속도 추적)
  if (params_.velocity_tracking_weight > 0.0) {
    cost_function_->addCost(std::make_unique<VelocityTrackingCost>(
      params_.velocity_tracking_weight, params_.reference_velocity, params_.dt));
    RCLCPP_INFO(node_->get_logger(),
      "VelocityTrackingCost enabled (weight=%.1f, ref_vel=%.2f m/s)",
      params_.velocity_tracking_weight, params_.reference_velocity);
  }

  // Initialize control sequence (nu from model)
  int nu = dynamics_->model().controlDim();
  control_sequence_ = Eigen::MatrixXd::Zero(params_.N, nu);
  current_velocity_ = Eigen::VectorXd::Zero(nu);

  // Savitzky-Golay Filter 초기화 (EMA 대체)
  if (params_.sg_filter_enabled) {
    sg_filter_ = std::make_unique<SavitzkyGolayFilter>(
      params_.sg_half_window, params_.sg_poly_order, nu);
    RCLCPP_INFO(node_->get_logger(),
      "SG Filter enabled (half_window=%d, poly_order=%d, window_size=%d)",
      params_.sg_half_window, params_.sg_poly_order, sg_filter_->windowSize());
  }

  // Create marker publisher
  marker_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
    plugin_name_ + "/mppi_markers", 10
  );

  // Register parameter callback for dynamic reconfiguration
  param_callback_handle_ = node_->add_on_set_parameters_callback(
    std::bind(&MPPIControllerPlugin::onSetParametersCallback, this, std::placeholders::_1)
  );

  // 성능 최적화: 사전 할당 버퍼 + OpenMP 설정
  allocateBuffers();
  Eigen::setNbThreads(1);  // Eigen 내부 멀티스레딩 비활성화 (OpenMP와 충돌 방지)

  if (params_.num_threads > 0) {
    omp_set_num_threads(params_.num_threads);
    RCLCPP_INFO(node_->get_logger(), "OpenMP threads set to %d", params_.num_threads);
  } else {
    RCLCPP_INFO(node_->get_logger(), "OpenMP threads: auto (%d available)",
      omp_get_max_threads());
  }

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
  current_velocity_ = dynamics_->model().twistToControl(velocity);

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
    Eigen::VectorXd odom_state = poseToState(pose);  // odom 프레임 (후방 체크용)

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

    Eigen::VectorXd current_state = poseToState(pose_in_plan_frame);

    // 2. Prune plan (이미 지나간 waypoint 제거) — map 프레임에서 수행
    prunePlan(current_state);

    // 3. Convert pruned path to reference trajectory (lookahead 기반)
    Eigen::MatrixXd reference_trajectory = pathToReferenceTrajectory(pruned_plan_, current_state);

    // 3.5. Goal distance 갱신 (goal approach 감속에 사용)
    if (!pruned_plan_.poses.empty()) {
      const auto& goal_pose = pruned_plan_.poses.back().pose.position;
      double dx = goal_pose.x - current_state(0);
      double dy = goal_pose.y - current_state(1);
      goal_dist_ = std::sqrt(dx * dx + dy * dy);
    }

    // 4. Update costmap TF (cost_function_ 재생성 없이 TF만 갱신)
    if (params_.use_costmap_cost) {
      updateCostmapObstacles();
    }

    // 4.5. CBF 장애물 갱신 (costmap lethal 셀 → point obstacles)
    if (params_.cbf_enabled) {
      updateCBFObstacles();
    }

    // 4.6. Conformal Predictor 업데이트 (동적 안전 마진)
    if (conformal_predictor_ && params_.cbf_enabled) {
      if (prev_predicted_state_valid_) {
        double error = (current_state.head(2) - prev_predicted_state_.head(2)).norm();
        conformal_predictor_->update(error);
        double margin = params_.cbf_safety_margin + conformal_predictor_->getMargin();
        barrier_set_.updateSafetyMargin(margin);
      }
      // 다음 스텝 예측 저장 (1-step ahead)
      if (control_sequence_.rows() > 0) {
        Eigen::MatrixXd s(1, current_state.size());
        s.row(0) = current_state.transpose();
        Eigen::MatrixXd c(1, control_sequence_.cols());
        c.row(0) = control_sequence_.row(0);
        prev_predicted_state_ = dynamics_->model().propagateBatch(
          s, c, params_.dt).row(0).transpose();
        prev_predicted_state_valid_ = true;
      }
    }

    // 5. Compute optimal control (measure computation time)
    auto start_time = std::chrono::high_resolution_clock::now();
    auto [u_opt, info] = computeControl(current_state, reference_trajectory);
    auto end_time = std::chrono::high_resolution_clock::now();
    double computation_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // 5.3. CBF Safety Filter (활성화된 경우)
    Eigen::VectorXd final_control = u_opt;
    if (params_.cbf_enabled && params_.cbf_use_safety_filter && cbf_safety_filter_) {
      auto [u_safe, filter_info] = cbf_safety_filter_->filter(
        current_state, u_opt, *dynamics_);
      final_control = u_safe;
      info.cbf_used = true;
      info.cbf_filter_info = filter_info;

      if (filter_info.filter_applied) {
        RCLCPP_DEBUG(node_->get_logger(),
          "CBF filter: %d active barriers, qp_success=%s, ||u_diff||=%.4f",
          filter_info.num_active_barriers,
          filter_info.qp_success ? "true" : "false",
          (u_safe - u_opt).norm());
      }
    }

    // 5.4. CBF 시각화 (활성화된 경우)
    if (params_.cbf_enabled && params_.visualize_cbf) {
      publishCBFVisualization(info, current_state);
    }

    // 5.5. Tube-MPPI 피드백 보정 (활성화된 경우)
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

    // 6. 출력 스무딩: SG Filter (우선) 또는 EMA (폴백)
    if (sg_filter_) {
      final_control = sg_filter_->apply(control_sequence_);
      sg_filter_->pushHistory(final_control);
    } else if (prev_cmd_valid_ && params_.control_smoothing_alpha < 1.0) {
      double a = params_.control_smoothing_alpha;
      final_control = a * final_control + (1.0 - a) * prev_cmd_;
    }
    prev_cmd_ = final_control;
    prev_cmd_valid_ = true;

    // 6.2. Apply speed limit & goal slowdown
    double v_cmd = final_control(0);

    // Goal approach 감속: sqrt 거리 비례로 v_max 스케일링 (부드러운 감속)
    if (goal_dist_ < params_.goal_slowdown_dist && params_.goal_slowdown_dist > 1e-6) {
      double ratio = goal_dist_ / params_.goal_slowdown_dist;
      double v_scale = std::clamp(std::sqrt(ratio), 0.2, 1.0);
      double effective_v_max = params_.v_max * v_scale;
      v_cmd = std::clamp(v_cmd, params_.v_min, effective_v_max);
      final_control(0) = v_cmd;
    }

    if (speed_limit_valid_) {
      v_cmd = std::clamp(v_cmd, -speed_limit_, speed_limit_);
      final_control(0) = v_cmd;
    }

    // 6.5. 후방 안전 검사: odom 프레임 좌표로 costmap 조회 (EMA 후에 적용 → 우회 불가)
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
            final_control(0) = 0.0;
          }
        }
      }
    }

    // 6.9. Non-Coaxial / Ackermann: delta 추적 갱신 (controlToTwist 전에 수행)
    if (params_.motion_model == "non_coaxial_swerve") {
      double delta_dot = final_control(2);
      last_delta_ += delta_dot * params_.dt;
      last_delta_ = std::clamp(last_delta_,
        -params_.max_steering_angle, params_.max_steering_angle);
      auto& nc_model = dynamic_cast<NonCoaxialSwerveModel&>(dynamics_->model());
      nc_model.setLastDelta(last_delta_);
    }
    else if (params_.motion_model == "ackermann") {
      double delta_dot = final_control(1);  // nu=2 → 인덱스 1
      last_delta_ += delta_dot * params_.dt;
      last_delta_ = std::clamp(last_delta_,
        -params_.max_steering_angle, params_.max_steering_angle);
      auto& ack_model = dynamic_cast<AckermannModel&>(dynamics_->model());
      ack_model.setLastDelta(last_delta_);
    }

    // 7. Build Twist message (MotionModel 기반 변환)
    cmd_vel.twist = dynamics_->model().controlToTwist(final_control);

    // 8. Publish visualization (plan 프레임에서)
    publishVisualization(info, current_state, reference_trajectory, info.weighted_avg_trajectory, computation_time_ms);

    // 8.5. Collision debug visualization (debug_collision_viz=true일 때만)
    if (params_.debug_collision_viz) {
      publishCollisionDebugVisualization(info, current_state, reference_trajectory, info.weighted_avg_trajectory);
    }

    RCLCPP_DEBUG(
      node_->get_logger(),
      "MPPI: twist.linear.x=%.3f, twist.angular.z=%.3f, min_cost=%.4f, ESS=%.1f/%d",
      cmd_vel.twist.linear.x, cmd_vel.twist.angular.z,
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
  prev_cmd_valid_ = false;
  last_delta_ = 0.0;  // 경로 리셋 시 steering angle 초기화
  if (sg_filter_) { sg_filter_->reset(); }  // SG 필터 이력 초기화
  if (conformal_predictor_) { conformal_predictor_->reset(); }  // Conformal 이력 초기화
  prev_predicted_state_valid_ = false;

  // Reset control sequence for new plan (warm start from zero)
  int nu = dynamics_ ? dynamics_->model().controlDim() : 2;
  control_sequence_ = Eigen::MatrixXd::Zero(params_.N, nu);

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

void MPPIControllerPlugin::allocateBuffers()
{
  int K = params_.K;
  int N = params_.N;
  int nu = dynamics_ ? dynamics_->model().controlDim() : 2;
  int nx = dynamics_ ? dynamics_->model().stateDim() : 3;

  noise_buffer_.resize(K);
  perturbed_buffer_.resize(K);
  trajectory_buffer_.resize(K);

  for (int k = 0; k < K; ++k) {
    noise_buffer_[k] = Eigen::MatrixXd::Zero(N, nu);
    perturbed_buffer_[k] = Eigen::MatrixXd::Zero(N, nu);
    trajectory_buffer_[k] = Eigen::MatrixXd::Zero(N + 1, nx);
  }

  RCLCPP_INFO(node_->get_logger(),
    "Pre-allocated buffers: K=%d, N=%d, nu=%d, nx=%d (%.1f KB)",
    K, N, nu, nx,
    static_cast<double>(K * (N * nu + N * nu + (N + 1) * nx) * sizeof(double)) / 1024.0);
}

std::pair<Eigen::VectorXd, MPPIInfo> MPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory
)
{
  int N = params_.N;
  int K = params_.K;
  int nu = dynamics_->model().controlDim();
  int nx = dynamics_->model().stateDim();

  // 1. Shift previous control sequence (warm start)
  for (int t = 0; t < N - 1; ++t) {
    control_sequence_.row(t) = control_sequence_.row(t + 1);
  }
  control_sequence_.row(N - 1) = control_sequence_.row(N - 2);

  // 2. Sample noise (in-place — 힙 할당 0)
  sampler_->sampleInPlace(noise_buffer_, K, N, nu);

  // 2.5. Goal 근처 noise 스케일링 (sampler 재생성 없이 정밀 제어)
  if (goal_dist_ < params_.goal_slowdown_dist && params_.goal_slowdown_dist > 1e-6) {
    double ratio = goal_dist_ / params_.goal_slowdown_dist;
    double noise_scale = std::clamp(std::sqrt(ratio), 0.2, 1.0);
    for (int k = 0; k < K; ++k) {
      noise_buffer_[k] *= noise_scale;
    }
  }

  // 3. Add noise to control sequence and clip (Exploitation/Exploration 분할, in-place)
  if (static_cast<int>(perturbed_buffer_.size()) != K) {
    perturbed_buffer_.resize(K, Eigen::MatrixXd::Zero(N, nu));
  }
  int K_exploit = static_cast<int>((1.0 - params_.exploration_ratio) * K);

  for (int k = 0; k < K; ++k) {
    if (k < K_exploit) {
      // Exploitation: 이전 최적 시퀀스 + 노이즈
      perturbed_buffer_[k].noalias() = control_sequence_ + noise_buffer_[k];
    } else {
      // Exploration: 순수 노이즈 (warm-start 없이)
      perturbed_buffer_[k] = noise_buffer_[k];
    }
    perturbed_buffer_[k] = dynamics_->clipControls(perturbed_buffer_[k]);
  }

  // 4. Batch rollout (in-place — 버퍼 재사용)
  dynamics_->rolloutBatchInPlace(
    current_state,
    perturbed_buffer_,
    params_.dt,
    trajectory_buffer_
  );
  const auto& trajectories = trajectory_buffer_;

  // 5. Compute costs (디버그 모드: 비용 분해 포함)
  Eigen::VectorXd costs;
  CostBreakdown cost_breakdown;
  if (params_.debug_collision_viz) {
    cost_breakdown = cost_function_->computeDetailed(
      trajectories, perturbed_buffer_, reference_trajectory);
    costs = cost_breakdown.total_costs;
  } else {
    costs = cost_function_->compute(
      trajectories, perturbed_buffer_, reference_trajectory);
  }

  // 5.5. Information-Theoretic 정규화 (KL-divergence 기반)
  if (params_.it_alpha < 1.0) {
    Eigen::VectorXd sigma_inv = params_.noise_sigma.cwiseInverse().cwiseAbs2();
    for (int k = 0; k < K; ++k) {
      double it_cost = 0.0;
      for (int t = 0; t < N; ++t) {
        Eigen::VectorXd u_prev_t = control_sequence_.row(t).transpose();
        Eigen::VectorXd u_k_t = perturbed_buffer_[k].row(t).transpose();
        it_cost += u_prev_t.dot(sigma_inv.cwiseProduct(u_k_t));
      }
      costs(k) += params_.lambda * (1.0 - params_.it_alpha) * it_cost;
    }
  }

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

  // 7. Update control sequence with weighted average of noise (OpenMP 스레드 로컬)
  {
    int n_threads = 1;
    #ifdef _OPENMP
    n_threads = omp_get_max_threads();
    #endif
    if (K <= 4096) { n_threads = 1; }

    std::vector<Eigen::MatrixXd> thread_noise_accum(n_threads, Eigen::MatrixXd::Zero(N, nu));
    #pragma omp parallel if(K > 4096)
    {
      int tid = 0;
      #ifdef _OPENMP
      tid = omp_get_thread_num();
      #endif
      #pragma omp for schedule(static)
      for (int k = 0; k < K; ++k) {
        thread_noise_accum[tid].noalias() += weights(k) * noise_buffer_[k];
      }
    }
    Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
    for (int t = 0; t < n_threads; ++t) {
      weighted_noise += thread_noise_accum[t];
    }
    control_sequence_ += weighted_noise;
  }

  // Clip updated control sequence
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // 8. Extract optimal control (first timestep)
  Eigen::VectorXd u_opt = control_sequence_.row(0).transpose();

  // Compute weighted average trajectory (OpenMP 스레드 로컬)
  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, nx);
  {
    int n_threads = 1;
    #ifdef _OPENMP
    n_threads = omp_get_max_threads();
    #endif
    if (K <= 4096) { n_threads = 1; }

    std::vector<Eigen::MatrixXd> thread_traj_accum(n_threads, Eigen::MatrixXd::Zero(N + 1, nx));
    #pragma omp parallel if(K > 4096)
    {
      int tid = 0;
      #ifdef _OPENMP
      tid = omp_get_thread_num();
      #endif
      #pragma omp for schedule(static)
      for (int k = 0; k < K; ++k) {
        thread_traj_accum[tid].noalias() += weights(k) * trajectories[k];
      }
    }
    for (int t = 0; t < n_threads; ++t) {
      weighted_traj += thread_traj_accum[t];
    }
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

  // 디버그 비용 분해
  if (params_.debug_collision_viz) {
    info.cost_breakdown = cost_breakdown;
  }

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

Eigen::VectorXd MPPIControllerPlugin::poseToState(const geometry_msgs::msg::PoseStamped& pose)
{
  int nx = dynamics_ ? dynamics_->model().stateDim() : 3;
  Eigen::VectorXd state = Eigen::VectorXd::Zero(nx);
  state(0) = pose.pose.position.x;
  state(1) = pose.pose.position.y;
  if (nx >= 3) {
    state(2) = quaternionToYaw(pose.pose.orientation);
  }
  // NonCoaxialSwerve (nx=4): state(3)=δ — pose에서 추출 불가, last_delta_로 추적
  if (nx >= 4) {
    state(3) = last_delta_;
  }
  return state;
}

void MPPIControllerPlugin::prunePlan(const Eigen::VectorXd& current_state)
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
  const nav_msgs::msg::Path& path, const Eigen::VectorXd& /*current_state*/)
{
  if (path.poses.empty()) {
    RCLCPP_WARN(node_->get_logger(), "Empty path provided");
    int nx = dynamics_ ? dynamics_->model().stateDim() : 3;
    return Eigen::MatrixXd::Zero(params_.N + 1, nx);
  }

  int nx = dynamics_ ? dynamics_->model().stateDim() : 3;
  int path_size = path.poses.size();
  Eigen::MatrixXd reference = Eigen::MatrixXd::Zero(params_.N + 1, nx);

  // lookahead 거리 계산: 0 = auto (v_max * N * dt)
  double lookahead = params_.lookahead_dist;
  if (lookahead <= 0.0) {
    lookahead = params_.v_max * params_.N * params_.dt;
  }

  // 경로를 따라 누적 arc-length 계산
  std::vector<double> arc_lengths(path_size, 0.0);
  for (int i = 1; i < path_size; ++i) {
    double dx = path.poses[i].pose.position.x - path.poses[i - 1].pose.position.x;
    double dy = path.poses[i].pose.position.y - path.poses[i - 1].pose.position.y;
    arc_lengths[i] = arc_lengths[i - 1] + std::sqrt(dx * dx + dy * dy);
  }

  double total_path_length = arc_lengths.back();
  double effective_lookahead = std::max(
    params_.min_lookahead,
    std::min(lookahead, total_path_length)
  );
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
      // 경로 끝 도달: 마지막 점 반복 (goal orientation 사용)
      reference(t, 0) = path.poses[path_size - 1].pose.position.x;
      reference(t, 1) = path.poses[path_size - 1].pose.position.y;
      reference(t, 2) = quaternionToYaw(path.poses[path_size - 1].pose.orientation);
    } else {
      // 구간 내 선형 보간
      double seg_len = arc_lengths[path_idx + 1] - arc_lengths[path_idx];
      double alpha = (seg_len > 1e-6) ?
        (target_arc - arc_lengths[path_idx]) / seg_len : 0.0;

      double x_interp = (1.0 - alpha) * path.poses[path_idx].pose.position.x +
                        alpha * path.poses[path_idx + 1].pose.position.x;
      double y_interp = (1.0 - alpha) * path.poses[path_idx].pose.position.y +
                        alpha * path.poses[path_idx + 1].pose.position.y;
      reference(t, 0) = x_interp;
      reference(t, 1) = y_interp;

      // 경로 접선 방향을 theta로 사용 (nav2 planner orientation은 goal 고정이므로)
      double dx = path.poses[path_idx + 1].pose.position.x
                - path.poses[path_idx].pose.position.x;
      double dy = path.poses[path_idx + 1].pose.position.y
                - path.poses[path_idx].pose.position.y;
      double tangent_len = std::sqrt(dx * dx + dy * dy);
      if (tangent_len > 1e-6) {
        reference(t, 2) = std::atan2(dy, dx);
      } else {
        // 정지 구간: 이전 스텝의 heading 유지
        reference(t, 2) = (t > 0) ? reference(t - 1, 2) : 0.0;
      }
    }
  }

  // Reference theta smoothing (circular moving average)
  if (params_.ref_theta_smooth_window >= 3) {
    int half_w = params_.ref_theta_smooth_window / 2;
    int num_pts = params_.N + 1;
    std::vector<double> raw_theta(num_pts);
    for (int t = 0; t < num_pts; ++t) raw_theta[t] = reference(t, 2);

    for (int t = 0; t < num_pts; ++t) {
      double sin_sum = 0.0, cos_sum = 0.0;
      int lo = std::max(0, t - half_w);
      int hi = std::min(num_pts - 1, t + half_w);
      for (int j = lo; j <= hi; ++j) {
        sin_sum += std::sin(raw_theta[j]);
        cos_sum += std::cos(raw_theta[j]);
      }
      reference(t, 2) = std::atan2(sin_sum, cos_sum);
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

void MPPIControllerPlugin::updateCBFObstacles()
{
  if (!costmap_ros_) {
    return;
  }

  auto costmap = costmap_ros_->getCostmap();
  if (!costmap) {
    return;
  }

  // Costmap lethal 셀을 point obstacles로 변환
  std::vector<Eigen::Vector3d> obstacles;
  double resolution = costmap->getResolution();
  unsigned int size_x = costmap->getSizeInCellsX();
  unsigned int size_y = costmap->getSizeInCellsY();

  // costmap 프레임 → plan 프레임 변환 준비
  // local costmap은 odom 프레임이지만 current_state는 map 프레임이므로
  // 장애물 좌표를 map 프레임으로 변환해야 CBF 거리 계산이 정확함
  std::string costmap_frame = costmap_ros_->getGlobalFrameID();
  std::string plan_frame = global_plan_.header.frame_id;
  if (plan_frame.empty()) { plan_frame = "map"; }

  bool need_transform = (costmap_frame != plan_frame);
  geometry_msgs::msg::TransformStamped tf_costmap_to_plan;
  if (need_transform && tf_buffer_) {
    try {
      tf_costmap_to_plan = tf_buffer_->lookupTransform(
        plan_frame, costmap_frame, tf2::TimePointZero);
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 2000,
        "CBF: Failed to get %s→%s transform: %s. Using costmap frame directly.",
        costmap_frame.c_str(), plan_frame.c_str(), ex.what());
      need_transform = false;
    }
  }

  for (unsigned int mx = 0; mx < size_x; ++mx) {
    for (unsigned int my = 0; my < size_y; ++my) {
      unsigned char cost = costmap->getCost(mx, my);
      if (cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
        double wx, wy;
        costmap->mapToWorld(mx, my, wx, wy);

        if (need_transform) {
          // odom→map 프레임 변환
          geometry_msgs::msg::PointStamped pt_in, pt_out;
          pt_in.point.x = wx;
          pt_in.point.y = wy;
          pt_in.point.z = 0.0;
          tf2::doTransform(pt_in, pt_out, tf_costmap_to_plan);
          wx = pt_out.point.x;
          wy = pt_out.point.y;
        }

        obstacles.emplace_back(wx, wy, resolution * 0.5);
      }
    }
  }

  // Dynamic Obstacle Tracker로 속도 추정 → C3BF 자동 활성화
  if (params_.dynamic_obstacle_tracking_enabled && obstacle_tracker_) {
    std::vector<Eigen::Vector2d> lethal_cells;
    lethal_cells.reserve(obstacles.size());
    for (const auto& obs : obstacles) {
      lethal_cells.emplace_back(obs(0), obs(1));
    }
    auto [tracked_obs, tracked_vels] = obstacle_tracker_->process(
      lethal_cells, resolution * 0.5, node_->now().seconds());
    barrier_set_.setObstaclesWithVelocity(tracked_obs, tracked_vels);
  } else {
    barrier_set_.setObstacles(obstacles);
  }
}

void MPPIControllerPlugin::publishVisualization(
  const MPPIInfo& info,
  const Eigen::VectorXd& current_state,
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

  // Motion model
  node_->declare_parameter(prefix + "motion_model", params_.motion_model);

  // MPPI parameters
  node_->declare_parameter(prefix + "N", params_.N);
  node_->declare_parameter(prefix + "dt", params_.dt);
  node_->declare_parameter(prefix + "K", params_.K);
  node_->declare_parameter(prefix + "lambda", params_.lambda);

  // Noise parameters (vy는 swerve, delta_dot은 non_coaxial에서 사용)
  node_->declare_parameter(prefix + "noise_sigma_v", params_.noise_sigma(0));
  node_->declare_parameter(prefix + "noise_sigma_vy", 0.5);
  node_->declare_parameter(prefix + "noise_sigma_omega", params_.noise_sigma(1));
  node_->declare_parameter(prefix + "noise_sigma_delta_dot", 0.3);  // non_coaxial 전용

  // Control limits
  node_->declare_parameter(prefix + "v_max", params_.v_max);
  node_->declare_parameter(prefix + "v_min", params_.v_min);
  node_->declare_parameter(prefix + "omega_max", params_.omega_max);
  node_->declare_parameter(prefix + "omega_min", params_.omega_min);
  node_->declare_parameter(prefix + "vy_max", params_.vy_max);

  // Cost weights - Q
  node_->declare_parameter(prefix + "Q_x", params_.Q(0, 0));
  node_->declare_parameter(prefix + "Q_y", params_.Q(1, 1));
  node_->declare_parameter(prefix + "Q_theta", params_.Q(2, 2));

  // Cost weights - Qf
  node_->declare_parameter(prefix + "Qf_x", params_.Qf(0, 0));
  node_->declare_parameter(prefix + "Qf_y", params_.Qf(1, 1));
  node_->declare_parameter(prefix + "Qf_theta", params_.Qf(2, 2));

  // Cost weights - R (vy는 swerve, delta_dot은 non_coaxial에서 사용)
  node_->declare_parameter(prefix + "R_v", params_.R(0, 0));
  node_->declare_parameter(prefix + "R_vy", 0.1);
  node_->declare_parameter(prefix + "R_omega", params_.R(1, 1));
  node_->declare_parameter(prefix + "R_delta_dot", 0.5);  // non_coaxial 전용

  // Cost weights - R_rate (vy는 swerve, delta_dot은 non_coaxial에서 사용)
  node_->declare_parameter(prefix + "R_rate_v", params_.R_rate(0, 0));
  node_->declare_parameter(prefix + "R_rate_vy", 1.0);
  node_->declare_parameter(prefix + "R_rate_omega", params_.R_rate(1, 1));
  node_->declare_parameter(prefix + "R_rate_delta_dot", 1.0);  // non_coaxial 전용

  // Obstacle avoidance
  node_->declare_parameter(prefix + "obstacle_weight", params_.obstacle_weight);
  node_->declare_parameter(prefix + "safety_distance", params_.safety_distance);

  // Forward preference
  node_->declare_parameter(prefix + "prefer_forward_weight", params_.prefer_forward_weight);
  node_->declare_parameter(prefix + "prefer_forward_linear_ratio", params_.prefer_forward_linear_ratio);
  node_->declare_parameter(prefix + "prefer_forward_velocity_incentive", params_.prefer_forward_velocity_incentive);

  // Control smoothing
  node_->declare_parameter(prefix + "control_smoothing_alpha", params_.control_smoothing_alpha);

  // Savitzky-Golay Filter
  node_->declare_parameter(prefix + "sg_filter_enabled", params_.sg_filter_enabled);
  node_->declare_parameter(prefix + "sg_half_window", params_.sg_half_window);
  node_->declare_parameter(prefix + "sg_poly_order", params_.sg_poly_order);

  // Information-Theoretic 정규화
  node_->declare_parameter(prefix + "it_alpha", params_.it_alpha);

  // Exploitation/Exploration 분할
  node_->declare_parameter(prefix + "exploration_ratio", params_.exploration_ratio);

  // Costmap obstacle cost
  node_->declare_parameter(prefix + "use_costmap_cost", params_.use_costmap_cost);
  node_->declare_parameter(prefix + "costmap_lethal_cost", params_.costmap_lethal_cost);
  node_->declare_parameter(prefix + "costmap_critical_cost", params_.costmap_critical_cost);
  node_->declare_parameter(prefix + "lookahead_dist", params_.lookahead_dist);
  node_->declare_parameter(prefix + "min_lookahead", params_.min_lookahead);
  node_->declare_parameter(prefix + "goal_slowdown_dist", params_.goal_slowdown_dist);
  node_->declare_parameter(prefix + "ref_theta_smooth_window", params_.ref_theta_smooth_window);

  // Velocity Tracking Cost
  node_->declare_parameter(prefix + "velocity_tracking_weight", params_.velocity_tracking_weight);
  node_->declare_parameter(prefix + "reference_velocity", params_.reference_velocity);

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
  node_->declare_parameter(prefix + "tube_nominal_reset_threshold", params_.tube_nominal_reset_threshold);
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

  // Biased-MPPI (RA-L 2024)
  node_->declare_parameter(prefix + "biased_enabled", params_.biased_enabled);
  node_->declare_parameter(prefix + "bias_ratio", params_.bias_ratio);
  node_->declare_parameter(prefix + "biased_braking", params_.biased_braking);
  node_->declare_parameter(prefix + "biased_goto_goal", params_.biased_goto_goal);
  node_->declare_parameter(prefix + "biased_path_following", params_.biased_path_following);
  node_->declare_parameter(prefix + "biased_previous_solution", params_.biased_previous_solution);
  node_->declare_parameter(prefix + "biased_goto_goal_gain", params_.biased_goto_goal_gain);
  node_->declare_parameter(prefix + "biased_path_following_gain", params_.biased_path_following_gain);

  // Non-Coaxial Swerve / Ackermann 공통 파라미터
  node_->declare_parameter(prefix + "max_steering_rate", params_.max_steering_rate);
  node_->declare_parameter(prefix + "max_steering_angle", params_.max_steering_angle);
  node_->declare_parameter(prefix + "wheelbase", params_.wheelbase);

  // CBF (Control Barrier Function)
  node_->declare_parameter(prefix + "cbf_enabled", params_.cbf_enabled);
  node_->declare_parameter(prefix + "cbf_gamma", params_.cbf_gamma);
  node_->declare_parameter(prefix + "cbf_safety_margin", params_.cbf_safety_margin);
  node_->declare_parameter(prefix + "cbf_robot_radius", params_.cbf_robot_radius);
  node_->declare_parameter(prefix + "cbf_activation_distance", params_.cbf_activation_distance);
  node_->declare_parameter(prefix + "cbf_cost_weight", params_.cbf_cost_weight);
  node_->declare_parameter(prefix + "cbf_use_safety_filter", params_.cbf_use_safety_filter);

  // Residual Dynamics
  node_->declare_parameter(prefix + "residual_enabled", params_.residual_enabled);
  node_->declare_parameter(prefix + "residual_weights_path", params_.residual_weights_path);
  node_->declare_parameter(prefix + "residual_alpha", params_.residual_alpha);

  // Ensemble Dynamics
  node_->declare_parameter(prefix + "ensemble_enabled", params_.ensemble_enabled);
  node_->declare_parameter(prefix + "ensemble_weights_dir", params_.ensemble_weights_dir);
  node_->declare_parameter(prefix + "ensemble_size", params_.ensemble_size);
  node_->declare_parameter(prefix + "ensemble_alpha", params_.ensemble_alpha);
  node_->declare_parameter(prefix + "uncertainty_cost_weight", params_.uncertainty_cost_weight);

  // C3BF
  node_->declare_parameter(prefix + "c3bf_enabled", params_.c3bf_enabled);
  node_->declare_parameter(prefix + "c3bf_alpha_safe", params_.c3bf_alpha_safe);
  node_->declare_parameter(prefix + "c3bf_cost_weight", params_.c3bf_cost_weight);

  // Adaptive Shield
  node_->declare_parameter(prefix + "adaptive_shield_enabled", params_.adaptive_shield_enabled);
  node_->declare_parameter(prefix + "adaptive_shield_alpha_min", params_.adaptive_shield_alpha_min);
  node_->declare_parameter(prefix + "adaptive_shield_alpha_max", params_.adaptive_shield_alpha_max);
  node_->declare_parameter(prefix + "adaptive_shield_k_d", params_.adaptive_shield_k_d);
  node_->declare_parameter(prefix + "adaptive_shield_k_v", params_.adaptive_shield_k_v);

  // Horizon-Weighted CBF
  node_->declare_parameter(prefix + "cbf_horizon_discount", params_.cbf_horizon_discount);

  // Online Data Buffer
  node_->declare_parameter(prefix + "online_data_enabled", params_.online_data_enabled);
  node_->declare_parameter(prefix + "online_data_capacity", params_.online_data_capacity);
  node_->declare_parameter(prefix + "online_data_export_path", params_.online_data_export_path);

  // Online Learning (Model Reload)
  node_->declare_parameter(prefix + "model_reload_enabled", params_.model_reload_enabled);
  node_->declare_parameter(prefix + "model_reload_interval_sec", params_.model_reload_interval_sec);

  // Safety Enhancement: BR-MPPI
  node_->declare_parameter(prefix + "barrier_rate_cost_weight", params_.barrier_rate_cost_weight);

  // Safety Enhancement: Conformal Predictor
  node_->declare_parameter(prefix + "conformal_enabled", params_.conformal_enabled);
  node_->declare_parameter(prefix + "conformal_coverage", params_.conformal_coverage);
  node_->declare_parameter(prefix + "conformal_window_size", params_.conformal_window_size);
  node_->declare_parameter(prefix + "conformal_initial_margin", params_.conformal_initial_margin);
  node_->declare_parameter(prefix + "conformal_min_margin", params_.conformal_min_margin);
  node_->declare_parameter(prefix + "conformal_max_margin", params_.conformal_max_margin);
  node_->declare_parameter(prefix + "conformal_decay_rate", params_.conformal_decay_rate);

  // LP-MPPI (Low-Pass Filtering)
  node_->declare_parameter(prefix + "lp_enabled", params_.lp_enabled);
  node_->declare_parameter(prefix + "lp_cutoff_frequency", params_.lp_cutoff_frequency);
  node_->declare_parameter(prefix + "lp_filter_all_samples", params_.lp_filter_all_samples);

  // Halton-MPPI (Halton low-discrepancy sequence)
  node_->declare_parameter(prefix + "halton_enabled", params_.halton_enabled);
  node_->declare_parameter(prefix + "halton_beta", params_.halton_beta);
  node_->declare_parameter(prefix + "halton_sequence_offset", params_.halton_sequence_offset);

  // Feedback-MPPI (F-MPPI, Riccati feedback gains)
  node_->declare_parameter(prefix + "feedback_mppi_enabled", params_.feedback_mppi_enabled);
  node_->declare_parameter(prefix + "feedback_gain_scale", params_.feedback_gain_scale);
  node_->declare_parameter(prefix + "feedback_recompute_interval", params_.feedback_recompute_interval);
  node_->declare_parameter(prefix + "feedback_regularization", params_.feedback_regularization);

  // Dynamic Obstacle Tracker
  node_->declare_parameter(prefix + "dynamic_obstacle_tracking_enabled", params_.dynamic_obstacle_tracking_enabled);
  node_->declare_parameter(prefix + "obstacle_cluster_distance", params_.obstacle_cluster_distance);
  node_->declare_parameter(prefix + "obstacle_min_cluster_size", params_.obstacle_min_cluster_size);
  node_->declare_parameter(prefix + "obstacle_velocity_ema_alpha", params_.obstacle_velocity_ema_alpha);
  node_->declare_parameter(prefix + "obstacle_max_association_distance", params_.obstacle_max_association_distance);
  node_->declare_parameter(prefix + "obstacle_track_timeout", params_.obstacle_track_timeout);

  // Safety Enhancement: Shield-MPPI
  node_->declare_parameter(prefix + "shield_cbf_stride", params_.shield_cbf_stride);
  node_->declare_parameter(prefix + "shield_max_iterations", params_.shield_max_iterations);

  // CLF-CBF-QP
  node_->declare_parameter(prefix + "clf_cbf_enabled", params_.clf_cbf_enabled);
  node_->declare_parameter(prefix + "clf_decay_rate", params_.clf_decay_rate);
  node_->declare_parameter(prefix + "clf_slack_penalty", params_.clf_slack_penalty);
  node_->declare_parameter(prefix + "clf_P_scale", params_.clf_P_scale);

  // CBF 합성
  node_->declare_parameter(prefix + "cbf_composition_enabled", params_.cbf_composition_enabled);
  node_->declare_parameter(prefix + "cbf_composition_method", params_.cbf_composition_method);
  node_->declare_parameter(prefix + "cbf_composition_alpha", params_.cbf_composition_alpha);

  // Predictive Safety Filter
  node_->declare_parameter(prefix + "predictive_safety_enabled", params_.predictive_safety_enabled);
  node_->declare_parameter(prefix + "predictive_safety_horizon", params_.predictive_safety_horizon);
  node_->declare_parameter(prefix + "predictive_safety_decay", params_.predictive_safety_decay);
  node_->declare_parameter(prefix + "predictive_safety_max_iterations", params_.predictive_safety_max_iterations);

  // Covariance Steering MPPI (CS-MPPI)
  node_->declare_parameter(prefix + "cs_enabled", params_.cs_enabled);
  node_->declare_parameter(prefix + "cs_scale_min", params_.cs_scale_min);
  node_->declare_parameter(prefix + "cs_scale_max", params_.cs_scale_max);
  node_->declare_parameter(prefix + "cs_feedback_enabled", params_.cs_feedback_enabled);
  node_->declare_parameter(prefix + "cs_feedback_gain", params_.cs_feedback_gain);

  // pi-MPPI (Projection MPPI)
  node_->declare_parameter(prefix + "pi_enabled", params_.pi_enabled);
  node_->declare_parameter(prefix + "pi_admm_iterations", params_.pi_admm_iterations);
  node_->declare_parameter(prefix + "pi_admm_rho", params_.pi_admm_rho);
  node_->declare_parameter(prefix + "pi_derivative_order", params_.pi_derivative_order);
  node_->declare_parameter(prefix + "pi_rate_max_v", params_.pi_rate_max_v);
  node_->declare_parameter(prefix + "pi_rate_max_omega", params_.pi_rate_max_omega);
  node_->declare_parameter(prefix + "pi_rate_max_vy", params_.pi_rate_max_vy);
  node_->declare_parameter(prefix + "pi_accel_max_v", params_.pi_accel_max_v);
  node_->declare_parameter(prefix + "pi_accel_max_omega", params_.pi_accel_max_omega);
  node_->declare_parameter(prefix + "pi_accel_max_vy", params_.pi_accel_max_vy);

  // Hybrid Swerve MPPI (MPPI-H)
  node_->declare_parameter(prefix + "hybrid_enabled", params_.hybrid_enabled);
  node_->declare_parameter(prefix + "hybrid_cdist_threshold", params_.hybrid_cdist_threshold);
  node_->declare_parameter(prefix + "hybrid_cangle_threshold", params_.hybrid_cangle_threshold);
  node_->declare_parameter(prefix + "hybrid_hysteresis_count", params_.hybrid_hysteresis_count);
  node_->declare_parameter(prefix + "hybrid_lf", params_.hybrid_lf);
  node_->declare_parameter(prefix + "hybrid_lr", params_.hybrid_lr);
  node_->declare_parameter(prefix + "hybrid_dl", params_.hybrid_dl);
  node_->declare_parameter(prefix + "hybrid_dr", params_.hybrid_dr);
  node_->declare_parameter(prefix + "hybrid_v_wheel_max", params_.hybrid_v_wheel_max);
  node_->declare_parameter(prefix + "hybrid_delta_max", params_.hybrid_delta_max);
  node_->declare_parameter(prefix + "hybrid_noise_sigma_vfl", params_.hybrid_noise_sigma_vfl);
  node_->declare_parameter(prefix + "hybrid_noise_sigma_vrr", params_.hybrid_noise_sigma_vrr);
  node_->declare_parameter(prefix + "hybrid_noise_sigma_dfl", params_.hybrid_noise_sigma_dfl);
  node_->declare_parameter(prefix + "hybrid_noise_sigma_drr", params_.hybrid_noise_sigma_drr);
  node_->declare_parameter(prefix + "hybrid_R_vfl", params_.hybrid_R_vfl);
  node_->declare_parameter(prefix + "hybrid_R_vrr", params_.hybrid_R_vrr);
  node_->declare_parameter(prefix + "hybrid_R_dfl", params_.hybrid_R_dfl);
  node_->declare_parameter(prefix + "hybrid_R_drr", params_.hybrid_R_drr);

  // iLQR Warm-Start
  node_->declare_parameter(prefix + "ilqr_enabled", params_.ilqr_enabled);
  node_->declare_parameter(prefix + "ilqr_max_iterations", params_.ilqr_max_iterations);
  node_->declare_parameter(prefix + "ilqr_regularization", params_.ilqr_regularization);
  node_->declare_parameter(prefix + "ilqr_line_search_steps", params_.ilqr_line_search_steps);
  node_->declare_parameter(prefix + "ilqr_cost_tolerance", params_.ilqr_cost_tolerance);

  // Auto-Selector MPPI
  node_->declare_parameter(prefix + "auto_selector_enabled", params_.auto_selector_enabled);
  node_->declare_parameter(prefix + "auto_selector_safety_threshold", params_.auto_selector_safety_threshold);
  node_->declare_parameter(prefix + "auto_selector_recovery_threshold", params_.auto_selector_recovery_threshold);
  node_->declare_parameter(prefix + "auto_selector_fast_threshold", params_.auto_selector_fast_threshold);
  node_->declare_parameter(prefix + "auto_selector_precision_dist", params_.auto_selector_precision_dist);
  node_->declare_parameter(prefix + "auto_selector_hysteresis", params_.auto_selector_hysteresis);
  node_->declare_parameter(prefix + "auto_selector_smoothing_alpha", params_.auto_selector_smoothing_alpha);

  // Robust MPPI
  node_->declare_parameter(prefix + "robust_enabled", params_.robust_enabled);
  node_->declare_parameter(prefix + "robust_alpha", params_.robust_alpha);
  node_->declare_parameter(prefix + "robust_penalty", params_.robust_penalty);
  node_->declare_parameter(prefix + "robust_wasserstein_radius", params_.robust_wasserstein_radius);
  node_->declare_parameter(prefix + "robust_adaptive_alpha", params_.robust_adaptive_alpha);

  // IT-MPPI (Information-Theoretic MPPI)
  node_->declare_parameter(prefix + "it_mppi_enabled", params_.it_mppi_enabled);
  node_->declare_parameter(prefix + "it_exploration_weight", params_.it_exploration_weight);
  node_->declare_parameter(prefix + "it_kl_weight", params_.it_kl_weight);
  node_->declare_parameter(prefix + "it_diversity_threshold", params_.it_diversity_threshold);
  node_->declare_parameter(prefix + "it_adaptive_exploration", params_.it_adaptive_exploration);
  node_->declare_parameter(prefix + "it_exploration_decay", params_.it_exploration_decay);

  // CEM-MPPI
  node_->declare_parameter(prefix + "cem_enabled", params_.cem_enabled);
  node_->declare_parameter(prefix + "cem_iterations", params_.cem_iterations);
  node_->declare_parameter(prefix + "cem_elite_ratio", params_.cem_elite_ratio);
  node_->declare_parameter(prefix + "cem_momentum", params_.cem_momentum);
  node_->declare_parameter(prefix + "cem_sigma_min", params_.cem_sigma_min);
  node_->declare_parameter(prefix + "cem_sigma_decay", params_.cem_sigma_decay);
  node_->declare_parameter(prefix + "cem_adaptive_enabled", params_.cem_adaptive_enabled);
  node_->declare_parameter(prefix + "cem_adaptive_cost_tol", params_.cem_adaptive_cost_tol);
  node_->declare_parameter(prefix + "cem_adaptive_min_iter", params_.cem_adaptive_min_iter);
  node_->declare_parameter(prefix + "cem_adaptive_max_iter", params_.cem_adaptive_max_iter);

  // Trajectory Library MPPI
  node_->declare_parameter(prefix + "traj_library_enabled", params_.traj_library_enabled);
  node_->declare_parameter(prefix + "traj_library_ratio", params_.traj_library_ratio);
  node_->declare_parameter(prefix + "traj_library_perturbation", params_.traj_library_perturbation);
  node_->declare_parameter(prefix + "traj_library_adaptive", params_.traj_library_adaptive);
  node_->declare_parameter(prefix + "traj_library_num_per_primitive", params_.traj_library_num_per_primitive);

  // Receding Horizon MPPI (RH-MPPI)
  node_->declare_parameter(prefix + "rh_mppi_enabled", params_.rh_mppi_enabled);
  node_->declare_parameter(prefix + "rh_N_min", params_.rh_N_min);
  node_->declare_parameter(prefix + "rh_N_max", params_.rh_N_max);
  node_->declare_parameter(prefix + "rh_speed_weight", params_.rh_speed_weight);
  node_->declare_parameter(prefix + "rh_obstacle_weight", params_.rh_obstacle_weight);
  node_->declare_parameter(prefix + "rh_error_weight", params_.rh_error_weight);
  node_->declare_parameter(prefix + "rh_obs_dist_threshold", params_.rh_obs_dist_threshold);
  node_->declare_parameter(prefix + "rh_error_threshold", params_.rh_error_threshold);
  node_->declare_parameter(prefix + "rh_smoothing_alpha", params_.rh_smoothing_alpha);

  // Constrained MPPI (Augmented Lagrangian)
  node_->declare_parameter(prefix + "constrained_enabled", params_.constrained_enabled);
  node_->declare_parameter(prefix + "constrained_mu_init", params_.constrained_mu_init);
  node_->declare_parameter(prefix + "constrained_mu_growth", params_.constrained_mu_growth);
  node_->declare_parameter(prefix + "constrained_mu_max", params_.constrained_mu_max);
  node_->declare_parameter(prefix + "constrained_accel_max_v", params_.constrained_accel_max_v);
  node_->declare_parameter(prefix + "constrained_accel_max_omega", params_.constrained_accel_max_omega);
  node_->declare_parameter(prefix + "constrained_clearance_min", params_.constrained_clearance_min);

  // CC-MPPI 파라미터
  node_->declare_parameter(prefix + "cc_mppi_enabled", params_.cc_mppi_enabled);
  node_->declare_parameter(prefix + "cc_risk_budget", params_.cc_risk_budget);
  node_->declare_parameter(prefix + "cc_penalty_weight", params_.cc_penalty_weight);
  node_->declare_parameter(prefix + "cc_adaptive_risk", params_.cc_adaptive_risk);
  node_->declare_parameter(prefix + "cc_tightening_rate", params_.cc_tightening_rate);
  node_->declare_parameter(prefix + "cc_quantile_smoothing", params_.cc_quantile_smoothing);
  node_->declare_parameter(prefix + "cc_cbf_projection_enabled", params_.cc_cbf_projection_enabled);

  // 성능 최적화 파라미터
  node_->declare_parameter(prefix + "num_threads", params_.num_threads);
  node_->declare_parameter(prefix + "costmap_eval_stride", params_.costmap_eval_stride);

  // Collision Debug Visualization
  node_->declare_parameter(prefix + "debug_collision_viz", params_.debug_collision_viz);
  node_->declare_parameter(prefix + "debug_cost_breakdown", params_.debug_cost_breakdown);
  node_->declare_parameter(prefix + "debug_collision_points", params_.debug_collision_points);
  node_->declare_parameter(prefix + "debug_safety_footprint", params_.debug_safety_footprint);
  node_->declare_parameter(prefix + "debug_cost_heatmap", params_.debug_cost_heatmap);
  node_->declare_parameter(prefix + "debug_footprint_radius", params_.debug_footprint_radius);
  node_->declare_parameter(prefix + "debug_heatmap_stride", params_.debug_heatmap_stride);

  // Visualization
  node_->declare_parameter(prefix + "visualize_samples", params_.visualize_samples);
  node_->declare_parameter(prefix + "visualize_best", params_.visualize_best);
  node_->declare_parameter(prefix + "visualize_weighted_avg", params_.visualize_weighted_avg);
  node_->declare_parameter(prefix + "visualize_reference", params_.visualize_reference);
  node_->declare_parameter(prefix + "visualize_text_info", params_.visualize_text_info);
  node_->declare_parameter(prefix + "visualize_control_sequence", params_.visualize_control_sequence);
  node_->declare_parameter(prefix + "visualize_tube", params_.visualize_tube);
  node_->declare_parameter(prefix + "visualize_cbf", params_.visualize_cbf);
  node_->declare_parameter(prefix + "max_visualized_samples", params_.max_visualized_samples);

  RCLCPP_INFO(node_->get_logger(), "MPPI parameters declared (M2+SOTA features included)");
}

void MPPIControllerPlugin::loadParameters()
{
  std::string prefix = plugin_name_ + ".";

  // Motion model
  params_.motion_model = node_->get_parameter(prefix + "motion_model").as_string();

  // nu 결정 및 noise_sigma/R/R_rate 동적 리사이즈
  int nu = (params_.motion_model == "diff_drive") ? 2 : 3;
  bool is_non_coaxial = (params_.motion_model == "non_coaxial_swerve");
  if (nu > static_cast<int>(params_.noise_sigma.size())) {
    params_.noise_sigma = Eigen::VectorXd::Zero(nu);
    params_.R = Eigen::MatrixXd::Zero(nu, nu);
    params_.R_rate = Eigen::MatrixXd::Zero(nu, nu);
  }

  // nx 결정 및 Q/Qf 동적 리사이즈 (non_coaxial_swerve: nx=4)
  int nx = is_non_coaxial ? 4 : 3;
  if (nx > static_cast<int>(params_.Q.rows())) {
    Eigen::MatrixXd Q_new = Eigen::MatrixXd::Zero(nx, nx);
    Q_new.topLeftCorner(params_.Q.rows(), params_.Q.cols()) = params_.Q;
    params_.Q = Q_new;
    Eigen::MatrixXd Qf_new = Eigen::MatrixXd::Zero(nx, nx);
    Qf_new.topLeftCorner(params_.Qf.rows(), params_.Qf.cols()) = params_.Qf;
    params_.Qf = Qf_new;
  }

  // MPPI parameters
  params_.N = node_->get_parameter(prefix + "N").as_int();
  params_.dt = node_->get_parameter(prefix + "dt").as_double();
  params_.K = node_->get_parameter(prefix + "K").as_int();
  params_.lambda = node_->get_parameter(prefix + "lambda").as_double();

  // Noise parameters — non_coaxial: [v, omega, delta_dot], swerve: [vx, vy, omega]
  params_.noise_sigma(0) = node_->get_parameter(prefix + "noise_sigma_v").as_double();
  if (is_non_coaxial) {
    params_.noise_sigma(1) = node_->get_parameter(prefix + "noise_sigma_omega").as_double();
    params_.noise_sigma(2) = node_->get_parameter(prefix + "noise_sigma_delta_dot").as_double();
  } else if (nu >= 3) {
    params_.noise_sigma(1) = node_->get_parameter(prefix + "noise_sigma_vy").as_double();
    params_.noise_sigma(2) = node_->get_parameter(prefix + "noise_sigma_omega").as_double();
  } else {
    params_.noise_sigma(1) = node_->get_parameter(prefix + "noise_sigma_omega").as_double();
  }

  // Control limits
  params_.v_max = node_->get_parameter(prefix + "v_max").as_double();
  params_.v_min = node_->get_parameter(prefix + "v_min").as_double();
  params_.omega_max = node_->get_parameter(prefix + "omega_max").as_double();
  params_.omega_min = node_->get_parameter(prefix + "omega_min").as_double();
  params_.vy_max = node_->get_parameter(prefix + "vy_max").as_double();

  // Cost weights - Q
  params_.Q(0, 0) = node_->get_parameter(prefix + "Q_x").as_double();
  params_.Q(1, 1) = node_->get_parameter(prefix + "Q_y").as_double();
  params_.Q(2, 2) = node_->get_parameter(prefix + "Q_theta").as_double();

  // Cost weights - Qf
  params_.Qf(0, 0) = node_->get_parameter(prefix + "Qf_x").as_double();
  params_.Qf(1, 1) = node_->get_parameter(prefix + "Qf_y").as_double();
  params_.Qf(2, 2) = node_->get_parameter(prefix + "Qf_theta").as_double();

  // Cost weights - R — non_coaxial: [v, omega, delta_dot], swerve: [vx, vy, omega]
  params_.R(0, 0) = node_->get_parameter(prefix + "R_v").as_double();
  if (is_non_coaxial) {
    params_.R(1, 1) = node_->get_parameter(prefix + "R_omega").as_double();
    params_.R(2, 2) = node_->get_parameter(prefix + "R_delta_dot").as_double();
  } else if (nu >= 3) {
    params_.R(1, 1) = node_->get_parameter(prefix + "R_vy").as_double();
    params_.R(2, 2) = node_->get_parameter(prefix + "R_omega").as_double();
  } else {
    params_.R(1, 1) = node_->get_parameter(prefix + "R_omega").as_double();
  }

  // Cost weights - R_rate — non_coaxial: [v, omega, delta_dot], swerve: [vx, vy, omega]
  params_.R_rate(0, 0) = node_->get_parameter(prefix + "R_rate_v").as_double();
  if (is_non_coaxial) {
    params_.R_rate(1, 1) = node_->get_parameter(prefix + "R_rate_omega").as_double();
    params_.R_rate(2, 2) = node_->get_parameter(prefix + "R_rate_delta_dot").as_double();
  } else if (nu >= 3) {
    params_.R_rate(1, 1) = node_->get_parameter(prefix + "R_rate_vy").as_double();
    params_.R_rate(2, 2) = node_->get_parameter(prefix + "R_rate_omega").as_double();
  } else {
    params_.R_rate(1, 1) = node_->get_parameter(prefix + "R_rate_omega").as_double();
  }

  // Obstacle avoidance
  params_.obstacle_weight = node_->get_parameter(prefix + "obstacle_weight").as_double();
  params_.safety_distance = node_->get_parameter(prefix + "safety_distance").as_double();

  // Forward preference
  params_.prefer_forward_weight = node_->get_parameter(prefix + "prefer_forward_weight").as_double();
  params_.prefer_forward_linear_ratio = node_->get_parameter(prefix + "prefer_forward_linear_ratio").as_double();
  params_.prefer_forward_velocity_incentive = node_->get_parameter(prefix + "prefer_forward_velocity_incentive").as_double();

  // Control smoothing
  params_.control_smoothing_alpha = node_->get_parameter(prefix + "control_smoothing_alpha").as_double();

  // Savitzky-Golay Filter
  params_.sg_filter_enabled = node_->get_parameter(prefix + "sg_filter_enabled").as_bool();
  params_.sg_half_window = node_->get_parameter(prefix + "sg_half_window").as_int();
  params_.sg_poly_order = node_->get_parameter(prefix + "sg_poly_order").as_int();

  // Information-Theoretic 정규화
  params_.it_alpha = node_->get_parameter(prefix + "it_alpha").as_double();

  // Exploitation/Exploration 분할
  params_.exploration_ratio = node_->get_parameter(prefix + "exploration_ratio").as_double();

  // Costmap obstacle cost
  params_.use_costmap_cost = node_->get_parameter(prefix + "use_costmap_cost").as_bool();
  params_.costmap_lethal_cost = node_->get_parameter(prefix + "costmap_lethal_cost").as_double();
  params_.costmap_critical_cost = node_->get_parameter(prefix + "costmap_critical_cost").as_double();
  params_.lookahead_dist = node_->get_parameter(prefix + "lookahead_dist").as_double();
  params_.min_lookahead = node_->get_parameter(prefix + "min_lookahead").as_double();
  params_.goal_slowdown_dist = node_->get_parameter(prefix + "goal_slowdown_dist").as_double();
  params_.ref_theta_smooth_window = node_->get_parameter(prefix + "ref_theta_smooth_window").as_int();

  // Velocity Tracking Cost
  params_.velocity_tracking_weight = node_->get_parameter(prefix + "velocity_tracking_weight").as_double();
  params_.reference_velocity = node_->get_parameter(prefix + "reference_velocity").as_double();

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
  params_.tube_nominal_reset_threshold = node_->get_parameter(prefix + "tube_nominal_reset_threshold").as_double();
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

  // Biased-MPPI (RA-L 2024)
  params_.biased_enabled = node_->get_parameter(prefix + "biased_enabled").as_bool();
  params_.bias_ratio = node_->get_parameter(prefix + "bias_ratio").as_double();
  params_.biased_braking = node_->get_parameter(prefix + "biased_braking").as_bool();
  params_.biased_goto_goal = node_->get_parameter(prefix + "biased_goto_goal").as_bool();
  params_.biased_path_following = node_->get_parameter(prefix + "biased_path_following").as_bool();
  params_.biased_previous_solution = node_->get_parameter(prefix + "biased_previous_solution").as_bool();
  params_.biased_goto_goal_gain = node_->get_parameter(prefix + "biased_goto_goal_gain").as_double();
  params_.biased_path_following_gain = node_->get_parameter(prefix + "biased_path_following_gain").as_double();

  // Non-Coaxial Swerve / Ackermann 공통 파라미터
  params_.max_steering_rate = node_->get_parameter(prefix + "max_steering_rate").as_double();
  params_.max_steering_angle = node_->get_parameter(prefix + "max_steering_angle").as_double();
  params_.wheelbase = node_->get_parameter(prefix + "wheelbase").as_double();

  // CBF (Control Barrier Function)
  params_.cbf_enabled = node_->get_parameter(prefix + "cbf_enabled").as_bool();
  params_.cbf_gamma = node_->get_parameter(prefix + "cbf_gamma").as_double();
  params_.cbf_safety_margin = node_->get_parameter(prefix + "cbf_safety_margin").as_double();
  params_.cbf_robot_radius = node_->get_parameter(prefix + "cbf_robot_radius").as_double();
  params_.cbf_activation_distance = node_->get_parameter(prefix + "cbf_activation_distance").as_double();
  params_.cbf_cost_weight = node_->get_parameter(prefix + "cbf_cost_weight").as_double();
  params_.cbf_use_safety_filter = node_->get_parameter(prefix + "cbf_use_safety_filter").as_bool();

  // Residual Dynamics
  params_.residual_enabled = node_->get_parameter(prefix + "residual_enabled").as_bool();
  params_.residual_weights_path = node_->get_parameter(prefix + "residual_weights_path").as_string();
  params_.residual_alpha = node_->get_parameter(prefix + "residual_alpha").as_double();

  // Ensemble Dynamics
  params_.ensemble_enabled = node_->get_parameter(prefix + "ensemble_enabled").as_bool();
  params_.ensemble_weights_dir = node_->get_parameter(prefix + "ensemble_weights_dir").as_string();
  params_.ensemble_size = node_->get_parameter(prefix + "ensemble_size").as_int();
  params_.ensemble_alpha = node_->get_parameter(prefix + "ensemble_alpha").as_double();
  params_.uncertainty_cost_weight = node_->get_parameter(prefix + "uncertainty_cost_weight").as_double();

  // C3BF
  params_.c3bf_enabled = node_->get_parameter(prefix + "c3bf_enabled").as_bool();
  params_.c3bf_alpha_safe = node_->get_parameter(prefix + "c3bf_alpha_safe").as_double();
  params_.c3bf_cost_weight = node_->get_parameter(prefix + "c3bf_cost_weight").as_double();

  // Adaptive Shield
  params_.adaptive_shield_enabled = node_->get_parameter(prefix + "adaptive_shield_enabled").as_bool();
  params_.adaptive_shield_alpha_min = node_->get_parameter(prefix + "adaptive_shield_alpha_min").as_double();
  params_.adaptive_shield_alpha_max = node_->get_parameter(prefix + "adaptive_shield_alpha_max").as_double();
  params_.adaptive_shield_k_d = node_->get_parameter(prefix + "adaptive_shield_k_d").as_double();
  params_.adaptive_shield_k_v = node_->get_parameter(prefix + "adaptive_shield_k_v").as_double();

  // Horizon-Weighted CBF
  params_.cbf_horizon_discount = node_->get_parameter(prefix + "cbf_horizon_discount").as_double();

  // Online Data Buffer
  params_.online_data_enabled = node_->get_parameter(prefix + "online_data_enabled").as_bool();
  params_.online_data_capacity = node_->get_parameter(prefix + "online_data_capacity").as_int();
  params_.online_data_export_path = node_->get_parameter(prefix + "online_data_export_path").as_string();

  // Online Learning (Model Reload)
  params_.model_reload_enabled = node_->get_parameter(prefix + "model_reload_enabled").as_bool();
  params_.model_reload_interval_sec = node_->get_parameter(prefix + "model_reload_interval_sec").as_double();

  // Safety Enhancement: BR-MPPI
  params_.barrier_rate_cost_weight = node_->get_parameter(prefix + "barrier_rate_cost_weight").as_double();

  // Safety Enhancement: Conformal Predictor
  params_.conformal_enabled = node_->get_parameter(prefix + "conformal_enabled").as_bool();
  params_.conformal_coverage = node_->get_parameter(prefix + "conformal_coverage").as_double();
  params_.conformal_window_size = node_->get_parameter(prefix + "conformal_window_size").as_int();
  params_.conformal_initial_margin = node_->get_parameter(prefix + "conformal_initial_margin").as_double();
  params_.conformal_min_margin = node_->get_parameter(prefix + "conformal_min_margin").as_double();
  params_.conformal_max_margin = node_->get_parameter(prefix + "conformal_max_margin").as_double();
  params_.conformal_decay_rate = node_->get_parameter(prefix + "conformal_decay_rate").as_double();

  // LP-MPPI (Low-Pass Filtering)
  params_.lp_enabled = node_->get_parameter(prefix + "lp_enabled").as_bool();
  params_.lp_cutoff_frequency = node_->get_parameter(prefix + "lp_cutoff_frequency").as_double();
  params_.lp_filter_all_samples = node_->get_parameter(prefix + "lp_filter_all_samples").as_bool();

  // Halton-MPPI (Halton low-discrepancy sequence)
  params_.halton_enabled = node_->get_parameter(prefix + "halton_enabled").as_bool();
  params_.halton_beta = node_->get_parameter(prefix + "halton_beta").as_double();
  params_.halton_sequence_offset = node_->get_parameter(prefix + "halton_sequence_offset").as_int();

  // Feedback-MPPI (F-MPPI, Riccati feedback gains)
  params_.feedback_mppi_enabled = node_->get_parameter(prefix + "feedback_mppi_enabled").as_bool();
  params_.feedback_gain_scale = node_->get_parameter(prefix + "feedback_gain_scale").as_double();
  params_.feedback_recompute_interval = node_->get_parameter(prefix + "feedback_recompute_interval").as_int();
  params_.feedback_regularization = node_->get_parameter(prefix + "feedback_regularization").as_double();

  // Dynamic Obstacle Tracker
  params_.dynamic_obstacle_tracking_enabled = node_->get_parameter(prefix + "dynamic_obstacle_tracking_enabled").as_bool();
  params_.obstacle_cluster_distance = node_->get_parameter(prefix + "obstacle_cluster_distance").as_double();
  params_.obstacle_min_cluster_size = node_->get_parameter(prefix + "obstacle_min_cluster_size").as_int();
  params_.obstacle_velocity_ema_alpha = node_->get_parameter(prefix + "obstacle_velocity_ema_alpha").as_double();
  params_.obstacle_max_association_distance = node_->get_parameter(prefix + "obstacle_max_association_distance").as_double();
  params_.obstacle_track_timeout = node_->get_parameter(prefix + "obstacle_track_timeout").as_double();

  // Safety Enhancement: Shield-MPPI
  params_.shield_cbf_stride = node_->get_parameter(prefix + "shield_cbf_stride").as_int();
  params_.shield_max_iterations = node_->get_parameter(prefix + "shield_max_iterations").as_int();

  // CLF-CBF-QP
  params_.clf_cbf_enabled = node_->get_parameter(prefix + "clf_cbf_enabled").as_bool();
  params_.clf_decay_rate = node_->get_parameter(prefix + "clf_decay_rate").as_double();
  params_.clf_slack_penalty = node_->get_parameter(prefix + "clf_slack_penalty").as_double();
  params_.clf_P_scale = node_->get_parameter(prefix + "clf_P_scale").as_double();

  // CBF 합성
  params_.cbf_composition_enabled = node_->get_parameter(prefix + "cbf_composition_enabled").as_bool();
  params_.cbf_composition_method = node_->get_parameter(prefix + "cbf_composition_method").as_int();
  params_.cbf_composition_alpha = node_->get_parameter(prefix + "cbf_composition_alpha").as_double();

  // Predictive Safety Filter
  params_.predictive_safety_enabled = node_->get_parameter(prefix + "predictive_safety_enabled").as_bool();
  params_.predictive_safety_horizon = node_->get_parameter(prefix + "predictive_safety_horizon").as_int();
  params_.predictive_safety_decay = node_->get_parameter(prefix + "predictive_safety_decay").as_double();
  params_.predictive_safety_max_iterations = node_->get_parameter(prefix + "predictive_safety_max_iterations").as_int();

  // Covariance Steering MPPI (CS-MPPI)
  params_.cs_enabled = node_->get_parameter(prefix + "cs_enabled").as_bool();
  params_.cs_scale_min = node_->get_parameter(prefix + "cs_scale_min").as_double();
  params_.cs_scale_max = node_->get_parameter(prefix + "cs_scale_max").as_double();
  params_.cs_feedback_enabled = node_->get_parameter(prefix + "cs_feedback_enabled").as_bool();
  params_.cs_feedback_gain = node_->get_parameter(prefix + "cs_feedback_gain").as_double();

  // pi-MPPI (Projection MPPI)
  params_.pi_enabled = node_->get_parameter(prefix + "pi_enabled").as_bool();
  params_.pi_admm_iterations = node_->get_parameter(prefix + "pi_admm_iterations").as_int();
  params_.pi_admm_rho = node_->get_parameter(prefix + "pi_admm_rho").as_double();
  params_.pi_derivative_order = node_->get_parameter(prefix + "pi_derivative_order").as_int();
  params_.pi_rate_max_v = node_->get_parameter(prefix + "pi_rate_max_v").as_double();
  params_.pi_rate_max_omega = node_->get_parameter(prefix + "pi_rate_max_omega").as_double();
  params_.pi_rate_max_vy = node_->get_parameter(prefix + "pi_rate_max_vy").as_double();
  params_.pi_accel_max_v = node_->get_parameter(prefix + "pi_accel_max_v").as_double();
  params_.pi_accel_max_omega = node_->get_parameter(prefix + "pi_accel_max_omega").as_double();
  params_.pi_accel_max_vy = node_->get_parameter(prefix + "pi_accel_max_vy").as_double();

  // Hybrid Swerve MPPI (MPPI-H)
  params_.hybrid_enabled = node_->get_parameter(prefix + "hybrid_enabled").as_bool();
  params_.hybrid_cdist_threshold = node_->get_parameter(prefix + "hybrid_cdist_threshold").as_double();
  params_.hybrid_cangle_threshold = node_->get_parameter(prefix + "hybrid_cangle_threshold").as_double();
  params_.hybrid_hysteresis_count = node_->get_parameter(prefix + "hybrid_hysteresis_count").as_int();
  params_.hybrid_lf = node_->get_parameter(prefix + "hybrid_lf").as_double();
  params_.hybrid_lr = node_->get_parameter(prefix + "hybrid_lr").as_double();
  params_.hybrid_dl = node_->get_parameter(prefix + "hybrid_dl").as_double();
  params_.hybrid_dr = node_->get_parameter(prefix + "hybrid_dr").as_double();
  params_.hybrid_v_wheel_max = node_->get_parameter(prefix + "hybrid_v_wheel_max").as_double();
  params_.hybrid_delta_max = node_->get_parameter(prefix + "hybrid_delta_max").as_double();
  params_.hybrid_noise_sigma_vfl = node_->get_parameter(prefix + "hybrid_noise_sigma_vfl").as_double();
  params_.hybrid_noise_sigma_vrr = node_->get_parameter(prefix + "hybrid_noise_sigma_vrr").as_double();
  params_.hybrid_noise_sigma_dfl = node_->get_parameter(prefix + "hybrid_noise_sigma_dfl").as_double();
  params_.hybrid_noise_sigma_drr = node_->get_parameter(prefix + "hybrid_noise_sigma_drr").as_double();
  params_.hybrid_R_vfl = node_->get_parameter(prefix + "hybrid_R_vfl").as_double();
  params_.hybrid_R_vrr = node_->get_parameter(prefix + "hybrid_R_vrr").as_double();
  params_.hybrid_R_dfl = node_->get_parameter(prefix + "hybrid_R_dfl").as_double();
  params_.hybrid_R_drr = node_->get_parameter(prefix + "hybrid_R_drr").as_double();

  // iLQR Warm-Start
  params_.ilqr_enabled = node_->get_parameter(prefix + "ilqr_enabled").as_bool();
  params_.ilqr_max_iterations = node_->get_parameter(prefix + "ilqr_max_iterations").as_int();
  params_.ilqr_regularization = node_->get_parameter(prefix + "ilqr_regularization").as_double();
  params_.ilqr_line_search_steps = node_->get_parameter(prefix + "ilqr_line_search_steps").as_int();
  params_.ilqr_cost_tolerance = node_->get_parameter(prefix + "ilqr_cost_tolerance").as_double();

  // Auto-Selector MPPI
  params_.auto_selector_enabled = node_->get_parameter(prefix + "auto_selector_enabled").as_bool();
  params_.auto_selector_safety_threshold = node_->get_parameter(prefix + "auto_selector_safety_threshold").as_double();
  params_.auto_selector_recovery_threshold = node_->get_parameter(prefix + "auto_selector_recovery_threshold").as_double();
  params_.auto_selector_fast_threshold = node_->get_parameter(prefix + "auto_selector_fast_threshold").as_double();
  params_.auto_selector_precision_dist = node_->get_parameter(prefix + "auto_selector_precision_dist").as_double();
  params_.auto_selector_hysteresis = node_->get_parameter(prefix + "auto_selector_hysteresis").as_int();
  params_.auto_selector_smoothing_alpha = node_->get_parameter(prefix + "auto_selector_smoothing_alpha").as_double();

  // Robust MPPI
  params_.robust_enabled = node_->get_parameter(prefix + "robust_enabled").as_bool();
  params_.robust_alpha = node_->get_parameter(prefix + "robust_alpha").as_double();
  params_.robust_penalty = node_->get_parameter(prefix + "robust_penalty").as_double();
  params_.robust_wasserstein_radius = node_->get_parameter(prefix + "robust_wasserstein_radius").as_double();
  params_.robust_adaptive_alpha = node_->get_parameter(prefix + "robust_adaptive_alpha").as_bool();

  // IT-MPPI (Information-Theoretic MPPI)
  params_.it_mppi_enabled = node_->get_parameter(prefix + "it_mppi_enabled").as_bool();
  params_.it_exploration_weight = node_->get_parameter(prefix + "it_exploration_weight").as_double();
  params_.it_kl_weight = node_->get_parameter(prefix + "it_kl_weight").as_double();
  params_.it_diversity_threshold = node_->get_parameter(prefix + "it_diversity_threshold").as_double();
  params_.it_adaptive_exploration = node_->get_parameter(prefix + "it_adaptive_exploration").as_bool();
  params_.it_exploration_decay = node_->get_parameter(prefix + "it_exploration_decay").as_double();

  // CEM-MPPI
  params_.cem_enabled = node_->get_parameter(prefix + "cem_enabled").as_bool();
  params_.cem_iterations = node_->get_parameter(prefix + "cem_iterations").as_int();
  params_.cem_elite_ratio = node_->get_parameter(prefix + "cem_elite_ratio").as_double();
  params_.cem_momentum = node_->get_parameter(prefix + "cem_momentum").as_double();
  params_.cem_sigma_min = node_->get_parameter(prefix + "cem_sigma_min").as_double();
  params_.cem_sigma_decay = node_->get_parameter(prefix + "cem_sigma_decay").as_double();
  params_.cem_adaptive_enabled = node_->get_parameter(prefix + "cem_adaptive_enabled").as_bool();
  params_.cem_adaptive_cost_tol = node_->get_parameter(prefix + "cem_adaptive_cost_tol").as_double();
  params_.cem_adaptive_min_iter = node_->get_parameter(prefix + "cem_adaptive_min_iter").as_int();
  params_.cem_adaptive_max_iter = node_->get_parameter(prefix + "cem_adaptive_max_iter").as_int();

  // Trajectory Library MPPI
  params_.traj_library_enabled = node_->get_parameter(prefix + "traj_library_enabled").as_bool();
  params_.traj_library_ratio = node_->get_parameter(prefix + "traj_library_ratio").as_double();
  params_.traj_library_perturbation = node_->get_parameter(prefix + "traj_library_perturbation").as_double();
  params_.traj_library_adaptive = node_->get_parameter(prefix + "traj_library_adaptive").as_bool();
  params_.traj_library_num_per_primitive = node_->get_parameter(prefix + "traj_library_num_per_primitive").as_int();

  // Receding Horizon MPPI (RH-MPPI)
  params_.rh_mppi_enabled = node_->get_parameter(prefix + "rh_mppi_enabled").as_bool();
  params_.rh_N_min = node_->get_parameter(prefix + "rh_N_min").as_int();
  params_.rh_N_max = node_->get_parameter(prefix + "rh_N_max").as_int();
  params_.rh_speed_weight = node_->get_parameter(prefix + "rh_speed_weight").as_double();
  params_.rh_obstacle_weight = node_->get_parameter(prefix + "rh_obstacle_weight").as_double();
  params_.rh_error_weight = node_->get_parameter(prefix + "rh_error_weight").as_double();
  params_.rh_obs_dist_threshold = node_->get_parameter(prefix + "rh_obs_dist_threshold").as_double();
  params_.rh_error_threshold = node_->get_parameter(prefix + "rh_error_threshold").as_double();
  params_.rh_smoothing_alpha = node_->get_parameter(prefix + "rh_smoothing_alpha").as_double();

  // Constrained MPPI (Augmented Lagrangian)
  params_.constrained_enabled = node_->get_parameter(prefix + "constrained_enabled").as_bool();
  params_.constrained_mu_init = node_->get_parameter(prefix + "constrained_mu_init").as_double();
  params_.constrained_mu_growth = node_->get_parameter(prefix + "constrained_mu_growth").as_double();
  params_.constrained_mu_max = node_->get_parameter(prefix + "constrained_mu_max").as_double();
  params_.constrained_accel_max_v = node_->get_parameter(prefix + "constrained_accel_max_v").as_double();
  params_.constrained_accel_max_omega = node_->get_parameter(prefix + "constrained_accel_max_omega").as_double();
  params_.constrained_clearance_min = node_->get_parameter(prefix + "constrained_clearance_min").as_double();

  // CC-MPPI 파라미터
  params_.cc_mppi_enabled = node_->get_parameter(prefix + "cc_mppi_enabled").as_bool();
  params_.cc_risk_budget = node_->get_parameter(prefix + "cc_risk_budget").as_double();
  params_.cc_penalty_weight = node_->get_parameter(prefix + "cc_penalty_weight").as_double();
  params_.cc_adaptive_risk = node_->get_parameter(prefix + "cc_adaptive_risk").as_bool();
  params_.cc_tightening_rate = node_->get_parameter(prefix + "cc_tightening_rate").as_double();
  params_.cc_quantile_smoothing = node_->get_parameter(prefix + "cc_quantile_smoothing").as_double();
  params_.cc_cbf_projection_enabled = node_->get_parameter(prefix + "cc_cbf_projection_enabled").as_bool();

  // 성능 최적화 파라미터
  params_.num_threads = node_->get_parameter(prefix + "num_threads").as_int();
  params_.costmap_eval_stride = node_->get_parameter(prefix + "costmap_eval_stride").as_int();

  // Collision Debug Visualization
  params_.debug_collision_viz = node_->get_parameter(prefix + "debug_collision_viz").as_bool();
  params_.debug_cost_breakdown = node_->get_parameter(prefix + "debug_cost_breakdown").as_bool();
  params_.debug_collision_points = node_->get_parameter(prefix + "debug_collision_points").as_bool();
  params_.debug_safety_footprint = node_->get_parameter(prefix + "debug_safety_footprint").as_bool();
  params_.debug_cost_heatmap = node_->get_parameter(prefix + "debug_cost_heatmap").as_bool();
  params_.debug_footprint_radius = node_->get_parameter(prefix + "debug_footprint_radius").as_double();
  params_.debug_heatmap_stride = node_->get_parameter(prefix + "debug_heatmap_stride").as_int();

  // Visualization
  params_.visualize_samples = node_->get_parameter(prefix + "visualize_samples").as_bool();
  params_.visualize_best = node_->get_parameter(prefix + "visualize_best").as_bool();
  params_.visualize_weighted_avg = node_->get_parameter(prefix + "visualize_weighted_avg").as_bool();
  params_.visualize_reference = node_->get_parameter(prefix + "visualize_reference").as_bool();
  params_.visualize_text_info = node_->get_parameter(prefix + "visualize_text_info").as_bool();
  params_.visualize_control_sequence = node_->get_parameter(prefix + "visualize_control_sequence").as_bool();
  params_.visualize_tube = node_->get_parameter(prefix + "visualize_tube").as_bool();
  params_.visualize_cbf = node_->get_parameter(prefix + "visualize_cbf").as_bool();
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
  RCLCPP_INFO(
    node_->get_logger(),
    "Trajectory stability: sg_filter=%s, it_alpha=%.3f, exploration_ratio=%.2f",
    params_.sg_filter_enabled ? "ON" : "OFF",
    params_.it_alpha, params_.exploration_ratio
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
      else if (short_name == "noise_sigma_vy") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "noise_sigma_vy must be >= 0.0";
          return result;
        }
        if (params_.noise_sigma.size() >= 3) {
          params_.noise_sigma(1) = value;
          need_recreate_sampler = true;
          RCLCPP_INFO(node_->get_logger(), "Updated noise_sigma_vy: %.3f", value);
        }
      }
      else if (short_name == "noise_sigma_omega") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "noise_sigma_omega must be >= 0.0";
          return result;
        }
        int omega_idx = params_.noise_sigma.size() - 1;
        params_.noise_sigma(omega_idx) = value;
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
      else if (short_name == "R_vy") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "R_vy must be >= 0.0";
          return result;
        }
        if (params_.R.rows() >= 3) {
          params_.R(1, 1) = value;
          need_recreate_cost_function = true;
          RCLCPP_INFO(node_->get_logger(), "Updated R_vy: %.3f", value);
        }
      }
      else if (short_name == "R_omega") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "R_omega must be >= 0.0";
          return result;
        }
        int omega_idx = params_.R.rows() - 1;
        params_.R(omega_idx, omega_idx) = value;
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
      else if (short_name == "R_rate_vy") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "R_rate_vy must be >= 0.0";
          return result;
        }
        if (params_.R_rate.rows() >= 3) {
          params_.R_rate(1, 1) = value;
          need_recreate_cost_function = true;
          RCLCPP_INFO(node_->get_logger(), "Updated R_rate_vy: %.3f", value);
        }
      }
      else if (short_name == "R_rate_omega") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "R_rate_omega must be >= 0.0";
          return result;
        }
        int omega_idx = params_.R_rate.rows() - 1;
        params_.R_rate(omega_idx, omega_idx) = value;
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
      // Collision Debug Visualization (런타임 ON/OFF)
      else if (short_name == "debug_collision_viz") {
        params_.debug_collision_viz = param.as_bool();
        RCLCPP_INFO(node_->get_logger(), "Updated debug_collision_viz: %s",
          params_.debug_collision_viz ? "ON" : "OFF");
      }
      else if (short_name == "debug_cost_breakdown") {
        params_.debug_cost_breakdown = param.as_bool();
      }
      else if (short_name == "debug_collision_points") {
        params_.debug_collision_points = param.as_bool();
      }
      else if (short_name == "debug_safety_footprint") {
        params_.debug_safety_footprint = param.as_bool();
      }
      else if (short_name == "debug_cost_heatmap") {
        params_.debug_cost_heatmap = param.as_bool();
      }
      else if (short_name == "debug_footprint_radius") {
        params_.debug_footprint_radius = param.as_double();
      }
      else if (short_name == "debug_heatmap_stride") {
        params_.debug_heatmap_stride = param.as_int();
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
      else if (short_name == "prefer_forward_velocity_incentive") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "prefer_forward_velocity_incentive must be >= 0.0";
          return result;
        }
        params_.prefer_forward_velocity_incentive = value;
        need_recreate_cost_function = true;
        RCLCPP_INFO(node_->get_logger(), "Updated prefer_forward_velocity_incentive: %.3f", value);
      }
      else if (short_name == "control_smoothing_alpha") {
        double value = param.as_double();
        if (value < 0.0 || value > 1.0) {
          result.successful = false;
          result.reason = "control_smoothing_alpha must be in [0.0, 1.0]";
          return result;
        }
        params_.control_smoothing_alpha = value;
        RCLCPP_INFO(node_->get_logger(), "Updated control_smoothing_alpha: %.3f", value);
      }
      // min_lookahead
      else if (short_name == "min_lookahead") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "min_lookahead must be >= 0.0";
          return result;
        }
        params_.min_lookahead = value;
        RCLCPP_INFO(node_->get_logger(), "Updated min_lookahead: %.3f", value);
      }
      // goal_slowdown_dist
      else if (short_name == "goal_slowdown_dist") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "goal_slowdown_dist must be >= 0.0";
          return result;
        }
        params_.goal_slowdown_dist = value;
        RCLCPP_INFO(node_->get_logger(), "Updated goal_slowdown_dist: %.3f", value);
      }
      // costmap_lethal_cost
      else if (short_name == "costmap_lethal_cost") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "costmap_lethal_cost must be >= 0.0";
          return result;
        }
        params_.costmap_lethal_cost = value;
        if (costmap_obstacle_cost_ptr_) {
          costmap_obstacle_cost_ptr_->setLethalCost(value);
        }
        RCLCPP_INFO(node_->get_logger(), "Updated costmap_lethal_cost: %.1f", value);
      }
      // costmap_critical_cost
      else if (short_name == "costmap_critical_cost") {
        double value = param.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = "costmap_critical_cost must be >= 0.0";
          return result;
        }
        params_.costmap_critical_cost = value;
        if (costmap_obstacle_cost_ptr_) {
          costmap_obstacle_cost_ptr_->setCriticalCost(value);
        }
        RCLCPP_INFO(node_->get_logger(), "Updated costmap_critical_cost: %.1f", value);
      }
      // visualize_cbf
      else if (short_name == "visualize_cbf") {
        params_.visualize_cbf = param.as_bool();
        RCLCPP_INFO(node_->get_logger(), "Updated visualize_cbf: %s",
          params_.visualize_cbf ? "ON" : "OFF");
      }
      // cbf_enabled — 런타임 활성화 시 CBF 컴포넌트 초기화
      else if (short_name == "cbf_enabled") {
        bool new_val = param.as_bool();
        if (new_val && !params_.cbf_enabled) {
          // OFF→ON: barrier_set_ 재구성 + safety filter 생성
          barrier_set_ = BarrierFunctionSet(
            params_.cbf_robot_radius, params_.cbf_safety_margin,
            params_.cbf_activation_distance);

          if (params_.cbf_use_safety_filter && dynamics_ && !cbf_safety_filter_) {
            int nu_dim = dynamics_->model().controlDim();
            Eigen::VectorXd u_min(nu_dim), u_max(nu_dim);
            bool is_nc = (params_.motion_model == "non_coaxial_swerve");
            bool is_ack = (params_.motion_model == "ackermann");
            if (is_nc) {
              u_min << params_.v_min, params_.omega_min, -params_.max_steering_rate;
              u_max << params_.v_max, params_.omega_max,  params_.max_steering_rate;
            } else if (is_ack) {
              u_min << params_.v_min, -params_.max_steering_rate;
              u_max << params_.v_max,  params_.max_steering_rate;
            } else if (nu_dim >= 3) {
              u_min << params_.v_min, -params_.v_max, params_.omega_min;
              u_max << params_.v_max,  params_.v_max, params_.omega_max;
            } else {
              u_min << params_.v_min, params_.omega_min;
              u_max << params_.v_max, params_.omega_max;
            }
            cbf_safety_filter_ = std::make_unique<CBFSafetyFilter>(
              &barrier_set_, params_.cbf_gamma, params_.dt, u_min, u_max);
          }
          RCLCPP_INFO(node_->get_logger(),
            "CBF enabled at runtime (safety_filter=%s)",
            cbf_safety_filter_ ? "ON" : "OFF");
        }
        params_.cbf_enabled = new_val;
        RCLCPP_INFO(node_->get_logger(), "Updated cbf_enabled: %s",
          new_val ? "ON" : "OFF");
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

void MPPIControllerPlugin::publishCollisionDebugVisualization(
  const MPPIInfo& info,
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& /*reference_trajectory*/,
  const Eigen::MatrixXd& weighted_avg_trajectory
)
{
  if (!marker_pub_ || marker_pub_->get_subscription_count() == 0) {
    return;
  }

  visualization_msgs::msg::MarkerArray marker_array;
  auto stamp = node_->now();
  std::string frame_id = global_plan_.header.frame_id.empty() ?
    "map" : global_plan_.header.frame_id;
  double marker_lifetime = 0.3;

  // (1) 비용 분해 텍스트 (로봇 위 1.5m)
  if (params_.debug_cost_breakdown && !info.cost_breakdown.component_costs.empty()) {
    visualization_msgs::msg::Marker text_marker;
    text_marker.header.stamp = stamp;
    text_marker.header.frame_id = frame_id;
    text_marker.ns = "mppi_debug_cost_text";
    text_marker.id = 0;
    text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::msg::Marker::ADD;
    text_marker.lifetime = rclcpp::Duration::from_seconds(marker_lifetime);

    text_marker.pose.position.x = current_state(0);
    text_marker.pose.position.y = current_state(1);
    text_marker.pose.position.z = 1.5;
    text_marker.pose.orientation.w = 1.0;

    std::string text;
    for (const auto& [name, costs] : info.cost_breakdown.component_costs) {
      double min_c = costs.minCoeff();
      double mean_c = costs.mean();
      double max_c = costs.maxCoeff();
      char buf[128];
      snprintf(buf, sizeof(buf), "%-16s %7.1f / %7.1f / %7.1f\n",
        name.c_str(), min_c, mean_c, max_c);
      text += buf;
    }

    text_marker.text = text;
    text_marker.scale.z = 0.15;
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 0.0;
    text_marker.color.a = 1.0;

    marker_array.markers.push_back(text_marker);
  }

  // (2) 충돌 지점 구체 (weighted_avg 궤적에서 costmap 쿼리)
  if (params_.debug_collision_points && costmap_obstacle_cost_ptr_ && costmap_ros_) {
    auto costmap = costmap_ros_->getCostmap();
    if (costmap && weighted_avg_trajectory.rows() > 0) {
      visualization_msgs::msg::Marker collision_marker;
      collision_marker.header.stamp = stamp;
      collision_marker.header.frame_id = frame_id;
      collision_marker.ns = "mppi_debug_collision";
      collision_marker.id = 0;
      collision_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
      collision_marker.action = visualization_msgs::msg::Marker::ADD;
      collision_marker.lifetime = rclcpp::Duration::from_seconds(marker_lifetime);
      collision_marker.pose.orientation.w = 1.0;
      collision_marker.scale.x = 0.12;
      collision_marker.scale.y = 0.12;
      collision_marker.scale.z = 0.12;

      // weighted_avg를 단일 궤적으로 computePerPoint
      std::vector<Eigen::MatrixXd> single_traj = {weighted_avg_trajectory};
      Eigen::MatrixXd per_point = costmap_obstacle_cost_ptr_->computePerPoint(single_traj);

      for (int t = 0; t < weighted_avg_trajectory.rows(); ++t) {
        double cost_val = per_point(0, t);
        if (cost_val <= 0.0) continue;

        geometry_msgs::msg::Point p;
        p.x = weighted_avg_trajectory(t, 0);
        p.y = weighted_avg_trajectory(t, 1);
        p.z = 0.1;
        collision_marker.points.push_back(p);

        std_msgs::msg::ColorRGBA color;
        color.a = 0.9;
        if (cost_val >= params_.costmap_lethal_cost * 0.9) {
          // LETHAL → 빨강
          color.r = 1.0; color.g = 0.0; color.b = 0.0;
        } else if (cost_val >= params_.costmap_critical_cost * 0.9) {
          // INSCRIBED → 주황
          color.r = 1.0; color.g = 0.5; color.b = 0.0;
        } else {
          // inflation → 노랑
          color.r = 1.0; color.g = 1.0; color.b = 0.0;
        }
        collision_marker.colors.push_back(color);
      }

      if (!collision_marker.points.empty()) {
        marker_array.markers.push_back(collision_marker);
      }
    }
  }

  // (3) 안전 영역 원 (내부: footprint_radius, 외부: footprint + safety_distance)
  if (params_.debug_safety_footprint) {
    auto make_circle = [&](double radius, int id, float r, float g, float b) {
      visualization_msgs::msg::Marker circle;
      circle.header.stamp = stamp;
      circle.header.frame_id = frame_id;
      circle.ns = "mppi_debug_footprint";
      circle.id = id;
      circle.type = visualization_msgs::msg::Marker::LINE_STRIP;
      circle.action = visualization_msgs::msg::Marker::ADD;
      circle.lifetime = rclcpp::Duration::from_seconds(marker_lifetime);
      circle.scale.x = 0.02;
      circle.color.r = r;
      circle.color.g = g;
      circle.color.b = b;
      circle.color.a = 0.8;

      int num_pts = 36;
      for (int i = 0; i <= num_pts; ++i) {
        double angle = 2.0 * M_PI * i / num_pts;
        geometry_msgs::msg::Point p;
        p.x = current_state(0) + radius * std::cos(angle);
        p.y = current_state(1) + radius * std::sin(angle);
        p.z = 0.05;
        circle.points.push_back(p);
      }
      return circle;
    };

    // 내부 원 (시안)
    marker_array.markers.push_back(
      make_circle(params_.debug_footprint_radius, 0, 0.0f, 1.0f, 1.0f));
    // 외부 원 (빨강)
    marker_array.markers.push_back(
      make_circle(params_.debug_footprint_radius + params_.safety_distance, 1, 1.0f, 0.0f, 0.0f));
  }

  // (4) best 궤적 비용 히트맵 (SPHERE_LIST)
  if (params_.debug_cost_heatmap && costmap_obstacle_cost_ptr_ && info.best_trajectory.rows() > 0) {
    visualization_msgs::msg::Marker heatmap;
    heatmap.header.stamp = stamp;
    heatmap.header.frame_id = frame_id;
    heatmap.ns = "mppi_debug_heatmap";
    heatmap.id = 0;
    heatmap.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    heatmap.action = visualization_msgs::msg::Marker::ADD;
    heatmap.lifetime = rclcpp::Duration::from_seconds(marker_lifetime);
    heatmap.pose.orientation.w = 1.0;
    heatmap.scale.x = 0.06;
    heatmap.scale.y = 0.06;
    heatmap.scale.z = 0.06;

    std::vector<Eigen::MatrixXd> best_traj = {info.best_trajectory};
    Eigen::MatrixXd per_point = costmap_obstacle_cost_ptr_->computePerPoint(best_traj);

    double max_cost = per_point.maxCoeff();
    if (max_cost < 1e-6) max_cost = 1.0;

    int stride = std::max(1, params_.debug_heatmap_stride);
    for (int t = 0; t < info.best_trajectory.rows(); t += stride) {
      geometry_msgs::msg::Point p;
      p.x = info.best_trajectory(t, 0);
      p.y = info.best_trajectory(t, 1);
      p.z = 0.15;
      heatmap.points.push_back(p);

      double ratio = std::min(per_point(0, t) / max_cost, 1.0);

      std_msgs::msg::ColorRGBA color;
      color.a = 0.8;
      if (ratio < 0.5) {
        // 녹색 → 노랑
        color.r = static_cast<float>(ratio * 2.0);
        color.g = 1.0f;
        color.b = 0.0f;
      } else {
        // 노랑 → 빨강
        color.r = 1.0f;
        color.g = static_cast<float>((1.0 - ratio) * 2.0);
        color.b = 0.0f;
      }
      heatmap.colors.push_back(color);
    }

    if (!heatmap.points.empty()) {
      marker_array.markers.push_back(heatmap);
    }
  }

  if (!marker_array.markers.empty()) {
    marker_pub_->publish(marker_array);
  }
}

void MPPIControllerPlugin::publishCBFVisualization(
  const MPPIInfo& info,
  const Eigen::VectorXd& current_state)
{
  if (!marker_pub_ || marker_pub_->get_subscription_count() == 0) {
    return;
  }

  visualization_msgs::msg::MarkerArray marker_array;
  auto stamp = node_->now();
  std::string frame_id = global_plan_.header.frame_id.empty() ?
    "map" : global_plan_.header.frame_id;
  const double lifetime_sec = 0.3;
  int marker_id = 0;

  const auto& barriers = barrier_set_.barriers();
  auto active_set = barrier_set_.getActiveBarriers(current_state);

  // Active barrier 포인터를 set로 변환 (빠른 lookup)
  std::set<const CircleBarrier*> active_ptrs(active_set.begin(), active_set.end());

  // 1. Barrier circles (CYLINDER) + h(x) 텍스트 (TEXT_VIEW_FACING)
  for (size_t i = 0; i < barriers.size(); ++i) {
    const auto& b = barriers[i];
    bool is_active = active_ptrs.count(&b) > 0;

    // Inactive barriers: skip (성능 최적화)
    if (!is_active) {
      continue;
    }

    double h = b.evaluate(current_state);
    double diameter = b.safeDistance() * 2.0;

    // Barrier disk marker
    visualization_msgs::msg::Marker disk;
    disk.header.stamp = stamp;
    disk.header.frame_id = frame_id;
    disk.ns = "mppi_cbf_barriers";
    disk.id = marker_id;
    disk.type = visualization_msgs::msg::Marker::CYLINDER;
    disk.action = visualization_msgs::msg::Marker::ADD;
    disk.pose.position.x = b.obsX();
    disk.pose.position.y = b.obsY();
    disk.pose.position.z = 0.01;
    disk.pose.orientation.w = 1.0;
    disk.scale.x = diameter;
    disk.scale.y = diameter;
    disk.scale.z = 0.02;
    disk.lifetime = rclcpp::Duration::from_seconds(lifetime_sec);

    // h(x) 기반 색상: h>2 green, 0<h≤2 yellow, h≤0 red
    if (h > 2.0) {
      disk.color.r = 0.0; disk.color.g = 1.0; disk.color.b = 0.0;
    } else if (h > 0.0) {
      // yellow → red 보간 (h: 2→0 → green→red)
      double t = h / 2.0;  // 1.0=safe, 0.0=boundary
      disk.color.r = 1.0 - t;
      disk.color.g = t + (1.0 - t) * 0.5;
      disk.color.b = 0.0;
    } else {
      disk.color.r = 1.0; disk.color.g = 0.0; disk.color.b = 0.0;
    }
    disk.color.a = 0.4;
    marker_array.markers.push_back(disk);

    // h(x) text label
    visualization_msgs::msg::Marker text;
    text.header.stamp = stamp;
    text.header.frame_id = frame_id;
    text.ns = "mppi_cbf_text";
    text.id = marker_id;
    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::msg::Marker::ADD;
    text.pose.position.x = b.obsX();
    text.pose.position.y = b.obsY();
    text.pose.position.z = 0.5;
    text.pose.orientation.w = 1.0;
    text.scale.z = 0.15;
    text.color.r = 1.0; text.color.g = 1.0; text.color.b = 1.0; text.color.a = 1.0;
    text.lifetime = rclcpp::Duration::from_seconds(lifetime_sec);

    char buf[32];
    std::snprintf(buf, sizeof(buf), "h=%.2f", h);
    text.text = buf;
    marker_array.markers.push_back(text);

    ++marker_id;
  }

  // 2. Correction arrow (filter_applied == true일 때만)
  const auto& fi = info.cbf_filter_info;
  if (fi.filter_applied && fi.u_mppi.size() > 0 && fi.u_safe.size() > 0) {
    Eigen::VectorXd du = fi.u_safe - fi.u_mppi;
    double theta = current_state(2);

    // body→world 변환: dv 방향을 로봇 heading으로 회전
    double dx = du(0) * std::cos(theta);
    double dy = du(0) * std::sin(theta);
    // omega 보정은 회전이므로 lateral로 표현
    if (du.size() > 1) {
      dx -= du(du.size() - 1) * std::sin(theta) * 0.3;
      dy += du(du.size() - 1) * std::cos(theta) * 0.3;
    }

    double arrow_len = std::sqrt(dx * dx + dy * dy);
    if (arrow_len > 0.01) {
      visualization_msgs::msg::Marker arrow;
      arrow.header.stamp = stamp;
      arrow.header.frame_id = frame_id;
      arrow.ns = "mppi_cbf_correction";
      arrow.id = 0;
      arrow.type = visualization_msgs::msg::Marker::ARROW;
      arrow.action = visualization_msgs::msg::Marker::ADD;
      arrow.lifetime = rclcpp::Duration::from_seconds(lifetime_sec);

      geometry_msgs::msg::Point start, end;
      start.x = current_state(0);
      start.y = current_state(1);
      start.z = 0.15;
      // 스케일링: 시각적으로 보기 좋게 제어 보정량을 확대
      double scale = std::min(2.0, 1.0 / arrow_len);
      end.x = start.x + dx * scale;
      end.y = start.y + dy * scale;
      end.z = 0.15;

      arrow.points.push_back(start);
      arrow.points.push_back(end);
      arrow.scale.x = 0.04;  // shaft diameter
      arrow.scale.y = 0.08;  // head diameter
      arrow.scale.z = 0.08;  // head length
      arrow.color.r = 1.0; arrow.color.g = 0.0; arrow.color.b = 1.0; arrow.color.a = 0.9;
      marker_array.markers.push_back(arrow);
    }
  }

  // 3. Status text summary
  {
    double min_h = std::numeric_limits<double>::max();
    for (double h : fi.barrier_values) {
      min_h = std::min(min_h, h);
    }

    visualization_msgs::msg::Marker status;
    status.header.stamp = stamp;
    status.header.frame_id = frame_id;
    status.ns = "mppi_cbf_status";
    status.id = 0;
    status.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    status.action = visualization_msgs::msg::Marker::ADD;
    status.pose.position.x = current_state(0);
    status.pose.position.y = current_state(1) + 0.6;
    status.pose.position.z = 0.8;
    status.pose.orientation.w = 1.0;
    status.scale.z = 0.12;
    status.color.r = 0.0; status.color.g = 1.0; status.color.b = 1.0; status.color.a = 1.0;
    status.lifetime = rclcpp::Duration::from_seconds(lifetime_sec);

    char buf[128];
    std::snprintf(buf, sizeof(buf),
      "CBF: %d/%zu active, Filter=[%s], min_h=%.2f",
      fi.num_active_barriers,
      barriers.size(),
      fi.filter_applied ? "APPLIED" : "pass",
      (min_h < 1e10) ? min_h : 0.0);
    status.text = buf;
    marker_array.markers.push_back(status);
  }

  marker_pub_->publish(marker_array);
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
