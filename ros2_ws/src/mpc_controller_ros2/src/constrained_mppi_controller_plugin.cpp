// =============================================================================
// Constrained MPPI Controller Plugin
//
// Augmented Lagrangian 기법으로 hard constraints 처리:
//   L(u,lambda,mu) = cost(u) + lambda^T g(u) + (mu/2)||max(0, g(u))||^2
//
// 제약 조건:
//   - 속도: max(0, |u_t| - u_limit) per control dim
//   - 가속도: max(0, |u_t - u_{t-1}|/dt - a_limit) per control dim
//   - 장애물 클리어런스: max(0, d_min - min_dist) (비용 프록시)
//
// Dual update:
//   lambda = max(0, lambda + mu * g)
//   mu = min(mu * growth, mu_max)  if violation > 0
// =============================================================================

#include "mpc_controller_ros2/constrained_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>
#include <algorithm>
#include <limits>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::ConstrainedMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void ConstrainedMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  // Initialize dual variables
  lambda_ = Eigen::Vector3d::Zero();
  mu_ = params_.constrained_mu_init;

  auto node = parent.lock();
  RCLCPP_INFO(
    node->get_logger(),
    "Constrained-MPPI plugin configured: enabled=%d, "
    "mu_init=%.2f, mu_growth=%.2f, mu_max=%.1f, "
    "accel_max_v=%.2f, accel_max_omega=%.2f, clearance_min=%.2f",
    params_.constrained_enabled,
    params_.constrained_mu_init,
    params_.constrained_mu_growth,
    params_.constrained_mu_max,
    params_.constrained_accel_max_v,
    params_.constrained_accel_max_omega,
    params_.constrained_clearance_min);
}

// =============================================================================
// 제약 위반 평가: (3,) [vel, accel, clearance]
// =============================================================================

Eigen::Vector3d ConstrainedMPPIControllerPlugin::evaluateConstraintViolation(
  const Eigen::MatrixXd& control_seq,
  const Eigen::MatrixXd& trajectory) const
{
  int N = control_seq.rows();
  int nu = control_seq.cols();
  double dt = params_.dt;

  double vel_violation = 0.0;
  double accel_violation = 0.0;
  double clearance_violation = 0.0;

  // --- 속도 제약 ---
  for (int t = 0; t < N; ++t) {
    // v (첫 번째 제어 차원)
    double v = std::abs(control_seq(t, 0));
    double v_limit = params_.v_max;
    vel_violation += std::max(0.0, v - v_limit);

    // omega (두 번째 제어 차원, 존재 시)
    if (nu >= 2) {
      double omega = std::abs(control_seq(t, 1));
      double omega_limit = params_.omega_max;
      vel_violation += std::max(0.0, omega - omega_limit);
    }
  }

  // --- 가속도 제약 ---
  for (int t = 1; t < N; ++t) {
    double dv = std::abs(control_seq(t, 0) - control_seq(t - 1, 0)) / dt;
    accel_violation += std::max(0.0, dv - params_.constrained_accel_max_v);

    if (nu >= 2) {
      double domega = std::abs(control_seq(t, 1) - control_seq(t - 1, 1)) / dt;
      accel_violation += std::max(0.0, domega - params_.constrained_accel_max_omega);
    }
  }

  // --- 장애물 클리어런스 제약 (간소화: costmap 기반) ---
  // 실제 장애물 검사는 costmap cost에 포함됨.
  // 여기서는 clearance_violation = 0 (costmap cost가 대신 처리)
  // 향후 확장: barrier_set_ obstacles 순회 가능
  (void)trajectory;  // clearance는 costmap cost로 대체

  return Eigen::Vector3d(vel_violation, accel_violation, clearance_violation);
}

// =============================================================================
// Augmented cost 계산
// =============================================================================

Eigen::VectorXd ConstrainedMPPIControllerPlugin::computeAugmentedCosts(
  const Eigen::VectorXd& base_costs,
  const std::vector<Eigen::MatrixXd>& perturbed_controls,
  const std::vector<Eigen::MatrixXd>& trajectories) const
{
  int K = static_cast<int>(base_costs.size());
  Eigen::VectorXd augmented_costs = base_costs;

  for (int k = 0; k < K; ++k) {
    Eigen::Vector3d g = evaluateConstraintViolation(
      perturbed_controls[k], trajectories[k]);

    // Augmented Lagrangian: lambda^T g + (mu/2) ||max(0, g)||^2
    double lagrangian_term = lambda_.dot(g);
    double penalty_term = 0.0;
    for (int i = 0; i < 3; ++i) {
      double gi = std::max(0.0, g(i));
      penalty_term += gi * gi;
    }
    penalty_term *= (mu_ / 2.0);

    augmented_costs(k) += lagrangian_term + penalty_term;
  }

  return augmented_costs;
}

// =============================================================================
// Dual variable 업데이트
// =============================================================================

void ConstrainedMPPIControllerPlugin::updateDualVariables(
  const Eigen::Vector3d& violation)
{
  // lambda update: lambda = max(0, lambda + mu * g)
  for (int i = 0; i < 3; ++i) {
    lambda_(i) = std::max(0.0, lambda_(i) + mu_ * violation(i));
  }

  // mu growth: if any violation > 0, grow mu
  bool any_violated = false;
  for (int i = 0; i < 3; ++i) {
    if (violation(i) > 1e-6) {
      any_violated = true;
      break;
    }
  }
  if (any_violated) {
    mu_ = std::min(mu_ * params_.constrained_mu_growth, params_.constrained_mu_max);
  }
}

// =============================================================================
// computeControl — Augmented Lagrangian MPPI
// =============================================================================

std::pair<Eigen::VectorXd, MPPIInfo> ConstrainedMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // 비활성 시 base 호출
  if (!params_.constrained_enabled) {
    return MPPIControllerPlugin::computeControl(current_state, reference_trajectory);
  }

  int N = params_.N;
  int K = params_.K;
  int nu = dynamics_->model().controlDim();
  int nx = dynamics_->model().stateDim();

  // ---- STEP 1: Warm-start (shift control sequence) ----
  for (int t = 0; t < N - 1; ++t) {
    control_sequence_.row(t) = control_sequence_.row(t + 1);
  }
  control_sequence_.row(N - 1) = control_sequence_.row(N - 2);

  // ---- STEP 2: Sample noise ----
  sampler_->sampleInPlace(noise_buffer_, K, N, nu);
  if (static_cast<int>(perturbed_buffer_.size()) != K) {
    perturbed_buffer_.resize(K, Eigen::MatrixXd::Zero(N, nu));
  }

  for (int k = 0; k < K; ++k) {
    perturbed_buffer_[k].noalias() = control_sequence_ + noise_buffer_[k];
    perturbed_buffer_[k] = dynamics_->clipControls(perturbed_buffer_[k]);
  }

  // ---- STEP 3: Batch rollout ----
  dynamics_->rolloutBatchInPlace(
    current_state, perturbed_buffer_, params_.dt, trajectory_buffer_);

  // ---- STEP 4: Standard costs ----
  Eigen::VectorXd base_costs = cost_function_->compute(
    trajectory_buffer_, perturbed_buffer_, reference_trajectory);

  // ---- STEP 5: Augmented Lagrangian costs ----
  Eigen::VectorXd augmented_costs = computeAugmentedCosts(
    base_costs, perturbed_buffer_, trajectory_buffer_);

  // ---- STEP 6: IT-normalization -> weights ----
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(augmented_costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);
  }
  Eigen::VectorXd weights = weight_computation_->compute(augmented_costs, current_lambda);

  // ---- STEP 7: Weighted update ----
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_noise += weights(k) * noise_buffer_[k];
  }
  control_sequence_ += weighted_noise;
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // ---- STEP 8: Dual update on applied control ----
  // Rollout applied control for constraint evaluation
  std::vector<Eigen::MatrixXd> applied_controls = {control_sequence_};
  std::vector<Eigen::MatrixXd> applied_traj(1, Eigen::MatrixXd::Zero(N + 1, nx));
  dynamics_->rolloutBatchInPlace(current_state, applied_controls, params_.dt, applied_traj);

  Eigen::Vector3d violation = evaluateConstraintViolation(
    control_sequence_, applied_traj[0]);
  updateDualVariables(violation);

  // ---- STEP 9: Extract optimal control ----
  Eigen::VectorXd u_opt = control_sequence_.row(0).transpose();

  // Weighted average trajectory
  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int k = 0; k < K; ++k) {
    weighted_traj += weights(k) * trajectory_buffer_[k];
  }

  // Best sample
  int best_idx;
  double min_cost = augmented_costs.minCoeff(&best_idx);
  double ess = computeESS(weights);

  // Count violated constraints
  int num_violated = 0;
  for (int i = 0; i < 3; ++i) {
    if (violation(i) > 1e-6) num_violated++;
  }

  // ---- STEP 10: Build info ----
  MPPIInfo info;
  info.sample_trajectories = trajectory_buffer_;
  info.sample_weights = weights;
  info.best_trajectory = trajectory_buffer_[best_idx];
  info.weighted_avg_trajectory = weighted_traj;
  info.temperature = (params_.adaptive_temperature && adaptive_temp_) ?
    adaptive_temp_->getLambda() : params_.lambda;
  info.ess = ess;
  info.costs = augmented_costs;

  info.colored_noise_used = params_.colored_noise;
  info.adaptive_temp_used = params_.adaptive_temperature;
  info.tube_mppi_used = params_.tube_enabled;

  // Constrained MPPI metrics
  info.constrained_total_violation = violation.sum();
  info.constrained_mu = mu_;
  info.constrained_num_violated = num_violated;

  RCLCPP_DEBUG(
    node_->get_logger(),
    "Constrained-MPPI: min_cost=%.4f, violation=[%.4f, %.4f, %.4f], "
    "lambda=[%.4f, %.4f, %.4f], mu=%.2f, num_violated=%d, ESS=%.1f/%d",
    min_cost,
    violation(0), violation(1), violation(2),
    lambda_(0), lambda_(1), lambda_(2),
    mu_, num_violated, ess, K);

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
