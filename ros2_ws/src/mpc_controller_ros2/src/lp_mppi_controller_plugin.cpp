// =============================================================================
// LP-MPPI (Low-Pass Filtering MPPI) Controller Plugin
//
// 2025 연구 기반: 샘플링된 제어 시퀀스에 1차 IIR Low-Pass 필터를 적용하여
// 고주파 chattering을 제거하고 제어 품질을 향상.
//
// 수식:
//   y[t] = α·x[t] + (1-α)·y[t-1]         ... (1) 1st-order IIR
//   α = dt / (τ + dt),  τ = 1/(2πf_c)     ... (2) 시간 상수
//
// Smooth-MPPI와 상보적:
//   Smooth-MPPI — 시간 도메인 (Δu 리파라미터화, jerk cost)
//   LP-MPPI     — 주파수 도메인 (IIR 필터, 컷오프 주파수)
// =============================================================================

#include "mpc_controller_ros2/lp_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::LPMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void LPMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  int nu = dynamics_->model().controlDim();
  u_prev_ = Eigen::VectorXd::Zero(nu);
  prev_sequence_ = Eigen::MatrixXd::Zero(params_.N, nu);

  // 수식 (2): α = dt / (τ + dt),  τ = 1/(2πf_c)
  if (params_.lp_enabled && params_.lp_cutoff_frequency > 0.0) {
    double tau = 1.0 / (2.0 * M_PI * params_.lp_cutoff_frequency);
    lp_alpha_ = params_.dt / (tau + params_.dt);
  } else {
    lp_alpha_ = 1.0;  // 필터 비활성화 (pass-through)
  }

  auto node = parent.lock();
  RCLCPP_INFO(
    node->get_logger(),
    "LP-MPPI plugin configured: f_c=%.1f Hz, alpha=%.4f, filter_all=%s",
    params_.lp_cutoff_frequency,
    lp_alpha_,
    params_.lp_filter_all_samples ? "true" : "false");
}

void LPMPPIControllerPlugin::applyLowPassFilter(
  Eigen::MatrixXd& sequence,
  double alpha,
  const Eigen::VectorXd& initial) const
{
  // 수식 (1): y[t] = α·x[t] + (1-α)·y[t-1]
  int N = sequence.rows();
  Eigen::VectorXd prev = initial;
  for (int t = 0; t < N; ++t) {
    Eigen::VectorXd filtered = alpha * sequence.row(t).transpose()
                              + (1.0 - alpha) * prev;
    sequence.row(t) = filtered.transpose();
    prev = filtered;
  }
}

std::pair<Eigen::VectorXd, MPPIInfo> LPMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  int N = params_.N;
  int K = params_.K;
  int nu = dynamics_->model().controlDim();
  int nx = dynamics_->model().stateDim();

  // ──── Step 1: Shift control sequence (warm start) ────
  for (int t = 0; t < N - 1; ++t) {
    control_sequence_.row(t) = control_sequence_.row(t + 1);
  }
  control_sequence_.row(N - 1).setZero();

  // ──── Step 2: 노이즈 샘플링 ────
  auto noise = sampler_->sample(K, N, nu);

  // ──── Step 3: Perturbed controls + LP 필터 적용 ────
  std::vector<Eigen::MatrixXd> perturbed_controls;
  perturbed_controls.reserve(K);
  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd u_k = control_sequence_ + noise[k];

    // 전체 샘플에 LP 필터 적용 (고주파 노이즈 제거)
    if (params_.lp_enabled && params_.lp_filter_all_samples) {
      applyLowPassFilter(u_k, lp_alpha_, u_prev_);
    }

    u_k = dynamics_->clipControls(u_k);
    perturbed_controls.push_back(u_k);
  }

  // ──── Step 4: Batch rollout ────
  auto trajectories = dynamics_->rolloutBatch(
    current_state, perturbed_controls, params_.dt);

  // ──── Step 5: Cost 계산 ────
  Eigen::VectorXd costs = cost_function_->compute(
    trajectories, perturbed_controls, reference_trajectory);

  // ──── Step 6: 가중치 계산 (adaptive temp 포함) ────
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);
  }
  Eigen::VectorXd weights = weight_computation_->compute(costs, current_lambda);

  // ──── Step 7: 가중 업데이트 ────
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_noise += weights(k) * noise[k];
  }
  control_sequence_ += weighted_noise;

  // ──── Step 8: 최적 시퀀스에 LP 필터 적용 ────
  if (params_.lp_enabled) {
    applyLowPassFilter(control_sequence_, lp_alpha_, u_prev_);
  }
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // ──── Step 9: 최적 제어 추출 ────
  Eigen::VectorXd u_opt = control_sequence_.row(0).transpose();
  u_prev_ = u_opt;

  // ──── Step 10: Info 구성 ────
  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int k = 0; k < K; ++k) {
    weighted_traj += weights(k) * trajectories[k];
  }

  int best_idx;
  double min_cost = costs.minCoeff(&best_idx);
  double ess = computeESS(weights);

  MPPIInfo info;
  info.sample_trajectories = trajectories;
  info.sample_weights = weights;
  info.best_trajectory = trajectories[best_idx];
  info.weighted_avg_trajectory = weighted_traj;
  info.temperature = (params_.adaptive_temperature && adaptive_temp_) ?
    adaptive_temp_->getLambda() : params_.lambda;
  info.ess = ess;
  info.costs = costs;

  info.colored_noise_used = params_.colored_noise;
  info.adaptive_temp_used = params_.adaptive_temperature;
  info.tube_mppi_used = false;

  RCLCPP_DEBUG(
    node_->get_logger(),
    "LP-MPPI: min_cost=%.4f, ESS=%.1f/%d, alpha=%.4f, f_c=%.1f Hz",
    min_cost, ess, K, lp_alpha_, params_.lp_cutoff_frequency);

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
