// =============================================================================
// Robust MPPI (Distributionally Robust) Controller Plugin
//
// Minimax 최적화: 분포 불확실성 하의 worst-case 기대 비용 최소화.
// CVaR-like worst-case 추정 + 분산 페널티 + Wasserstein 보수성.
//
// 성능 최적화:
//   - sampleInPlace / rolloutBatchInPlace 재사용 (힙 할당 0)
//   - nth_element O(K) 부분 정렬 (full sort 불필요)
//   - Wasserstein 페널티: 이웃 비용 차분 기반 gradient norm 근사
// =============================================================================

#include "mpc_controller_ros2/robust_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::RobustMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void RobustMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  auto node = parent.lock();

  RCLCPP_INFO(
    node->get_logger(),
    "Robust-MPPI plugin configured: enabled=%d, alpha=%.2f, penalty=%.2f, "
    "wasserstein_radius=%.3f, adaptive_alpha=%d",
    params_.robust_enabled,
    params_.robust_alpha,
    params_.robust_penalty,
    params_.robust_wasserstein_radius,
    params_.robust_adaptive_alpha);
}

// =============================================================================
// Wasserstein 페널티: 비용 기울기 norm 근사 (이웃 비용 차분)
// =============================================================================

Eigen::VectorXd RobustMPPIControllerPlugin::computeWassersteinPenalty(
  const Eigen::VectorXd& costs) const
{
  int K = static_cast<int>(costs.size());
  Eigen::VectorXd penalty = Eigen::VectorXd::Zero(K);

  if (K < 2) return penalty;

  // 정렬된 인덱스로 이웃 비용 차분 기반 gradient norm 근사
  std::vector<int> sorted_idx(K);
  std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
  std::sort(sorted_idx.begin(), sorted_idx.end(),
    [&costs](int a, int b) { return costs(a) < costs(b); });

  for (int i = 0; i < K; ++i) {
    int idx = sorted_idx[i];
    double grad_est = 0.0;
    if (i > 0 && i < K - 1) {
      // 중앙 차분
      grad_est = std::abs(costs(sorted_idx[i + 1]) - costs(sorted_idx[i - 1])) * 0.5;
    } else if (i == 0 && K > 1) {
      grad_est = std::abs(costs(sorted_idx[1]) - costs(sorted_idx[0]));
    } else if (i == K - 1 && K > 1) {
      grad_est = std::abs(costs(sorted_idx[K - 1]) - costs(sorted_idx[K - 2]));
    }
    penalty(idx) = grad_est;
  }

  return penalty;
}

// =============================================================================
// Robust 처리: 분산 페널티 + Wasserstein + CVaR worst-alpha 추정
// =============================================================================

std::pair<double, double> RobustMPPIControllerPlugin::applyRobustProcessing(
  Eigen::VectorXd& costs) const
{
  int K = static_cast<int>(costs.size());
  if (K == 0) return {0.0, params_.robust_alpha};

  // (a) 비용 분산 계산
  double mean_cost = costs.mean();
  double variance = (costs.array() - mean_cost).square().mean();

  // (b) 분산 페널티 추가: robust_cost[k] = cost[k] + penalty * variance
  if (params_.robust_penalty > 0.0) {
    costs.array() += params_.robust_penalty * variance;
  }

  // (c) Wasserstein 페널티
  if (params_.robust_wasserstein_radius > 0.0) {
    Eigen::VectorXd w_penalty = computeWassersteinPenalty(costs);
    costs += params_.robust_wasserstein_radius * w_penalty;
  }

  // (d) Adaptive alpha: 비용 스프레드에 따라 조정
  double effective_alpha = params_.robust_alpha;
  if (params_.robust_adaptive_alpha) {
    double cost_range = costs.maxCoeff() - costs.minCoeff();
    double normalized_spread = cost_range / (std::abs(mean_cost) + 1e-8);

    // 높은 스프레드 → 작은 alpha (더 보수적)
    // 낮은 스프레드 → 큰 alpha (덜 보수적)
    double spread_factor = std::exp(-normalized_spread);
    effective_alpha = std::clamp(
      params_.robust_alpha * (0.5 + spread_factor),
      0.05, 1.0);
  }

  // (e) CVaR worst-alpha 추정: worst alpha 분율의 평균 비용
  int worst_count = std::max(1, static_cast<int>(std::ceil(effective_alpha * K)));
  worst_count = std::min(worst_count, K);

  // 상위(최악) worst_count 비용 찾기
  std::vector<int> indices(K);
  std::iota(indices.begin(), indices.end(), 0);
  std::nth_element(indices.begin(), indices.begin() + (K - worst_count), indices.end(),
    [&costs](int a, int b) { return costs(a) < costs(b); });

  double worst_case_cost = 0.0;
  for (int i = K - worst_count; i < K; ++i) {
    worst_case_cost += costs(indices[i]);
  }
  worst_case_cost /= worst_count;

  return {worst_case_cost, effective_alpha};
}

// =============================================================================
// computeControl — Robust MPPI 파이프라인
// =============================================================================

std::pair<Eigen::VectorXd, MPPIInfo> RobustMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // Robust 비활성 시 base 호출
  if (!params_.robust_enabled) {
    return MPPIControllerPlugin::computeControl(current_state, reference_trajectory);
  }

  int N = params_.N;
  int K = params_.K;
  int nu = dynamics_->model().controlDim();
  int nx = dynamics_->model().stateDim();

  // ──── STEP 1: Warm-start (shift control sequence) ────
  for (int t = 0; t < N - 1; ++t) {
    control_sequence_.row(t) = control_sequence_.row(t + 1);
  }
  control_sequence_.row(N - 1) = control_sequence_.row(N - 2);

  // ──── STEP 2: Sample noise ────
  sampler_->sampleInPlace(noise_buffer_, K, N, nu);
  if (static_cast<int>(perturbed_buffer_.size()) != K) {
    perturbed_buffer_.resize(K, Eigen::MatrixXd::Zero(N, nu));
  }

  for (int k = 0; k < K; ++k) {
    perturbed_buffer_[k].noalias() = control_sequence_ + noise_buffer_[k];
    perturbed_buffer_[k] = dynamics_->clipControls(perturbed_buffer_[k]);
  }

  // ──── STEP 3: Batch rollout + costs ────
  dynamics_->rolloutBatchInPlace(
    current_state, perturbed_buffer_, params_.dt, trajectory_buffer_);

  Eigen::VectorXd costs = cost_function_->compute(
    trajectory_buffer_, perturbed_buffer_, reference_trajectory);

  // ──── STEP 4: ROBUST PROCESSING ────
  Eigen::VectorXd original_costs = costs;  // 원본 보존 (info용)
  auto [worst_case_cost, effective_alpha] = applyRobustProcessing(costs);

  // ──── STEP 5: IT-normalization on robust costs → weights ────
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);
  }
  Eigen::VectorXd weights = weight_computation_->compute(costs, current_lambda);

  // ──── STEP 6: Weighted update ────
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_noise += weights(k) * noise_buffer_[k];
  }
  control_sequence_ += weighted_noise;
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // ──── STEP 7: Extract optimal control ────
  Eigen::VectorXd u_opt = control_sequence_.row(0).transpose();

  // Weighted average trajectory
  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int k = 0; k < K; ++k) {
    weighted_traj += weights(k) * trajectory_buffer_[k];
  }

  // Best sample
  int best_idx;
  double min_cost = original_costs.minCoeff(&best_idx);
  double ess = computeESS(weights);

  // Build info
  MPPIInfo info;
  info.sample_trajectories = trajectory_buffer_;
  info.sample_weights = weights;
  info.best_trajectory = trajectory_buffer_[best_idx];
  info.weighted_avg_trajectory = weighted_traj;
  info.temperature = (params_.adaptive_temperature && adaptive_temp_) ?
    adaptive_temp_->getLambda() : params_.lambda;
  info.ess = ess;
  info.costs = original_costs;

  info.colored_noise_used = params_.colored_noise;
  info.adaptive_temp_used = params_.adaptive_temperature;
  info.tube_mppi_used = params_.tube_enabled;

  info.robust_worst_case_cost = worst_case_cost;
  info.robust_effective_alpha = effective_alpha;

  RCLCPP_DEBUG(
    node_->get_logger(),
    "Robust-MPPI: min_cost=%.4f, worst_case=%.4f, eff_alpha=%.3f, ESS=%.1f/%d",
    min_cost, worst_case_cost, effective_alpha, ess, K);

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
