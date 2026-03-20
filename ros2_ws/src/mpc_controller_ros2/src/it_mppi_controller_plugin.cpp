// =============================================================================
// IT-MPPI (Information-Theoretic MPPI) Controller Plugin
//
// 정보이론적 비용항으로 탐색-활용 균형을 제어하는 MPPI 변형.
//
// 핵심:
//   - Sample diversity bonus: 궤적 다양성 보상
//   - KL divergence regularization: 사전 분포와의 거리 페널티
//   - Adaptive exploration: 시간에 따른 탐색 감쇠
//
// 성능 최적화:
//   - sampleInPlace / rolloutBatchInPlace 재사용 (힙 할당 0)
//   - diversity 계산: O(K²) 이지만 최종 상태만 비교 (nx << N*nx)
// =============================================================================

#include "mpc_controller_ros2/it_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>
#include <algorithm>
#include <limits>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::ITMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void ITMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  // 적응형 탐색 가중치 초기화
  current_exploration_weight_ = params_.it_exploration_weight;

  auto node = parent.lock();
  RCLCPP_INFO(
    node->get_logger(),
    "IT-MPPI plugin configured: enabled=%d, exploration_weight=%.3f, "
    "kl_weight=%.4f, diversity_threshold=%.2f, "
    "adaptive=%d (decay=%.4f)",
    params_.it_mppi_enabled,
    params_.it_exploration_weight,
    params_.it_kl_weight,
    params_.it_diversity_threshold,
    params_.it_adaptive_exploration,
    params_.it_exploration_decay);
}

// =============================================================================
// Sample diversity bonus: 각 샘플 최종 상태의 평균 쌍별 L2 거리
// =============================================================================

Eigen::VectorXd ITMPPIControllerPlugin::computeDiversityBonus(
  const std::vector<Eigen::MatrixXd>& trajectories) const
{
  int K = static_cast<int>(trajectories.size());
  Eigen::VectorXd diversity(K);
  diversity.setZero();

  if (K <= 1) return diversity;

  // 최종 상태 추출 (각 궤적의 마지막 행)
  int last_row = static_cast<int>(trajectories[0].rows()) - 1;
  int nx = static_cast<int>(trajectories[0].cols());

  // 평균 쌍별 거리
  double inv_K_minus_1 = 1.0 / (K - 1);
  for (int k = 0; k < K; ++k) {
    double sum_dist = 0.0;
    Eigen::VectorXd xk = trajectories[k].row(last_row).transpose();
    for (int j = 0; j < K; ++j) {
      if (j == k) continue;
      Eigen::VectorXd xj = trajectories[j].row(last_row).transpose();
      sum_dist += (xk - xj).norm();
    }
    diversity(k) = sum_dist * inv_K_minus_1;
  }

  return diversity;
}

// =============================================================================
// KL divergence 페널티: 0.5 * ||noise||² / sigma² (가우시안 사전 분포)
// =============================================================================

Eigen::VectorXd ITMPPIControllerPlugin::computeKLPenalty(
  const std::vector<Eigen::MatrixXd>& noise) const
{
  int K = static_cast<int>(noise.size());
  Eigen::VectorXd kl(K);
  kl.setZero();

  if (K == 0) return kl;

  int N = static_cast<int>(noise[0].rows());
  int nu = static_cast<int>(noise[0].cols());

  // sigma² 역수 벡터
  Eigen::VectorXd inv_sigma_sq(nu);
  for (int j = 0; j < nu; ++j) {
    double s = params_.noise_sigma(j);
    inv_sigma_sq(j) = (s > 1e-8) ? 1.0 / (s * s) : 1.0;
  }

  for (int k = 0; k < K; ++k) {
    double sq_sum = 0.0;
    for (int t = 0; t < N; ++t) {
      for (int j = 0; j < nu; ++j) {
        double e = noise[k](t, j);
        sq_sum += e * e * inv_sigma_sq(j);
      }
    }
    kl(k) = 0.5 * sq_sum / N;  // 시간 평균으로 정규화
  }

  return kl;
}

// =============================================================================
// computeControl — Information-Theoretic MPPI
// =============================================================================

std::pair<Eigen::VectorXd, MPPIInfo> ITMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // IT-MPPI 비활성 시 base 호출
  if (!params_.it_mppi_enabled) {
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

  // ──── STEP 3: Batch rollout + standard costs ────
  dynamics_->rolloutBatchInPlace(
    current_state, perturbed_buffer_, params_.dt, trajectory_buffer_);

  Eigen::VectorXd costs = cost_function_->compute(
    trajectory_buffer_, perturbed_buffer_, reference_trajectory);

  // ──── STEP 4: Information-Theoretic Processing ────
  Eigen::VectorXd diversity_bonus = computeDiversityBonus(trajectory_buffer_);
  Eigen::VectorXd kl_penalty = computeKLPenalty(noise_buffer_);

  double ew = current_exploration_weight_;

  // diversity threshold 보너스: 다양성이 임계값 이하이면 추가 보너스
  double mean_diversity = diversity_bonus.mean();
  double diversity_scale = 1.0;
  if (mean_diversity < params_.it_diversity_threshold && mean_diversity > 1e-8) {
    diversity_scale = params_.it_diversity_threshold / mean_diversity;
  }

  // info_cost 계산
  Eigen::VectorXd info_costs(K);
  for (int k = 0; k < K; ++k) {
    info_costs(k) = costs(k)
      - ew * diversity_scale * diversity_bonus(k)
      + params_.it_kl_weight * kl_penalty(k);
  }

  // Adaptive exploration decay
  if (params_.it_adaptive_exploration) {
    current_exploration_weight_ *= params_.it_exploration_decay;
  }

  // ──── STEP 5: MPPI weighted update ────
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(info_costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);
  }
  Eigen::VectorXd weights = weight_computation_->compute(info_costs, current_lambda);

  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_noise += weights(k) * noise_buffer_[k];
  }
  control_sequence_ += weighted_noise;
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // ──── STEP 6: Extract optimal control ────
  Eigen::VectorXd u_opt = control_sequence_.row(0).transpose();

  // Weighted average trajectory
  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int k = 0; k < K; ++k) {
    weighted_traj += weights(k) * trajectory_buffer_[k];
  }

  // Best sample
  int best_idx;
  double min_cost = info_costs.minCoeff(&best_idx);
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
  info.costs = info_costs;

  info.colored_noise_used = params_.colored_noise;
  info.adaptive_temp_used = params_.adaptive_temperature;
  info.tube_mppi_used = params_.tube_enabled;

  // IT-MPPI 메트릭
  info.it_exploration_bonus = ew * diversity_scale * diversity_bonus.mean();
  info.it_diversity_score = mean_diversity;
  info.it_kl_divergence = kl_penalty.mean();

  RCLCPP_DEBUG(
    node_->get_logger(),
    "IT-MPPI: min_cost=%.4f, diversity=%.4f, kl=%.4f, ew=%.4f, ESS=%.1f/%d",
    min_cost, mean_diversity, kl_penalty.mean(), ew, ess, K);

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
