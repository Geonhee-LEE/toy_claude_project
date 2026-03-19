// =============================================================================
// CEM-MPPI Controller Plugin
//
// Reference: Pinneri et al. (2021) "Sample-Efficient Cross-Entropy Method"
//
// CEM 반복으로 샘플링 분포(μ,σ)를 정제 → 마지막 반복에서 MPPI 가중 업데이트.
// DIAL-MPPI와 구조적 유사성: 둘 다 다중 inner loop. DIAL=노이즈 감쇠, CEM=분포 정제.
//
// 성능 최적화:
//   - sampleInPlace / rolloutBatchInPlace 재사용
//   - Elite 선택: std::nth_element O(K) (full sort 불필요)
//   - 마지막 반복 결과 시각화 재사용 (추가 롤아웃 없음)
// =============================================================================

#include "mpc_controller_ros2/cem_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::CemMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void CemMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  auto node = parent.lock();

  int max_iter = params_.cem_adaptive_enabled ?
    params_.cem_adaptive_max_iter : params_.cem_iterations;

  RCLCPP_INFO(
    node->get_logger(),
    "CEM-MPPI plugin configured: enabled=%d, iterations=%d, "
    "elite_ratio=%.2f, momentum=%.2f, sigma_min=%.4f, sigma_decay=%.2f, "
    "adaptive=%d (tol=%.4f, min=%d, max=%d)",
    params_.cem_enabled,
    max_iter,
    params_.cem_elite_ratio,
    params_.cem_momentum,
    params_.cem_sigma_min,
    params_.cem_sigma_decay,
    params_.cem_adaptive_enabled,
    params_.cem_adaptive_cost_tol,
    params_.cem_adaptive_min_iter,
    params_.cem_adaptive_max_iter);
}

// =============================================================================
// Elite 선택: costs 기준 top-p% 인덱스 반환
// =============================================================================

std::vector<int> CemMPPIControllerPlugin::selectElites(
  const Eigen::VectorXd& costs, int num_elites) const
{
  int K = static_cast<int>(costs.size());
  num_elites = std::clamp(num_elites, 1, K);

  // 인덱스 배열 생성
  std::vector<int> indices(K);
  std::iota(indices.begin(), indices.end(), 0);

  // nth_element로 top-p% 부분 정렬 (O(K))
  std::nth_element(indices.begin(), indices.begin() + num_elites, indices.end(),
    [&costs](int a, int b) { return costs(a) < costs(b); });

  indices.resize(num_elites);
  return indices;
}

// =============================================================================
// Elite로부터 μ, σ refit
// =============================================================================

void CemMPPIControllerPlugin::refitDistribution(
  const std::vector<Eigen::MatrixXd>& perturbed_controls,
  const std::vector<int>& elite_indices,
  Eigen::MatrixXd& mean_out,
  Eigen::VectorXd& sigma_out) const
{
  if (elite_indices.empty()) return;

  int N = mean_out.rows();
  int nu = mean_out.cols();
  int M = static_cast<int>(elite_indices.size());
  double inv_M = 1.0 / M;

  // μ = mean(elites)
  mean_out.setZero();
  for (int idx : elite_indices) {
    mean_out += perturbed_controls[idx];
  }
  mean_out *= inv_M;

  // σ = std(elites) — per-nu, time-averaged
  sigma_out.setZero();
  for (int idx : elite_indices) {
    Eigen::MatrixXd diff = perturbed_controls[idx] - mean_out;
    for (int j = 0; j < nu; ++j) {
      sigma_out(j) += diff.col(j).squaredNorm();
    }
  }
  sigma_out *= inv_M / N;  // variance per nu, averaged over time
  sigma_out = sigma_out.cwiseSqrt();
}

// =============================================================================
// computeControl — CEM iteration + MPPI final update
// =============================================================================

std::pair<Eigen::VectorXd, MPPIInfo> CemMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // CEM 비활성 시 base 호출
  if (!params_.cem_enabled) {
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

  // ──── STEP 2: CEM 분포 초기화 ────
  Eigen::MatrixXd mu = control_sequence_;           // (N, nu)
  Eigen::VectorXd sigma = params_.noise_sigma;      // (nu,)

  // ──── STEP 3: CEM 반복 루프 ────
  int max_iter = params_.cem_adaptive_enabled ?
    params_.cem_adaptive_max_iter : params_.cem_iterations;
  double prev_elite_cost = std::numeric_limits<double>::infinity();
  int actual_iterations = 0;

  // 마지막 반복 결과 저장
  Eigen::VectorXd last_costs;
  Eigen::VectorXd last_weights;
  double last_elite_mean_cost = 0.0;

  for (int i = 1; i <= max_iter; ++i) {
    // 3a: Sample K from N(μ, diag(σ²))
    sampler_->sampleInPlace(noise_buffer_, K, N, nu);
    if (static_cast<int>(perturbed_buffer_.size()) != K) {
      perturbed_buffer_.resize(K, Eigen::MatrixXd::Zero(N, nu));
    }

    for (int k = 0; k < K; ++k) {
      // σ 스케일링
      for (int h = 0; h < N; ++h) {
        noise_buffer_[k].row(h).array() *= sigma.transpose().array();
      }
      perturbed_buffer_[k].noalias() = mu + noise_buffer_[k];
      perturbed_buffer_[k] = dynamics_->clipControls(perturbed_buffer_[k]);
    }

    // 3b: Batch rollout + cost
    dynamics_->rolloutBatchInPlace(
      current_state, perturbed_buffer_, params_.dt, trajectory_buffer_);

    Eigen::VectorXd costs = cost_function_->compute(
      trajectory_buffer_, perturbed_buffer_, reference_trajectory);

    // 3c: Elite 선택
    int num_elites = std::max(1,
      static_cast<int>(std::floor(params_.cem_elite_ratio * K)));
    auto elite_indices = selectElites(costs, num_elites);

    // Elite mean cost
    double elite_mean_cost = 0.0;
    for (int idx : elite_indices) {
      elite_mean_cost += costs(idx);
    }
    elite_mean_cost /= num_elites;
    last_elite_mean_cost = elite_mean_cost;

    // 3d: Refit μ, σ from elites
    Eigen::MatrixXd mu_new = Eigen::MatrixXd::Zero(N, nu);
    Eigen::VectorXd sigma_new = Eigen::VectorXd::Zero(nu);
    refitDistribution(perturbed_buffer_, elite_indices, mu_new, sigma_new);

    // 3e: Momentum blending
    mu = (1.0 - params_.cem_momentum) * mu_new + params_.cem_momentum * mu;

    // 3f: σ 감쇠 + floor
    sigma = sigma_new * params_.cem_sigma_decay;
    for (int j = 0; j < nu; ++j) {
      sigma(j) = std::max(sigma(j), params_.cem_sigma_min);
    }

    actual_iterations = i;
    last_costs = std::move(costs);

    // 3g: Adaptive 조기 종료
    if (params_.cem_adaptive_enabled && i >= params_.cem_adaptive_min_iter) {
      double improvement = (prev_elite_cost - elite_mean_cost) /
        (std::abs(prev_elite_cost) + 1e-8);
      if (improvement < params_.cem_adaptive_cost_tol) {
        break;
      }
    }
    prev_elite_cost = elite_mean_cost;
  }

  // ──── STEP 4: 마지막 반복에서 MPPI 가중 업데이트 ────
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(last_costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);
  }
  last_weights = weight_computation_->compute(last_costs, current_lambda);

  // noise_for_weight = perturbed - mu (마지막 반복의 mu 기준)
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_noise += last_weights(k) * (perturbed_buffer_[k] - mu);
  }
  control_sequence_ = mu + weighted_noise;
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // ──── STEP 5: Extract optimal control ────
  Eigen::VectorXd u_opt = control_sequence_.row(0).transpose();

  // Weighted average trajectory
  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int k = 0; k < K; ++k) {
    weighted_traj += last_weights(k) * trajectory_buffer_[k];
  }

  // Best sample
  int best_idx;
  double min_cost = last_costs.minCoeff(&best_idx);
  double ess = computeESS(last_weights);

  // Build info
  MPPIInfo info;
  info.sample_trajectories = trajectory_buffer_;
  info.sample_weights = last_weights;
  info.best_trajectory = trajectory_buffer_[best_idx];
  info.weighted_avg_trajectory = weighted_traj;
  info.temperature = (params_.adaptive_temperature && adaptive_temp_) ?
    adaptive_temp_->getLambda() : params_.lambda;
  info.ess = ess;
  info.costs = last_costs;

  info.colored_noise_used = params_.colored_noise;
  info.adaptive_temp_used = params_.adaptive_temperature;
  info.tube_mppi_used = params_.tube_enabled;

  info.cem_iterations_used = actual_iterations;
  info.cem_elite_mean_cost = last_elite_mean_cost;

  RCLCPP_DEBUG(
    node_->get_logger(),
    "CEM-MPPI: iter=%d/%d, min_cost=%.4f, elite_mean=%.4f, ESS=%.1f/%d",
    actual_iterations, max_iter, min_cost, last_elite_mean_cost, ess, K);

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
