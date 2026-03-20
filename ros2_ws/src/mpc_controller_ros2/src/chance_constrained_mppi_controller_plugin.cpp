// =============================================================================
// Chance-Constrained MPPI Controller Plugin
//
// Blackmore et al. (JGCD 2011) inspired: P(g(x) <= 0) >= 1-epsilon
//
// 핵심:
//   1. K 샘플 기반 위반 확률 추정: p_hat = count(g>0) / K
//   2. Risk 분배: Bonferroni (eps/M) 또는 Adaptive (slack 재분배)
//   3. Quantile tightening: 위반 확률 초과 시 페널티
//   4. EMA smoothed empirical quantile
//
// vs Constrained MPPI:
//   - 확률적 접근 (dual variable 불필요)
//   - sample-based violation probability
//   - risk budget allocation across constraints
// =============================================================================

#include "mpc_controller_ros2/chance_constrained_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::ChanceConstrainedMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void ChanceConstrainedMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  smoothed_quantiles_ = Eigen::Vector3d::Zero();

  auto node = parent.lock();
  RCLCPP_INFO(
    node->get_logger(),
    "CC-MPPI plugin configured: enabled=%d, "
    "risk_budget=%.3f, penalty_weight=%.1f, adaptive_risk=%d, "
    "tightening_rate=%.2f, quantile_smoothing=%.2f",
    params_.cc_mppi_enabled,
    params_.cc_risk_budget,
    params_.cc_penalty_weight,
    params_.cc_adaptive_risk,
    params_.cc_tightening_rate,
    params_.cc_quantile_smoothing);
}

// =============================================================================
// K 샘플의 per-constraint 위반량 평가: (K x 3)
// =============================================================================

Eigen::MatrixXd ChanceConstrainedMPPIControllerPlugin::evaluateSampleViolations(
  const std::vector<Eigen::MatrixXd>& perturbed_controls,
  const std::vector<Eigen::MatrixXd>& trajectories) const
{
  int K = static_cast<int>(perturbed_controls.size());
  Eigen::MatrixXd violations(K, 3);

  double dt = params_.dt;

  for (int k = 0; k < K; ++k) {
    const auto& ctrl = perturbed_controls[k];
    int N = ctrl.rows();
    int nu = ctrl.cols();

    double vel_violation = 0.0;
    double accel_violation = 0.0;

    // Velocity constraint
    for (int t = 0; t < N; ++t) {
      double v = std::abs(ctrl(t, 0));
      vel_violation += std::max(0.0, v - params_.v_max);
      if (nu >= 2) {
        double omega = std::abs(ctrl(t, 1));
        vel_violation += std::max(0.0, omega - params_.omega_max);
      }
    }

    // Acceleration constraint
    for (int t = 1; t < N; ++t) {
      double dv = std::abs(ctrl(t, 0) - ctrl(t - 1, 0)) / dt;
      accel_violation += std::max(0.0, dv - params_.constrained_accel_max_v);
      if (nu >= 2) {
        double domega = std::abs(ctrl(t, 1) - ctrl(t - 1, 1)) / dt;
        accel_violation += std::max(0.0, domega - params_.constrained_accel_max_omega);
      }
    }

    // Clearance constraint (costmap proxy)
    (void)trajectories;
    double clearance_violation = 0.0;

    violations(k, 0) = vel_violation;
    violations(k, 1) = accel_violation;
    violations(k, 2) = clearance_violation;
  }

  return violations;
}

// =============================================================================
// 위반 확률 추정: p_hat = count(g > 0) / K
// =============================================================================

Eigen::Vector3d ChanceConstrainedMPPIControllerPlugin::estimateViolationProbabilities(
  const Eigen::MatrixXd& violations) const
{
  int K = violations.rows();
  Eigen::Vector3d p_hat = Eigen::Vector3d::Zero();

  for (int i = 0; i < 3; ++i) {
    int count = 0;
    for (int k = 0; k < K; ++k) {
      if (violations(k, i) > 1e-6) {
        count++;
      }
    }
    p_hat(i) = static_cast<double>(count) / K;
  }

  return p_hat;
}

// =============================================================================
// Risk 분배: Bonferroni 또는 Adaptive
// =============================================================================

Eigen::Vector3d ChanceConstrainedMPPIControllerPlugin::allocateRisk(
  const Eigen::Vector3d& violation_probs) const
{
  double eps = params_.cc_risk_budget;
  constexpr int M = 3;
  double eps_bonf = eps / M;

  if (!params_.cc_adaptive_risk) {
    // Bonferroni: 균등 분배
    return Eigen::Vector3d(eps_bonf, eps_bonf, eps_bonf);
  }

  // Adaptive: 위반이 적은 제약에서 여유분을 재분배
  Eigen::Vector3d allocated = Eigen::Vector3d::Constant(eps_bonf);

  // Compute slack for each constraint
  Eigen::Vector3d slack = Eigen::Vector3d::Zero();
  double total_slack = 0.0;
  int num_violated = 0;

  for (int i = 0; i < M; ++i) {
    if (violation_probs(i) < eps_bonf) {
      slack(i) = eps_bonf - violation_probs(i);
      total_slack += slack(i);
    } else {
      num_violated++;
    }
  }

  // Redistribute slack to violated constraints
  if (num_violated > 0 && total_slack > 0) {
    double redistribution = total_slack / num_violated;
    for (int i = 0; i < M; ++i) {
      if (violation_probs(i) >= eps_bonf) {
        allocated(i) += redistribution;
      } else {
        allocated(i) -= slack(i);
      }
    }
  }

  // Ensure non-negative and total <= eps
  for (int i = 0; i < M; ++i) {
    allocated(i) = std::max(1e-8, allocated(i));
  }

  return allocated;
}

// =============================================================================
// Empirical quantile (O(K) nth_element)
// =============================================================================

double ChanceConstrainedMPPIControllerPlugin::empiricalQuantile(
  const Eigen::VectorXd& values, double quantile_level) const
{
  int n = values.size();
  if (n == 0) return 0.0;

  std::vector<double> sorted(n);
  for (int i = 0; i < n; ++i) {
    sorted[i] = values(i);
  }

  int idx = std::min(static_cast<int>(std::ceil(quantile_level * n)) - 1, n - 1);
  idx = std::max(0, idx);

  std::nth_element(sorted.begin(), sorted.begin() + idx, sorted.end());
  return sorted[idx];
}

// =============================================================================
// Chance-constrained augmented costs
// =============================================================================

Eigen::VectorXd ChanceConstrainedMPPIControllerPlugin::computeChanceConstrainedCosts(
  const Eigen::VectorXd& base_costs,
  const Eigen::MatrixXd& violations,
  const Eigen::Vector3d& allocated_risk) const
{
  int K = static_cast<int>(base_costs.size());
  Eigen::VectorXd augmented_costs = base_costs;

  Eigen::Vector3d p_hat = estimateViolationProbabilities(violations);

  for (int i = 0; i < 3; ++i) {
    if (p_hat(i) <= allocated_risk(i)) {
      continue;  // Within risk budget — no penalty
    }

    // Compute quantile for this constraint
    Eigen::VectorXd col = violations.col(i);
    double q = empiricalQuantile(col, 1.0 - allocated_risk(i));

    // Penalty for samples that exceed the quantile threshold
    double excess = p_hat(i) - allocated_risk(i);
    for (int k = 0; k < K; ++k) {
      if (violations(k, i) > 1e-6) {
        double sample_penalty = params_.cc_penalty_weight * excess *
          std::max(violations(k, i), q);
        augmented_costs(k) += sample_penalty;
      }
    }
  }

  return augmented_costs;
}

// =============================================================================
// computeControl — Chance-Constrained MPPI
// =============================================================================

std::pair<Eigen::VectorXd, MPPIInfo> ChanceConstrainedMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  if (!params_.cc_mppi_enabled) {
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

  // ---- STEP 5: CC processing ----
  Eigen::MatrixXd violations = evaluateSampleViolations(
    perturbed_buffer_, trajectory_buffer_);

  Eigen::Vector3d p_hat = estimateViolationProbabilities(violations);
  Eigen::Vector3d eps_allocated = allocateRisk(p_hat);

  Eigen::VectorXd augmented_costs = computeChanceConstrainedCosts(
    base_costs, violations, eps_allocated);

  // Update smoothed quantiles (EMA)
  for (int i = 0; i < 3; ++i) {
    Eigen::VectorXd col = violations.col(i);
    double q = empiricalQuantile(col, 1.0 - eps_allocated(i));
    double alpha = params_.cc_quantile_smoothing;
    smoothed_quantiles_(i) = alpha * q + (1.0 - alpha) * smoothed_quantiles_(i);
  }

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

  // ---- STEP 8: Extract optimal control ----
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

  // Count tightened constraints
  int num_tightened = 0;
  for (int i = 0; i < 3; ++i) {
    if (p_hat(i) > eps_allocated(i)) num_tightened++;
  }

  // ---- STEP 9: Build info ----
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

  // CC-MPPI metrics
  info.cc_violation_probability = p_hat.maxCoeff();
  info.cc_effective_risk = eps_allocated.sum();
  info.cc_num_tightened = num_tightened;

  RCLCPP_DEBUG(
    node_->get_logger(),
    "CC-MPPI: min_cost=%.4f, p_hat=[%.4f, %.4f, %.4f], "
    "eps_alloc=[%.4f, %.4f, %.4f], num_tightened=%d, ESS=%.1f/%d",
    min_cost,
    p_hat(0), p_hat(1), p_hat(2),
    eps_allocated(0), eps_allocated(1), eps_allocated(2),
    num_tightened, ess, K);

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
