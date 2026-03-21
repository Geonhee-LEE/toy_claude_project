// =============================================================================
// CC-CBF-MPPI Controller Plugin
//
// Chance-Constrained CBF-MPPI: 확률적 risk budget + 기하학적 barrier 통합
//
// 핵심:
//   1. 4종 제약: velocity, acceleration, clearance (barrier h<0), CBF rate
//   2. K 샘플 궤적에서 barrier 위반 확률 sample-based 추정
//   3. Risk budget 분배 + quantile tightening (CC-MPPI 프레임워크)
//   4. 선택적 CBF 투영 (Shield-MPPI 스타일 안전 필터)
//
// vs CC-MPPI:  clearance = barrier_set_ 기반 (placeholder 아님)
// vs Shield:   확률적 접근 + CBF 투영은 선택적
//
// 참고: Blackmore et al. (JGCD 2011) + Ames et al. (2019)
// =============================================================================

#include "mpc_controller_ros2/cc_cbf_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::CCCBFMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void CCCBFMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  smoothed_quantiles_ = Eigen::Vector4d::Zero();

  auto node = parent.lock();
  RCLCPP_INFO(
    node->get_logger(),
    "CC-CBF-MPPI plugin configured: "
    "risk_budget=%.3f, penalty_weight=%.1f, adaptive_risk=%d, "
    "cbf_enabled=%d, cbf_gamma=%.2f, "
    "projection=%d",
    params_.cc_risk_budget,
    params_.cc_penalty_weight,
    params_.cc_adaptive_risk,
    params_.cbf_enabled,
    params_.cbf_gamma,
    params_.cc_cbf_projection_enabled);
}

// =============================================================================
// K 샘플의 per-constraint 위반량 평가: (K x 4)
// [velocity, acceleration, clearance(barrier), cbf_rate(dh/dt+γh)]
// =============================================================================

Eigen::MatrixXd CCCBFMPPIControllerPlugin::evaluateSampleViolations(
  const std::vector<Eigen::MatrixXd>& perturbed_controls,
  const std::vector<Eigen::MatrixXd>& trajectories) const
{
  int K = static_cast<int>(perturbed_controls.size());
  constexpr int M = 4;  // 4종 제약
  Eigen::MatrixXd violations(K, M);

  double dt = params_.dt;
  double cbf_gamma = params_.cbf_gamma;
  bool has_barriers = !barrier_set_.empty();

  for (int k = 0; k < K; ++k) {
    const auto& ctrl = perturbed_controls[k];
    const auto& traj = trajectories[k];
    int N = ctrl.rows();
    int nu = ctrl.cols();

    double vel_violation = 0.0;
    double accel_violation = 0.0;
    double clearance_violation = 0.0;
    double cbf_rate_violation = 0.0;

    // ---- Velocity constraint (worst-step) ----
    for (int t = 0; t < N; ++t) {
      double v = std::abs(ctrl(t, 0));
      double step_viol = std::max(0.0, v - params_.v_max);
      if (nu >= 2) {
        double omega = std::abs(ctrl(t, 1));
        step_viol = std::max(step_viol, std::max(0.0, omega - params_.omega_max));
      }
      vel_violation = std::max(vel_violation, step_viol);
    }

    // ---- Acceleration constraint (worst-step) ----
    for (int t = 1; t < N; ++t) {
      double dv = std::abs(ctrl(t, 0) - ctrl(t - 1, 0)) / dt;
      double step_viol = std::max(0.0, dv - params_.constrained_accel_max_v);
      if (nu >= 2) {
        double domega = std::abs(ctrl(t, 1) - ctrl(t - 1, 1)) / dt;
        step_viol = std::max(step_viol, std::max(0.0, domega - params_.constrained_accel_max_omega));
      }
      accel_violation = std::max(accel_violation, step_viol);
    }

    // ---- Clearance constraint (barrier h(x) < 0 → 위반) ----
    // ---- CBF rate constraint (dh/dt + γh < 0 → 위반) ----
    if (has_barriers) {
      int N_traj = traj.rows();  // N+1
      for (int t = 0; t < N_traj; ++t) {
        Eigen::VectorXd state = traj.row(t).transpose();

        // Clearance: h(x) < 0 이면 장애물 영역
        Eigen::VectorXd h_values = barrier_set_.evaluateAll(state);
        for (int b = 0; b < h_values.size(); ++b) {
          if (h_values(b) < 0.0) {
            clearance_violation = std::max(clearance_violation, -h_values(b));
          }
        }

        // CBF rate: Δh/dt + γh < 0 이면 위반 (이산 근사)
        if (t > 0) {
          Eigen::VectorXd prev_state = traj.row(t - 1).transpose();
          Eigen::VectorXd h_prev = barrier_set_.evaluateAll(prev_state);
          for (int b = 0; b < std::min(h_values.size(), h_prev.size()); ++b) {
            double dh_dt = (h_values(b) - h_prev(b)) / dt;
            double cbf_cond = dh_dt + cbf_gamma * h_values(b);
            if (cbf_cond < 0.0) {
              cbf_rate_violation = std::max(cbf_rate_violation, -cbf_cond);
            }
          }
        }
      }
    }

    violations(k, 0) = vel_violation;
    violations(k, 1) = accel_violation;
    violations(k, 2) = clearance_violation;
    violations(k, 3) = cbf_rate_violation;
  }

  return violations;
}

// =============================================================================
// 위반 확률 추정: p_hat = count(g > 0) / K
// =============================================================================

Eigen::Vector4d CCCBFMPPIControllerPlugin::estimateViolationProbabilities(
  const Eigen::MatrixXd& violations) const
{
  int K = violations.rows();
  Eigen::Vector4d p_hat = Eigen::Vector4d::Zero();

  for (int i = 0; i < 4; ++i) {
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
// Risk 분배: Bonferroni 또는 Adaptive (4종 제약)
// =============================================================================

Eigen::Vector4d CCCBFMPPIControllerPlugin::allocateRisk(
  const Eigen::Vector4d& violation_probs) const
{
  double eps = params_.cc_risk_budget;
  constexpr int M = 4;
  double eps_bonf = eps / M;

  if (!params_.cc_adaptive_risk) {
    return Eigen::Vector4d::Constant(eps_bonf);
  }

  // Adaptive: 위반이 적은 제약에서 여유분을 재분배
  Eigen::Vector4d allocated = Eigen::Vector4d::Constant(eps_bonf);

  Eigen::Vector4d slack = Eigen::Vector4d::Zero();
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

  // Ensure non-negative
  for (int i = 0; i < M; ++i) {
    allocated(i) = std::max(1e-8, allocated(i));
  }

  // Normalize: sum(allocated) <= eps
  double total = allocated.sum();
  if (total > eps) {
    allocated *= (eps / total);
  }

  return allocated;
}

// =============================================================================
// Empirical quantile (O(K) nth_element)
// =============================================================================

double CCCBFMPPIControllerPlugin::empiricalQuantile(
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
// Chance-constrained augmented costs (4종 제약)
// =============================================================================

Eigen::VectorXd CCCBFMPPIControllerPlugin::computeChanceConstrainedCosts(
  const Eigen::VectorXd& base_costs,
  const Eigen::MatrixXd& violations,
  const Eigen::Vector4d& allocated_risk) const
{
  int K = static_cast<int>(base_costs.size());
  Eigen::VectorXd augmented_costs = base_costs;

  Eigen::Vector4d p_hat = estimateViolationProbabilities(violations);

  for (int i = 0; i < 4; ++i) {
    if (p_hat(i) <= allocated_risk(i)) {
      continue;  // Within risk budget
    }

    Eigen::VectorXd col = violations.col(i);
    double q = empiricalQuantile(col, 1.0 - allocated_risk(i));

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
// CBF 투영 (Shield-MPPI 스타일)
// =============================================================================

Eigen::VectorXd CCCBFMPPIControllerPlugin::computeXdot(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& u) const
{
  Eigen::MatrixXd s(1, state.size());
  s.row(0) = state.transpose();
  Eigen::MatrixXd c(1, u.size());
  c.row(0) = u.transpose();
  return dynamics_->model().dynamicsBatch(s, c).row(0).transpose();
}

Eigen::VectorXd CCCBFMPPIControllerPlugin::projectControlCBF(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& u) const
{
  auto active_barriers = barrier_set_.getActiveBarriers(state);
  if (active_barriers.empty()) {
    return u;
  }

  Eigen::VectorXd u_proj = u;
  constexpr int max_iterations = 10;
  constexpr double step_size = 0.1;

  for (int iter = 0; iter < max_iterations; ++iter) {
    bool all_satisfied = true;

    for (const auto* barrier : active_barriers) {
      double h = barrier->evaluate(state);
      Eigen::VectorXd grad_h = barrier->gradient(state);
      Eigen::VectorXd x_dot = computeXdot(state, u_proj);
      double h_dot = grad_h.dot(x_dot);

      double constraint = h_dot + params_.cbf_gamma * h;

      if (constraint < 0.0) {
        all_satisfied = false;

        int nu_dim = u.size();
        Eigen::VectorXd dhdot_du(nu_dim);

        if (nu_dim == 2 && state.size() >= 3) {
          double theta = state(2);
          dhdot_du(0) = grad_h(0) * std::cos(theta) + grad_h(1) * std::sin(theta);
          dhdot_du(1) = (grad_h.size() > 2) ? grad_h(2) : 0.0;
        } else {
          constexpr double eps = 1e-4;
          for (int j = 0; j < nu_dim; ++j) {
            Eigen::VectorXd u_plus = u_proj;
            u_plus(j) += eps;
            double h_dot_plus = grad_h.dot(computeXdot(state, u_plus));
            dhdot_du(j) = (h_dot_plus - h_dot) / eps;
          }
        }

        double dhdot_du_norm_sq = dhdot_du.squaredNorm();
        if (dhdot_du_norm_sq > 1e-12) {
          double step = step_size * (-constraint) / dhdot_du_norm_sq;
          u_proj += step * dhdot_du;
        }

        u_proj = dynamics_->clipControls(
          Eigen::MatrixXd(u_proj.transpose())).row(0).transpose();
      }
    }

    if (all_satisfied) {
      break;
    }
  }

  return u_proj;
}

// =============================================================================
// computeControl — CC-CBF-MPPI
// =============================================================================

std::pair<Eigen::VectorXd, MPPIInfo> CCCBFMPPIControllerPlugin::computeControl(
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

  // ---- STEP 1: Warm-start ----
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

  // ---- STEP 5: CC-CBF processing (4종 제약) ----
  Eigen::MatrixXd violations = evaluateSampleViolations(
    perturbed_buffer_, trajectory_buffer_);

  Eigen::Vector4d p_hat = estimateViolationProbabilities(violations);
  Eigen::Vector4d eps_allocated = allocateRisk(p_hat);

  Eigen::VectorXd augmented_costs = computeChanceConstrainedCosts(
    base_costs, violations, eps_allocated);

  // Finite check
  for (int k = 0; k < K; ++k) {
    if (!std::isfinite(augmented_costs(k))) {
      augmented_costs(k) = base_costs(k) + 1e6;
    }
  }

  // Update smoothed quantiles (EMA) + tightening
  for (int i = 0; i < 4; ++i) {
    Eigen::VectorXd col = violations.col(i);
    double q = empiricalQuantile(col, 1.0 - eps_allocated(i));
    double alpha = params_.cc_quantile_smoothing;
    smoothed_quantiles_(i) = alpha * q + (1.0 - alpha) * smoothed_quantiles_(i);
    if (p_hat(i) > eps_allocated(i)) {
      smoothed_quantiles_(i) *= params_.cc_tightening_rate;
    }
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

  // ---- STEP 8: Optional CBF projection ----
  bool any_projected = false;
  if (params_.cc_cbf_projection_enabled && params_.cbf_enabled && !barrier_set_.empty()) {
    auto active = barrier_set_.getActiveBarriers(current_state);
    if (!active.empty()) {
      int shield_steps = std::min(params_.shield_cbf_stride, N);
      shield_steps = std::max(1, shield_steps);
      Eigen::VectorXd state_k = current_state;

      for (int t = 0; t < shield_steps; ++t) {
        Eigen::VectorXd u_t = control_sequence_.row(t).transpose();
        Eigen::VectorXd u_safe = projectControlCBF(state_k, u_t);

        if ((u_safe - u_t).squaredNorm() > 1e-12) {
          control_sequence_.row(t) = u_safe.transpose();
          any_projected = true;
        }

        Eigen::MatrixXd state_mat(1, nx);
        state_mat.row(0) = state_k.transpose();
        Eigen::MatrixXd ctrl_mat(1, nu);
        ctrl_mat.row(0) = control_sequence_.row(t);
        state_k = dynamics_->model().propagateBatch(
          state_mat, ctrl_mat, params_.dt).row(0).transpose();
      }
    }
  }

  // ---- STEP 9: Extract optimal control ----
  Eigen::VectorXd u_opt = control_sequence_.row(0).transpose();

  // Weighted average trajectory
  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int k = 0; k < K; ++k) {
    weighted_traj += weights(k) * trajectory_buffer_[k];
  }

  // Projected trajectory (시각화)
  if (any_projected) {
    Eigen::MatrixXd proj_traj(N + 1, nx);
    proj_traj.row(0) = current_state.transpose();
    Eigen::VectorXd s = current_state;
    for (int t = 0; t < N; ++t) {
      Eigen::MatrixXd sm(1, nx);
      sm.row(0) = s.transpose();
      Eigen::MatrixXd cm(1, nu);
      cm.row(0) = control_sequence_.row(t);
      s = dynamics_->model().propagateBatch(sm, cm, params_.dt).row(0).transpose();
      proj_traj.row(t + 1) = s.transpose();
    }
    weighted_traj = proj_traj;
  }

  int best_idx;
  double min_cost = augmented_costs.minCoeff(&best_idx);
  double ess = computeESS(weights);

  int num_tightened = 0;
  for (int i = 0; i < 4; ++i) {
    if (p_hat(i) > eps_allocated(i)) num_tightened++;
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

  // CC-CBF metrics
  info.cc_violation_probability = p_hat.maxCoeff();
  info.cc_effective_risk = eps_allocated.sum();
  info.cc_num_tightened = num_tightened;

  RCLCPP_DEBUG(
    node_->get_logger(),
    "CC-CBF-MPPI: min_cost=%.4f, p_hat=[%.4f,%.4f,%.4f,%.4f], "
    "tightened=%d, projected=%d, ESS=%.1f/%d",
    min_cost,
    p_hat(0), p_hat(1), p_hat(2), p_hat(3),
    num_tightened, any_projected, ess, K);

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
