// =============================================================================
// Biased-MPPI Controller Plugin
//
// Reference: Trevisan & Alonso-Mora (2024) "Biased-MPPI: Informing
//            Sampling-Based Model Predictive Control by Fusing Ancillary
//            Controllers" IEEE RA-L
//
// K개 샘플을 J_total개(ancillary 결정적) + (K-J_total)개(Gaussian)로 분할.
// 수정된 비용함수 S~ = S + lambda*log(p/q_s)에서 밀도비가 소거되므로
// 가중치 공식은 Vanilla와 동일 -> 기존 WeightComputation 전략과 100% 호환.
//
// Ancillary 컨트롤러 (4종):
//   1. Braking       — zero 제어열 (긴급 정지)
//   2. GoToGoal      — 목표 방향 P-제어 (open-loop rollout)
//   3. PathFollowing — 경로 접선 추종 (open-loop rollout)
//   4. PreviousSolution — 이전 최적 시퀀스 복제
//
// Python 대응: 해당 없음 (C++ only 신규 구현)
// =============================================================================

#include "mpc_controller_ros2/biased_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>
#include <algorithm>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::BiasedMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void BiasedMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  auto node = parent.lock();

  // 활성 ancillary 수 카운트
  int num_ancillary = 0;
  if (params_.biased_braking) ++num_ancillary;
  if (params_.biased_goto_goal) ++num_ancillary;
  if (params_.biased_path_following) ++num_ancillary;
  if (params_.biased_previous_solution) ++num_ancillary;

  RCLCPP_INFO(
    node->get_logger(),
    "Biased-MPPI plugin configured: enabled=%d, bias_ratio=%.3f, "
    "ancillary=[braking=%d, goto_goal=%d, path_following=%d, prev_solution=%d] (%d active)",
    params_.biased_enabled,
    params_.bias_ratio,
    params_.biased_braking,
    params_.biased_goto_goal,
    params_.biased_path_following,
    params_.biased_previous_solution,
    num_ancillary);
}

// =============================================================================
// Ancillary 시퀀스 생성기
// =============================================================================

Eigen::MatrixXd BiasedMPPIControllerPlugin::generateBrakingSequence(
  int N, int nu) const
{
  return Eigen::MatrixXd::Zero(N, nu);
}

Eigen::MatrixXd BiasedMPPIControllerPlugin::generateGoToGoalSequence(
  const Eigen::VectorXd& state,
  const Eigen::MatrixXd& ref_traj,
  int N, int nu, double dt) const
{
  Eigen::MatrixXd seq = Eigen::MatrixXd::Zero(N, nu);
  const double gain = params_.biased_goto_goal_gain;
  const double v_max = params_.v_max;
  const double omega_max = params_.omega_max;

  // 목표 = reference trajectory의 마지막 행
  Eigen::VectorXd goal = ref_traj.row(ref_traj.rows() - 1).transpose();

  // Open-loop rollout with P-control toward goal
  Eigen::VectorXd predicted = state;
  for (int t = 0; t < N; ++t) {
    double dx = goal(0) - predicted(0);
    double dy = goal(1) - predicted(1);
    double dist = std::sqrt(dx * dx + dy * dy);
    double angle_to_goal = std::atan2(dy, dx);
    double heading_error = normalizeAngle(angle_to_goal - predicted(2));

    // P-control
    double v_cmd = gain * std::min(dist, v_max);
    double omega_cmd = gain * heading_error;

    // Clip
    v_cmd = std::clamp(v_cmd, params_.v_min, v_max);
    omega_cmd = std::clamp(omega_cmd, params_.omega_min, omega_max);

    seq(t, 0) = v_cmd;
    if (nu >= 2) {
      seq(t, 1) = omega_cmd;
    }
    // Swerve (nu=3): vy는 0으로 유지

    // 1-step forward prediction (Euler)
    double theta = predicted(2);
    predicted(0) += v_cmd * std::cos(theta) * dt;
    predicted(1) += v_cmd * std::sin(theta) * dt;
    if (predicted.size() >= 3) {
      predicted(2) += (nu >= 2 ? omega_cmd : 0.0) * dt;
      predicted(2) = normalizeAngle(predicted(2));
    }
  }

  return seq;
}

Eigen::MatrixXd BiasedMPPIControllerPlugin::generatePathFollowingSequence(
  const Eigen::VectorXd& state,
  const Eigen::MatrixXd& ref_traj,
  int N, int nu, double dt) const
{
  Eigen::MatrixXd seq = Eigen::MatrixXd::Zero(N, nu);
  const double gain = params_.biased_path_following_gain;
  const double v_max = params_.v_max;
  const double omega_max = params_.omega_max;

  Eigen::VectorXd predicted = state;
  for (int t = 0; t < N; ++t) {
    // 경로 접선 계산 (ref_traj는 N+1 x nx)
    int t_ref = std::min(t + 1, static_cast<int>(ref_traj.rows()) - 1);
    int t_ref_prev = std::min(t, static_cast<int>(ref_traj.rows()) - 1);
    if (t_ref == t_ref_prev && t_ref > 0) {
      t_ref_prev = t_ref - 1;
    }

    Eigen::VectorXd tangent = ref_traj.row(t_ref).transpose()
                            - ref_traj.row(t_ref_prev).transpose();
    double tangent_norm = std::sqrt(tangent(0) * tangent(0) + tangent(1) * tangent(1));
    double target_theta = std::atan2(tangent(1), tangent(0));
    double heading_error = normalizeAngle(target_theta - predicted(2));

    // P-control
    double v_cmd = gain * std::max(tangent_norm, 0.0);
    double omega_cmd = gain * heading_error;

    // Clip
    v_cmd = std::clamp(v_cmd, params_.v_min, v_max);
    omega_cmd = std::clamp(omega_cmd, params_.omega_min, omega_max);

    seq(t, 0) = v_cmd;
    if (nu >= 2) {
      seq(t, 1) = omega_cmd;
    }

    // 1-step forward prediction (Euler)
    double theta = predicted(2);
    predicted(0) += v_cmd * std::cos(theta) * dt;
    predicted(1) += v_cmd * std::sin(theta) * dt;
    if (predicted.size() >= 3) {
      predicted(2) += (nu >= 2 ? omega_cmd : 0.0) * dt;
      predicted(2) = normalizeAngle(predicted(2));
    }
  }

  return seq;
}

Eigen::MatrixXd BiasedMPPIControllerPlugin::generatePreviousSolutionSequence(
  int N, int nu) const
{
  // control_sequence_가 이미 (N, nu) → 그대로 복제
  if (control_sequence_.rows() == N && control_sequence_.cols() == nu) {
    return control_sequence_;
  }
  // 크기 불일치 시 zero fallback
  return Eigen::MatrixXd::Zero(N, nu);
}

// =============================================================================
// computeControl — base 파이프라인 + biased 샘플 주입
// =============================================================================

std::pair<Eigen::VectorXd, MPPIInfo> BiasedMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // biased 비활성 시 base 호출
  if (!params_.biased_enabled) {
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

  // ──── STEP 2: Noise 샘플링 ────
  auto noise_samples = sampler_->sample(K, N, nu);

  // ──── STEP 2.5: Ancillary 시퀀스 생성 ────
  std::vector<Eigen::MatrixXd> ancillary_seqs;
  if (params_.biased_braking) {
    ancillary_seqs.push_back(generateBrakingSequence(N, nu));
  }
  if (params_.biased_goto_goal) {
    ancillary_seqs.push_back(
      generateGoToGoalSequence(current_state, reference_trajectory, N, nu, params_.dt));
  }
  if (params_.biased_path_following) {
    ancillary_seqs.push_back(
      generatePathFollowingSequence(current_state, reference_trajectory, N, nu, params_.dt));
  }
  if (params_.biased_previous_solution) {
    ancillary_seqs.push_back(generatePreviousSolutionSequence(N, nu));
  }

  int num_ancillary = static_cast<int>(ancillary_seqs.size());
  int J = static_cast<int>(std::floor(params_.bias_ratio * K));  // 각 ancillary당 J개
  int J_total = J * num_ancillary;
  // J_total이 K를 초과하지 않도록 보호
  if (J_total >= K) {
    J = (num_ancillary > 0) ? (K - 1) / num_ancillary : 0;
    J_total = J * num_ancillary;
  }
  int K_gaussian = K - J_total;

  // ──── STEP 3: Biased 샘플 구성 ────
  std::vector<Eigen::MatrixXd> perturbed_controls;
  perturbed_controls.reserve(K);

  // noise_for_weight: Step 7에서 control_sequence_ 업데이트에 사용
  std::vector<Eigen::MatrixXd> noise_for_weight;
  noise_for_weight.reserve(K);

  // (a) Ancillary 결정적 샘플 (J_total개)
  for (int a = 0; a < num_ancillary; ++a) {
    for (int j = 0; j < J; ++j) {
      Eigen::MatrixXd anc_ctrl = dynamics_->clipControls(ancillary_seqs[a]);
      perturbed_controls.push_back(anc_ctrl);
      // noise_for_weight = ancillary - control_sequence_ (warm-started)
      noise_for_weight.push_back(anc_ctrl - control_sequence_);
    }
  }

  // (b) Gaussian 샘플 (K_gaussian개) — Exploitation/Exploration 분할
  int K_exploit = static_cast<int>((1.0 - params_.exploration_ratio) * K_gaussian);
  for (int k = 0; k < K_gaussian; ++k) {
    Eigen::MatrixXd perturbed;
    if (k < K_exploit) {
      perturbed = control_sequence_ + noise_samples[k];
    } else {
      perturbed = noise_samples[k];
    }
    perturbed = dynamics_->clipControls(perturbed);
    perturbed_controls.push_back(perturbed);
    noise_for_weight.push_back(noise_samples[k]);
  }

  // ──── STEP 4: Batch rollout ────
  auto trajectories = dynamics_->rolloutBatch(
    current_state, perturbed_controls, params_.dt);

  // ──── STEP 5: Cost 계산 ────
  Eigen::VectorXd costs;
  CostBreakdown cost_breakdown;
  if (params_.debug_collision_viz) {
    cost_breakdown = cost_function_->computeDetailed(
      trajectories, perturbed_controls, reference_trajectory);
    costs = cost_breakdown.total_costs;
  } else {
    costs = cost_function_->compute(
      trajectories, perturbed_controls, reference_trajectory);
  }

  // ──── STEP 5.5: IT 정규화 ────
  if (params_.it_alpha < 1.0) {
    Eigen::VectorXd sigma_inv = params_.noise_sigma.cwiseInverse().cwiseAbs2();
    for (int k = 0; k < K; ++k) {
      double it_cost = 0.0;
      for (int t = 0; t < N; ++t) {
        Eigen::VectorXd u_prev_t = control_sequence_.row(t).transpose();
        Eigen::VectorXd u_k_t = perturbed_controls[k].row(t).transpose();
        it_cost += u_prev_t.dot(sigma_inv.cwiseProduct(u_k_t));
      }
      costs(k) += params_.lambda * (1.0 - params_.it_alpha) * it_cost;
    }
  }

  // ──── STEP 6: Weight 계산 ────
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);
  }
  Eigen::VectorXd weights = weight_computation_->compute(costs, current_lambda);

  // ──── STEP 7: control_sequence_ += sum(w[k] * noise_for_weight[k]) ────
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_noise += weights(k) * noise_for_weight[k];
  }
  control_sequence_ += weighted_noise;
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // ──── STEP 8: Extract optimal control ────
  Eigen::VectorXd u_opt = control_sequence_.row(0).transpose();

  // Weighted average trajectory
  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int k = 0; k < K; ++k) {
    weighted_traj += weights(k) * trajectories[k];
  }

  // Best sample
  int best_idx;
  double min_cost = costs.minCoeff(&best_idx);
  double ess = computeESS(weights);

  // Build info
  MPPIInfo info;
  info.sample_trajectories = trajectories;
  info.sample_weights = weights;
  info.best_trajectory = trajectories[best_idx];
  info.weighted_avg_trajectory = weighted_traj;
  info.temperature = (params_.adaptive_temperature && adaptive_temp_) ?
    adaptive_temp_->getLambda() : params_.lambda;
  info.ess = ess;
  info.costs = costs;

  if (params_.debug_collision_viz) {
    info.cost_breakdown = cost_breakdown;
  }

  info.colored_noise_used = params_.colored_noise;
  info.adaptive_temp_used = params_.adaptive_temperature;
  info.tube_mppi_used = params_.tube_enabled;

  RCLCPP_DEBUG(
    node_->get_logger(),
    "Biased-MPPI: min_cost=%.4f, ESS=%.1f/%d, J_total=%d, K_gauss=%d",
    min_cost, ess, K, J_total, K_gaussian);

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
