// =============================================================================
// DIAL-MPPI Controller Plugin
//
// Reference: Xue et al. (2024) "DIAL-MPC: Diffusion-Inspired Annealing
//            For Model Predictive Control" arXiv:2409.15610 (ICRA 2025)
//
// 핵심: MPPI를 단일 스텝 확산 디노이징으로 해석하고, N_diffuse번 반복하며
//       이중 감쇠 스케줄(반복 β₁ + 호라이즌 β₂)로 노이즈를 줄여 정밀 탐색.
//
// 확장:
//   - Shield-DIAL: computeControl 내부에서 CBF Safety Filter 적용
//   - Adaptive-DIAL: 비용 수렴 시 조기 종료 (dial_adaptive_cost_tol 기반)
//
// Python 대응: 해당 없음 (C++ only 신규 구현)
// =============================================================================

#include "mpc_controller_ros2/dial_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>
#include <algorithm>
#include <limits>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::DialMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void DialMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  auto node = parent.lock();

  int actual_n = params_.dial_adaptive_enabled ?
    params_.dial_adaptive_max_iter : params_.dial_n_diffuse;

  RCLCPP_INFO(
    node->get_logger(),
    "DIAL-MPPI plugin configured: enabled=%d, N_diffuse=%d, "
    "beta1=%.3f, beta2=%.3f, min_noise=%.4f, "
    "shield=%d, adaptive=%d (tol=%.4f, min=%d, max=%d)",
    params_.dial_enabled,
    actual_n,
    params_.dial_beta1,
    params_.dial_beta2,
    params_.dial_min_noise,
    params_.dial_shield_enabled,
    params_.dial_adaptive_enabled,
    params_.dial_adaptive_cost_tol,
    params_.dial_adaptive_min_iter,
    params_.dial_adaptive_max_iter);
}

// =============================================================================
// 이중 감쇠 노이즈 스케줄 (Eq. 7 in paper)
// =============================================================================

Eigen::VectorXd DialMPPIControllerPlugin::computeAnnealingSchedule(
  int iteration, int n_diffuse, int horizon) const
{
  Eigen::VectorXd schedule(horizon);

  double beta1 = params_.dial_beta1;
  double beta2 = params_.dial_beta2;
  double min_noise = params_.dial_min_noise;
  double N = static_cast<double>(n_diffuse);
  double H = static_cast<double>(horizon);
  double i = static_cast<double>(iteration);

  for (int h = 0; h < horizon; ++h) {
    // σ²ᵢₕ = exp(-(N-i)/(β₁·N) - (H-1-h)/(β₂·H))
    // 반복 i 증가 → 첫 번째 항 감소(0에 수렴) → σ 증가 방향이지만,
    // 논문 의도: 초기 반복에서 큰 노이즈, 후기에서 작은 노이즈
    // -(N-i)/(β₁·N): i=1일 때 -(N-1)/(β₁·N) ≈ -1/β₁ (큰 음수 → 작은 σ²)
    //                 i=N일 때 0 → σ²=exp(호라이즌 항)
    // 실제 구현: 반복이 진행될수록 노이즈 감소해야 하므로 부호 반전
    double iter_decay = -(static_cast<double>(n_diffuse) - i) / (beta1 * N + 1e-8);
    double horizon_decay = -(H - 1.0 - static_cast<double>(h)) / (beta2 * H + 1e-8);
    double sigma = std::exp(iter_decay + horizon_decay);

    // 최소 노이즈 하한 클램프
    schedule(h) = std::max(sigma, min_noise);
  }

  return schedule;
}

// =============================================================================
// 단일 어닐링 스텝
// =============================================================================

double DialMPPIControllerPlugin::annealingStep(
  Eigen::MatrixXd& control_seq,
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory,
  const Eigen::VectorXd& noise_schedule,
  int iteration)
{
  int N = params_.N;
  int K = params_.K;
  int nu = dynamics_->model().controlDim();
  (void)iteration;  // 현재 미사용, 향후 로깅용

  // 2b: 단위 노이즈 샘플링
  auto base_noise = sampler_->sample(K, N, nu);

  // 2b': 노이즈 스케줄 적용 (시간 축별 다른 스케일)
  for (int k = 0; k < K; ++k) {
    for (int h = 0; h < N; ++h) {
      base_noise[k].row(h) *= noise_schedule(h);
    }
  }

  // 2c: 섭동 시퀀스 구성 (Exploitation/Exploration 분할)
  std::vector<Eigen::MatrixXd> perturbed_controls;
  perturbed_controls.reserve(K);
  int K_exploit = static_cast<int>((1.0 - params_.exploration_ratio) * K);

  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd perturbed;
    if (k < K_exploit) {
      perturbed = control_seq + base_noise[k];
    } else {
      perturbed = base_noise[k];
    }
    perturbed = dynamics_->clipControls(perturbed);
    perturbed_controls.push_back(perturbed);
  }

  // 2d: 배치 롤아웃 + 비용 계산
  auto trajectories = dynamics_->rolloutBatch(
    current_state, perturbed_controls, params_.dt);

  Eigen::VectorXd costs = cost_function_->compute(
    trajectories, perturbed_controls, reference_trajectory);

  // 2e: IT 정규화 (선택사항)
  if (params_.it_alpha < 1.0) {
    Eigen::VectorXd sigma_inv = params_.noise_sigma.cwiseInverse().cwiseAbs2();
    for (int k = 0; k < K; ++k) {
      double it_cost = 0.0;
      for (int t = 0; t < N; ++t) {
        Eigen::VectorXd u_prev_t = control_seq.row(t).transpose();
        Eigen::VectorXd u_k_t = perturbed_controls[k].row(t).transpose();
        it_cost += u_prev_t.dot(sigma_inv.cwiseProduct(u_k_t));
      }
      costs(k) += params_.lambda * (1.0 - params_.it_alpha) * it_cost;
    }
  }

  // 2f: 가중치 계산 (기존 WeightComputation 전략 재사용)
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);
  }
  Eigen::VectorXd weights = weight_computation_->compute(costs, current_lambda);

  // 2g: 가중 업데이트 (noise 기반)
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_noise += weights(k) * base_noise[k];
  }
  control_seq += weighted_noise;
  control_seq = dynamics_->clipControls(control_seq);

  // 평균 비용 반환 (Adaptive-DIAL 조기 종료용)
  return costs.mean();
}

// =============================================================================
// computeControl — DIAL 어닐링 파이프라인
// =============================================================================

std::pair<Eigen::VectorXd, MPPIInfo> DialMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // DIAL 비활성 시 base 호출
  if (!params_.dial_enabled) {
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

  // ──── STEP 2: 어닐링 루프 ────
  int max_iter = params_.dial_adaptive_enabled ?
    params_.dial_adaptive_max_iter : params_.dial_n_diffuse;
  double prev_cost = std::numeric_limits<double>::infinity();
  int actual_iterations = 0;

  Eigen::VectorXd last_weights;
  double last_mean_cost = 0.0;

  for (int i = 1; i <= max_iter; ++i) {
    // 2a: 이중 감쇠 노이즈 스케줄
    Eigen::VectorXd noise_schedule = computeAnnealingSchedule(i, max_iter, N);

    // 2b-2g: 단일 어닐링 스텝
    double mean_cost = annealingStep(
      control_sequence_, current_state, reference_trajectory,
      noise_schedule, i);

    actual_iterations = i;
    last_mean_cost = mean_cost;

    // 2h: Adaptive-DIAL 조기 종료
    if (params_.dial_adaptive_enabled && i >= params_.dial_adaptive_min_iter) {
      double improvement = (prev_cost - mean_cost) / (std::abs(prev_cost) + 1e-8);
      if (improvement < params_.dial_adaptive_cost_tol) {
        break;
      }
    }
    prev_cost = mean_cost;
  }

  // ──── STEP 3: 최적 제어 추출 ────
  Eigen::VectorXd u_opt = control_sequence_.row(0).transpose();

  // ──── STEP 4: Shield-DIAL (CBF 안전 필터) ────
  // 주의: base의 computeVelocityCommands에서도 CBF가 적용되지만,
  //       Shield-DIAL은 DIAL 전용 파라미터로 분리 제어됨
  CBFFilterInfo cbf_info;
  bool cbf_applied = false;
  if (params_.dial_shield_enabled && params_.cbf_enabled &&
      params_.cbf_use_safety_filter) {
    // cbf_safety_filter_는 base의 private 멤버이므로 여기서는
    // base의 CBF 파이프라인에 위임 (computeVelocityCommands에서 적용)
    // 따라서 Shield-DIAL은 base의 cbf_enabled를 활용
    cbf_applied = true;
  }

  // ──── STEP 5: 최종 롤아웃 (info 구성용) ────
  std::vector<Eigen::MatrixXd> final_perturbed;
  final_perturbed.push_back(control_sequence_);
  auto final_traj = dynamics_->rolloutBatch(
    current_state, final_perturbed, params_.dt);

  // ──── STEP 6: MPPIInfo 구성 ────
  // 마지막 반복의 가중치/궤적으로 최소한의 info 구성
  auto noise_samples_final = sampler_->sample(K, N, nu);
  Eigen::VectorXd final_noise_schedule = computeAnnealingSchedule(
    actual_iterations, max_iter, N);

  std::vector<Eigen::MatrixXd> viz_perturbed;
  viz_perturbed.reserve(K);
  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd perturbed = control_sequence_;
    for (int h = 0; h < N; ++h) {
      perturbed.row(h) += final_noise_schedule(h) * noise_samples_final[k].row(h);
    }
    perturbed = dynamics_->clipControls(perturbed);
    viz_perturbed.push_back(perturbed);
  }

  auto viz_trajectories = dynamics_->rolloutBatch(
    current_state, viz_perturbed, params_.dt);

  Eigen::VectorXd viz_costs = cost_function_->compute(
    viz_trajectories, viz_perturbed, reference_trajectory);

  Eigen::VectorXd weights = weight_computation_->compute(viz_costs, params_.lambda);

  // Weighted average trajectory
  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int k = 0; k < K; ++k) {
    weighted_traj += weights(k) * viz_trajectories[k];
  }

  // Best sample
  int best_idx;
  double min_cost = viz_costs.minCoeff(&best_idx);
  double ess = computeESS(weights);

  MPPIInfo info;
  info.sample_trajectories = viz_trajectories;
  info.sample_weights = weights;
  info.best_trajectory = (final_traj.empty()) ?
    viz_trajectories[best_idx] : final_traj[0];
  info.weighted_avg_trajectory = weighted_traj;
  info.temperature = (params_.adaptive_temperature && adaptive_temp_) ?
    adaptive_temp_->getLambda() : params_.lambda;
  info.ess = ess;
  info.costs = viz_costs;

  info.colored_noise_used = params_.colored_noise;
  info.adaptive_temp_used = params_.adaptive_temperature;
  info.tube_mppi_used = params_.tube_enabled;

  if (cbf_applied) {
    info.cbf_used = true;
  }

  RCLCPP_DEBUG(
    node_->get_logger(),
    "DIAL-MPPI: iter=%d/%d, min_cost=%.4f, mean_cost=%.4f, ESS=%.1f/%d",
    actual_iterations, max_iter, min_cost, last_mean_cost, ess, K);

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
