// =============================================================================
// DIAL-MPPI Controller Plugin
//
// Reference: Xue et al. (2024) "DIAL-MPC: Diffusion-Inspired Annealing
//            For Model Predictive Control" arXiv:2409.15610 (ICRA 2025)
//
// 핵심: MPPI를 단일 스텝 확산 디노이징으로 해석하고, N_diffuse번 반복하며
//       이중 감쇠 스케줄(반복 β₁ + 호라이즌 β₂)로 노이즈를 줄여 정밀 탐색.
//
// 성능 최적화:
//   - 마지막 어닐링 스텝의 trajectories/weights를 시각화에 재사용
//   - 추가 K 롤아웃 없이 info 구성 (총 롤아웃 = N_diffuse × K)
//   - sampleInPlace / rolloutBatchInPlace / OpenMP 적용
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
#include <omp.h>

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
    double iter_decay = -(static_cast<double>(n_diffuse) - i) / (beta1 * N + 1e-8);
    double horizon_decay = -(H - 1.0 - static_cast<double>(h)) / (beta2 * H + 1e-8);
    double sigma = std::exp(iter_decay + horizon_decay);
    schedule(h) = std::max(sigma, min_noise);
  }

  return schedule;
}

// =============================================================================
// 단일 어닐링 스텝 — AnnealingResult 반환 (시각화 재사용)
// =============================================================================

AnnealingResult DialMPPIControllerPlugin::annealingStep(
  Eigen::MatrixXd& control_seq,
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory,
  const Eigen::VectorXd& noise_schedule,
  int iteration)
{
  int N = params_.N;
  int K = params_.K;
  int nu = dynamics_->model().controlDim();
  (void)iteration;

  // 2b: 단위 노이즈 샘플링 + 스케줄 적용 (in-place)
  sampler_->sampleInPlace(noise_buffer_, K, N, nu);
  for (int k = 0; k < K; ++k) {
    for (int h = 0; h < N; ++h) {
      noise_buffer_[k].row(h) *= noise_schedule(h);
    }
  }

  // 2c: 섭동 시퀀스 구성 (in-place)
  if (static_cast<int>(perturbed_buffer_.size()) != K) {
    perturbed_buffer_.resize(K, Eigen::MatrixXd::Zero(N, nu));
  }
  int K_exploit = static_cast<int>((1.0 - params_.exploration_ratio) * K);

  for (int k = 0; k < K; ++k) {
    if (k < K_exploit) {
      perturbed_buffer_[k].noalias() = control_seq + noise_buffer_[k];
    } else {
      perturbed_buffer_[k] = noise_buffer_[k];
    }
    perturbed_buffer_[k] = dynamics_->clipControls(perturbed_buffer_[k]);
  }

  // 2d: 배치 롤아웃 + 비용 계산 (in-place)
  dynamics_->rolloutBatchInPlace(
    current_state, perturbed_buffer_, params_.dt, trajectory_buffer_);

  Eigen::VectorXd costs = cost_function_->compute(
    trajectory_buffer_, perturbed_buffer_, reference_trajectory);

  // 2e: IT 정규화
  if (params_.it_alpha < 1.0) {
    Eigen::VectorXd sigma_inv = params_.noise_sigma.cwiseInverse().cwiseAbs2();
    for (int k = 0; k < K; ++k) {
      double it_cost = 0.0;
      for (int t = 0; t < N; ++t) {
        Eigen::VectorXd u_prev_t = control_seq.row(t).transpose();
        Eigen::VectorXd u_k_t = perturbed_buffer_[k].row(t).transpose();
        it_cost += u_prev_t.dot(sigma_inv.cwiseProduct(u_k_t));
      }
      costs(k) += params_.lambda * (1.0 - params_.it_alpha) * it_cost;
    }
  }

  // 2f: 가중치 계산
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);
  }
  Eigen::VectorXd weights = weight_computation_->compute(costs, current_lambda);

  // 2g: 가중 업데이트 (OpenMP 스레드 로컬 누적)
  {
    int n_threads = 1;
    #ifdef _OPENMP
    n_threads = omp_get_max_threads();
    #endif
    if (K <= 4096) { n_threads = 1; }

    std::vector<Eigen::MatrixXd> thread_accum(n_threads, Eigen::MatrixXd::Zero(N, nu));
    #pragma omp parallel if(K > 4096)
    {
      int tid = 0;
      #ifdef _OPENMP
      tid = omp_get_thread_num();
      #endif
      #pragma omp for schedule(static)
      for (int k = 0; k < K; ++k) {
        thread_accum[tid].noalias() += weights(k) * noise_buffer_[k];
      }
    }
    Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
    for (int t = 0; t < n_threads; ++t) {
      weighted_noise += thread_accum[t];
    }
    control_seq += weighted_noise;
  }
  control_seq = dynamics_->clipControls(control_seq);

  // 결과 반환 (궤적/가중치를 시각화에 재사용)
  AnnealingResult result;
  result.mean_cost = costs.mean();
  result.trajectories = trajectory_buffer_;  // copy for visualization
  result.weights = std::move(weights);
  result.costs = std::move(costs);
  return result;
}

// =============================================================================
// computeControl — DIAL 어닐링 파이프라인 (최적화: 추가 롤아웃 없음)
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

  AnnealingResult last_result;

  for (int i = 1; i <= max_iter; ++i) {
    Eigen::VectorXd noise_schedule = computeAnnealingSchedule(i, max_iter, N);

    last_result = annealingStep(
      control_sequence_, current_state, reference_trajectory,
      noise_schedule, i);

    actual_iterations = i;

    // Adaptive-DIAL 조기 종료
    if (params_.dial_adaptive_enabled && i >= params_.dial_adaptive_min_iter) {
      double improvement = (prev_cost - last_result.mean_cost) /
        (std::abs(prev_cost) + 1e-8);
      if (improvement < params_.dial_adaptive_cost_tol) {
        break;
      }
    }
    prev_cost = last_result.mean_cost;
  }

  // ──── STEP 3: 최적 제어 추출 ────
  Eigen::VectorXd u_opt = control_sequence_.row(0).transpose();

  // ──── STEP 4: Shield-DIAL (CBF) ────
  bool cbf_applied = false;
  if (params_.dial_shield_enabled && params_.cbf_enabled &&
      params_.cbf_use_safety_filter) {
    cbf_applied = true;
  }

  // ──── STEP 5: MPPIInfo 구성 (마지막 반복 결과 재사용 — 추가 롤아웃 없음) ────
  const auto& trajectories = last_result.trajectories;
  const auto& weights = last_result.weights;
  const auto& costs = last_result.costs;

  // Weighted average trajectory (OpenMP 스레드 로컬)
  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, nx);
  {
    int n_threads = 1;
    #ifdef _OPENMP
    n_threads = omp_get_max_threads();
    #endif
    if (K <= 4096) { n_threads = 1; }

    std::vector<Eigen::MatrixXd> thread_accum(n_threads, Eigen::MatrixXd::Zero(N + 1, nx));
    #pragma omp parallel if(K > 4096)
    {
      int tid = 0;
      #ifdef _OPENMP
      tid = omp_get_thread_num();
      #endif
      #pragma omp for schedule(static)
      for (int k = 0; k < K; ++k) {
        thread_accum[tid].noalias() += weights(k) * trajectories[k];
      }
    }
    for (int t = 0; t < n_threads; ++t) {
      weighted_traj += thread_accum[t];
    }
  }

  // Best sample
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
  info.tube_mppi_used = params_.tube_enabled;

  if (cbf_applied) {
    info.cbf_used = true;
  }

  RCLCPP_DEBUG(
    node_->get_logger(),
    "DIAL-MPPI: iter=%d/%d, min_cost=%.4f, mean_cost=%.4f, ESS=%.1f/%d",
    actual_iterations, max_iter, min_cost, last_result.mean_cost, ess, K);

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
