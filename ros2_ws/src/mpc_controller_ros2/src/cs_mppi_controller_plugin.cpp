// =============================================================================
// CS-MPPI (Covariance Steering MPPI) Controller Plugin
//
// Reference: Le Cleac'h et al. (2023) "CoVO-MPC: Theoretical Analysis of
//            Sampling-based MPC and Optimal Covariance Design" (CoRL 2023)
//
// 핵심: 동역학 Jacobian B_t의 Frobenius 노름으로 각 시간 스텝의
//       노이즈 스케일을 적응적으로 조절. 감도가 높은 구간은 더 넓게
//       탐색하고, 낮은 구간은 정밀하게 제어.
//
// 성능: getLinearization() N회 호출 → ~0.03ms (DiffDrive), ~0.09ms (Swerve)
//       나머지는 기존 MPPI 파이프라인 그대로 재사용.
//
// 기존 인프라 재사용:
//   - MotionModel::getLinearization() (iLQR-MPPI, PR #142)
//   - noise_buffer_, perturbed_buffer_, trajectory_buffer_ (base plugin)
//   - DIAL-MPPI computeControl override 패턴
// =============================================================================

#include "mpc_controller_ros2/cs_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::CSMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void CSMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  int N = params_.N;
  int nx = dynamics_->model().stateDim();

  // 사전 할당
  cs_scale_buffer_ = Eigen::VectorXd::Ones(N);
  nominal_states_ = Eigen::MatrixXd::Zero(N + 1, nx);

  auto node = parent.lock();
  RCLCPP_INFO(
    node->get_logger(),
    "CS-MPPI plugin configured: enabled=%d, scale_min=%.2f, scale_max=%.2f, "
    "feedback=%d (gain=%.2f)",
    params_.cs_enabled,
    params_.cs_scale_min,
    params_.cs_scale_max,
    params_.cs_feedback_enabled,
    params_.cs_feedback_gain);
}

// =============================================================================
// 동역학 Jacobian B_t 감도 → 노이즈 스케일 팩터
// =============================================================================

Eigen::VectorXd CSMPPIControllerPlugin::computeCovarianceScaling(
  const Eigen::VectorXd& x0,
  const Eigen::MatrixXd& ctrl)
{
  int N = params_.N;
  int nx = dynamics_->model().stateDim();
  int nu = dynamics_->model().controlDim();
  double dt = params_.dt;

  // 1. Nominal rollout → nominal_states_
  nominal_states_.resize(N + 1, nx);
  nominal_states_.row(0) = x0.transpose();

  for (int t = 0; t < N; ++t) {
    Eigen::MatrixXd state_mat(1, nx);
    state_mat.row(0) = nominal_states_.row(t);
    Eigen::MatrixXd ctrl_mat(1, nu);
    ctrl_mat.row(0) = ctrl.row(t);
    nominal_states_.row(t + 1) = dynamics_->model().propagateBatch(
      state_mat, ctrl_mat, dt).row(0);
  }

  // 2. Linearize at each step → ||B_t||_F
  Eigen::VectorXd sensitivities(N);
  double sum_sens = 0.0;

  for (int t = 0; t < N; ++t) {
    Eigen::VectorXd x_t = nominal_states_.row(t).transpose();
    Eigen::VectorXd u_t = ctrl.row(t).transpose();

    Linearization lin = dynamics_->model().getLinearization(x_t, u_t, dt);

    // Frobenius 노름: ||B_t||_F
    double sens = lin.B.norm();  // Eigen .norm() = Frobenius for matrices
    sensitivities(t) = sens;
    sum_sens += sens;
  }

  // 3. 스케일 팩터: clamp(sens_t / mean_sens, min, max)
  double mean_sens = sum_sens / N;
  Eigen::VectorXd scale_factors(N);

  if (mean_sens < 1e-10) {
    // 감도가 거의 0 → 균일 스케일
    scale_factors.setOnes();
  } else {
    for (int t = 0; t < N; ++t) {
      double raw = sensitivities(t) / mean_sens;
      scale_factors(t) = std::clamp(raw, params_.cs_scale_min, params_.cs_scale_max);
    }
  }

  return scale_factors;
}

// =============================================================================
// noise_buffer_에 per-step 스케일 적용
// =============================================================================

void CSMPPIControllerPlugin::applyAdaptedNoise(const Eigen::VectorXd& scale_factors)
{
  int K = params_.K;
  int N = params_.N;

  for (int k = 0; k < K; ++k) {
    for (int t = 0; t < N; ++t) {
      noise_buffer_[k].row(t) *= scale_factors(t);
    }
  }
}

// =============================================================================
// computeControl — CS-MPPI 파이프라인
// =============================================================================

std::pair<Eigen::VectorXd, MPPIInfo> CSMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // CS 비활성 시 base 호출
  if (!params_.cs_enabled) {
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

  // ──── STEP 2: Covariance scaling (B_t 감도 분석) ────
  cs_scale_buffer_ = computeCovarianceScaling(current_state, control_sequence_);

  // ──── STEP 3: Sample noise ────
  sampler_->sampleInPlace(noise_buffer_, K, N, nu);

  // Goal 근처 noise 스케일링 (base 동작 유지)
  if (goal_dist_ < params_.goal_slowdown_dist && params_.goal_slowdown_dist > 1e-6) {
    double ratio = goal_dist_ / params_.goal_slowdown_dist;
    double noise_scale = std::clamp(std::sqrt(ratio), 0.2, 1.0);
    for (int k = 0; k < K; ++k) {
      noise_buffer_[k] *= noise_scale;
    }
  }

  // ──── STEP 4: Apply per-step covariance scaling ────
  applyAdaptedNoise(cs_scale_buffer_);

  // ──── STEP 5: Perturb + Clip ────
  if (static_cast<int>(perturbed_buffer_.size()) != K) {
    perturbed_buffer_.resize(K, Eigen::MatrixXd::Zero(N, nu));
  }
  int K_exploit = static_cast<int>((1.0 - params_.exploration_ratio) * K);

  for (int k = 0; k < K; ++k) {
    if (k < K_exploit) {
      perturbed_buffer_[k].noalias() = control_sequence_ + noise_buffer_[k];
    } else {
      perturbed_buffer_[k] = noise_buffer_[k];
    }
    perturbed_buffer_[k] = dynamics_->clipControls(perturbed_buffer_[k]);
  }

  // ──── STEP 6: Batch rollout ────
  dynamics_->rolloutBatchInPlace(
    current_state, perturbed_buffer_, params_.dt, trajectory_buffer_);
  const auto& trajectories = trajectory_buffer_;

  // ──── STEP 7: Cost computation ────
  Eigen::VectorXd costs = cost_function_->compute(
    trajectories, perturbed_buffer_, reference_trajectory);

  // IT 정규화
  if (params_.it_alpha < 1.0) {
    Eigen::VectorXd sigma_inv = params_.noise_sigma.cwiseInverse().cwiseAbs2();
    for (int k = 0; k < K; ++k) {
      double it_cost = 0.0;
      for (int t = 0; t < N; ++t) {
        Eigen::VectorXd u_prev_t = control_sequence_.row(t).transpose();
        Eigen::VectorXd u_k_t = perturbed_buffer_[k].row(t).transpose();
        it_cost += u_prev_t.dot(sigma_inv.cwiseProduct(u_k_t));
      }
      costs(k) += params_.lambda * (1.0 - params_.it_alpha) * it_cost;
    }
  }

  // ──── STEP 8: Weights (Adaptive Temperature 포함) ────
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);
  }
  Eigen::VectorXd weights = weight_computation_->compute(costs, current_lambda);

  // ──── STEP 9: Weighted update (OpenMP) ────
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
    control_sequence_ += weighted_noise;
  }
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // ──── STEP 10: Extract optimal control ────
  Eigen::VectorXd u_opt = control_sequence_.row(0).transpose();

  // ──── STEP 11: Build MPPIInfo ────
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

  RCLCPP_DEBUG(
    node_->get_logger(),
    "CS-MPPI: min_cost=%.4f, mean_cost=%.4f, ESS=%.1f/%d, "
    "scale_range=[%.2f, %.2f]",
    min_cost, costs.mean(), ess, K,
    cs_scale_buffer_.minCoeff(), cs_scale_buffer_.maxCoeff());

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
