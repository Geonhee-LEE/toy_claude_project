// =============================================================================
// pi-MPPI (Projection MPPI) Controller Plugin
//
// Reference: Andrejev et al. (2025) "pi-MPPI: A Projection-based MPPI
//            Scheme for Smooth Optimal Control of Fixed-Wing Aerial Vehicles"
//            (IEEE RA-L 2025, arXiv 2504.10962)
//
// 핵심: ADMM QP 투영 필터로 제어 시퀀스의 크기/변화율/가속도 hard bounds를
//       보장. 샘플 투영(pre-filter) + 최종 투영(post-filter) 2단계.
//
// 성능: K=512, N=30, nu=2, 10 ADMM iter → < 0.5ms 오버헤드
//
// 기존 인프라 재사용:
//   - noise_buffer_, perturbed_buffer_, trajectory_buffer_ (base plugin)
//   - CS-MPPI/DIAL-MPPI computeControl override 패턴
// =============================================================================

#include "mpc_controller_ros2/pi_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::PiMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

// =============================================================================
// ADMMProjector Implementation
// =============================================================================

ADMMProjector::ADMMProjector(int N, double dt, double rho, int max_iter, int derivative_order)
  : N_(N), dt_(dt), rho_(rho), max_iter_(max_iter), derivative_order_(derivative_order)
{
  // Build D1: (N-1) x N  first-order finite difference
  // D1[i,i] = -1/dt, D1[i,i+1] = 1/dt
  D1_ = Eigen::MatrixXd::Zero(N - 1, N);
  for (int i = 0; i < N - 1; ++i) {
    D1_(i, i) = -1.0 / dt_;
    D1_(i, i + 1) = 1.0 / dt_;
  }

  // Build D2: (N-2) x N  second-order finite difference
  // D2[i,i] = 1/dt^2, D2[i,i+1] = -2/dt^2, D2[i,i+2] = 1/dt^2
  D2_ = Eigen::MatrixXd::Zero(N - 2, N);
  double inv_dt2 = 1.0 / (dt_ * dt_);
  for (int i = 0; i < N - 2; ++i) {
    D2_(i, i) = inv_dt2;
    D2_(i, i + 1) = -2.0 * inv_dt2;
    D2_(i, i + 2) = inv_dt2;
  }

  // Stack A = [I; D1; D2] or [I; D1] depending on derivative_order
  int rows_I = N;
  int rows_D1 = N - 1;
  int rows_D2 = (derivative_order >= 2) ? (N - 2) : 0;
  m_ = rows_I + rows_D1 + rows_D2;

  A_ = Eigen::MatrixXd::Zero(m_, N);
  A_.topRows(rows_I) = Eigen::MatrixXd::Identity(N, N);
  A_.middleRows(rows_I, rows_D1) = D1_;
  if (derivative_order >= 2) {
    A_.bottomRows(rows_D2) = D2_;
  }

  // Precompute LLT of P = I + rho * A^T * A
  Eigen::MatrixXd P = Eigen::MatrixXd::Identity(N, N) + rho * A_.transpose() * A_;
  kkt_llt_.compute(P);
}

void ADMMProjector::projectDimension(
  const Eigen::VectorXd& v_raw,
  Eigen::VectorXd& v_out,
  double u_min, double u_max,
  double rate_max, double accel_max) const
{
  int N = N_;

  // Build bounds vector for z = A * v_tilde
  // z_bounds: [u_min..u_max for I rows; -rate_max..rate_max for D1; -accel_max..accel_max for D2]
  Eigen::VectorXd z_lo(m_), z_hi(m_);

  // I block: control bounds
  z_lo.head(N).setConstant(u_min);
  z_hi.head(N).setConstant(u_max);

  // D1 block: rate bounds
  z_lo.segment(N, N - 1).setConstant(-rate_max);
  z_hi.segment(N, N - 1).setConstant(rate_max);

  // D2 block: accel bounds (if order >= 2)
  if (derivative_order_ >= 2) {
    z_lo.tail(N - 2).setConstant(-accel_max);
    z_hi.tail(N - 2).setConstant(accel_max);
  }

  // ADMM iterations
  Eigen::VectorXd z = A_ * v_raw;  // initialize z
  Eigen::VectorXd y = Eigen::VectorXd::Zero(m_);  // dual variable
  Eigen::VectorXd v_tilde = v_raw;

  for (int iter = 0; iter < max_iter_; ++iter) {
    // Primal update: v_tilde = P^{-1} * (v_raw + rho * A^T * (z - y))
    Eigen::VectorXd rhs = v_raw + rho_ * A_.transpose() * (z - y);
    v_tilde = kkt_llt_.solve(rhs);

    // Dual projection: z = clamp(A * v_tilde + y, bounds)
    Eigen::VectorXd Av = A_ * v_tilde;
    Eigen::VectorXd w = Av + y;
    z = w.cwiseMax(z_lo).cwiseMin(z_hi);

    // Dual ascent: y += A * v_tilde - z
    y += Av - z;
  }

  // Post-clamp to guarantee control bounds (ADMM is approximate with finite iter)
  v_out = v_tilde.cwiseMax(u_min).cwiseMin(u_max);
}

void ADMMProjector::projectSequence(
  const Eigen::MatrixXd& in,
  Eigen::MatrixXd& out,
  const Eigen::VectorXd& u_min,
  const Eigen::VectorXd& u_max,
  const Eigen::VectorXd& rate_max,
  const Eigen::VectorXd& accel_max) const
{
  int nu = static_cast<int>(in.cols());
  out.resize(in.rows(), in.cols());

  for (int d = 0; d < nu; ++d) {
    Eigen::VectorXd col_in = in.col(d);
    Eigen::VectorXd col_out(N_);
    projectDimension(col_in, col_out, u_min(d), u_max(d), rate_max(d), accel_max(d));
    out.col(d) = col_out;
  }
}

// =============================================================================
// PiMPPIControllerPlugin Implementation
// =============================================================================

void PiMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  int N = params_.N;
  int nu = dynamics_->model().controlDim();

  // Build ADMM projector
  projector_ = std::make_unique<ADMMProjector>(
    N, params_.dt, params_.pi_admm_rho,
    params_.pi_admm_iterations, params_.pi_derivative_order);

  // Build per-dimension bounds from motion model limits + pi params
  pi_u_min_ = Eigen::VectorXd::Zero(nu);
  pi_u_max_ = Eigen::VectorXd::Zero(nu);
  pi_rate_max_ = Eigen::VectorXd::Zero(nu);
  pi_accel_max_ = Eigen::VectorXd::Zero(nu);

  // Map control limits based on motion model
  std::string model = params_.motion_model;
  if (model == "diff_drive" || model == "ackermann") {
    // nu=2: [v, omega] or [v, delta_dot]
    pi_u_min_(0) = params_.v_min;
    pi_u_max_(0) = params_.v_max;
    pi_u_min_(1) = params_.omega_min;
    pi_u_max_(1) = params_.omega_max;
    pi_rate_max_(0) = params_.pi_rate_max_v;
    pi_rate_max_(1) = params_.pi_rate_max_omega;
    pi_accel_max_(0) = params_.pi_accel_max_v;
    pi_accel_max_(1) = params_.pi_accel_max_omega;
  } else {
    // swerve/non_coaxial: nu=3: [vx, vy, omega] or [v, omega, delta_dot]
    pi_u_min_(0) = params_.v_min;
    pi_u_max_(0) = params_.v_max;
    double vy_lim = (params_.vy_max > 0) ? params_.vy_max : params_.v_max;
    pi_u_min_(1) = -vy_lim;
    pi_u_max_(1) = vy_lim;
    pi_u_min_(2) = params_.omega_min;
    pi_u_max_(2) = params_.omega_max;
    pi_rate_max_(0) = params_.pi_rate_max_v;
    pi_rate_max_(1) = params_.pi_rate_max_vy;
    pi_rate_max_(2) = params_.pi_rate_max_omega;
    pi_accel_max_(0) = params_.pi_accel_max_v;
    pi_accel_max_(1) = params_.pi_accel_max_vy;
    pi_accel_max_(2) = params_.pi_accel_max_omega;
  }

  auto node = parent.lock();
  RCLCPP_INFO(
    node->get_logger(),
    "pi-MPPI plugin configured: enabled=%d, admm_iter=%d, rho=%.2f, "
    "order=%d, rate_max_v=%.2f, accel_max_v=%.2f",
    params_.pi_enabled,
    params_.pi_admm_iterations,
    params_.pi_admm_rho,
    params_.pi_derivative_order,
    params_.pi_rate_max_v,
    params_.pi_accel_max_v);
}

// =============================================================================
// projectAllSamples — K개 샘플 병렬 투영
// =============================================================================

void PiMPPIControllerPlugin::projectAllSamples()
{
  int K = params_.K;

  #pragma omp parallel for schedule(static) if(K > 256)
  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd projected;
    projector_->projectSequence(
      perturbed_buffer_[k], projected,
      pi_u_min_, pi_u_max_, pi_rate_max_, pi_accel_max_);
    perturbed_buffer_[k] = projected;
  }
}

// =============================================================================
// computeControl — pi-MPPI 파이프라인
// =============================================================================

std::pair<Eigen::VectorXd, MPPIInfo> PiMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // pi 비활성 시 base 호출
  if (!params_.pi_enabled) {
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

  // Goal 근처 noise 스케일링 (base 동작 유지)
  if (goal_dist_ < params_.goal_slowdown_dist && params_.goal_slowdown_dist > 1e-6) {
    double ratio = goal_dist_ / params_.goal_slowdown_dist;
    double noise_scale = std::clamp(std::sqrt(ratio), 0.2, 1.0);
    for (int k = 0; k < K; ++k) {
      noise_buffer_[k] *= noise_scale;
    }
  }

  // ──── STEP 3: Perturb ────
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
  }

  // ──── STEP 4: ★ ADMM Projection on all samples ────
  projectAllSamples();

  // ──── STEP 5: Recompute noise after projection ────
  for (int k = 0; k < K; ++k) {
    noise_buffer_[k] = perturbed_buffer_[k] - control_sequence_;
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

  // ──── STEP 10: ★ Final projection on control_sequence_ ────
  {
    Eigen::MatrixXd projected;
    projector_->projectSequence(
      control_sequence_, projected,
      pi_u_min_, pi_u_max_, pi_rate_max_, pi_accel_max_);
    control_sequence_ = projected;
  }

  // ──── STEP 11: Extract optimal control ────
  Eigen::VectorXd u_opt = control_sequence_.row(0).transpose();

  // ──── STEP 12: Build MPPIInfo ────
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
    "pi-MPPI: min_cost=%.4f, mean_cost=%.4f, ESS=%.1f/%d",
    min_cost, costs.mean(), ess, K);

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
