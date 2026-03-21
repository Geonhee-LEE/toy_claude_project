// =============================================================================
// Composable MPPI Controller Plugin
//
// 파이프라인 기반 단일 플러그인: 35종 플러그인의 기능을 YAML on/off로 조합.
// 각 Phase는 active_.xxx 조건문으로 선택적 실행.
//
// Phase 0: Adaptation     ── RH-MPPI(동적 N) + CS-MPPI(공분산 스케일링)
// Phase 1: Warm-Start     ── iLQR solve + TrajLib primitive inject
// Phase 2: Sampling       ── sampler_->sampleInPlace + CS noise scaling
// Phase 3: Pre-Filter     ── π-MPPI ADMM projection on K samples
// Phase 4: Core MPPI      ── Rollout → Cost → IT → Adaptive Temp → Weights → Update
// Phase 5: Post-Filter    ── LP IIR filter + π-MPPI ADMM on optimal seq
// Phase 6: Safety         ── Shield CBF projection (per-step)
// Phase 7: Output Correct ── Feedback Riccati K_0·dx
// Phase 8: Restore        ── RH-MPPI N 복원
// =============================================================================

#include "mpc_controller_ros2/composable_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>
#include <algorithm>
#include <omp.h>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::ComposableMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

// =============================================================================
// configure()
// =============================================================================

void ComposableMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  // Base plugin 초기화 (params_, dynamics_, cost_function_, sampler_ 등)
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  int nx = dynamics_->model().stateDim();
  int nu = dynamics_->model().controlDim();

  // ──── ActiveLayers 캐싱 ────
  active_.rh_mppi = params_.rh_mppi_enabled;
  active_.cs_mppi = params_.cs_enabled;
  active_.ilqr = params_.ilqr_enabled;
  active_.traj_library = params_.traj_library_enabled;
  active_.halton = params_.halton_enabled;
  active_.pi_mppi = params_.pi_enabled;
  active_.lp_filter = params_.lp_enabled;
  active_.shield_cbf = params_.cbf_enabled && params_.cbf_use_safety_filter;
  active_.feedback = params_.feedback_mppi_enabled;

  // ──── Phase 0: Halton Sampler 교체 ────
  if (active_.halton) {
    sampler_ = std::make_unique<HaltonSampler>(
      params_.noise_sigma, params_.halton_beta, params_.halton_sequence_offset);
    allocateBuffers();
  }

  // ──── Phase 0: RH-MPPI ────
  if (active_.rh_mppi) {
    N_max_ = params_.N;
    int effective_N_max = (params_.rh_N_max > 0) ?
      std::min(params_.rh_N_max, N_max_) : N_max_;
    int effective_N_min = std::min(params_.rh_N_min, effective_N_max);

    horizon_manager_ = std::make_unique<AdaptiveHorizonManager>(
      effective_N_min, effective_N_max,
      params_.rh_speed_weight, params_.rh_obstacle_weight, params_.rh_error_weight,
      params_.rh_obs_dist_threshold, params_.rh_error_threshold,
      params_.rh_smoothing_alpha);
  }

  // ──── Phase 0: CS-MPPI ────
  if (active_.cs_mppi) {
    cs_scale_buffer_ = Eigen::VectorXd::Ones(params_.N);
    cs_nominal_states_ = Eigen::MatrixXd::Zero(params_.N + 1, nx);
  }

  // ──── Phase 1: iLQR ────
  if (active_.ilqr) {
    ILQRParams ilqr_params;
    ilqr_params.max_iterations = params_.ilqr_max_iterations;
    ilqr_params.regularization = params_.ilqr_regularization;
    ilqr_params.line_search_steps = params_.ilqr_line_search_steps;
    ilqr_params.cost_tolerance = params_.ilqr_cost_tolerance;
    ilqr_solver_ = std::make_unique<ILQRSolver>(ilqr_params, nx, nu);
  }

  // ──── Phase 1: Trajectory Library ────
  if (active_.traj_library) {
    traj_library_.generate(
      params_.N, nu, params_.dt,
      params_.v_max, params_.v_min, params_.omega_max);
  }

  // ──── Phase 3/5: π-MPPI ADMM ────
  if (active_.pi_mppi) {
    projector_ = std::make_unique<ADMMProjector>(
      params_.N, params_.dt, params_.pi_admm_rho,
      params_.pi_admm_iterations, params_.pi_derivative_order);

    pi_u_min_ = Eigen::VectorXd(nu);
    pi_u_max_ = Eigen::VectorXd(nu);
    pi_rate_max_ = Eigen::VectorXd(nu);
    pi_accel_max_ = Eigen::VectorXd(nu);

    if (nu >= 1) {
      pi_u_min_(0) = params_.v_min;
      pi_u_max_(0) = params_.v_max;
      pi_rate_max_(0) = params_.pi_rate_max_v;
      pi_accel_max_(0) = params_.pi_accel_max_v;
    }
    if (nu >= 2) {
      pi_u_min_(1) = params_.omega_min;
      pi_u_max_(1) = params_.omega_max;
      pi_rate_max_(1) = params_.pi_rate_max_omega;
      pi_accel_max_(1) = params_.pi_accel_max_omega;
    }
    if (nu >= 3) {
      double vy_limit = (params_.vy_max > 0) ? params_.vy_max : params_.v_max;
      pi_u_min_(2) = -vy_limit;
      pi_u_max_(2) = vy_limit;
      pi_rate_max_(2) = params_.pi_rate_max_vy;
      pi_accel_max_(2) = params_.pi_accel_max_vy;
    }
  }

  // ──── Phase 5: LP 필터 ────
  if (active_.lp_filter && params_.lp_cutoff_frequency > 0.0) {
    double tau = 1.0 / (2.0 * M_PI * params_.lp_cutoff_frequency);
    lp_alpha_ = params_.dt / (tau + params_.dt);
  }
  lp_u_prev_ = Eigen::VectorXd::Zero(nu);

  // ──── Phase 6: Shield CBF ────
  if (active_.shield_cbf) {
    shield_cbf_stride_ = std::max(1, params_.shield_cbf_stride);
    shield_max_iterations_ = std::max(1, params_.shield_max_iterations);
  }

  // ──── Phase 7: Feedback ────
  if (active_.feedback) {
    gain_computer_ = std::make_unique<FeedbackGainComputer>(
      nx, nu, params_.feedback_regularization);
  }

  // ──── 로그 ────
  RCLCPP_INFO(node_->get_logger(),
    "Composable MPPI configured: "
    "RH=%d CS=%d iLQR=%d TrajLib=%d Halton=%d Pi=%d LP=%d Shield=%d Feedback=%d",
    active_.rh_mppi, active_.cs_mppi, active_.ilqr, active_.traj_library,
    active_.halton, active_.pi_mppi, active_.lp_filter, active_.shield_cbf,
    active_.feedback);
}

// =============================================================================
// CS-MPPI: Covariance Scaling
// =============================================================================

Eigen::VectorXd ComposableMPPIControllerPlugin::computeCovarianceScaling(
  const Eigen::VectorXd& x0,
  const Eigen::MatrixXd& ctrl)
{
  int N = params_.N;
  int nx = dynamics_->model().stateDim();
  int nu = dynamics_->model().controlDim();
  double dt = params_.dt;

  cs_nominal_states_.resize(N + 1, nx);
  cs_nominal_states_.row(0) = x0.transpose();

  for (int t = 0; t < N; ++t) {
    Eigen::MatrixXd state_mat(1, nx);
    state_mat.row(0) = cs_nominal_states_.row(t);
    Eigen::MatrixXd ctrl_mat(1, nu);
    ctrl_mat.row(0) = ctrl.row(t);
    cs_nominal_states_.row(t + 1) = dynamics_->model().propagateBatch(
      state_mat, ctrl_mat, dt).row(0);
  }

  Eigen::VectorXd sensitivities(N);
  double sum_sens = 0.0;

  for (int t = 0; t < N; ++t) {
    Eigen::VectorXd x_t = cs_nominal_states_.row(t).transpose();
    Eigen::VectorXd u_t = ctrl.row(t).transpose();
    Linearization lin = dynamics_->model().getLinearization(x_t, u_t, dt);
    double sens = lin.B.norm();
    sensitivities(t) = sens;
    sum_sens += sens;
  }

  double mean_sens = sum_sens / N;
  Eigen::VectorXd scale_factors(N);

  if (mean_sens < 1e-10) {
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
// LP 필터
// =============================================================================

void ComposableMPPIControllerPlugin::applyLowPassFilter(
  Eigen::MatrixXd& sequence,
  double alpha,
  const Eigen::VectorXd& initial) const
{
  int N = sequence.rows();
  Eigen::VectorXd prev = initial;
  for (int t = 0; t < N; ++t) {
    Eigen::VectorXd filtered = alpha * sequence.row(t).transpose()
                              + (1.0 - alpha) * prev;
    sequence.row(t) = filtered.transpose();
    prev = filtered;
  }
}

// =============================================================================
// Shield CBF 투영
// =============================================================================

Eigen::VectorXd ComposableMPPIControllerPlugin::projectControlCBF(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& u) const
{
  auto active_barriers = barrier_set_.getActiveBarriers(state);
  if (active_barriers.empty()) {
    return u;
  }

  Eigen::VectorXd u_proj = u;

  for (int iter = 0; iter < shield_max_iterations_; ++iter) {
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
          double step = shield_step_size_ * (-constraint) / dhdot_du_norm_sq;
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

Eigen::VectorXd ComposableMPPIControllerPlugin::computeXdot(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& u) const
{
  Eigen::MatrixXd s(1, state.size());
  s.row(0) = state.transpose();
  Eigen::MatrixXd c(1, u.size());
  c.row(0) = u.transpose();
  return dynamics_->model().dynamicsBatch(s, c).row(0).transpose();
}

// =============================================================================
// computeControl() — 8-Phase 파이프라인
// =============================================================================

std::pair<Eigen::VectorXd, MPPIInfo> ComposableMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // 모든 레이어 비활성 시 → base 호출 (zero overhead)
  if (!active_.rh_mppi && !active_.cs_mppi && !active_.ilqr &&
      !active_.traj_library && !active_.halton && !active_.pi_mppi &&
      !active_.lp_filter && !active_.shield_cbf && !active_.feedback) {
    return MPPIControllerPlugin::computeControl(current_state, reference_trajectory);
  }

  int N = params_.N;
  int K = params_.K;
  int nu = dynamics_->model().controlDim();
  int nx = dynamics_->model().stateDim();

  // ════════════════════════════════════════════════════════════════════════
  // Phase 0: Adaptation
  // ════════════════════════════════════════════════════════════════════════

  int effective_N = N;
  int N_saved = N;

  // Phase 0a: RH-MPPI — 동적 horizon
  if (active_.rh_mppi && horizon_manager_) {
    double speed = (control_sequence_.rows() > 0 && control_sequence_.cols() > 0) ?
      std::abs(control_sequence_(0, 0)) : 0.0;

    double min_obs_dist = params_.rh_obs_dist_threshold;
    if (params_.cbf_enabled && barrier_set_.size() > 0) {
      Eigen::VectorXd h_vals = barrier_set_.evaluateAll(current_state);
      if (h_vals.size() > 0) {
        min_obs_dist = std::max(h_vals.minCoeff(), 0.0);
      }
    }

    double tracking_error = 0.0;
    if (reference_trajectory.rows() > 0 && current_state.size() >= 2) {
      double dx = current_state(0) - reference_trajectory(0, 0);
      double dy = current_state(1) - reference_trajectory(0, 1);
      tracking_error = std::sqrt(dx * dx + dy * dy);
    }

    effective_N = horizon_manager_->computeEffectiveN(
      speed, params_.v_max, min_obs_dist, tracking_error);

    // control_sequence_ 리사이즈
    int current_rows = static_cast<int>(control_sequence_.rows());
    if (effective_N != current_rows) {
      Eigen::MatrixXd new_seq = Eigen::MatrixXd::Zero(effective_N, nu);
      int copy_rows = std::min(effective_N, current_rows);
      if (copy_rows > 0) {
        new_seq.topRows(copy_rows) = control_sequence_.topRows(copy_rows);
      }
      control_sequence_ = new_seq;

      // 버퍼 리사이즈 (noise, perturbed, trajectory)
      noise_buffer_.resize(K, Eigen::MatrixXd::Zero(effective_N, nu));
      perturbed_buffer_.resize(K, Eigen::MatrixXd::Zero(effective_N, nu));
      trajectory_buffer_.resize(K, Eigen::MatrixXd::Zero(effective_N + 1, nx));
    }
    params_.N = effective_N;
    N = effective_N;
  }

  // ════════════════════════════════════════════════════════════════════════
  // Phase 1: Warm-Start — Shift + iLQR + TrajLib
  // ════════════════════════════════════════════════════════════════════════

  // Shift previous control sequence
  for (int t = 0; t < N - 1; ++t) {
    control_sequence_.row(t) = control_sequence_.row(t + 1);
  }
  control_sequence_.row(N - 1) = control_sequence_.row(N - 2);

  // Phase 1a: iLQR warm-start
  if (active_.ilqr && ilqr_solver_) {
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nx, nx);
    int q_size = std::min(static_cast<int>(params_.Q.rows()), nx);
    Q.topLeftCorner(q_size, q_size) = params_.Q.topLeftCorner(q_size, q_size);

    Eigen::MatrixXd Qf = Eigen::MatrixXd::Zero(nx, nx);
    int qf_size = std::min(static_cast<int>(params_.Qf.rows()), nx);
    Qf.topLeftCorner(qf_size, qf_size) = params_.Qf.topLeftCorner(qf_size, qf_size);

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(nu, nu);
    int r_size = std::min(static_cast<int>(params_.R.rows()), nu);
    R.topLeftCorner(r_size, r_size) = params_.R.topLeftCorner(r_size, r_size);

    ilqr_solver_->solve(
      current_state, control_sequence_, reference_trajectory,
      dynamics_->model(), Q, Qf, R, params_.dt);
  }

  // Phase 1b: Trajectory Library 업데이트
  if (active_.traj_library) {
    traj_library_.updatePreviousSolution(control_sequence_);
  }

  // ════════════════════════════════════════════════════════════════════════
  // Phase 0b: CS-MPPI — 공분산 스케일링 계산 (warm-start 이후)
  // ════════════════════════════════════════════════════════════════════════

  if (active_.cs_mppi) {
    cs_scale_buffer_ = computeCovarianceScaling(current_state, control_sequence_);
  }

  // ════════════════════════════════════════════════════════════════════════
  // Phase 2: Sampling
  // ════════════════════════════════════════════════════════════════════════

  sampler_->sampleInPlace(noise_buffer_, K, N, nu);

  // Goal 근처 noise 스케일링
  if (goal_dist_ < params_.goal_slowdown_dist && params_.goal_slowdown_dist > 1e-6) {
    double ratio = goal_dist_ / params_.goal_slowdown_dist;
    double noise_scale = std::clamp(std::sqrt(ratio), 0.2, 1.0);
    for (int k = 0; k < K; ++k) {
      noise_buffer_[k] *= noise_scale;
    }
  }

  // CS-MPPI: per-step noise scaling
  if (active_.cs_mppi) {
    for (int k = 0; k < K; ++k) {
      for (int t = 0; t < N; ++t) {
        noise_buffer_[k].row(t) *= cs_scale_buffer_(t);
      }
    }
  }

  // ════════════════════════════════════════════════════════════════════════
  // Phase 2b: Perturb controls (TrajLib 또는 기본)
  // ════════════════════════════════════════════════════════════════════════

  if (static_cast<int>(perturbed_buffer_.size()) != K) {
    perturbed_buffer_.resize(K, Eigen::MatrixXd::Zero(N, nu));
  }

  int lib_samples = 0;

  if (active_.traj_library) {
    const auto& primitives = traj_library_.getPrimitives();
    int num_prims = traj_library_.numPrimitives();
    int samples_per_prim = params_.traj_library_num_per_primitive;
    if (samples_per_prim <= 0) {
      int total_lib = static_cast<int>(std::floor(params_.traj_library_ratio * K));
      samples_per_prim = (num_prims > 0) ? std::max(1, total_lib / num_prims) : 0;
    }
    lib_samples = samples_per_prim * num_prims;
    if (lib_samples >= K) {
      samples_per_prim = (num_prims > 0) ? (K - 1) / num_prims : 0;
      lib_samples = samples_per_prim * num_prims;
    }

    // 라이브러리 샘플
    int idx = 0;
    for (int p = 0; p < num_prims && idx < lib_samples; ++p) {
      for (int j = 0; j < samples_per_prim && idx < lib_samples; ++j) {
        perturbed_buffer_[idx] = primitives[p].control_sequence;
        if (params_.traj_library_perturbation > 0.0) {
          perturbed_buffer_[idx] += params_.traj_library_perturbation * noise_buffer_[idx];
        }
        perturbed_buffer_[idx] = dynamics_->clipControls(perturbed_buffer_[idx]);
        // noise_buffer_를 라이브러리 delta로 교체 (가중 업데이트용)
        noise_buffer_[idx] = perturbed_buffer_[idx] - control_sequence_;
        ++idx;
      }
    }
  }

  // Gaussian 샘플 (나머지)
  int K_exploit = static_cast<int>((1.0 - params_.exploration_ratio) * (K - lib_samples));
  for (int k = lib_samples; k < K; ++k) {
    int local_k = k - lib_samples;
    if (local_k < K_exploit) {
      perturbed_buffer_[k].noalias() = control_sequence_ + noise_buffer_[k];
    } else {
      perturbed_buffer_[k] = noise_buffer_[k];
    }
    perturbed_buffer_[k] = dynamics_->clipControls(perturbed_buffer_[k]);
  }

  // ════════════════════════════════════════════════════════════════════════
  // Phase 3: Pre-Filter — π-MPPI ADMM on K samples
  // ════════════════════════════════════════════════════════════════════════

  if (active_.pi_mppi && projector_) {
    Eigen::MatrixXd projected(N, nu);
    for (int k = 0; k < K; ++k) {
      projector_->projectSequence(
        perturbed_buffer_[k], projected,
        pi_u_min_, pi_u_max_, pi_rate_max_, pi_accel_max_);
      perturbed_buffer_[k] = projected;
    }
  }

  // ════════════════════════════════════════════════════════════════════════
  // Phase 4: Core MPPI — Rollout → Cost → Weights → Update
  // ════════════════════════════════════════════════════════════════════════

  // 4a: Batch rollout
  dynamics_->rolloutBatchInPlace(
    current_state, perturbed_buffer_, params_.dt, trajectory_buffer_);
  const auto& trajectories = trajectory_buffer_;

  // 4b: Cost computation
  Eigen::VectorXd costs;
  CostBreakdown cost_breakdown;
  if (params_.debug_collision_viz) {
    cost_breakdown = cost_function_->computeDetailed(
      trajectories, perturbed_buffer_, reference_trajectory);
    costs = cost_breakdown.total_costs;
  } else {
    costs = cost_function_->compute(
      trajectories, perturbed_buffer_, reference_trajectory);
  }

  // 4c: IT 정규화
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

  // 4d: Adaptive Temperature + Weights
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);
  }
  Eigen::VectorXd weights = weight_computation_->compute(costs, current_lambda);

  // 4e: Weighted noise update (OpenMP)
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

  // ════════════════════════════════════════════════════════════════════════
  // Phase 5: Post-Filter — LP + π-MPPI
  // ════════════════════════════════════════════════════════════════════════

  // 5a: LP IIR filter
  if (active_.lp_filter) {
    applyLowPassFilter(control_sequence_, lp_alpha_, lp_u_prev_);
    control_sequence_ = dynamics_->clipControls(control_sequence_);
  }

  // 5b: π-MPPI ADMM on optimal sequence (LP 이후 → bound 보장)
  if (active_.pi_mppi && projector_) {
    Eigen::MatrixXd projected(N, nu);
    projector_->projectSequence(
      control_sequence_, projected,
      pi_u_min_, pi_u_max_, pi_rate_max_, pi_accel_max_);
    control_sequence_ = projected;
  }

  // ════════════════════════════════════════════════════════════════════════
  // Phase 6: Safety — Shield CBF 투영
  // ════════════════════════════════════════════════════════════════════════

  bool any_cbf_projected = false;
  if (active_.shield_cbf && !barrier_set_.empty()) {
    auto active_barriers = barrier_set_.getActiveBarriers(current_state);
    if (!active_barriers.empty()) {
      int shield_steps = std::min(shield_cbf_stride_, N);
      Eigen::VectorXd state_k = current_state;

      for (int t = 0; t < shield_steps; ++t) {
        Eigen::VectorXd u_t = control_sequence_.row(t).transpose();
        Eigen::VectorXd u_safe = projectControlCBF(state_k, u_t);

        if ((u_safe - u_t).squaredNorm() > 1e-12) {
          control_sequence_.row(t) = u_safe.transpose();
          any_cbf_projected = true;
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

  // ════════════════════════════════════════════════════════════════════════
  // Phase 7: Output — Extract u_opt + Feedback correction
  // ════════════════════════════════════════════════════════════════════════

  Eigen::VectorXd u_opt = control_sequence_.row(0).transpose();

  // LP: u_prev 업데이트
  if (active_.lp_filter) {
    lp_u_prev_ = u_opt;
  }

  // Phase 7: Feedback Riccati correction
  if (active_.feedback && gain_computer_) {
    if (cycle_counter_ % params_.feedback_recompute_interval == 0) {
      cached_nominal_trajectory_ = Eigen::MatrixXd::Zero(N + 1, nx);
      cached_nominal_trajectory_.row(0) = current_state.transpose();

      for (int t = 0; t < N; ++t) {
        Eigen::MatrixXd s(1, nx);
        s.row(0) = cached_nominal_trajectory_.row(t);
        Eigen::MatrixXd c(1, nu);
        c.row(0) = control_sequence_.row(t);
        cached_nominal_trajectory_.row(t + 1) =
          dynamics_->model().propagateBatch(s, c, params_.dt).row(0);
      }

      Eigen::MatrixXd Q_use = Eigen::MatrixXd::Zero(nx, nx);
      int q_size = std::min(static_cast<int>(params_.Q.rows()), nx);
      Q_use.topLeftCorner(q_size, q_size) = params_.Q.topLeftCorner(q_size, q_size);

      Eigen::MatrixXd Qf_use = Eigen::MatrixXd::Zero(nx, nx);
      int qf_size = std::min(static_cast<int>(params_.Qf.rows()), nx);
      Qf_use.topLeftCorner(qf_size, qf_size) = params_.Qf.topLeftCorner(qf_size, qf_size);

      Eigen::MatrixXd R_use = Eigen::MatrixXd::Zero(nu, nu);
      int r_size = std::min(static_cast<int>(params_.R.rows()), nu);
      R_use.topLeftCorner(r_size, r_size) = params_.R.topLeftCorner(r_size, r_size);

      cached_gains_ = std::vector<Eigen::MatrixXd>(
        gain_computer_->computeGains(
          cached_nominal_trajectory_, control_sequence_,
          dynamics_->model(), Q_use, Qf_use, R_use, params_.dt));
    }
    cycle_counter_++;

    if (!cached_gains_.empty()) {
      Eigen::VectorXd dx = current_state - cached_nominal_trajectory_.row(0).transpose();

      auto angle_idx = dynamics_->model().angleIndices();
      for (int idx : angle_idx) {
        if (idx < dx.size()) {
          dx(idx) = std::atan2(std::sin(dx(idx)), std::cos(dx(idx)));
        }
      }

      Eigen::VectorXd du = params_.feedback_gain_scale * cached_gains_[0] * dx;
      u_opt += du;

      Eigen::MatrixXd u_mat(1, nu);
      u_mat.row(0) = u_opt.transpose();
      u_opt = dynamics_->clipControls(u_mat).row(0).transpose();
    }
  }

  // ════════════════════════════════════════════════════════════════════════
  // Phase 4 (cont): Build MPPIInfo
  // ════════════════════════════════════════════════════════════════════════

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

  // CBF 투영된 경우 → 시각화 궤적 재생성
  if (any_cbf_projected) {
    Eigen::MatrixXd projected_traj(N + 1, nx);
    projected_traj.row(0) = current_state.transpose();
    Eigen::VectorXd s = current_state;
    for (int t = 0; t < N; ++t) {
      Eigen::MatrixXd sm(1, nx);
      sm.row(0) = s.transpose();
      Eigen::MatrixXd cm(1, nu);
      cm.row(0) = control_sequence_.row(t);
      s = dynamics_->model().propagateBatch(sm, cm, params_.dt).row(0).transpose();
      projected_traj.row(t + 1) = s.transpose();
    }
    weighted_traj = projected_traj;
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

  if (params_.debug_collision_viz) {
    info.cost_breakdown = cost_breakdown;
  }

  info.colored_noise_used = params_.colored_noise;
  info.adaptive_temp_used = params_.adaptive_temperature;
  info.tube_mppi_used = false;
  info.cbf_used = any_cbf_projected;

  if (active_.rh_mppi) {
    info.effective_horizon = effective_N;
  }

  // ════════════════════════════════════════════════════════════════════════
  // Phase 8: Restore — RH-MPPI N 복원
  // ════════════════════════════════════════════════════════════════════════

  if (active_.rh_mppi) {
    params_.N = N_saved;

    if (static_cast<int>(control_sequence_.rows()) != N_max_) {
      Eigen::MatrixXd restored = Eigen::MatrixXd::Zero(N_max_, nu);
      int copy_rows = std::min(static_cast<int>(control_sequence_.rows()), N_max_);
      if (copy_rows > 0) {
        restored.topRows(copy_rows) = control_sequence_.topRows(copy_rows);
      }
      control_sequence_ = restored;
    }
  }

  RCLCPP_DEBUG(
    node_->get_logger(),
    "Composable-MPPI: min_cost=%.4f, ESS=%.1f/%d, N=%d, cbf_proj=%d",
    min_cost, ess, K, N, any_cbf_projected);

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
