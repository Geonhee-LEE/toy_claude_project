// =============================================================================
// MPPI-H (Hybrid Swerve MPPI) Controller Plugin
//
// Reference: MizuhoAOKI/mppi_swerve_drive_ros (IROS 2024, arXiv:2409.08648)
// "Switching Sampling Space of MPPI to Balance Efficiency and Safety
//  in 4WIDS Vehicle Navigation"
//
// 핵심: Low-D(body velocity) ↔ 4D(대각 바퀴) 실시간 전환
// - Low-D 모드 (cdist < 0.3m AND cangle < 0.3rad): 효율적 주행
// - 4D 모드 (otherwise): 높은 자유도로 안전한 회피
// - 히스테리시스: 채터링 방지 (기본 3 cycles)
//
// 기존 인프라 재사용:
//   - base class의 dynamics_, sampler_, control_sequence_ → Low-D 모드
//   - dynamics_4d_, sampler_4d_, control_seq_4d_ → 4D 모드
//   - WheelLevel4DModel::FK/IK → warm-start 변환
// =============================================================================

#include "mpc_controller_ros2/hybrid_swerve_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>
#include <algorithm>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::HybridSwerveMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void HybridSwerveMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  // Base class 초기화 (Low-D 모델 생성)
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  // Non-Coaxial 여부 판별
  is_non_coaxial_ = (params_.motion_model == "non_coaxial_swerve");

  int N = params_.N;
  int K = params_.K;

  // ─── 4D 모드 초기화 ───

  // 4D MotionModel 생성
  MPPIParams params_4d = params_;
  params_4d.motion_model = "wheel_level_4d";
  auto model_4d = MotionModelFactory::create("wheel_level_4d", params_4d);
  dynamics_4d_ = std::make_unique<BatchDynamicsWrapper>(params_4d, std::move(model_4d));

  // 4D Sampler: nu=4 노이즈 σ
  Eigen::VectorXd noise_sigma_4d(4);
  noise_sigma_4d(0) = params_.hybrid_noise_sigma_vfl;
  noise_sigma_4d(1) = params_.hybrid_noise_sigma_vrr;
  noise_sigma_4d(2) = params_.hybrid_noise_sigma_dfl;
  noise_sigma_4d(3) = params_.hybrid_noise_sigma_drr;
  sampler_4d_ = std::make_unique<GaussianSampler>(noise_sigma_4d);

  // 4D Cost function: Q(3x3 공유), R(4x4)
  int nx_4d = 3;  // WheelLevel4DModel는 항상 nx=3
  Eigen::MatrixXd Q_4d = params_.Q.topLeftCorner(nx_4d, nx_4d);
  Eigen::MatrixXd Qf_4d = params_.Qf.topLeftCorner(nx_4d, nx_4d);
  Eigen::MatrixXd R_4d = Eigen::MatrixXd::Zero(4, 4);
  R_4d(0, 0) = params_.hybrid_R_vfl;
  R_4d(1, 1) = params_.hybrid_R_vrr;
  R_4d(2, 2) = params_.hybrid_R_dfl;
  R_4d(3, 3) = params_.hybrid_R_drr;

  cost_function_4d_ = std::make_unique<CompositeMPPICost>();
  cost_function_4d_->addCost(std::make_unique<StateTrackingCost>(Q_4d));
  cost_function_4d_->addCost(std::make_unique<TerminalCost>(Qf_4d));
  cost_function_4d_->addCost(std::make_unique<ControlEffortCost>(R_4d));

  // 4D 버퍼 사전 할당
  control_seq_4d_ = Eigen::MatrixXd::Zero(N, 4);
  noise_buf_4d_.resize(K, Eigen::MatrixXd::Zero(N, 4));
  perturbed_buf_4d_.resize(K, Eigen::MatrixXd::Zero(N, 4));
  traj_buf_4d_.resize(K, Eigen::MatrixXd::Zero(N + 1, nx_4d));

  // 초기 상태
  current_mode_ = Mode::LOW_D;
  mode_switch_counter_ = 0;
  tracked_delta_ = 0.0;
  last_ctrl_4d_ = Eigen::Vector4d::Zero();

  auto node = parent.lock();
  RCLCPP_INFO(
    node->get_logger(),
    "MPPI-H plugin configured: enabled=%d, model=%s (non_coaxial=%d), "
    "cdist_thr=%.2f, cangle_thr=%.2f, hysteresis=%d, "
    "lf=%.2f lr=%.2f dl=%.2f dr=%.2f",
    params_.hybrid_enabled,
    params_.motion_model.c_str(),
    is_non_coaxial_,
    params_.hybrid_cdist_threshold,
    params_.hybrid_cangle_threshold,
    params_.hybrid_hysteresis_count,
    params_.hybrid_lf, params_.hybrid_lr,
    params_.hybrid_dl, params_.hybrid_dr);
}

// =============================================================================
// 모드 결정
// =============================================================================

HybridSwerveMPPIControllerPlugin::Mode
HybridSwerveMPPIControllerPlugin::determineMode(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory) const
{
  // 참조 궤적의 첫 번째 미래 점과의 오차
  int ref_idx = std::min(5, static_cast<int>(reference_trajectory.rows()) - 1);
  double dx = reference_trajectory(ref_idx, 0) - current_state(0);
  double dy = reference_trajectory(ref_idx, 1) - current_state(1);
  double cdist = std::sqrt(dx * dx + dy * dy);

  double ref_theta = reference_trajectory(ref_idx, 2);
  double cur_theta = current_state(2);
  double cangle = std::abs(normalizeAngle(ref_theta - cur_theta));

  if (cdist < params_.hybrid_cdist_threshold &&
      cangle < params_.hybrid_cangle_threshold) {
    return Mode::LOW_D;
  }
  return Mode::FOUR_D;
}

// =============================================================================
// Warm-start 변환: Low-D → 4D (IK)
// =============================================================================

void HybridSwerveMPPIControllerPlugin::convertLowTo4D()
{
  int N = params_.N;
  int nu_low = dynamics_->model().controlDim();
  double lf = params_.hybrid_lf;
  double lr = params_.hybrid_lr;
  double dl = params_.hybrid_dl;
  double dr = params_.hybrid_dr;

  for (int t = 0; t < N; ++t) {
    Eigen::Vector3d body_vel;
    if (is_non_coaxial_) {
      // NonCoaxial [v, omega, delta_dot] → body velocity [vx, vy, omega]
      // vx = v, vy ≈ 0 (non-holonomic), omega = omega
      body_vel(0) = control_sequence_(t, 0);
      body_vel(1) = 0.0;
      body_vel(2) = (nu_low >= 2) ? control_sequence_(t, 1) : 0.0;
    } else {
      // Swerve [vx, vy, omega] → body velocity 직접
      body_vel(0) = control_sequence_(t, 0);
      body_vel(1) = (nu_low >= 2) ? control_sequence_(t, 1) : 0.0;
      body_vel(2) = (nu_low >= 3) ? control_sequence_(t, 2) : 0.0;
    }

    Eigen::Vector4d u4d = WheelLevel4DModel::inverseKinematics(body_vel, lf, lr, dl, dr);
    control_seq_4d_.row(t) = u4d.transpose();
  }
}

// =============================================================================
// Warm-start 변환: 4D → Low-D (FK)
// =============================================================================

void HybridSwerveMPPIControllerPlugin::convert4DToLow()
{
  int N = params_.N;
  int nu_low = dynamics_->model().controlDim();
  double dl = params_.hybrid_dl;
  double dr = params_.hybrid_dr;

  for (int t = 0; t < N; ++t) {
    Eigen::Vector4d u4d = control_seq_4d_.row(t).transpose();
    Eigen::Vector3d body_vel = WheelLevel4DModel::forwardKinematics(u4d, dl, dr);

    if (is_non_coaxial_) {
      // body velocity [vx, vy, omega] → NonCoaxial [v, omega, delta_dot]
      control_sequence_(t, 0) = body_vel(0);  // v = vx
      if (nu_low >= 2) control_sequence_(t, 1) = body_vel(2);  // omega
      if (nu_low >= 3) control_sequence_(t, 2) = 0.0;  // delta_dot ≈ 0
    } else {
      // body velocity → Swerve [vx, vy, omega]
      control_sequence_(t, 0) = body_vel(0);
      if (nu_low >= 2) control_sequence_(t, 1) = body_vel(1);
      if (nu_low >= 3) control_sequence_(t, 2) = body_vel(2);
    }
  }
  control_sequence_ = dynamics_->clipControls(control_sequence_);
}

// =============================================================================
// 4D 모드 MPPI 파이프라인
// =============================================================================

std::pair<Eigen::VectorXd, MPPIInfo>
HybridSwerveMPPIControllerPlugin::computeControl4D(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  int N = params_.N;
  int K = params_.K;
  int nu_4d = 4;
  int nx_4d = 3;

  // 4D 상태 추출: WheelLevel4DModel은 nx=3 [x,y,θ]
  Eigen::VectorXd state_4d;
  if (is_non_coaxial_) {
    // NonCoaxial [x,y,θ,δ] → 4D [x,y,θ]
    state_4d = current_state.head(3);
  } else {
    state_4d = current_state;
  }

  // 4D 참조 궤적: nx=3 부분만
  Eigen::MatrixXd ref_4d = reference_trajectory.leftCols(
    std::min(nx_4d, static_cast<int>(reference_trajectory.cols())));

  // ──── STEP 1: Warm-start (shift control sequence) ────
  for (int t = 0; t < N - 1; ++t) {
    control_seq_4d_.row(t) = control_seq_4d_.row(t + 1);
  }
  control_seq_4d_.row(N - 1) = control_seq_4d_.row(N - 2);

  // ──── STEP 2: Sample noise ────
  sampler_4d_->sampleInPlace(noise_buf_4d_, K, N, nu_4d);

  // ──── STEP 3: Perturb + Clip ────
  if (static_cast<int>(perturbed_buf_4d_.size()) != K) {
    perturbed_buf_4d_.resize(K, Eigen::MatrixXd::Zero(N, nu_4d));
  }
  int K_exploit = static_cast<int>((1.0 - params_.exploration_ratio) * K);

  for (int k = 0; k < K; ++k) {
    if (k < K_exploit) {
      perturbed_buf_4d_[k].noalias() = control_seq_4d_ + noise_buf_4d_[k];
    } else {
      perturbed_buf_4d_[k] = noise_buf_4d_[k];
    }
    perturbed_buf_4d_[k] = dynamics_4d_->clipControls(perturbed_buf_4d_[k]);
  }

  // ──── STEP 4: Batch rollout ────
  dynamics_4d_->rolloutBatchInPlace(
    state_4d, perturbed_buf_4d_, params_.dt, traj_buf_4d_);

  // ──── STEP 5: Cost computation ────
  Eigen::VectorXd costs = cost_function_4d_->compute(
    traj_buf_4d_, perturbed_buf_4d_, ref_4d);

  // ──── STEP 6: Weights ────
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);
  }
  Eigen::VectorXd weights = weight_computation_->compute(costs, current_lambda);

  // ──── STEP 7: Weighted update ────
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu_4d);
  for (int k = 0; k < K; ++k) {
    weighted_noise.noalias() += weights(k) * noise_buf_4d_[k];
  }
  control_seq_4d_ += weighted_noise;
  control_seq_4d_ = dynamics_4d_->clipControls(control_seq_4d_);

  // ──── STEP 8: δ 추적 (Non-Coaxial용) ────
  last_ctrl_4d_ = control_seq_4d_.row(0).transpose();
  tracked_delta_ = (last_ctrl_4d_(2) + last_ctrl_4d_(3)) / 2.0;

  // ──── STEP 9: FK → body velocity → Twist 변환용 제어 ────
  Eigen::Vector4d u_opt_4d = control_seq_4d_.row(0).transpose();
  Eigen::Vector3d body_vel = WheelLevel4DModel::forwardKinematics(
    u_opt_4d, params_.hybrid_dl, params_.hybrid_dr);

  // Low-D 형식으로 결과 반환 (Twist 출력은 body velocity)
  Eigen::VectorXd u_opt;
  int nu_low = dynamics_->model().controlDim();
  if (is_non_coaxial_) {
    u_opt = Eigen::VectorXd::Zero(nu_low);
    u_opt(0) = body_vel(0);  // v
    if (nu_low >= 2) u_opt(1) = body_vel(2);  // omega
    if (nu_low >= 3) u_opt(2) = 0.0;  // delta_dot ≈ 0
  } else {
    u_opt = Eigen::VectorXd::Zero(nu_low);
    u_opt(0) = body_vel(0);  // vx
    if (nu_low >= 2) u_opt(1) = body_vel(1);  // vy
    if (nu_low >= 3) u_opt(2) = body_vel(2);  // omega
  }

  // ──── STEP 10: MPPIInfo ────
  int best_idx;
  double min_cost = costs.minCoeff(&best_idx);
  double ess = computeESS(weights);

  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, nx_4d);
  for (int k = 0; k < K; ++k) {
    weighted_traj.noalias() += weights(k) * traj_buf_4d_[k];
  }

  MPPIInfo info;
  info.sample_trajectories = traj_buf_4d_;
  info.sample_weights = weights;
  info.best_trajectory = traj_buf_4d_[best_idx];
  info.weighted_avg_trajectory = weighted_traj;
  info.temperature = (params_.adaptive_temperature && adaptive_temp_) ?
    adaptive_temp_->getLambda() : params_.lambda;
  info.ess = ess;
  info.costs = costs;

  if (node_) {
    RCLCPP_DEBUG(
      node_->get_logger(),
      "MPPI-H [4D]: min_cost=%.4f, ESS=%.1f/%d",
      min_cost, ess, K);
  }

  (void)min_cost;  // suppress unused variable warning in Release

  return {u_opt, info};
}

// =============================================================================
// computeControl — MPPI-H 파이프라인 (모드 전환 + 히스테리시스)
// =============================================================================

std::pair<Eigen::VectorXd, MPPIInfo>
HybridSwerveMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // Hybrid 비활성 시 Low-D base 호출
  if (!params_.hybrid_enabled) {
    return MPPIControllerPlugin::computeControl(current_state, reference_trajectory);
  }

  // ──── 모드 결정 (히스테리시스) ────
  Mode desired_mode = determineMode(current_state, reference_trajectory);

  if (desired_mode != current_mode_) {
    mode_switch_counter_++;
    if (mode_switch_counter_ >= params_.hybrid_hysteresis_count) {
      // 모드 전환 실행
      if (desired_mode == Mode::FOUR_D) {
        convertLowTo4D();
        // Non-Coaxial: δ 초기화
        if (is_non_coaxial_ && current_state.size() >= 4) {
          tracked_delta_ = current_state(3);
          for (int t = 0; t < params_.N; ++t) {
            control_seq_4d_(t, 2) = tracked_delta_;  // δ_fl
            control_seq_4d_(t, 3) = tracked_delta_;  // δ_rr
          }
        }
      } else {
        convert4DToLow();
        // Non-Coaxial: δ_avg 복원
        if (is_non_coaxial_) {
          tracked_delta_ = (last_ctrl_4d_(2) + last_ctrl_4d_(3)) / 2.0;
        }
      }
      current_mode_ = desired_mode;
      mode_switch_counter_ = 0;

      if (node_) {
        RCLCPP_DEBUG(
          node_->get_logger(),
          "MPPI-H: mode switch → %s",
          (current_mode_ == Mode::FOUR_D) ? "4D" : "Low-D");
      }
    }
  } else {
    mode_switch_counter_ = 0;
  }

  // ──── 선택된 모드로 MPPI 실행 ────
  if (current_mode_ == Mode::FOUR_D) {
    return computeControl4D(current_state, reference_trajectory);
  } else {
    return MPPIControllerPlugin::computeControl(current_state, reference_trajectory);
  }
}

}  // namespace mpc_controller_ros2
