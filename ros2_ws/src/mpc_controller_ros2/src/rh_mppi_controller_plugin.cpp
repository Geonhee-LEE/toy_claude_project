// =============================================================================
// RH-MPPI (Receding Horizon MPPI) Controller Plugin
//
// 동적 예측 horizon N 조정: 속도/장애물/추적 오차에 따라 N을 적응적으로 변경.
// params_.N을 임시로 effective_N으로 교체한 뒤 parent::computeControl() 호출.
// sampler, rollout, cost function 모두 행렬 행 수에서 N을 자동 추론하므로 안전.
//
// 알고리즘:
//   combined = weighted_avg(speed_f, obstacle_f, error_f)
//   effective_N = N_min + (N_max - N_min) * combined   (EMA 스무딩)
// =============================================================================

#include "mpc_controller_ros2/rh_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::RHMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void RHMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  // N_max = YAML에서 로드된 N (최대 horizon)
  N_max_ = params_.N;

  // AdaptiveHorizonManager 초기화
  // rh_N_max는 YAML N을 오버라이드하지 않고 params_.rh_N_max를 사용
  int effective_N_max = (params_.rh_N_max > 0) ?
    std::min(params_.rh_N_max, N_max_) : N_max_;
  int effective_N_min = std::min(params_.rh_N_min, effective_N_max);

  horizon_manager_ = std::make_unique<AdaptiveHorizonManager>(
    effective_N_min, effective_N_max,
    params_.rh_speed_weight, params_.rh_obstacle_weight, params_.rh_error_weight,
    params_.rh_obs_dist_threshold, params_.rh_error_threshold,
    params_.rh_smoothing_alpha);

  auto node = parent.lock();
  RCLCPP_INFO(
    node->get_logger(),
    "RH-MPPI plugin configured: N_min=%d, N_max=%d, "
    "speed_w=%.1f, obs_w=%.1f, err_w=%.1f, "
    "obs_thresh=%.1f m, err_thresh=%.1f m, alpha=%.2f",
    effective_N_min, effective_N_max,
    params_.rh_speed_weight, params_.rh_obstacle_weight, params_.rh_error_weight,
    params_.rh_obs_dist_threshold, params_.rh_error_threshold,
    params_.rh_smoothing_alpha);
}

std::pair<Eigen::VectorXd, MPPIInfo> RHMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // ──── Step 0: RH-MPPI 비활성화 시 parent 직접 호출 ────
  if (!params_.rh_mppi_enabled) {
    return MPPIControllerPlugin::computeControl(current_state, reference_trajectory);
  }

  int nu = dynamics_->model().controlDim();

  // ──── Step 1: 환경 컨텍스트 수집 ────
  // 속도: 이전 최적 제어의 선속도 (row(0)의 첫 번째 요소)
  double speed = (control_sequence_.rows() > 0 && control_sequence_.cols() > 0) ?
    std::abs(control_sequence_(0, 0)) : 0.0;

  // 장애물 거리: barrier_set_ 평가 (CBF 비활성화 시 임계값 반환)
  double min_obs_dist = params_.rh_obs_dist_threshold;  // 기본값: 영향 없음
  if (params_.cbf_enabled && barrier_set_.size() > 0) {
    Eigen::VectorXd h_vals = barrier_set_.evaluateAll(current_state);
    if (h_vals.size() > 0) {
      min_obs_dist = h_vals.minCoeff();
      // h(x) = dist - safety_radius 형태이므로, 실제 거리 ≈ h + safety
      min_obs_dist = std::max(min_obs_dist, 0.0);
    }
  }

  // 추적 오차: state[:2]와 reference.row(0)[:2] 사이의 L2 거리
  double tracking_error = 0.0;
  if (reference_trajectory.rows() > 0 && current_state.size() >= 2) {
    double dx = current_state(0) - reference_trajectory(0, 0);
    double dy = current_state(1) - reference_trajectory(0, 1);
    tracking_error = std::sqrt(dx * dx + dy * dy);
  }

  // ──── Step 2: effective_N 계산 ────
  int effective_N = horizon_manager_->computeEffectiveN(
    speed, params_.v_max, min_obs_dist, tracking_error);

  // ──── Step 3: control_sequence_ 리사이즈 ────
  int current_N = static_cast<int>(control_sequence_.rows());
  if (effective_N != current_N) {
    Eigen::MatrixXd new_seq = Eigen::MatrixXd::Zero(effective_N, nu);
    int copy_rows = std::min(effective_N, current_N);
    if (copy_rows > 0) {
      new_seq.topRows(copy_rows) = control_sequence_.topRows(copy_rows);
    }
    control_sequence_ = new_seq;
  }

  // ──── Step 4: params_.N 임시 교체 ────
  params_.N = effective_N;

  // ──── Step 5: parent::computeControl() ────
  auto [u_opt, info] = MPPIControllerPlugin::computeControl(
    current_state, reference_trajectory);

  // ──── Step 6: params_.N 복원 ────
  params_.N = N_max_;

  // ──── Step 7: control_sequence_ 복원 (N_max, zero-pad) ────
  if (static_cast<int>(control_sequence_.rows()) != N_max_) {
    Eigen::MatrixXd restored = Eigen::MatrixXd::Zero(N_max_, nu);
    int copy_rows = std::min(static_cast<int>(control_sequence_.rows()), N_max_);
    if (copy_rows > 0) {
      restored.topRows(copy_rows) = control_sequence_.topRows(copy_rows);
    }
    control_sequence_ = restored;
  }

  // ──── Step 8: info에 effective_horizon 기록 ────
  info.effective_horizon = effective_N;

  RCLCPP_DEBUG(
    node_->get_logger(),
    "RH-MPPI: effective_N=%d, speed=%.2f, obs_dist=%.2f, track_err=%.2f",
    effective_N, speed, min_obs_dist, tracking_error);

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
