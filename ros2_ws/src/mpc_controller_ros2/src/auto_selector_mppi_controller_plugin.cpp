// =============================================================================
// Auto-Selector MPPI Controller Plugin
//
// 런타임 컨텍스트 기반 MPPI 전략 자동 전환 메타 컨트롤러.
// params_ 프로파일 교체 → parent::computeControl() → baseline 복원.
//
// 전략: CRUISE / PRECISE / AGGRESSIVE / RECOVERY / SAFE
// 우선순위: SAFE > RECOVERY > AGGRESSIVE > PRECISE > CRUISE
// =============================================================================

#include "mpc_controller_ros2/auto_selector_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::AutoSelectorMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void AutoSelectorMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  // baseline 저장 (deep copy — Eigen MatrixXd 포함)
  baseline_params_ = params_;
  baseline_N_ = params_.N;

  // StrategySelector 초기화
  selector_ = std::make_unique<StrategySelector>(
    params_.auto_selector_safety_threshold,
    params_.auto_selector_recovery_threshold,
    params_.auto_selector_fast_threshold,
    params_.auto_selector_precision_dist,
    params_.auto_selector_hysteresis,
    params_.auto_selector_smoothing_alpha);

  auto node = parent.lock();
  RCLCPP_INFO(
    node->get_logger(),
    "Auto-Selector MPPI configured: enabled=%s, "
    "safety=%.2fm, recovery=%.2fm, fast=%.1f, precise=%.2fm, "
    "hysteresis=%d, alpha=%.2f",
    params_.auto_selector_enabled ? "true" : "false",
    params_.auto_selector_safety_threshold,
    params_.auto_selector_recovery_threshold,
    params_.auto_selector_fast_threshold,
    params_.auto_selector_precision_dist,
    params_.auto_selector_hysteresis,
    params_.auto_selector_smoothing_alpha);
}

std::pair<Eigen::VectorXd, MPPIInfo> AutoSelectorMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // ──── Step 0: 비활성화 시 vanilla ────
  if (!params_.auto_selector_enabled) {
    return MPPIControllerPlugin::computeControl(current_state, reference_trajectory);
  }

  int nu = dynamics_->model().controlDim();

  // ──── Step 1: 컨텍스트 수집 ────
  double speed = (control_sequence_.rows() > 0 && control_sequence_.cols() > 0) ?
    std::abs(control_sequence_(0, 0)) : 0.0;

  double min_obs_dist = 10.0;  // 기본: 영향 없음
  if (barrier_set_.size() > 0) {
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

  double goal_dist = goal_dist_;  // 부모 클래스의 protected 멤버

  // ──── Step 2: 전략 선택 ────
  MPPIStrategy strategy = selector_->update(
    speed, baseline_params_.v_max, min_obs_dist, tracking_error, goal_dist);

  // ──── Step 3: 프로파일 적용 ────
  applyProfile(strategy);

  // ──── Step 4: N 변경 시 control_sequence_ 리사이즈 ────
  int effective_N = params_.N;
  int current_N = static_cast<int>(control_sequence_.rows());
  if (effective_N != current_N) {
    Eigen::MatrixXd new_seq = Eigen::MatrixXd::Zero(effective_N, nu);
    int copy_rows = std::min(effective_N, current_N);
    if (copy_rows > 0) {
      new_seq.topRows(copy_rows) = control_sequence_.topRows(copy_rows);
    }
    control_sequence_ = new_seq;
  }

  // ──── Step 5: parent::computeControl() ────
  auto [u_opt, info] = MPPIControllerPlugin::computeControl(
    current_state, reference_trajectory);

  // ──── Step 6: baseline 복원 ────
  restoreBaseline();

  // ──── Step 7: control_sequence_ 복원 ────
  if (static_cast<int>(control_sequence_.rows()) != baseline_N_) {
    Eigen::MatrixXd restored = Eigen::MatrixXd::Zero(baseline_N_, nu);
    int copy_rows = std::min(static_cast<int>(control_sequence_.rows()), baseline_N_);
    if (copy_rows > 0) {
      restored.topRows(copy_rows) = control_sequence_.topRows(copy_rows);
    }
    control_sequence_ = restored;
  }

  // ──── Step 8: info 기록 ────
  info.active_strategy = strategyToString(strategy);
  info.effective_horizon = effective_N;

  RCLCPP_DEBUG(
    node_->get_logger(),
    "Auto-Selector: strategy=%s, speed=%.2f, obs=%.2f, err=%.2f, goal=%.2f",
    info.active_strategy.c_str(), speed, min_obs_dist, tracking_error, goal_dist);

  return {u_opt, info};
}

void AutoSelectorMPPIControllerPlugin::applyProfile(MPPIStrategy strategy)
{
  switch (strategy) {
    case MPPIStrategy::SAFE:
      params_.cbf_enabled = true;
      params_.shield_cbf_stride = 1;
      params_.predictive_safety_enabled = true;
      break;

    case MPPIStrategy::RECOVERY: {
      params_.exploration_ratio = 0.3;
      params_.lambda = baseline_params_.lambda * 3.0;
      int half_N = std::max(10, baseline_N_ / 2);
      params_.N = half_N;
      break;
    }

    case MPPIStrategy::AGGRESSIVE:
      params_.lp_enabled = true;
      params_.lp_cutoff_frequency = 8.0;
      break;

    case MPPIStrategy::PRECISE:
      params_.feedback_mppi_enabled = true;
      params_.Q = baseline_params_.Q * 1.5;
      params_.Qf = baseline_params_.Qf * 1.5;
      break;

    case MPPIStrategy::CRUISE:
    default:
      // baseline 그대로 — 특수 기능 없음
      break;
  }
}

void AutoSelectorMPPIControllerPlugin::restoreBaseline()
{
  params_ = baseline_params_;
}

}  // namespace mpc_controller_ros2
