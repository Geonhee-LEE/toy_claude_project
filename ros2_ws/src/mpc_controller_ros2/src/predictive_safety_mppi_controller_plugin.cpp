#include "mpc_controller_ros2/predictive_safety_mppi_controller_plugin.hpp"
#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::PredictiveSafetyMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void PredictiveSafetyMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  ShieldMPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  predictive_safety_enabled_ = params_.predictive_safety_enabled;
  predictive_safety_horizon_ = params_.predictive_safety_horizon;

  if (predictive_safety_enabled_) {
    int nu = dynamics_->model().controlDim();
    Eigen::VectorXd u_min(nu), u_max(nu);
    if (nu >= 2) {
      u_min(0) = params_.v_min;
      u_max(0) = params_.v_max;
      u_min(1) = params_.omega_min;
      u_max(1) = params_.omega_max;
    }
    if (nu >= 3) {
      u_min(2) = params_.omega_min;
      u_max(2) = params_.omega_max;
    }

    predictive_filter_ = std::make_unique<PredictiveSafetyFilter>(
      &barrier_set_, params_.cbf_gamma, params_.dt,
      u_min, u_max);

    predictive_filter_->setMaxIterations(params_.predictive_safety_max_iterations);
    predictive_filter_->setHorizonDecay(params_.predictive_safety_decay);

    RCLCPP_INFO(node_->get_logger(),
      "Predictive Safety MPPI configured (horizon=%d, decay=%.2f, max_iter=%d)",
      predictive_safety_horizon_,
      params_.predictive_safety_decay,
      params_.predictive_safety_max_iterations);
  }
}

std::pair<Eigen::VectorXd, MPPIInfo> PredictiveSafetyMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // 1. Shield-MPPI (MPPI + per-step CBF)
  auto [u_opt, info] = ShieldMPPIControllerPlugin::computeControl(
    current_state, reference_trajectory);

  // 2. Predictive Safety 비활성화 시 반환
  if (!predictive_safety_enabled_ || !predictive_filter_) {
    return {u_opt, info};
  }

  // 3. 최적 제어 시퀀스 추출 (control_sequence_에서)
  // control_sequence_: (N x nu) from MPPI solver
  if (control_sequence_.rows() < 2) {
    return {u_opt, info};
  }

  int N = control_sequence_.rows();
  if (predictive_safety_horizon_ > 0 && predictive_safety_horizon_ < N) {
    N = predictive_safety_horizon_;
  }

  // N-step forward safety filter
  Eigen::MatrixXd ctrl_seq = control_sequence_.topRows(N);
  auto result = predictive_filter_->filter(
    current_state, ctrl_seq, dynamics_->model());

  // 첫 스텝 제어를 u_opt으로 교체
  if (result.feasible || result.num_corrected_steps > 0) {
    u_opt = result.u_safe_sequence.row(0).transpose();
    // 나머지 시퀀스도 업데이트 (warm-start용)
    control_sequence_.topRows(N) = result.u_safe_sequence;
  }

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
