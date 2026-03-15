#include "mpc_controller_ros2/clf_cbf_mppi_controller_plugin.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::CLFCBFMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void CLFCBFMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  // 부모 configure (Shield-MPPI 전체 초기화)
  ShieldMPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  clf_cbf_enabled_ = params_.clf_cbf_enabled;

  if (clf_cbf_enabled_) {
    int nx = dynamics_->model().stateDim();
    int nu = dynamics_->model().controlDim();

    // P 행렬: Q 가중치에 scale 적용
    Eigen::MatrixXd P = params_.clf_P_scale * params_.Q;
    // P가 nx보다 작으면 확장
    if (P.rows() < nx) {
      Eigen::MatrixXd P_full = Eigen::MatrixXd::Identity(nx, nx);
      P_full.topLeftCorner(P.rows(), P.cols()) = P;
      P = P_full;
    }

    // CLF 생성
    auto angle_indices = dynamics_->model().angleIndices();
    clf_ = std::make_unique<CLFFunction>(
      P, params_.clf_decay_rate, angle_indices);

    // 제어 bounds
    Eigen::VectorXd u_min(nu), u_max(nu);
    if (nu >= 2) {
      u_min(0) = params_.v_min;
      u_max(0) = params_.v_max;
      u_min(1) = params_.omega_min;
      u_max(1) = params_.omega_max;
    }
    if (nu >= 3) {
      u_min(2) = params_.omega_min;  // vy_min for swerve
      u_max(2) = params_.omega_max;  // vy_max for swerve
    }

    // CLF-CBF-QP 솔버 생성
    clf_cbf_solver_ = std::make_unique<CLFCBFQPSolver>(
      clf_.get(), &barrier_set_,
      params_.cbf_gamma,
      params_.clf_slack_penalty,
      u_min, u_max);

    RCLCPP_INFO(node_->get_logger(),
      "CLF-CBF-MPPI configured (c=%.2f, p=%.1f, P_scale=%.1f)",
      params_.clf_decay_rate, params_.clf_slack_penalty, params_.clf_P_scale);
  }
}

std::pair<Eigen::VectorXd, MPPIInfo> CLFCBFMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // 1. Shield-MPPI 계산 (MPPI + CBF 투영)
  auto [u_opt, info] = ShieldMPPIControllerPlugin::computeControl(
    current_state, reference_trajectory);

  // 2. CLF-CBF 비활성화이면 바로 반환
  if (!clf_cbf_enabled_ || !clf_ || !clf_cbf_solver_) {
    return {u_opt, info};
  }

  // 3. 참조 궤적에서 x_des 추출 (다음 스텝)
  int nx = dynamics_->model().stateDim();
  Eigen::VectorXd x_des = Eigen::VectorXd::Zero(nx);
  if (reference_trajectory.rows() > 1) {
    int ref_cols = std::min((int)reference_trajectory.cols(), nx);
    x_des.head(ref_cols) = reference_trajectory.row(1).head(ref_cols).transpose();
  } else if (reference_trajectory.rows() == 1) {
    int ref_cols = std::min((int)reference_trajectory.cols(), nx);
    x_des.head(ref_cols) = reference_trajectory.row(0).head(ref_cols).transpose();
  }

  // 4. CLF-CBF-QP 해결
  CLFCBFQPResult qp_result;
  if (barrier_set_.empty() || barrier_set_.getActiveBarriers(current_state).empty()) {
    // 장애물 없음 → CLF만
    qp_result = clf_cbf_solver_->solveCLFOnly(
      current_state, x_des, u_opt, *dynamics_);
  } else {
    // CLF + CBF 통합 QP
    qp_result = clf_cbf_solver_->solve(
      current_state, x_des, u_opt, *dynamics_);
  }

  if (qp_result.feasible) {
    u_opt = qp_result.u_safe;
  }

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
