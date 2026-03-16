#include "mpc_controller_ros2/tube_mppi_controller_plugin.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::TubeMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void TubeMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  // 부모 configure 호출 (기존 MPPI 전체 초기화 + TubeMPPI 인스턴스 생성)
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  nominal_reset_threshold_ = params_.tube_nominal_reset_threshold;
  nominal_initialized_ = false;

  RCLCPP_INFO(node_->get_logger(),
    "Tube-MPPI Plugin configured (tube_width=%.2f, reset_threshold=%.2f, "
    "k_fwd=%.2f, k_lat=%.2f, k_ang=%.2f)",
    params_.tube_width, nominal_reset_threshold_,
    params_.k_forward, params_.k_lateral, params_.k_angle);
}

std::pair<Eigen::VectorXd, MPPIInfo> TubeMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // Tube-MPPI 비활성화 시 기본 MPPI 사용
  if (!params_.tube_enabled) {
    return MPPIControllerPlugin::computeControl(current_state, reference_trajectory);
  }

  int nx = dynamics_->model().stateDim();
  int nu = dynamics_->model().controlDim();

  // 1. Nominal state 초기화
  if (!nominal_initialized_ || nominal_state_.size() != nx) {
    nominal_state_ = current_state;
    nominal_initialized_ = true;
  }

  // 2. Nominal-actual 편차 확인 → 임계값 초과 시 리셋
  double deviation = (nominal_state_.head(2) - current_state.head(2)).norm();
  if (deviation > nominal_reset_threshold_) {
    nominal_state_ = current_state;
  }

  // 3. Nominal state에서 MPPI 실행 (부모 호출)
  auto [u_nominal, info] = MPPIControllerPlugin::computeControl(
    nominal_state_, reference_trajectory);

  // 4. Body frame 오차 계산 + 피드백 보정
  //    TubeMPPI 인스턴스(tube_mppi_) 내부의 AncillaryController 사용
  Eigen::VectorXd body_error = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd feedback_correction = Eigen::VectorXd::Zero(nu);

  // Body frame 오차: nominal → actual
  double theta_nom = nominal_state_(2);
  double dx = current_state(0) - nominal_state_(0);
  double dy = current_state(1) - nominal_state_(1);
  double dtheta = current_state(2) - nominal_state_(2);
  // 각도 정규화
  while (dtheta > M_PI) { dtheta -= 2.0 * M_PI; }
  while (dtheta < -M_PI) { dtheta += 2.0 * M_PI; }

  // World → Body frame 변환
  body_error(0) = std::cos(theta_nom) * dx + std::sin(theta_nom) * dy;   // e_forward
  body_error(1) = -std::sin(theta_nom) * dx + std::cos(theta_nom) * dy;  // e_lateral
  body_error(2) = dtheta;  // e_angle

  // 피드백 보정: K_fb @ body_error
  Eigen::MatrixXd K_fb = params_.getFeedbackGainMatrix();
  if (K_fb.rows() == nu && K_fb.cols() >= 3) {
    feedback_correction = K_fb * body_error;
  } else if (nu >= 2) {
    // 폴백: 기본 DiffDrive 게인
    feedback_correction(0) = params_.k_forward * body_error(0);
    feedback_correction(1) = params_.k_lateral * body_error(1) +
                             params_.k_angle * body_error(2);
  }

  // 5. u_applied = u_nominal + feedback (클리핑)
  Eigen::VectorXd u_applied = u_nominal + feedback_correction;

  // 클리핑
  Eigen::MatrixXd u_mat(1, nu);
  u_mat.row(0) = u_applied.transpose();
  u_applied = dynamics_->clipControls(u_mat).row(0).transpose();

  // 6. Nominal state 전파 (u_nominal으로)
  Eigen::MatrixXd state_mat(1, nx);
  state_mat.row(0) = nominal_state_.transpose();
  Eigen::MatrixXd ctrl_mat(1, nu);
  ctrl_mat.row(0) = u_nominal.transpose();
  nominal_state_ = dynamics_->model().propagateBatch(
    state_mat, ctrl_mat, params_.dt).row(0).transpose();

  // 7. Info 갱신
  info.tube_mppi_used = true;
  info.tube_info.nominal_state = nominal_state_;
  info.tube_info.nominal_control = u_nominal;
  info.tube_info.body_error = body_error;
  info.tube_info.feedback_correction = feedback_correction;
  info.tube_info.applied_control = u_applied;
  info.tube_info.tube_width = params_.tube_width;

  // 7.5. 시각화 정렬: nominal→actual offset 적용
  // publishVisualization()은 current_state(actual)를 기준으로 마커를 그리지만,
  // info 내 궤적은 nominal 기준이므로 offset을 적용하여 actual 중심으로 맞춤
  double offset_x = current_state(0) - nominal_state_(0) + u_nominal(0) * params_.dt * std::cos(nominal_state_(2));
  double offset_y = current_state(1) - nominal_state_(1) + u_nominal(0) * params_.dt * std::sin(nominal_state_(2));
  // nominal_state_는 이미 전파되었으므로, 전파 전 nominal에서의 offset 사용
  // (nominal은 step 6에서 이미 전파됨)
  offset_x = current_state(0) - (nominal_state_(0) - u_nominal(0) * params_.dt * std::cos(current_state(2)));
  offset_y = current_state(1) - (nominal_state_(1) - u_nominal(0) * params_.dt * std::sin(current_state(2)));

  // 간단한 방식: actual - old_nominal (전파 전)
  // old_nominal은 이미 사라졌으므로, body_error에서 역산
  offset_x = dx;  // current - old_nominal (이미 계산됨)
  offset_y = dy;

  auto shiftTrajectory = [&](Eigen::MatrixXd& traj) {
    if (traj.rows() > 0 && traj.cols() >= 2) {
      traj.col(0).array() += offset_x;
      traj.col(1).array() += offset_y;
    }
  };

  shiftTrajectory(info.weighted_avg_trajectory);
  shiftTrajectory(info.best_trajectory);
  for (auto& traj : info.sample_trajectories) {
    shiftTrajectory(traj);
  }

  // Tube 경계 계산 (시각화용)
  if (info.weighted_avg_trajectory.rows() > 1) {
    double tw = params_.tube_width;
    for (int i = 0; i < info.weighted_avg_trajectory.rows(); ++i) {
      double x = info.weighted_avg_trajectory(i, 0);
      double y = info.weighted_avg_trajectory(i, 1);
      double theta = (info.weighted_avg_trajectory.cols() >= 3)
        ? info.weighted_avg_trajectory(i, 2) : 0.0;
      double px = -std::sin(theta);
      double py = std::cos(theta);
      Eigen::VectorXd left(3), right(3);
      left << x + tw * px, y + tw * py, theta;
      right << x - tw * px, y - tw * py, theta;
      info.tube_info.tube_boundary.push_back(left);
      info.tube_info.tube_boundary.push_back(right);
    }
  }

  return {u_applied, info};
}

}  // namespace mpc_controller_ros2
