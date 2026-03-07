#include "mpc_controller_ros2/ilqr_mppi_controller_plugin.hpp"
#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::IlqrMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void IlqrMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  // 부모 configure 호출 (기존 MPPI 전체 초기화)
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  // iLQR 파라미터 로드
  ilqr_enabled_ = params_.ilqr_enabled;

  int nx = dynamics_->model().stateDim();
  int nu = dynamics_->model().controlDim();

  ILQRParams ilqr_params;
  ilqr_params.max_iterations = params_.ilqr_max_iterations;
  ilqr_params.regularization = params_.ilqr_regularization;
  ilqr_params.line_search_steps = params_.ilqr_line_search_steps;
  ilqr_params.cost_tolerance = params_.ilqr_cost_tolerance;

  ilqr_solver_ = std::make_unique<ILQRSolver>(ilqr_params, nx, nu);

  RCLCPP_INFO(node_->get_logger(),
    "iLQR-MPPI configured (enabled=%s, max_iter=%d, reg=%.1e)",
    ilqr_enabled_ ? "true" : "false",
    ilqr_params.max_iterations, ilqr_params.regularization);
}

std::pair<Eigen::VectorXd, MPPIInfo> IlqrMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // iLQR warm-start: 기존 control_sequence_를 개선
  if (ilqr_enabled_ && ilqr_solver_) {
    int nx = dynamics_->model().stateDim();
    int nu = dynamics_->model().controlDim();

    // Q, Qf, R 행렬을 모델 차원에 맞게 추출
    Eigen::MatrixXd Q = params_.Q.topLeftCorner(
      std::min(static_cast<int>(params_.Q.rows()), nx),
      std::min(static_cast<int>(params_.Q.cols()), nx));
    if (Q.rows() < nx || Q.cols() < nx) {
      Eigen::MatrixXd Q_full = Eigen::MatrixXd::Zero(nx, nx);
      Q_full.topLeftCorner(Q.rows(), Q.cols()) = Q;
      Q = Q_full;
    }

    Eigen::MatrixXd Qf = params_.Qf.topLeftCorner(
      std::min(static_cast<int>(params_.Qf.rows()), nx),
      std::min(static_cast<int>(params_.Qf.cols()), nx));
    if (Qf.rows() < nx || Qf.cols() < nx) {
      Eigen::MatrixXd Qf_full = Eigen::MatrixXd::Zero(nx, nx);
      Qf_full.topLeftCorner(Qf.rows(), Qf.cols()) = Qf;
      Qf = Qf_full;
    }

    Eigen::MatrixXd R = params_.R.topLeftCorner(
      std::min(static_cast<int>(params_.R.rows()), nu),
      std::min(static_cast<int>(params_.R.cols()), nu));
    if (R.rows() < nu || R.cols() < nu) {
      Eigen::MatrixXd R_full = Eigen::MatrixXd::Zero(nu, nu);
      R_full.topLeftCorner(R.rows(), R.cols()) = R;
      R = R_full;
    }

    ilqr_solver_->solve(
      current_state, control_sequence_, reference_trajectory,
      dynamics_->model(), Q, Qf, R, params_.dt);
  }

  // 기본 MPPI 계산 (갱신된 control_sequence_ 기반)
  return MPPIControllerPlugin::computeControl(current_state, reference_trajectory);
}

}  // namespace mpc_controller_ros2
