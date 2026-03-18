#include "mpc_controller_ros2/feedback_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::FeedbackMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void FeedbackMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  int nx = dynamics_->model().stateDim();
  int nu = dynamics_->model().controlDim();
  gain_computer_ = std::make_unique<FeedbackGainComputer>(
    nx, nu, params_.feedback_regularization);

  auto node = parent.lock();
  RCLCPP_INFO(
    node->get_logger(),
    "Feedback-MPPI plugin configured: gain_scale=%.2f, recompute_interval=%d, reg=%.1e",
    params_.feedback_gain_scale,
    params_.feedback_recompute_interval,
    params_.feedback_regularization);
}

std::pair<Eigen::VectorXd, MPPIInfo> FeedbackMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // Step 1: 부모 MPPI로 u* 계산
  auto [u_opt, info] = MPPIControllerPlugin::computeControl(
    current_state, reference_trajectory);

  if (!params_.feedback_mppi_enabled) {
    return {u_opt, info};
  }

  int N = params_.N;
  int nx = dynamics_->model().stateDim();
  int nu = dynamics_->model().controlDim();

  // Step 2: Recompute gains at specified interval
  if (cycle_counter_ % params_.feedback_recompute_interval == 0) {
    // Rollout nominal trajectory from current state using optimal control sequence
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

    // Ensure Q, Qf, R dimensions match model
    Eigen::MatrixXd Q_use = Eigen::MatrixXd::Zero(nx, nx);
    int q_size = std::min(static_cast<int>(params_.Q.rows()), nx);
    Q_use.topLeftCorner(q_size, q_size) = params_.Q.topLeftCorner(q_size, q_size);

    Eigen::MatrixXd Qf_use = Eigen::MatrixXd::Zero(nx, nx);
    int qf_size = std::min(static_cast<int>(params_.Qf.rows()), nx);
    Qf_use.topLeftCorner(qf_size, qf_size) = params_.Qf.topLeftCorner(qf_size, qf_size);

    Eigen::MatrixXd R_use = Eigen::MatrixXd::Zero(nu, nu);
    int r_size = std::min(static_cast<int>(params_.R.rows()), nu);
    R_use.topLeftCorner(r_size, r_size) = params_.R.topLeftCorner(r_size, r_size);

    // Step 3: Riccati backward pass -> K_t
    cached_gains_ = std::vector<Eigen::MatrixXd>(
      gain_computer_->computeGains(
        cached_nominal_trajectory_, control_sequence_,
        dynamics_->model(), Q_use, Qf_use, R_use, params_.dt));
  }
  cycle_counter_++;

  // Step 4: Apply feedback correction at first step
  // u = u* + gain_scale * K_0 * (x_actual - x*_0)
  if (!cached_gains_.empty()) {
    Eigen::VectorXd dx = current_state - cached_nominal_trajectory_.row(0).transpose();

    // Angle normalization
    auto angle_idx = dynamics_->model().angleIndices();
    for (int idx : angle_idx) {
      if (idx < dx.size()) {
        dx(idx) = std::atan2(std::sin(dx(idx)), std::cos(dx(idx)));
      }
    }

    Eigen::VectorXd du = params_.feedback_gain_scale * cached_gains_[0] * dx;
    u_opt += du;

    // Clip to control bounds
    Eigen::MatrixXd u_mat(1, nu);
    u_mat.row(0) = u_opt.transpose();
    u_opt = dynamics_->clipControls(u_mat).row(0).transpose();
  }

  RCLCPP_DEBUG(
    node_->get_logger(),
    "F-MPPI: gain_scale=%.2f, recompute=%d, cycle=%d",
    params_.feedback_gain_scale,
    params_.feedback_recompute_interval,
    cycle_counter_);

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
