#include "mpc_controller_ros2/adaptive_shield_mppi_controller_plugin.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::AdaptiveShieldMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void AdaptiveShieldMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  // 부모 configure 호출 (Shield-MPPI 전체 초기화)
  ShieldMPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  // Adaptive Shield 전용 파라미터
  alpha_min_ = params_.adaptive_shield_alpha_min;
  alpha_max_ = params_.adaptive_shield_alpha_max;
  k_d_ = params_.adaptive_shield_k_d;
  k_v_ = params_.adaptive_shield_k_v;

  RCLCPP_INFO(node_->get_logger(),
    "AdaptiveShield-MPPI configured (alpha=[%.2f, %.2f], k_d=%.2f, k_v=%.2f)",
    alpha_min_, alpha_max_, k_d_, k_v_);
}

std::pair<Eigen::VectorXd, MPPIInfo> AdaptiveShieldMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // 적응형 alpha 계산
  if (params_.cbf_enabled && !barrier_set_.empty()) {
    double d_min = computeMinObstacleDistance(current_state);

    // 로봇 속도 추정 (이전 제어 시퀀스의 첫 번째)
    double robot_speed = 0.0;
    if (control_sequence_.rows() > 0) {
      robot_speed = std::abs(control_sequence_(0, 0));
      if (control_sequence_.cols() >= 3) {
        // swerve: sqrt(vx² + vy²)
        double vx = control_sequence_(0, 0);
        double vy = control_sequence_(0, 1);
        robot_speed = std::sqrt(vx * vx + vy * vy);
      }
    }

    // 동적 gamma 갱신
    params_.cbf_gamma = computeAdaptiveAlpha(d_min, robot_speed);
  }

  // Shield-MPPI 호출 (동적 gamma 적용)
  return ShieldMPPIControllerPlugin::computeControl(
    current_state, reference_trajectory);
}

double AdaptiveShieldMPPIControllerPlugin::computeAdaptiveAlpha(
  double min_distance, double robot_speed) const
{
  // α(d,v) = α_min + (α_max - α_min) · exp(-k_d · d) · (1 + k_v · ||v||)
  double alpha = alpha_min_ +
    (alpha_max_ - alpha_min_) *
    std::exp(-k_d_ * min_distance) *
    (1.0 + k_v_ * robot_speed);

  return std::clamp(alpha, alpha_min_, alpha_max_);
}

double AdaptiveShieldMPPIControllerPlugin::computeMinObstacleDistance(
  const Eigen::VectorXd& state) const
{
  double min_dist = std::numeric_limits<double>::max();
  for (const auto& barrier : barrier_set_.barriers()) {
    double dx = state(0) - barrier.obsX();
    double dy = state(1) - barrier.obsY();
    double dist = std::sqrt(dx * dx + dy * dy) - barrier.safeDistance();
    min_dist = std::min(min_dist, dist);
  }
  return std::max(min_dist, 0.0);
}

}  // namespace mpc_controller_ros2
