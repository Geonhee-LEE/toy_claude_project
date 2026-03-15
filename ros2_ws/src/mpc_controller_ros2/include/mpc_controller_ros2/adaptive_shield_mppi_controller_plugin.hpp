#ifndef MPC_CONTROLLER_ROS2__ADAPTIVE_SHIELD_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__ADAPTIVE_SHIELD_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/shield_mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Adaptive Shield-MPPI Controller Plugin
 *
 * ShieldMPPI를 상속하여 거리/속도 기반 동적 CBF gamma를 적용합니다.
 *
 * α(d,v) = α_min + (α_max - α_min) · exp(-k_d · d) · (1 + k_v · ||v||)
 *
 * - 장애물 가까움 + 고속 → 높은 alpha (강한 CBF)
 * - 장애물 멀리 + 저속 → 낮은 alpha (약한 CBF)
 */
class AdaptiveShieldMPPIControllerPlugin : public ShieldMPPIControllerPlugin
{
public:
  AdaptiveShieldMPPIControllerPlugin() = default;
  ~AdaptiveShieldMPPIControllerPlugin() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros
  ) override;

protected:
  std::pair<Eigen::VectorXd, MPPIInfo> computeControl(
    const Eigen::VectorXd& current_state,
    const Eigen::MatrixXd& reference_trajectory
  ) override;

private:
  /**
   * @brief 적응형 alpha 계산
   * @param min_distance 최소 장애물 거리
   * @param robot_speed 현재 로봇 속도
   * @return 동적 alpha 값
   */
  double computeAdaptiveAlpha(double min_distance, double robot_speed) const;

  /**
   * @brief 현재 상태에서 최소 장애물 거리 계산
   */
  double computeMinObstacleDistance(const Eigen::VectorXd& state) const;

  double alpha_min_{0.1};
  double alpha_max_{1.0};
  double k_d_{1.0};
  double k_v_{0.5};
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__ADAPTIVE_SHIELD_MPPI_CONTROLLER_PLUGIN_HPP_
