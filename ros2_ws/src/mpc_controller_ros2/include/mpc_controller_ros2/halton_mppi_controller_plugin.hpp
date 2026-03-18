#ifndef MPC_CONTROLLER_ROS2__HALTON_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__HALTON_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Halton-MPPI nav2 Controller Plugin
 *
 * MDPI Drones 2026 기반: Halton 저불일치 시퀀스로 제어 공간을
 * 균일하게 커버하여 적은 샘플로도 빠른 수렴 달성.
 *
 * Gaussian/Colored Noise 대신 Halton 시퀀스 + 역정규 CDF를 사용.
 * OU 시간 상관은 선택적으로 적용 (halton_beta > 0).
 */
class HaltonMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  HaltonMPPIControllerPlugin() = default;
  ~HaltonMPPIControllerPlugin() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros
  ) override;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__HALTON_MPPI_CONTROLLER_PLUGIN_HPP_
