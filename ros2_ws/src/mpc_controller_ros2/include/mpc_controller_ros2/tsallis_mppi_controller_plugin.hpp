#ifndef MPC_CONTROLLER_ROS2__TSALLIS_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__TSALLIS_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Tsallis-MPPI nav2 Controller Plugin
 *
 * MPPIControllerPlugin을 상속하고 가중치 계산 전략을
 * TsallisMPPIWeights로 교체한 플러그인.
 *
 * q-exponential 기반 가중치로 탐색/집중 균형 조절:
 * - q > 1: heavy-tail (탐색 증가, 다양한 샘플 반영)
 * - q < 1: light-tail (집중 증가, 최적 샘플에 집중)
 * - q = 1: Vanilla MPPI와 동일 (표준 softmax)
 */
class TsallisMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  TsallisMPPIControllerPlugin() = default;
  ~TsallisMPPIControllerPlugin() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros
  ) override;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__TSALLIS_MPPI_CONTROLLER_PLUGIN_HPP_
