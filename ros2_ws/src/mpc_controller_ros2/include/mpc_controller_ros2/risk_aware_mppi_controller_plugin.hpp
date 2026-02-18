#ifndef MPC_CONTROLLER_ROS2__RISK_AWARE_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__RISK_AWARE_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Risk-Aware (CVaR) MPPI nav2 Controller Plugin
 *
 * MPPIControllerPlugin을 상속하고 가중치 계산 전략을
 * RiskAwareMPPIWeights로 교체한 플러그인.
 *
 * Conditional Value at Risk 기반 가중치 절단:
 * - alpha = 1.0: 모든 샘플 사용 (Vanilla 동일)
 * - alpha < 1.0: 최저 비용 ceil(alpha*K)개만 사용 (risk-averse)
 * - alpha = 0.1: 상위 10% 최저 비용 샘플만 반영 (매우 보수적)
 */
class RiskAwareMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  RiskAwareMPPIControllerPlugin() = default;
  ~RiskAwareMPPIControllerPlugin() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros
  ) override;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__RISK_AWARE_MPPI_CONTROLLER_PLUGIN_HPP_
