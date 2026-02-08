#ifndef MPC_CONTROLLER_ROS2__LOG_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__LOG_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Log-MPPI nav2 Controller Plugin
 *
 * MPPIControllerPlugin을 상속하고 가중치 계산 전략만
 * LogMPPIWeights로 교체한 플러그인.
 *
 * Vanilla MPPI와 수학적으로 동일하나, log-space 가중치 조작이 필요한
 * 후속 확장(importance sampling 보정 등)의 기반 클래스로 활용.
 */
class LogMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  LogMPPIControllerPlugin();
  ~LogMPPIControllerPlugin() override = default;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__LOG_MPPI_CONTROLLER_PLUGIN_HPP_
