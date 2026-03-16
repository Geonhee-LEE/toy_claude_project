#ifndef MPC_CONTROLLER_ROS2__TUBE_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__TUBE_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Tube-MPPI Controller Plugin
 *
 * 기존 post-hoc 보정 방식과 달리, computeControl()을 override하여
 * nominal state에서 MPPI를 실행하는 진정한 Tube-MPPI 구현.
 *
 * 흐름:
 *   nominal_state → MPPI → u_nominal
 *   body_error = ancillary(nominal, actual)
 *   u_applied = u_nominal + K_fb · body_error
 *   nominal_state = propagate(nominal, u_nominal)
 *
 * 기존 TubeMPPI 클래스의 AncillaryController, computeTubeBoundary,
 * isInsideTube 메서드를 재사용합니다.
 */
class TubeMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  TubeMPPIControllerPlugin() = default;
  ~TubeMPPIControllerPlugin() override = default;

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
  Eigen::VectorXd nominal_state_;
  bool nominal_initialized_{false};
  double nominal_reset_threshold_{1.0};
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__TUBE_MPPI_CONTROLLER_PLUGIN_HPP_
