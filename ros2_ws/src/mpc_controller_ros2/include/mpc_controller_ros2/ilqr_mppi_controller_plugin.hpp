#ifndef MPC_CONTROLLER_ROS2__ILQR_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__ILQR_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/ilqr_solver.hpp"
#include <memory>

namespace mpc_controller_ros2
{

/**
 * @brief iLQR Warm-Start MPPI Controller Plugin
 *
 * MPPI 샘플링 전에 iLQR(1-2 iter)로 control_sequence_를 개선하여
 * 더 나은 nominal trajectory를 warm-start로 제공합니다.
 *
 * 흐름:
 *   shift(u_prev) → iLQR(1-2iter) → MPPI sample/rollout/weight → u_opt
 *
 * 오버헤드: N=30 기준 ~0.016ms (~1.6%)
 *
 * 근거: Williams et al. IT-MPC (2018), Cho et al. MPPI-IPDDP (2022),
 *       Feedback-MPPI (IEEE RA-L 2025)
 */
class IlqrMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  IlqrMPPIControllerPlugin() = default;
  ~IlqrMPPIControllerPlugin() override = default;

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
  std::unique_ptr<ILQRSolver> ilqr_solver_;
  bool ilqr_enabled_{true};
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__ILQR_MPPI_CONTROLLER_PLUGIN_HPP_
