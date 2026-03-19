#ifndef MPC_CONTROLLER_ROS2__TRAJECTORY_LIBRARY_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__TRAJECTORY_LIBRARY_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/trajectory_library.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Trajectory Library MPPI nav2 Controller Plugin
 *
 * 사전 계산된 7종 제어 시퀀스 프리미티브 라이브러리를 결정적 샘플로 주입하여
 * warm-start 다양성과 수렴 속도를 향상.
 *
 * L개 라이브러리 샘플 + (K-L)개 Gaussian 샘플로 분할.
 * Biased-MPPI와 동일한 밀도비 소거 → 가중치 공식 100% 호환.
 */
class TrajectoryLibraryMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  TrajectoryLibraryMPPIControllerPlugin() = default;
  ~TrajectoryLibraryMPPIControllerPlugin() override = default;

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

  TrajectoryLibrary library_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__TRAJECTORY_LIBRARY_MPPI_CONTROLLER_PLUGIN_HPP_
