#ifndef MPC_CONTROLLER_ROS2__SVG_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__SVG_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/svmpc_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief SVG-MPPI (Stein Variational Guided MPPI) nav2 Controller Plugin
 *
 * Kondo et al., ICRA 2024 기반. G개 guide particle만 SVGD로 최적화한 뒤,
 * 나머지 K-G개 샘플을 guide 주변에서 리샘플링하여 효율적으로 다중 모드 탐색.
 *
 * SVMPCControllerPlugin을 상속하여 computeSVGDForce(), computeDiversity(),
 * medianBandwidth() 메서드를 재사용.
 *
 * 알고리즘 플로우:
 *   Phase 1: 전체 K개 sample → rollout → cost
 *   Phase 2: 비용 최저 G개 → guide particle 선택
 *   Phase 3: SVGD loop (L회): G×G 커널로 guide만 최적화
 *   Phase 4: 각 guide 주변 followers 리샘플링
 *   Phase 5: 전체 K개 rollout → cost → weight → U 업데이트
 */
class SVGMPPIControllerPlugin : public SVMPCControllerPlugin
{
public:
  SVGMPPIControllerPlugin() = default;
  ~SVGMPPIControllerPlugin() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros
  ) override;

protected:
  std::pair<Eigen::Vector2d, MPPIInfo> computeControl(
    const Eigen::Vector3d& current_state,
    const Eigen::MatrixXd& reference_trajectory
  ) override;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__SVG_MPPI_CONTROLLER_PLUGIN_HPP_
