#ifndef MPC_CONTROLLER_ROS2__SVG_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__SVG_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/svmpc_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief SVG-MPPI (Stein Variational Guided MPPI) nav2 Controller Plugin
 *
 * Reference: Kondo et al. (2024) "SVG-MPPI: Steering Stein Variational
 *            Guided MPPI for Efficient Navigation" — ICRA 2024
 *
 * 핵심 수식:
 *   Guide 선택:  {x_g} = argmin_G S(x)               — Top-G by cost
 *   SVGD force:  φ*(x_i) = Σ_j [w_j·k(x_j,x_i)·(x_j-x_i)  (attractive)
 *                         + (1/G)·k(x_j,x_i)·(x_j-x_i)/h²]  (repulsive)
 *   RBF kernel:  k(x_i,x_j) = exp(-‖x_i-x_j‖²/2h²)
 *   Bandwidth:   h = √(median(‖x_i-x_j‖²) / 2·log(G+1))  (median heuristic)
 *   Follower:    x_f ~ N(x_guide, σ²_resample·I)     — Resampling
 *   복잡도:      O(G²D) + O(KND) << O(K²D) (SVMPC 대비)
 *
 * SVMPCControllerPlugin 상속 → computeSVGDForce(), computeDiversity(),
 * medianBandwidth() 재사용.
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
