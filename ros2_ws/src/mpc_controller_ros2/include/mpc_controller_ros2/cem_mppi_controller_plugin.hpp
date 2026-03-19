#ifndef MPC_CONTROLLER_ROS2__CEM_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__CEM_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief CEM-MPPI nav2 Controller Plugin
 *
 * Reference: Pinneri et al. (2021) "Sample-Efficient Cross-Entropy Method for MPC"
 *
 * Cross-Entropy Method의 반복적 분포 정제(elite selection + refit μ,σ)와
 * MPPI의 가중 업데이트를 결합한 하이브리드 컨트롤러.
 *
 * 알고리즘:
 *   for i = 1..cem_iterations:
 *     1. Sample K from N(μ, diag(σ²))
 *     2. Rollout + Cost
 *     3. Select top elite_ratio% → elite set
 *     4. Refit: μ_new = mean(elites), σ_new = std(elites)
 *     5. μ = (1-momentum)*μ_new + momentum*μ
 *   Final: MPPI weighted update on last samples
 */
class CemMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  CemMPPIControllerPlugin() = default;
  ~CemMPPIControllerPlugin() override = default;

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

  std::vector<int> selectElites(
    const Eigen::VectorXd& costs, int num_elites) const;

  void refitDistribution(
    const std::vector<Eigen::MatrixXd>& perturbed_controls,
    const std::vector<int>& elite_indices,
    Eigen::MatrixXd& mean_out,
    Eigen::VectorXd& sigma_out) const;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__CEM_MPPI_CONTROLLER_PLUGIN_HPP_
