#ifndef MPC_CONTROLLER_ROS2__CONSTRAINED_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__CONSTRAINED_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Constrained MPPI nav2 Controller Plugin
 *
 * Augmented Lagrangian 기법을 사용하여 hard constraints를 명시적으로 처리:
 *   - 속도 제약: |v| <= v_max, |omega| <= omega_max
 *   - 가속도 제약: |dv/dt| <= a_max, |domega/dt| <= alpha_max
 *   - 장애물 클리어런스: min_dist >= d_min
 *
 * 알고리즘:
 *   1. Sample K trajectories + rollout + standard costs
 *   2. Augmented cost: L(u,lambda,mu) = cost + lambda^T g + (mu/2)||max(0,g)||^2
 *   3. MPPI weighted update on augmented costs
 *   4. Dual update: lambda = max(0, lambda + mu*g), mu *= growth
 */
class ConstrainedMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  ConstrainedMPPIControllerPlugin() = default;
  ~ConstrainedMPPIControllerPlugin() override = default;

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

  /**
   * @brief 단일 제어 시퀀스의 제약 위반 벡터 계산
   * @return (3,) [vel_violation, accel_violation, clearance_violation]
   */
  Eigen::Vector3d evaluateConstraintViolation(
    const Eigen::MatrixXd& control_sequence,
    const Eigen::MatrixXd& trajectory) const;

  /**
   * @brief K 샘플 전체의 augmented cost 계산
   */
  Eigen::VectorXd computeAugmentedCosts(
    const Eigen::VectorXd& base_costs,
    const std::vector<Eigen::MatrixXd>& perturbed_controls,
    const std::vector<Eigen::MatrixXd>& trajectories) const;

  /**
   * @brief Dual variable (lambda, mu) 업데이트
   */
  void updateDualVariables(const Eigen::Vector3d& violation);

  // Lagrange multipliers (3: vel, accel, clearance)
  Eigen::Vector3d lambda_{Eigen::Vector3d::Zero()};
  // Penalty parameter
  double mu_{1.0};
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__CONSTRAINED_MPPI_CONTROLLER_PLUGIN_HPP_
