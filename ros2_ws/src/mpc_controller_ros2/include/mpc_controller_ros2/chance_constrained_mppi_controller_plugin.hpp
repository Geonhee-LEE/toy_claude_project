#ifndef MPC_CONTROLLER_ROS2__CHANCE_CONSTRAINED_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__CHANCE_CONSTRAINED_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief CC-MPPI (Chance-Constrained MPPI) nav2 Controller Plugin
 *
 * Blackmore et al. (JGCD 2011) 영감: 확률적 제약 만족 보장.
 *
 * 핵심 메커니즘:
 *   1. K 샘플 기반 per-constraint 위반 확률 추정: p_hat = count(g>0)/K
 *   2. Risk budget 분배: Bonferroni (ε/M) 또는 Adaptive (slack 재분배)
 *   3. Quantile tightening: P(g(x)≤0) ≥ 1-ε 위반 시 페널티
 *   4. EMA smoothed empirical quantile로 안정적 tightening
 *
 * vs Constrained MPPI:
 *   - Constrained: deterministic Augmented Lagrangian + dual variables
 *   - CC-MPPI: probabilistic sample-based + risk allocation (no dual)
 */
class ChanceConstrainedMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  ChanceConstrainedMPPIControllerPlugin() = default;
  ~ChanceConstrainedMPPIControllerPlugin() override = default;

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
   * @brief K 샘플의 per-constraint 위반량 평가
   * @return (K x 3) 행렬: [vel, accel, clearance] per sample
   */
  Eigen::MatrixXd evaluateSampleViolations(
    const std::vector<Eigen::MatrixXd>& perturbed_controls,
    const std::vector<Eigen::MatrixXd>& trajectories) const;

  /**
   * @brief Per-constraint 위반 확률 추정
   * @return (3,): p_hat_i = count(g_i > 0) / K
   */
  Eigen::Vector3d estimateViolationProbabilities(
    const Eigen::MatrixXd& violations) const;

  /**
   * @brief Risk budget 분배 (Bonferroni 또는 Adaptive)
   * @return (3,): per-constraint risk allocation ε_i
   */
  Eigen::Vector3d allocateRisk(
    const Eigen::Vector3d& violation_probs) const;

  /**
   * @brief Chance-constrained augmented costs 계산
   */
  Eigen::VectorXd computeChanceConstrainedCosts(
    const Eigen::VectorXd& base_costs,
    const Eigen::MatrixXd& violations,
    const Eigen::Vector3d& allocated_risk) const;

  /**
   * @brief Empirical quantile 계산 (nth_element O(K))
   */
  double empiricalQuantile(
    const Eigen::VectorXd& values, double quantile_level) const;

  /// EMA smoothed quantiles (3,)
  Eigen::Vector3d smoothed_quantiles_{Eigen::Vector3d::Zero()};
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__CHANCE_CONSTRAINED_MPPI_CONTROLLER_PLUGIN_HPP_
