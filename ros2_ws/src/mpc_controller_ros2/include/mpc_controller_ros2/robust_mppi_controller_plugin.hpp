#ifndef MPC_CONTROLLER_ROS2__ROBUST_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__ROBUST_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Robust MPPI (Distributionally Robust) nav2 Controller Plugin
 *
 * Minimax 최적화를 통해 분포 불확실성 하의 worst-case 기대 비용 최소화.
 * 표준 MPPI의 softmin 가중 평균 대신 CVaR-like worst-case 비용 추정 사용.
 *
 * 알고리즘:
 *   1. K 궤적 샘플링 + 비용 계산 (기존 MPPI 동일)
 *   2. Robust 처리:
 *      a. 비용 분산 계산 → robust_cost[k] = cost[k] + penalty * variance
 *      b. Wasserstein radius > 0: 추가 보수 페널티
 *      c. 비용 정렬, worst-alpha 경험적 CVaR 추정
 *      d. Adaptive alpha: 비용 분산에 따라 alpha 조정
 *   3. Robust 비용 기반 IT-normalization → 가중 업데이트
 */
class RobustMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  RobustMPPIControllerPlugin() = default;
  ~RobustMPPIControllerPlugin() override = default;

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
   * @brief 비용 벡터에 robust 처리 적용
   * @param costs 원본 비용 벡터 (K,) — in-place 수정
   * @return worst-case CVaR 비용, effective alpha
   */
  std::pair<double, double> applyRobustProcessing(
    Eigen::VectorXd& costs) const;

  /**
   * @brief Wasserstein 페널티 계산 (비용 기울기 norm 근사)
   * @param costs 비용 벡터
   * @return 페널티 벡터
   */
  Eigen::VectorXd computeWassersteinPenalty(
    const Eigen::VectorXd& costs) const;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__ROBUST_MPPI_CONTROLLER_PLUGIN_HPP_
