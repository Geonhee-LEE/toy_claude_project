#ifndef MPC_CONTROLLER_ROS2__IT_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__IT_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief IT-MPPI (Information-Theoretic MPPI) nav2 Controller Plugin
 *
 * 정보이론적 비용항을 추가하여 탐색(exploration)과 활용(exploitation)의 균형을 맞춤.
 *
 * 핵심 메커니즘:
 *   1. Sample diversity bonus: 각 샘플의 최종 상태와 다른 샘플들 간의 평균 거리
 *   2. KL divergence regularization: 노이즈 크기 기반 사전 분포와의 거리 페널티
 *   3. Information-theoretic cost: cost - exploration_weight * diversity + kl_weight * kl
 *   4. Adaptive exploration: 호출마다 exploration_weight를 감쇠시켜 점진적 수렴
 *
 * 알고리즘:
 *   1. Warm-start shift
 *   2. Sample noise + rollout + compute standard costs
 *   3. For each sample k:
 *      - diversity_bonus[k] = mean L2 distance to other samples' final states
 *      - kl_penalty[k] = 0.5 * ||noise[k]||² / sigma²
 *   4. info_cost[k] = cost[k] - exploration_weight * diversity_bonus[k]
 *                    + kl_weight * kl_penalty[k]
 *   5. MPPI weighted update on info_costs
 */
class ITMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  ITMPPIControllerPlugin() = default;
  ~ITMPPIControllerPlugin() override = default;

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
   * @brief 샘플 다양성 보너스 계산
   * @param trajectories K개 궤적 (N+1, nx)
   * @return K개 다양성 보너스 벡터
   */
  Eigen::VectorXd computeDiversityBonus(
    const std::vector<Eigen::MatrixXd>& trajectories) const;

  /**
   * @brief KL divergence 페널티 계산
   * @param noise K개 노이즈 행렬 (N, nu)
   * @return K개 KL 페널티 벡터
   */
  Eigen::VectorXd computeKLPenalty(
    const std::vector<Eigen::MatrixXd>& noise) const;

  /// 적응형 탐색 가중치 (런타임 감쇠)
  double current_exploration_weight_{0.1};
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__IT_MPPI_CONTROLLER_PLUGIN_HPP_
