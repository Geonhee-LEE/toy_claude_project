#ifndef MPC_CONTROLLER_ROS2__CC_CBF_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__CC_CBF_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief CC-CBF-MPPI (Chance-Constrained CBF-MPPI) nav2 Controller Plugin
 *
 * CC-MPPI의 확률적 제약 프레임워크 + CBF 기반 barrier clearance 통합.
 * P(충돌) ≤ ε 보증을 달성하는 하이브리드 플러그인.
 *
 * 핵심 메커니즘:
 *   1. K 샘플 궤적에서 barrier h(x_t) < 0 여부를 sample-based 추정
 *   2. 4종 제약: velocity, acceleration, clearance (barrier), CBF rate (dh/dt)
 *   3. Risk budget 분배: Bonferroni (ε/M) 또는 Adaptive
 *   4. 선택적 CBF 투영: 최적 제어에 Shield-MPPI 스타일 안전 필터
 *
 * vs CC-MPPI:
 *   - clearance 제약이 barrier_set_ 기반 실제 평가 (placeholder 아님)
 *   - 4번째 제약: CBF rate (dh/dt + γh < 0 위반)
 *   - 선택적 CBF 투영으로 hard safety 보장
 *
 * vs Shield-MPPI:
 *   - 확률적 접근 (P(h<0) ≤ ε) — soft cost + quantile tightening
 *   - CBF 투영은 선택적 후처리
 *
 * 참고: Blackmore et al. (JGCD 2011) + Ames et al. (2019)
 */
class CCCBFMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  CCCBFMPPIControllerPlugin() = default;
  ~CCCBFMPPIControllerPlugin() override = default;

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
   * @return (K x 4) 행렬: [vel, accel, clearance, cbf_rate] per sample
   */
  Eigen::MatrixXd evaluateSampleViolations(
    const std::vector<Eigen::MatrixXd>& perturbed_controls,
    const std::vector<Eigen::MatrixXd>& trajectories) const;

  /**
   * @brief Per-constraint 위반 확률 추정
   * @return (4,): p_hat_i = count(g_i > 0) / K
   */
  Eigen::Vector4d estimateViolationProbabilities(
    const Eigen::MatrixXd& violations) const;

  /**
   * @brief Risk budget 분배 (Bonferroni 또는 Adaptive)
   * @return (4,): per-constraint risk allocation ε_i
   */
  Eigen::Vector4d allocateRisk(
    const Eigen::Vector4d& violation_probs) const;

  /**
   * @brief Chance-constrained augmented costs 계산
   */
  Eigen::VectorXd computeChanceConstrainedCosts(
    const Eigen::VectorXd& base_costs,
    const Eigen::MatrixXd& violations,
    const Eigen::Vector4d& allocated_risk) const;

  /**
   * @brief Empirical quantile 계산 (nth_element O(K))
   */
  double empiricalQuantile(
    const Eigen::VectorXd& values, double quantile_level) const;

  /**
   * @brief CBF 투영 (Shield-MPPI 스타일)
   */
  Eigen::VectorXd projectControlCBF(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& u) const;

  /**
   * @brief 단일 상태 동역학: f(x, u) → x_dot
   */
  Eigen::VectorXd computeXdot(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& u) const;

  /// 4-constraint EMA smoothed quantiles
  Eigen::Vector4d smoothed_quantiles_{Eigen::Vector4d::Zero()};
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__CC_CBF_MPPI_CONTROLLER_PLUGIN_HPP_
