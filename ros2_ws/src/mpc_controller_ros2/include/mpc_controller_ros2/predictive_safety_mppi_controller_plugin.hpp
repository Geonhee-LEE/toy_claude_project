#ifndef MPC_CONTROLLER_ROS2__PREDICTIVE_SAFETY_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__PREDICTIVE_SAFETY_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/shield_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/predictive_safety_filter.hpp"
#include <memory>

namespace mpc_controller_ros2
{

/**
 * @brief Predictive Safety MPPI Controller Plugin
 *
 * ShieldMPPI를 상속하여 N-step 예측 안전 필터를 추가합니다.
 *
 * ShieldMPPI와의 차이:
 *   - ShieldMPPI: 최적 u* 첫 스텝만 CBF 투영
 *   - PredictiveSafety: 전체 horizon N스텝 forward rollout + CBF 투영
 *   → recursive feasibility 보장
 *
 * 파이프라인:
 *   1. MPPI 샘플링 → optimal u* sequence (부모 MPPIControllerPlugin)
 *   2. Shield CBF per-step 투영 (부모 ShieldMPPIControllerPlugin)
 *   3. N-step Predictive Safety Filter (이 클래스)
 *
 * 파라미터:
 *   predictive_safety_enabled: true/false
 *   predictive_safety_horizon: N (투영 horizon, 기본=전체)
 *   predictive_safety_decay: γ 감쇠 (기본 1.0=균일)
 *   predictive_safety_max_iterations: 스텝당 최대 투영 반복
 */
class PredictiveSafetyMPPIControllerPlugin : public ShieldMPPIControllerPlugin
{
public:
  PredictiveSafetyMPPIControllerPlugin() = default;
  ~PredictiveSafetyMPPIControllerPlugin() override = default;

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
  std::unique_ptr<PredictiveSafetyFilter> predictive_filter_;
  bool predictive_safety_enabled_{false};
  int predictive_safety_horizon_{0};  // 0 = 전체 horizon
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__PREDICTIVE_SAFETY_MPPI_CONTROLLER_PLUGIN_HPP_
