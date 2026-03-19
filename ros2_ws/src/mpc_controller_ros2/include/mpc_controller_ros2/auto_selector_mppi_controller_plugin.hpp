#ifndef MPC_CONTROLLER_ROS2__AUTO_SELECTOR_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__AUTO_SELECTOR_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/strategy_selector.hpp"
#include <memory>

namespace mpc_controller_ros2
{

/**
 * @brief Auto-Selector MPPI nav2 Controller Plugin
 *
 * 런타임 컨텍스트(속도, 장애물, 추적 오차, 목표 거리)에 따라
 * MPPI params_ 프로파일을 자동 전환하는 메타 컨트롤러.
 *
 * 5가지 전략:
 *   CRUISE     — 기본 (특수 기능 모두 비활성)
 *   PRECISE    — 목표 근접 (feedback + Q*1.5)
 *   AGGRESSIVE — 고속 주행 (LP 필터)
 *   RECOVERY   — 큰 추적 오차 (탐색 강화, N 축소)
 *   SAFE       — 장애물 근접 (CBF 활성화, 최우선)
 *
 * computeControl() 흐름:
 * ┌───────────────────────────────────────────────────┐
 * │ 1. 컨텍스트 수집 (speed, obs_dist, error, goal)   │
 * │ 2. selector.update() → strategy                   │
 * │ 3. applyProfile(strategy) → params_ 조작          │
 * │ 4. N 변경 시 control_sequence_ 리사이즈            │
 * │ 5. parent::computeControl()                       │
 * │ 6. restoreBaseline() → params_ 원본 복원           │
 * │ 7. N 변경 시 control_sequence_ 복원               │
 * │ 8. info.active_strategy = strategy name           │
 * └───────────────────────────────────────────────────┘
 */
class AutoSelectorMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  AutoSelectorMPPIControllerPlugin() = default;
  ~AutoSelectorMPPIControllerPlugin() override = default;

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
  void applyProfile(MPPIStrategy strategy);
  void restoreBaseline();

  std::unique_ptr<StrategySelector> selector_;
  MPPIParams baseline_params_;   // configure 직후 저장된 원본
  int baseline_N_;               // 원래 params_.N
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__AUTO_SELECTOR_MPPI_CONTROLLER_PLUGIN_HPP_
