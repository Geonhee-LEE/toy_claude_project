#ifndef MPC_CONTROLLER_ROS2__RH_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__RH_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/adaptive_horizon_manager.hpp"
#include <memory>

namespace mpc_controller_ros2
{

/**
 * @brief RH-MPPI (Receding Horizon MPPI) nav2 Controller Plugin
 *
 * 동적 예측 horizon N 조정 기능:
 *   - 고속 → 긴 horizon (먼 미래 예측)
 *   - 저속/장애물 근접/큰 추적 오차 → 짧은 horizon (빠른 반응)
 *   - EMA 스무딩으로 N 변화 시 jitter 방지
 *
 * 핵심 전략: params_.N을 임시로 effective_N으로 교체한 뒤
 * parent::computeControl() 호출. sampler, rollout, cost function 모두
 * 행렬 행 수에서 N을 자동 추론하므로 안전.
 *
 * computeControl() 흐름:
 *   1. 환경 수집: speed, min_obs_dist, tracking_err
 *   2. effective_N = horizon_manager(환경)
 *   3. control_sequence_ 리사이즈 (effective_N)
 *   4. params_.N = effective_N (임시)
 *   5. parent::computeControl() ← 자동 적응
 *   6. params_.N = N_max (복원)
 *   7. control_sequence_ 복원 (N_max, zero-pad)
 *   8. info.effective_horizon = effective_N
 */
class RHMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  RHMPPIControllerPlugin() = default;
  ~RHMPPIControllerPlugin() override = default;

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
  std::unique_ptr<AdaptiveHorizonManager> horizon_manager_;
  int N_max_;  // 원래 params_.N (최대 horizon)
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__RH_MPPI_CONTROLLER_PLUGIN_HPP_
