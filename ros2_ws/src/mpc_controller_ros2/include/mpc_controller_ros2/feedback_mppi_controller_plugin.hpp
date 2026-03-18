#ifndef MPC_CONTROLLER_ROS2__FEEDBACK_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__FEEDBACK_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/feedback_gain_computer.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Feedback-MPPI (F-MPPI) nav2 Controller Plugin
 *
 * RA-L 2026 기반: MPPI 최적 롤아웃에서 Riccati 역방향 패스로
 * 시변 피드백 게인 K_t를 계산하고, 사이클 간 선형 피드백 보정 적용.
 *
 * 알고리즘:
 *   1. MPPI로 u* 계산 (부모 computeControl)
 *   2. u*로 nominal 궤적 x* 롤아웃
 *   3. (A_t, B_t) 선형화 -> Riccati backward -> K_t
 *   4. 첫 스텝 보정: u = u* + gain_scale * K_0 * (x_actual - x*_0)
 *
 * Tube-MPPI 대비 장점:
 *   - nominal state 유지 불필요 (매 사이클 재계산)
 *   - 시변 게인 K_t (고정 K_fb 대비 정밀)
 *   - 게인 재계산 주기 조절 가능 (feedback_recompute_interval)
 */
class FeedbackMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  FeedbackMPPIControllerPlugin() = default;
  ~FeedbackMPPIControllerPlugin() override = default;

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
  std::unique_ptr<FeedbackGainComputer> gain_computer_;
  std::vector<Eigen::MatrixXd> cached_gains_;
  Eigen::MatrixXd cached_nominal_trajectory_;
  int cycle_counter_{0};
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__FEEDBACK_MPPI_CONTROLLER_PLUGIN_HPP_
