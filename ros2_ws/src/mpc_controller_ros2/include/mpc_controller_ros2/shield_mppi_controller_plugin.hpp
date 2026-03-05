#ifndef MPC_CONTROLLER_ROS2__SHIELD_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__SHIELD_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Shield-MPPI Controller Plugin
 *
 * per-step CBF 투영을 통해 최적 제어 시퀀스의 안전성을 보장합니다.
 *
 * 성능 최적화 전략:
 *   - 최적 제어 시퀀스(u*)에만 CBF 투영 적용 (K개 샘플 전체 X)
 *   - 해석적 ∂ḣ/∂u 계산 (유한차분 제거)
 *   - Active barrier 없으면 즉시 skip
 *
 * 이를 통해 K=256, N=30에서도 10Hz 제어 루프 유지 가능.
 */
class ShieldMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  ShieldMPPIControllerPlugin() = default;
  ~ShieldMPPIControllerPlugin() override = default;

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
  /**
   * @brief 단일 제어를 CBF 제약으로 투영 (해석적 gradient)
   * @param state 현재 상태 (nx,)
   * @param u 제어 입력 (nu,)
   * @return 투영된 제어 (nu,)
   */
  Eigen::VectorXd projectControlCBF(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& u) const;

  /** @brief 단일 상태 동역학: f(x, u) → x_dot */
  Eigen::VectorXd computeXdot(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& u) const;

  int shield_cbf_stride_{1};
  int shield_max_iterations_{10};
  double shield_step_size_{0.1};
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__SHIELD_MPPI_CONTROLLER_PLUGIN_HPP_
