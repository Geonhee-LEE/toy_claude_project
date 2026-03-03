#ifndef MPC_CONTROLLER_ROS2__SHIELD_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__SHIELD_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Shield-MPPI Controller Plugin
 *
 * per-step CBF 투영을 통해 모든 샘플 궤적의 안전성을 보장합니다.
 *
 * 알고리즘:
 *   for each sample k = 1..K:
 *     for each timestep t = 0..N-1 (stride 적용):
 *       u_k_t = projectControlCBF(x_k_t, u_k_t)
 *       x_k_{t+1} = propagate(x_k_t, u_k_t)
 *
 * 모든 궤적이 h(x) > 0을 만족하도록 보장합니다.
 *
 * 성능: stride=2일 때 K=512 기준 ~1ms 추가 오버헤드.
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
   * @brief 단일 제어를 CBF 제약으로 투영 (경량 버전)
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
