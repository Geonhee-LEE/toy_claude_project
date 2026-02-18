#ifndef MPC_CONTROLLER_ROS2__SMOOTH_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__SMOOTH_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Smooth-MPPI nav2 Controller Plugin
 *
 * Reference: Kim et al. (2021) "Smooth Model Predictive Path Integral
 *            Control without Smoothing" — pytorch_mppi v0.8.0+
 *
 * 핵심 수식 (Input-Lifting / Δu Reparameterization):
 *   u[t] = u_prev + Σ_{i=0}^{t} Δu[i]              — cumsum 복원
 *   J_jerk = w · Σ_t R_jerk · ‖Δu[t+1] - Δu[t]‖²  — jerk cost (2차 차분)
 *   ΔU* ← ΔU + Σ_k w_k · ε_k                        — Δu-space 가중 업데이트
 *
 * 알고리즘 플로우:
 *   1. Δu space에서 노이즈 샘플링
 *   2. cumsum으로 u 시퀀스 복원
 *   3. 비용 계산 (tracking + terminal + effort + jerk)
 *   4. Δu space에서 가중 평균 업데이트
 */
class SmoothMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  SmoothMPPIControllerPlugin() = default;
  ~SmoothMPPIControllerPlugin() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros
  ) override;

protected:
  std::pair<Eigen::Vector2d, MPPIInfo> computeControl(
    const Eigen::Vector3d& current_state,
    const Eigen::MatrixXd& reference_trajectory
  ) override;

private:
  Eigen::MatrixXd delta_u_sequence_;   // (N, 2) Δu warm-start
  Eigen::Vector2d u_prev_;             // 이전 제어 (cumsum 기준점)
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__SMOOTH_MPPI_CONTROLLER_PLUGIN_HPP_
