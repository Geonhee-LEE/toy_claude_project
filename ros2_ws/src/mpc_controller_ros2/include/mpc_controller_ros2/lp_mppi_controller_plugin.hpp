#ifndef MPC_CONTROLLER_ROS2__LP_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__LP_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief LP-MPPI (Low-Pass Filtering MPPI) nav2 Controller Plugin
 *
 * 2025 연구 기반: 샘플링된 제어 시퀀스에 1차 IIR Low-Pass 필터를
 * 적용하여 고주파 chattering을 제거하고 제어 품질을 향상.
 *
 * 핵심 수식 (1st-order IIR Low-Pass Filter):
 *   y[t] = α·x[t] + (1-α)·y[t-1]
 *   α = dt / (τ + dt),  τ = 1/(2πf_c)   (f_c: 컷오프 주파수 Hz)
 *
 * Smooth-MPPI와 상보적:
 *   - Smooth-MPPI: 시간 도메인 (Δu 리파라미터화)
 *   - LP-MPPI: 주파수 도메인 (IIR 필터)
 *
 * 알고리즘:
 *   1. K개 노이즈 샘플에 LP 필터 적용 (lp_filter_all_samples=true)
 *   2. 필터링된 제어로 rollout + cost 계산
 *   3. 가중 업데이트된 최적 시퀀스에도 LP 필터 적용
 */
class LPMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  LPMPPIControllerPlugin() = default;
  ~LPMPPIControllerPlugin() override = default;

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
   * @brief 제어 시퀀스에 causal IIR LP 필터 적용
   * @param sequence (N, nu) 제어 시퀀스 (in-place 수정)
   * @param alpha 필터 계수 (0: 완전 여과, 1: 필터 없음)
   * @param initial (nu,) 필터 초기값 (이전 제어)
   */
  void applyLowPassFilter(
    Eigen::MatrixXd& sequence,
    double alpha,
    const Eigen::VectorXd& initial) const;

  Eigen::VectorXd u_prev_;         // (nu,) 이전 적용 제어
  Eigen::MatrixXd prev_sequence_;  // (N, nu) 이전 최적 시퀀스 (warm-start)
  double lp_alpha_{1.0};           // 계산된 필터 계수
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__LP_MPPI_CONTROLLER_PLUGIN_HPP_
