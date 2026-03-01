#ifndef MPC_CONTROLLER_ROS2__DIAL_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__DIAL_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief DIAL-MPPI nav2 Controller Plugin
 *
 * Reference: Xue et al. (2024) "DIAL-MPC: Diffusion-Inspired Annealing
 *            For Model Predictive Control" arXiv:2409.15610 (ICRA 2025)
 *
 * 핵심: MPPI = 단일 스텝 확산 디노이징 → DIAL = 다중 스텝 어닐링.
 * 내부 루프에서 N_diffuse번 반복하며 이중 감쇠 스케줄로 노이즈를 줄여
 * 더 정밀한 최적 제어 시퀀스를 탐색합니다.
 *
 * 확장:
 *   - Shield-DIAL: CBF Safety Filter 통합 (기존 cbf_safety_filter_ 재사용)
 *   - Adaptive-DIAL: 비용 수렴 기반 적응형 반복 횟수 (조기 종료)
 */
class DialMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  DialMPPIControllerPlugin() = default;
  ~DialMPPIControllerPlugin() override = default;

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
   * @brief 이중 감쇠 노이즈 스케줄 계산
   * σ²ᵢₕ = exp(-(N-i)/(β₁·N) - (H-h)/(β₂·H)), clamped ≥ min_noise
   * @param iteration 현재 어닐링 반복 인덱스 (1-based)
   * @param n_diffuse 총 어닐링 반복 횟수
   * @param horizon 예측 호라이즌 길이 H
   * @return (H,) 벡터: 각 시간 스텝별 노이즈 스케일
   */
  Eigen::VectorXd computeAnnealingSchedule(
    int iteration, int n_diffuse, int horizon) const;

  /**
   * @brief 단일 어닐링 스텝 (샘플링 → 롤아웃 → 가중 업데이트)
   * @return 이번 반복의 평균 비용
   */
  double annealingStep(
    Eigen::MatrixXd& control_seq,
    const Eigen::VectorXd& current_state,
    const Eigen::MatrixXd& reference_trajectory,
    const Eigen::VectorXd& noise_schedule,
    int iteration);
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__DIAL_MPPI_CONTROLLER_PLUGIN_HPP_
