#ifndef MPC_CONTROLLER_ROS2__BIASED_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__BIASED_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Biased-MPPI nav2 Controller Plugin
 *
 * Reference: Trevisan & Alonso-Mora (2024) "Biased-MPPI: Informing
 *            Sampling-Based Model Predictive Control by Fusing Ancillary
 *            Controllers" IEEE RA-L
 *
 * K개 샘플을 J_total개(ancillary 결정적 시퀀스) + (K-J_total)개(Gaussian)로 분할.
 * 수정 비용함수 S~ = S + lambda*log(p/q_s)에서 밀도비가 소거되므로
 * 가중치 공식은 Vanilla와 동일 -> 기존 WeightComputation 전략과 100% 호환.
 *
 * Ancillary 컨트롤러 (4종):
 *   1. Braking       — zero 제어열 (긴급 정지)
 *   2. GoToGoal      — 목표 방향 P-제어
 *   3. PathFollowing — 경로 접선 추종
 *   4. PreviousSolution — 이전 최적 시퀀스 복제
 */
class BiasedMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  BiasedMPPIControllerPlugin() = default;
  ~BiasedMPPIControllerPlugin() override = default;

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

  Eigen::MatrixXd generateBrakingSequence(int N, int nu) const;
  Eigen::MatrixXd generateGoToGoalSequence(
    const Eigen::VectorXd& state, const Eigen::MatrixXd& ref_traj,
    int N, int nu, double dt) const;
  Eigen::MatrixXd generatePathFollowingSequence(
    const Eigen::VectorXd& state, const Eigen::MatrixXd& ref_traj,
    int N, int nu, double dt) const;
  Eigen::MatrixXd generatePreviousSolutionSequence(int N, int nu) const;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__BIASED_MPPI_CONTROLLER_PLUGIN_HPP_
