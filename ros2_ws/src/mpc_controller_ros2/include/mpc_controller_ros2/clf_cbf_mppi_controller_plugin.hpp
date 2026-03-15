#ifndef MPC_CONTROLLER_ROS2__CLF_CBF_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__CLF_CBF_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/shield_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/clf_function.hpp"
#include "mpc_controller_ros2/clf_cbf_qp_solver.hpp"
#include <memory>

namespace mpc_controller_ros2
{

/**
 * @brief CLF-CBF-MPPI Controller Plugin
 *
 * ShieldMPPI를 상속하여 CLF-CBF 통합 QP 안전 필터를 추가합니다.
 *
 * 파이프라인:
 *   1. MPPI 샘플링 → u_mppi (부모 MPPIControllerPlugin)
 *   2. Shield CBF 투영 (부모 ShieldMPPIControllerPlugin)
 *   3. CLF-CBF-QP 필터 → u_safe (이 클래스)
 *
 * CLF는 reference trajectory의 다음 스텝을 x_des로 사용하여
 * 목표 수렴을 보장하고, CBF는 안전을 보장합니다.
 * 충돌 시 slack δ로 CLF를 완화하여 안전 우선.
 *
 * 파라미터:
 *   clf_cbf_enabled: true/false
 *   clf_decay_rate: c (CLF V̇ + c·V ≤ δ)
 *   clf_slack_penalty: p (min p·δ²)
 *   clf_P_scale: P = scale · Q (상태 추적 가중치 재사용)
 */
class CLFCBFMPPIControllerPlugin : public ShieldMPPIControllerPlugin
{
public:
  CLFCBFMPPIControllerPlugin() = default;
  ~CLFCBFMPPIControllerPlugin() override = default;

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
  std::unique_ptr<CLFFunction> clf_;
  std::unique_ptr<CLFCBFQPSolver> clf_cbf_solver_;
  bool clf_cbf_enabled_{false};
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__CLF_CBF_MPPI_CONTROLLER_PLUGIN_HPP_
