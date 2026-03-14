#ifndef MPC_CONTROLLER_ROS2__CS_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__CS_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief CS-MPPI (Covariance Steering MPPI) nav2 Controller Plugin
 *
 * Reference: Le Cleac'h et al. (2023) "CoVO-MPC: Theoretical Analysis of
 *            Sampling-based MPC and Optimal Covariance Design" (CoRL 2023)
 *
 * Vanilla MPPI는 모든 시간 스텝에서 동일한 등방성 노이즈 ε ~ N(0, σ²I)를
 * 사용하지만, CS-MPPI는 동역학 Jacobian B_t의 Frobenius 노름(감도)을
 * 분석하여 시간 스텝별 노이즈 공분산을 적응적으로 조절합니다.
 *
 *   scale_t = clamp(||B_t||_F / mean(||B||_F), cs_scale_min, cs_scale_max)
 *   ε_t ~ N(0, scale_t · σ²I)
 *
 * 감도가 높은 스텝 → 큰 노이즈(탐색 강화), 낮은 스텝 → 작은 노이즈(정밀 제어).
 * 기존 getLinearization() 인프라(iLQR-MPPI, PR #142) 재사용.
 *
 * 성능 오버헤드: ~0.03ms (DiffDrive/Ackermann), ~0.09ms (Swerve) → < 5%
 */
class CSMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  CSMPPIControllerPlugin() = default;
  ~CSMPPIControllerPlugin() override = default;

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
   * @brief 동역학 Jacobian B_t 감도 기반 노이즈 스케일 팩터 계산
   * @param x0 현재 상태
   * @param ctrl 현재 control sequence (N x nu)
   * @return scale factors (N,) — clamp(||B_t||_F / mean, min, max)
   */
  Eigen::VectorXd computeCovarianceScaling(
    const Eigen::VectorXd& x0,
    const Eigen::MatrixXd& ctrl);

  /**
   * @brief noise_buffer_에 per-step 스케일 팩터 적용
   * @param scale_factors (N,)
   */
  void applyAdaptedNoise(const Eigen::VectorXd& scale_factors);

  // Buffers
  Eigen::VectorXd cs_scale_buffer_;   // (N,)
  Eigen::MatrixXd nominal_states_;    // (N+1, nx)
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__CS_MPPI_CONTROLLER_PLUGIN_HPP_
