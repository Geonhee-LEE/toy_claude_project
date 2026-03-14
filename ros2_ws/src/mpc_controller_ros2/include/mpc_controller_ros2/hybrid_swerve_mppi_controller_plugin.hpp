#ifndef MPC_CONTROLLER_ROS2__HYBRID_SWERVE_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__HYBRID_SWERVE_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/wheel_level_4d_model.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief MPPI-H (Hybrid Swerve MPPI) nav2 Controller Plugin
 *
 * Reference: MizuhoAOKI/mppi_swerve_drive_ros (IROS 2024, arXiv:2409.08648)
 * "Switching Sampling Space of MPPI to Balance Efficiency and Safety
 *  in 4WIDS Vehicle Navigation"
 *
 * 핵심: MPPI 샘플링 공간을 Low-D(body velocity)↔4D(대각 바퀴) 실시간 전환하여
 * 효율성(빠른 주행)과 안전성(높은 성공률) 균형 달성.
 *
 * 논문 결과 (100 에피소드, Cylinder Garden):
 *   MPPI-3D(a): 성공률 76%, 시간 36.4s
 *   MPPI-4D:    성공률 100%, 시간 38.4s
 *   MPPI-H:     성공률 99%, 시간 31.2s ← 최고 균형
 *
 * 지원 구성:
 *   A. Coaxial Swerve   (Low-D: SwerveDriveModel nx=3,nu=3)
 *   B. Non-Coaxial Swerve (Low-D: NonCoaxialSwerveModel nx=4,nu=3)
 *   공통 4D: WheelLevel4DModel (nx=3, nu=4)
 */
class HybridSwerveMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  enum class Mode { LOW_D, FOUR_D };

  HybridSwerveMPPIControllerPlugin() = default;
  ~HybridSwerveMPPIControllerPlugin() override = default;

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
   * @brief 추적 오차 기반 모드 결정
   * @param current_state 현재 상태
   * @param reference_trajectory 참조 궤적
   * @return 목표 모드
   */
  Mode determineMode(
    const Eigen::VectorXd& current_state,
    const Eigen::MatrixXd& reference_trajectory) const;

  /**
   * @brief Low-D → 4D 제어열 변환 (IK warm-start)
   */
  void convertLowTo4D();

  /**
   * @brief 4D → Low-D 제어열 변환 (FK warm-start)
   */
  void convert4DToLow();

  /**
   * @brief 4D 모드 MPPI 파이프라인 실행
   */
  std::pair<Eigen::VectorXd, MPPIInfo> computeControl4D(
    const Eigen::VectorXd& current_state,
    const Eigen::MatrixXd& reference_trajectory);

  // ─── 4D 모드 전용 멤버 ───
  std::unique_ptr<BatchDynamicsWrapper> dynamics_4d_;
  std::unique_ptr<BaseSampler> sampler_4d_;
  std::unique_ptr<CompositeMPPICost> cost_function_4d_;
  Eigen::MatrixXd control_seq_4d_;         // N x 4
  std::vector<Eigen::MatrixXd> noise_buf_4d_;       // K x (N x 4)
  std::vector<Eigen::MatrixXd> perturbed_buf_4d_;   // K x (N x 4)
  std::vector<Eigen::MatrixXd> traj_buf_4d_;         // K x (N+1 x 3)

  // ─── 전환 상태 ───
  Mode current_mode_{Mode::LOW_D};
  int mode_switch_counter_{0};
  bool is_non_coaxial_{false};

  // ─── Non-Coaxial δ 관리 ───
  double tracked_delta_{0.0};         // 4D 모드 중 추적된 δ_avg
  Eigen::Vector4d last_ctrl_4d_;      // 마지막 4D 제어 (δfl/δrr 참조용)
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__HYBRID_SWERVE_MPPI_CONTROLLER_PLUGIN_HPP_
