#ifndef MPC_CONTROLLER_ROS2__COMPOSABLE_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__COMPOSABLE_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/halton_sampler.hpp"
#include "mpc_controller_ros2/ilqr_solver.hpp"
#include "mpc_controller_ros2/trajectory_library.hpp"
#include "mpc_controller_ros2/pi_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/feedback_gain_computer.hpp"
#include "mpc_controller_ros2/adaptive_horizon_manager.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief 활성 레이어 캐시 구조체
 *
 * configure() 시 params_ boolean 플래그를 한 번 캐싱하여
 * computeControl() 내 반복 접근을 최소화.
 */
struct ActiveLayers
{
  // Phase 0: Adaptation
  bool rh_mppi{false};            // 동적 horizon N 적응
  bool cs_mppi{false};            // 공분산 스케일링

  // Phase 1: Warm-Start
  bool ilqr{false};               // iLQR warm-start
  bool traj_library{false};       // 프리미티브 라이브러리 주입

  // Phase 2: Sampling
  bool halton{false};             // Halton 저불일치 시퀀스

  // Phase 3/5: Filter
  bool pi_mppi{false};            // ADMM QP 투영
  bool lp_filter{false};          // IIR Low-Pass 필터

  // Phase 6: Safety
  bool shield_cbf{false};         // Shield CBF 투영

  // Phase 7: Output Correction
  bool feedback{false};           // Riccati 피드백 보정
};

/**
 * @brief Composable MPPI Controller Plugin
 *
 * 35종 단일 관심사 플러그인의 기능을 파이프라인 기반으로 조합.
 * YAML에서 각 레이어의 enabled 플래그를 on/off하여 임의의 조합 가능.
 *
 * 파이프라인:
 *   Phase 0: Adaptation     ── RH-MPPI(동적 N) + CS-MPPI(공분산 스케일링)
 *   Phase 1: Warm-Start     ── iLQR solve + TrajLib primitive inject
 *   Phase 2: Sampling       ── sampler_->sampleInPlace + CS noise scaling + goal slowdown
 *   Phase 3: Pre-Filter     ── π-MPPI ADMM projection on K samples
 *   Phase 4: Core MPPI      ── Rollout → Cost → IT reg → Adaptive Temp → Weights → Update
 *   Phase 5: Post-Filter    ── LP IIR filter + π-MPPI ADMM on optimal seq
 *   Phase 6: Safety         ── Shield CBF projection (per-step)
 *   Phase 7: Output Correct ── Feedback Riccati K_0·dx
 *   Phase 8: Restore        ── RH-MPPI N 복원
 */
class ComposableMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  ComposableMPPIControllerPlugin() = default;
  ~ComposableMPPIControllerPlugin() override = default;

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

  // ──── Phase 6: Shield CBF 투영 ────
  Eigen::VectorXd projectControlCBF(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& u) const;

  Eigen::VectorXd computeXdot(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& u) const;

  // ──── Phase 5: LP 필터 ────
  void applyLowPassFilter(
    Eigen::MatrixXd& sequence,
    double alpha,
    const Eigen::VectorXd& initial) const;

  // ──── 활성 레이어 ────
  ActiveLayers active_;

  // ──── 컴포넌트 포인터 ────
  // Phase 0
  std::unique_ptr<AdaptiveHorizonManager> horizon_manager_;
  int N_max_{30};
  Eigen::VectorXd cs_scale_buffer_;
  Eigen::MatrixXd cs_nominal_states_;

  // Phase 1
  std::unique_ptr<ILQRSolver> ilqr_solver_;
  TrajectoryLibrary traj_library_;

  // Phase 3/5
  std::unique_ptr<ADMMProjector> projector_;
  Eigen::VectorXd pi_u_min_, pi_u_max_, pi_rate_max_, pi_accel_max_;

  // Phase 5: LP
  double lp_alpha_{1.0};
  Eigen::VectorXd lp_u_prev_;

  // Phase 6: Shield
  int shield_cbf_stride_{1};
  int shield_max_iterations_{10};
  double shield_step_size_{0.1};

  // Phase 7: Feedback
  std::unique_ptr<FeedbackGainComputer> gain_computer_;
  std::vector<Eigen::MatrixXd> cached_gains_;
  Eigen::MatrixXd cached_nominal_trajectory_;
  int cycle_counter_{0};

  // ──── CS-MPPI 유틸 ────
  Eigen::VectorXd computeCovarianceScaling(
    const Eigen::VectorXd& x0,
    const Eigen::MatrixXd& ctrl);
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__COMPOSABLE_MPPI_CONTROLLER_PLUGIN_HPP_
