#ifndef MPC_CONTROLLER_ROS2__SPLINE_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__SPLINE_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Spline-MPPI nav2 Controller Plugin
 *
 * Reference: Yamada et al. (2024) "Spline-Based Model Predictive Path
 *            Integral Control" — ICRA 2024
 *
 * 핵심 수식 (Parameterized Sampling via B-spline):
 *   U_k = B · C_k,  B ∈ R^{N×P}, C_k ∈ R^{P×nu}   — B-spline 보간
 *   C_k = C + ε_k,  ε_k ~ N(0, Σ)                   — Knot perturbation
 *   C* ← C + Σ_k w_k · ε_k                           — Knot-space 업데이트
 *
 * B-spline basis (Cox-de Boor recursion):
 *   N_{i,0}(t) = { 1 if t_i ≤ t < t_{i+1}, 0 otherwise }
 *   N_{i,k}(t) = (t-t_i)/(t_{i+k}-t_i)·N_{i,k-1}(t)
 *              + (t_{i+k+1}-t)/(t_{i+k+1}-t_{i+1})·N_{i+1,k-1}(t)
 *
 * 알고리즘 플로우:
 *   1. P개 knot에 노이즈 샘플링 (K, P, nu)
 *   2. B-spline basis (N, P) 행렬로 보간 → (K, N, nu)
 *   3. 보간된 제어로 rollout/cost
 *   4. Knot space에서 가중 평균 업데이트
 */
class SplineMPPIControllerPlugin : public MPPIControllerPlugin
{
public:
  SplineMPPIControllerPlugin() = default;
  ~SplineMPPIControllerPlugin() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros
  ) override;

  /**
   * @brief B-spline basis matrix 계산
   *
   * de Boor 재귀 알고리즘, clamped uniform knot vector.
   * 행 정규화 (partition of unity).
   *
   * @param N 보간 시점 수 (출력 행 수)
   * @param P 제어점 수 (출력 열 수)
   * @param degree B-spline 차수 (기본 3 = cubic)
   * @return (N, P) basis matrix
   */
  static Eigen::MatrixXd computeBSplineBasis(int N, int P, int degree);

protected:
  std::pair<Eigen::Vector2d, MPPIInfo> computeControl(
    const Eigen::Vector3d& current_state,
    const Eigen::MatrixXd& reference_trajectory
  ) override;

private:
  Eigen::MatrixXd basis_;       // (N, P) 사전 계산
  Eigen::MatrixXd u_knots_;     // (P, 2) knot warm-start
  int P_;                       // 제어점 수
  int degree_;                  // B-spline 차수
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__SPLINE_MPPI_CONTROLLER_PLUGIN_HPP_
