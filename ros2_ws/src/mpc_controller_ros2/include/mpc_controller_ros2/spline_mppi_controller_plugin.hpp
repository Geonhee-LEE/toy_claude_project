#ifndef MPC_CONTROLLER_ROS2__SPLINE_MPPI_CONTROLLER_PLUGIN_HPP_
#define MPC_CONTROLLER_ROS2__SPLINE_MPPI_CONTROLLER_PLUGIN_HPP_

#include "mpc_controller_ros2/mppi_controller_plugin.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Spline-MPPI nav2 Controller Plugin
 *
 * ICRA 2024 기반. P개 제어점(knot)에만 노이즈를 부여하고,
 * B-spline basis matrix로 N개 시점으로 보간하여 구조적으로 부드러운 제어를 생성.
 *
 * 알고리즘 플로우:
 *   1. P개 knot에 노이즈 샘플링 (K, P, nu)
 *   2. B-spline basis (N, P) 행렬로 보간
 *   3. 보간된 (K, N, nu) 제어로 rollout/cost
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
