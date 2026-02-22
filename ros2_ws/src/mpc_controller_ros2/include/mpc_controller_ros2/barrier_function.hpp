#ifndef MPC_CONTROLLER_ROS2__BARRIER_FUNCTION_HPP_
#define MPC_CONTROLLER_ROS2__BARRIER_FUNCTION_HPP_

#include <Eigen/Dense>
#include <vector>

namespace mpc_controller_ros2
{

/**
 * @brief 원형 장애물 CBF (Control Barrier Function)
 *
 * h(x) = ||p - p_obs||² - d_safe²
 * d_safe = obstacle_radius + robot_radius + safety_margin
 *
 * h > 0: 안전 (장애물 밖)
 * h = 0: 경계 (barrier 위)
 * h < 0: 위험 (장애물 내부)
 */
class CircleBarrier
{
public:
  CircleBarrier(double obs_x, double obs_y, double obs_radius,
                double robot_radius, double safety_margin);

  /** @brief h(x) 평가 — 단일 상태 */
  double evaluate(const Eigen::VectorXd& state) const;

  /** @brief h(x) 배치 평가 — states (M x nx) → (M,) */
  Eigen::VectorXd evaluateBatch(const Eigen::MatrixXd& states) const;

  /** @brief ∇h(x) — (nx,) gradient */
  Eigen::VectorXd gradient(const Eigen::VectorXd& state) const;

  double obsX() const { return obs_x_; }
  double obsY() const { return obs_y_; }
  double safeDistance() const { return d_safe_; }

private:
  double obs_x_;
  double obs_y_;
  double d_safe_;
  double d_safe_sq_;
};

/**
 * @brief 다중 장애물 CBF 관리
 *
 * activation_distance 내의 장애물만 활성화하여 QP 제약 수 최소화
 */
class BarrierFunctionSet
{
public:
  explicit BarrierFunctionSet(double robot_radius = 0.2,
                              double safety_margin = 0.3,
                              double activation_distance = 3.0);

  /** @brief 장애물 목록 설정 (Vector3d = [x, y, radius]) */
  void setObstacles(const std::vector<Eigen::Vector3d>& obstacles);

  /** @brief 현재 상태에서 활성 barrier 목록 반환 */
  std::vector<const CircleBarrier*> getActiveBarriers(
    const Eigen::VectorXd& state) const;

  /** @brief 모든 barrier의 h(x) 값 반환 */
  Eigen::VectorXd evaluateAll(const Eigen::VectorXd& state) const;

  /** @brief 장애물 수 */
  size_t size() const { return barriers_.size(); }

  /** @brief 장애물 목록 비어있는지 */
  bool empty() const { return barriers_.empty(); }

  const std::vector<CircleBarrier>& barriers() const { return barriers_; }

private:
  double robot_radius_;
  double safety_margin_;
  double activation_distance_;
  std::vector<CircleBarrier> barriers_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__BARRIER_FUNCTION_HPP_
