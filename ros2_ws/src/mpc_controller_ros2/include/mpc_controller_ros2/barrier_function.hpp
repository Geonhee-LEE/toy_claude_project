#ifndef MPC_CONTROLLER_ROS2__BARRIER_FUNCTION_HPP_
#define MPC_CONTROLLER_ROS2__BARRIER_FUNCTION_HPP_

#include <Eigen/Dense>
#include <vector>
#include "mpc_controller_ros2/c3bf_barrier.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief CBF 합성 전략
 *
 * 다중 CBF를 하나의 합성 CBF로 결합하여 제약 수를 줄이고
 * QP 수렴을 개선합니다.
 */
enum class CBFCompositionMethod {
  MIN,           ///< h_c = min(h_1, ..., h_n) — 가장 보수적, 비미분
  SMOOTH_MIN,    ///< h_c = -1/α · log(Σ exp(-α·h_i)) — 미분 가능 근사
  LOG_SUM_EXP,   ///< h_c = -log(Σ exp(-h_i)) — smooth min 특수 경우 (α=1)
  PRODUCT        ///< h_c = Π h_i — 모든 배리어 동시 반영
};

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

  /**
   * @brief 안전 마진 동적 갱신 (Conformal Predictor 통합)
   *
   * 저장된 장애물 목록을 새로운 마진으로 재구축합니다.
   * @param new_margin 새로운 안전 마진 (m)
   */
  void updateSafetyMargin(double new_margin);

  /** @brief 현재 안전 마진 반환 */
  double safetyMargin() const { return safety_margin_; }

  /** @brief 장애물 수 */
  size_t size() const { return barriers_.size(); }

  /** @brief 장애물 목록 비어있는지 */
  bool empty() const { return barriers_.empty(); }

  const std::vector<CircleBarrier>& barriers() const { return barriers_; }

  /**
   * @brief 장애물 + 속도 설정 (C3BF용)
   * @param obstacles [x, y, radius]
   * @param velocities [vx, vy]
   */
  void setObstaclesWithVelocity(
    const std::vector<Eigen::Vector3d>& obstacles,
    const std::vector<Eigen::Vector2d>& velocities);

  /**
   * @brief 활성 C3BF barrier 반환
   * @param state 현재 상태
   * @return activation_distance 내의 C3BF barrier 포인터
   */
  std::vector<const C3BFBarrier*> getActiveC3BFBarriers(
    const Eigen::VectorXd& state) const;

  /** @brief C3BF barrier 목록 */
  const std::vector<C3BFBarrier>& c3bfBarriers() const { return c3bf_barriers_; }

  /** @brief alpha_safe setter (C3BF 콘 반각) */
  void setAlphaSafe(double alpha_safe) { alpha_safe_ = alpha_safe; }

  // =========================================================================
  // CBF 합성 메서드 (다중 CBF → 단일 합성 CBF)
  // =========================================================================

  /**
   * @brief 합성 CBF 값: h_c(x)
   *
   * 활성 barrier들을 합성하여 단일 CBF 값을 반환합니다.
   * @param state 현재 상태 (nx,)
   * @param method 합성 방법
   * @param alpha smooth-min 파라미터 (클수록 min에 가까움)
   * @return 합성 CBF 값 h_c(x)
   */
  double evaluateComposite(const Eigen::VectorXd& state,
                           CBFCompositionMethod method = CBFCompositionMethod::SMOOTH_MIN,
                           double alpha = 10.0) const;

  /**
   * @brief 합성 CBF gradient: ∇h_c(x)
   *
   * 활성 barrier들의 합성 gradient를 반환합니다.
   * @param state 현재 상태 (nx,)
   * @param method 합성 방법
   * @param alpha smooth-min 파라미터
   * @return ∇h_c(x) (nx,)
   */
  Eigen::VectorXd compositeGradient(const Eigen::VectorXd& state,
                                     CBFCompositionMethod method = CBFCompositionMethod::SMOOTH_MIN,
                                     double alpha = 10.0) const;

private:
  double robot_radius_;
  double safety_margin_;
  double activation_distance_;
  std::vector<CircleBarrier> barriers_;

  // C3BF barriers (속도 인식)
  std::vector<C3BFBarrier> c3bf_barriers_;
  double alpha_safe_{0.7854};  // π/4

  // 장애물 원본 저장 (마진 변경 시 재구축용)
  std::vector<Eigen::Vector3d> obstacles_raw_;
  std::vector<Eigen::Vector2d> obstacle_velocities_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__BARRIER_FUNCTION_HPP_
