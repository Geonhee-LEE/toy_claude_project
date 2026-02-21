#ifndef MPC_CONTROLLER_ROS2__CBF_SAFETY_FILTER_HPP_
#define MPC_CONTROLLER_ROS2__CBF_SAFETY_FILTER_HPP_

#include <Eigen/Dense>
#include <vector>
#include <utility>

#include "mpc_controller_ros2/barrier_function.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief CBF Safety Filter 결과 정보
 */
struct CBFFilterInfo
{
  int num_active_barriers{0};
  bool filter_applied{false};
  bool qp_success{false};
  std::vector<double> barrier_values;
  std::vector<double> constraint_margins;
};

/**
 * @brief CBF Safety Filter — Post-hoc QP
 *
 * min_u ||u - u_mppi||²
 * s.t.  ḣ_i(x,u) + γ·h_i(x) ≥ 0   ∀ active barrier i
 *       u_min ≤ u ≤ u_max
 *
 * nu=2~3 (DiffDrive/Swerve)이므로 projected gradient descent로 해결.
 * 추가 라이브러리 의존성 없음 (Eigen만 사용).
 */
class CBFSafetyFilter
{
public:
  CBFSafetyFilter(BarrierFunctionSet* barrier_set,
                  double gamma, double dt,
                  const Eigen::VectorXd& u_min,
                  const Eigen::VectorXd& u_max);

  /**
   * @brief MPPI 출력 u_mppi를 CBF 안전 필터링
   * @param state 현재 상태 (nx,)
   * @param u_mppi MPPI 출력 제어 (nu,)
   * @param dynamics 동역학 래퍼 (ḣ 계산에 사용)
   * @return {u_safe, info}
   */
  std::pair<Eigen::VectorXd, CBFFilterInfo> filter(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& u_mppi,
    const BatchDynamicsWrapper& dynamics) const;

private:
  /** @brief u_mppi가 이미 모든 CBF 제약을 만족하는지 빠른 확인 */
  bool isSafe(const Eigen::VectorXd& state,
              const Eigen::VectorXd& u,
              const std::vector<const CircleBarrier*>& active_barriers,
              const BatchDynamicsWrapper& dynamics) const;

  /** @brief 단일 상태에서 연속 동역학 f(x,u) 계산 */
  Eigen::VectorXd computeXdot(const Eigen::VectorXd& state,
                               const Eigen::VectorXd& u,
                               const BatchDynamicsWrapper& dynamics) const;

  /** @brief 제어 입력을 bounds 내로 클리핑 */
  Eigen::VectorXd clipToBounds(const Eigen::VectorXd& u) const;

  BarrierFunctionSet* barrier_set_;
  double gamma_;
  double dt_;
  Eigen::VectorXd u_min_;
  Eigen::VectorXd u_max_;

  // Projected gradient 파라미터
  static constexpr int kMaxIterations = 50;
  static constexpr double kStepSize = 0.1;
  static constexpr double kTolerance = 1e-6;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__CBF_SAFETY_FILTER_HPP_
