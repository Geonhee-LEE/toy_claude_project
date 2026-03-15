#ifndef MPC_CONTROLLER_ROS2__CLF_CBF_QP_SOLVER_HPP_
#define MPC_CONTROLLER_ROS2__CLF_CBF_QP_SOLVER_HPP_

#include <Eigen/Dense>
#include <vector>

#include "mpc_controller_ros2/clf_function.hpp"
#include "mpc_controller_ros2/barrier_function.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief CLF-CBF-QP 솔버 결과
 */
struct CLFCBFQPResult
{
  Eigen::VectorXd u_safe;       // QP 최적해 (nu,)
  double slack{0.0};            // CLF 완화 slack δ
  bool feasible{false};         // QP 실현 가능 여부
  int iterations{0};            // 반복 횟수
  double clf_value{0.0};        // V(x)
  double clf_constraint{0.0};   // L_f V + L_g V·u + c·V - δ
  std::vector<double> cbf_margins;  // 각 CBF 제약 마진
};

/**
 * @brief CLF-CBF-QP 통합 솔버
 *
 * min_u  (1/2)||u - u_ref||² + p·δ²
 * s.t.   L_f V + L_g V·u + c·V ≤ δ          (CLF, relaxed)
 *        L_f h_i + L_g h_i·u + γ·h_i ≥ 0    (CBF, hard)
 *        u_min ≤ u ≤ u_max
 *
 * nu=2~3의 소규모 QP → Projected gradient descent (Eigen only)
 *
 * 핵심: CLF는 slack δ로 완화되어 CBF(안전)과 충돌 시 안전 우선.
 * Ames et al., "Control Barrier Functions: Theory and Applications" (2019)
 */
class CLFCBFQPSolver
{
public:
  /**
   * @brief 솔버 생성
   * @param clf CLF 함수 (비소유)
   * @param barrier_set 장애물 barrier 집합 (비소유)
   * @param gamma CBF class-K 함수 계수
   * @param slack_penalty CLF slack 페널티 p (클수록 CLF 강제)
   * @param u_min 최소 제어 (nu,)
   * @param u_max 최대 제어 (nu,)
   */
  CLFCBFQPSolver(const CLFFunction* clf,
                 BarrierFunctionSet* barrier_set,
                 double gamma,
                 double slack_penalty,
                 const Eigen::VectorXd& u_min,
                 const Eigen::VectorXd& u_max);

  /**
   * @brief CLF-CBF-QP 해결
   * @param state 현재 상태 (nx,)
   * @param x_des 목표 상태 (nx,)
   * @param u_ref MPPI 참조 제어 (nu,)
   * @param dynamics 동역학 래퍼
   * @return QP 결과
   */
  CLFCBFQPResult solve(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& x_des,
    const Eigen::VectorXd& u_ref,
    const BatchDynamicsWrapper& dynamics) const;

  /**
   * @brief CLF만으로 QP 해결 (CBF 제약 없음)
   */
  CLFCBFQPResult solveCLFOnly(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& x_des,
    const Eigen::VectorXd& u_ref,
    const BatchDynamicsWrapper& dynamics) const;

  /** @brief slack penalty 설정 */
  void setSlackPenalty(double p) { slack_penalty_ = p; }

  /** @brief gamma 설정 */
  void setGamma(double gamma) { gamma_ = gamma; }

private:
  /** @brief 단일 상태 동역학 f(x,u) */
  Eigen::VectorXd computeXdot(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& u,
    const BatchDynamicsWrapper& dynamics) const;

  /** @brief 수치적 B 행렬 계산 (∂f/∂u) */
  Eigen::MatrixXd computeB(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& x_dot,
    const BatchDynamicsWrapper& dynamics) const;

  /** @brief 제어 클리핑 */
  Eigen::VectorXd clipToBounds(const Eigen::VectorXd& u) const;

  const CLFFunction* clf_;
  BarrierFunctionSet* barrier_set_;
  double gamma_;
  double slack_penalty_;
  Eigen::VectorXd u_min_;
  Eigen::VectorXd u_max_;

  // Solver 파라미터
  static constexpr int kMaxIterations = 100;
  static constexpr double kStepSize = 0.05;
  static constexpr double kTolerance = 1e-6;
  static constexpr double kFiniteDiffDelta = 1e-4;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__CLF_CBF_QP_SOLVER_HPP_
