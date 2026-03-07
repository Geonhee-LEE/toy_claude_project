#ifndef MPC_CONTROLLER_ROS2__ILQR_SOLVER_HPP_
#define MPC_CONTROLLER_ROS2__ILQR_SOLVER_HPP_

#include <Eigen/Dense>
#include <vector>
#include "mpc_controller_ros2/motion_model.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief iLQR Solver 파라미터
 */
struct ILQRParams {
  int max_iterations{2};        // warm-start이므로 1-2회
  double regularization{1e-6};  // Q_uu 정규화 (rho)
  int line_search_steps{4};     // alpha 후보 수
  double cost_tolerance{1e-4};  // 수렴 판정
};

/**
 * @brief iLQR (iterative Linear Quadratic Regulator) Solver
 *
 * MPPI의 control_sequence_ warm-start에 사용.
 * 1-2회 반복으로 더 나은 nominal trajectory를 생성하여
 * MPPI 샘플링 효율을 높입니다.
 *
 * 수식:
 *   Backward pass: Riccati 재귀 → K_t (feedback gain), k_t (feedforward)
 *   Forward pass: line search → u_new = u_bar + alpha*k_t + K_t*(x_new - x_bar)
 *
 * 근거: Williams et al. IT-MPC (2018), Cho et al. MPPI-IPDDP (2022)
 */
class ILQRSolver
{
public:
  ILQRSolver(const ILQRParams& params, int nx, int nu);

  /**
   * @brief iLQR 풀이 (control_sequence를 in-place 갱신)
   * @param x0 초기 상태 (nx,)
   * @param control_sequence (N, nu) — in/out
   * @param reference (N+1, nx) 참조 궤적
   * @param model 동역학 모델
   * @param Q (nx, nx) 상태 추적 가중치
   * @param Qf (nx, nx) 터미널 가중치
   * @param R (nu, nu) 제어 가중치
   * @param dt 시간 간격
   * @return 최종 비용
   */
  double solve(
    const Eigen::VectorXd& x0,
    Eigen::MatrixXd& control_sequence,
    const Eigen::MatrixXd& reference,
    const MotionModel& model,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& Qf,
    const Eigen::MatrixXd& R,
    double dt);

  /** @brief 파라미터 접근 */
  const ILQRParams& params() const { return params_; }

private:
  /**
   * @brief 순방향 rollout: x0 + control_sequence → 궤적 생성
   */
  void rolloutNominal(
    const Eigen::VectorXd& x0,
    const Eigen::MatrixXd& U,
    const MotionModel& model,
    double dt,
    Eigen::MatrixXd& X_out) const;

  /**
   * @brief Backward pass: Riccati 재귀 → k_t, K_t 계산
   * @return expected cost reduction (dV)
   */
  double backwardPass(
    const Eigen::MatrixXd& X_bar,
    const Eigen::MatrixXd& U_bar,
    const Eigen::MatrixXd& ref,
    const MotionModel& model,
    double dt);

  /**
   * @brief Forward pass: line search로 비용 감소 확인
   * @return 새로운 비용 (실패 시 -1)
   */
  double forwardPass(
    const Eigen::VectorXd& x0,
    const Eigen::MatrixXd& X_bar,
    const Eigen::MatrixXd& U_bar,
    const MotionModel& model,
    double dt,
    double alpha);

  /**
   * @brief 궤적 비용 계산
   */
  double computeTrajectoryCost(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& U,
    const Eigen::MatrixXd& ref) const;

  // 사전 할당 버퍼
  std::vector<Eigen::VectorXd> k_;    // feedforward (N개, nu)
  std::vector<Eigen::MatrixXd> K_;    // feedback gain (N개, nu x nx)
  Eigen::VectorXd V_x_;               // (nx,)
  Eigen::MatrixXd V_xx_;              // (nx, nx)

  // forward pass 임시 버퍼
  Eigen::MatrixXd X_new_;             // (N+1, nx)
  Eigen::MatrixXd U_new_;             // (N, nu)

  ILQRParams params_;
  Eigen::MatrixXd Q_, Qf_, R_;
  int nx_, nu_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__ILQR_SOLVER_HPP_
