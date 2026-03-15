#ifndef MPC_CONTROLLER_ROS2__CLF_FUNCTION_HPP_
#define MPC_CONTROLLER_ROS2__CLF_FUNCTION_HPP_

#include <Eigen/Dense>
#include <vector>

namespace mpc_controller_ros2
{

/**
 * @brief Control Lyapunov Function (CLF)
 *
 * V(x) = (x - x_des)^T P (x - x_des)
 *
 * CLF 조건: V̇(x,u) + c·V(x) ≤ δ  (δ = slack, 안전과 충돌 시 완화)
 *
 * Lie derivative 분해 (control-affine 동역학 ẋ = f(x) + g(x)u):
 *   V̇ = ∇V · ẋ = L_f V + L_g V · u
 *   L_f V = ∇V · f(x)       (drift term)
 *   L_g V = ∇V · g(x)       (control term, 1 x nu)
 *
 * QP에서 CLF 제약:
 *   L_f V + L_g V · u + c · V ≤ δ
 */
class CLFFunction
{
public:
  /**
   * @brief CLF 생성
   * @param P 양의 정부호 행렬 (nx × nx) — Lyapunov 가중치
   * @param c CLF decay rate (c > 0)
   * @param angle_indices 각도 상태 인덱스 (wrapping 적용)
   */
  CLFFunction(const Eigen::MatrixXd& P, double c,
              const std::vector<int>& angle_indices = {});

  /** @brief V(x) = (x - x_des)^T P (x - x_des) */
  double evaluate(const Eigen::VectorXd& state,
                  const Eigen::VectorXd& x_des) const;

  /** @brief ∇V = 2 P (x - x_des), shape (nx,) */
  Eigen::VectorXd gradient(const Eigen::VectorXd& state,
                            const Eigen::VectorXd& x_des) const;

  /**
   * @brief Lie derivative 분해: V̇ = L_f V + L_g V · u
   * @param state 현재 상태 (nx,)
   * @param x_des 목표 상태 (nx,)
   * @param A 연속 시간 ∂f/∂x (nx × nx) 또는 이산 시간 Jacobian
   * @param B 연속 시간 ∂f/∂u (nx × nu) 또는 이산 시간 Jacobian
   * @param x_dot 현재 상태에서 동역학 f(x,u) (nx,)
   * @return {L_f_V, L_g_V} — L_f_V: scalar, L_g_V: (nu,)
   */
  std::pair<double, Eigen::VectorXd> lieDerivatives(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& x_des,
    const Eigen::VectorXd& x_dot,
    const Eigen::MatrixXd& B) const;

  /** @brief CLF decay rate */
  double c() const { return c_; }

  /** @brief P 행렬 */
  const Eigen::MatrixXd& P() const { return P_; }

  /** @brief 상태 차원 */
  int stateDim() const { return nx_; }

private:
  /** @brief angle wrapping된 상태 오차 */
  Eigen::VectorXd stateError(const Eigen::VectorXd& state,
                              const Eigen::VectorXd& x_des) const;

  Eigen::MatrixXd P_;
  double c_;
  int nx_;
  std::vector<int> angle_indices_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__CLF_FUNCTION_HPP_
