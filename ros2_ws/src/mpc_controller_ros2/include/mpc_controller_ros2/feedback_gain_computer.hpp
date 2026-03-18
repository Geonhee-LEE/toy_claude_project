#ifndef MPC_CONTROLLER_ROS2__FEEDBACK_GAIN_COMPUTER_HPP_
#define MPC_CONTROLLER_ROS2__FEEDBACK_GAIN_COMPUTER_HPP_

#include <Eigen/Dense>
#include <vector>
#include "mpc_controller_ros2/motion_model.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Feedback-MPPI 용 시변 피드백 게인 계산기
 *
 * RA-L 2026 기반: MPPI 롤아웃 궤적에서 Riccati 역방향 패스를 수행하여
 * 시변 피드백 게인 K_t를 계산. 사이클 간 선형 피드백 보정에 사용.
 *
 * 수식:
 *   V_xx[N] = Qf
 *   Q_xx = Q + A^T V_xx A,  Q_ux = B^T V_xx A,  Q_uu = R + B^T V_xx B
 *   K_t = -Q_uu^{-1} Q_ux
 *   V_xx = Q_xx - Q_ux^T K_t
 *
 * iLQRSolver와 분리된 이유:
 *   - iLQR: 최적화 (반복 solve + forward pass + line search)
 *   - FeedbackGainComputer: 1회 backward pass만 (게인 추출 전용)
 */
class FeedbackGainComputer
{
public:
  /**
   * @param nx 상태 차원
   * @param nu 제어 차원
   * @param regularization Q_uu 정규화 (수치 안정성)
   */
  FeedbackGainComputer(int nx, int nu, double regularization = 1e-4);

  /**
   * @brief 시변 피드백 게인 K_t 계산 (Riccati backward pass)
   *
   * @param nominal_trajectory (N+1, nx) 공칭 궤적
   * @param control_sequence (N, nu) 공칭 제어
   * @param model 동역학 모델 (선형화용)
   * @param Q (nx, nx) 상태 추적 가중치
   * @param Qf (nx, nx) 터미널 가중치
   * @param R (nu, nu) 제어 가중치
   * @param dt 시간 간격
   * @return K_t 벡터 [N개, 각각 (nu x nx)]
   */
  const std::vector<Eigen::MatrixXd>& computeGains(
    const Eigen::MatrixXd& nominal_trajectory,
    const Eigen::MatrixXd& control_sequence,
    const MotionModel& model,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& Qf,
    const Eigen::MatrixXd& R,
    double dt);

  /** @brief 마지막 계산된 게인 접근 */
  const std::vector<Eigen::MatrixXd>& gains() const { return K_; }

  /** @brief 정규화 계수 설정 */
  void setRegularization(double reg) { regularization_ = reg; }

private:
  std::vector<Eigen::MatrixXd> K_;  // (N개, nu x nx)
  Eigen::MatrixXd V_xx_;            // (nx, nx)
  int nx_, nu_;
  double regularization_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__FEEDBACK_GAIN_COMPUTER_HPP_
