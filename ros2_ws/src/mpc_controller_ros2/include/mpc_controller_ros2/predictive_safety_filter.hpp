#ifndef MPC_CONTROLLER_ROS2__PREDICTIVE_SAFETY_FILTER_HPP_
#define MPC_CONTROLLER_ROS2__PREDICTIVE_SAFETY_FILTER_HPP_

#include <Eigen/Dense>
#include <vector>
#include "mpc_controller_ros2/barrier_function.hpp"
#include "mpc_controller_ros2/motion_model.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief N-step 예측 안전 필터 결과
 */
struct PredictiveSafetyResult {
  Eigen::MatrixXd u_safe_sequence;   // (N x nu) 안전 제어 시퀀스
  Eigen::MatrixXd safe_trajectory;   // ((N+1) x nx) 안전 궤적
  bool feasible{false};
  int num_corrected_steps{0};        // CBF 투영이 적용된 스텝 수
  std::vector<double> min_barrier_values;  // 각 스텝의 최소 barrier 값
};

/**
 * @brief N-step Predictive Safety Filter (MPC-CBF)
 *
 * 전체 제어 시퀀스를 forward rollout하면서 각 스텝에서
 * CBF 제약 위반 시 투영 보정. 이전 스텝의 보정이 이후 스텝에
 * 전파되어 recursive feasibility를 보장합니다.
 *
 * ShieldMPPI와의 차이:
 *   - ShieldMPPI: 처음 몇 스텝만 투영 (stride 기반)
 *   - PredictiveSafety: 전체 horizon 투영 → recursive feasibility 보장
 *
 * 알고리즘:
 *   for t = 0..N-1:
 *     γ_t = gamma * decay^t
 *     for each active barrier:
 *       if ḣ + γ_t·h < 0:
 *         u_t ← projected gradient step
 *     x_{t+1} = f(x_t, u_t)
 */
class PredictiveSafetyFilter {
public:
  /**
   * @brief 생성자
   * @param barrier_set 장애물 barrier 집합 (비소유)
   * @param gamma CBF class-K 함수 계수
   * @param dt 시간 간격 (초)
   * @param u_min 제어 입력 하한 (nu,)
   * @param u_max 제어 입력 상한 (nu,)
   */
  PredictiveSafetyFilter(BarrierFunctionSet* barrier_set,
                         double gamma, double dt,
                         const Eigen::VectorXd& u_min,
                         const Eigen::VectorXd& u_max);

  /**
   * @brief N-step 예측 안전 필터
   *
   * 전체 제어 시퀀스를 forward rollout하면서
   * 각 스텝에서 CBF 제약 위반 시 투영 보정.
   *
   * @param x0 초기 상태 (nx,)
   * @param control_sequence 제어 시퀀스 (N x nu)
   * @param model 동역학 모델
   * @return 안전 제어 시퀀스 + 궤적
   */
  PredictiveSafetyResult filter(
    const Eigen::VectorXd& x0,
    const Eigen::MatrixXd& control_sequence,
    const MotionModel& model) const;

  /**
   * @brief 궤적 안전성 검증 (투영 없이 확인만)
   * @return 모든 스텝에서 CBF 만족 여부
   */
  bool verifyTrajectory(
    const Eigen::VectorXd& x0,
    const Eigen::MatrixXd& control_sequence,
    const MotionModel& model) const;

  /** @brief 최대 투영 반복 설정 */
  void setMaxIterations(int max_iter) { max_iterations_ = max_iter; }

  /** @brief 감쇠 감마 설정 (시간에 따른 CBF 강도 조절) */
  void setHorizonDecay(double decay) { horizon_decay_ = decay; }

  /** @brief 현재 max_iterations 반환 */
  int maxIterations() const { return max_iterations_; }

  /** @brief 현재 horizon_decay 반환 */
  double horizonDecay() const { return horizon_decay_; }

private:
  /**
   * @brief 단일 스텝 CBF 투영 (ShieldMPPI의 projectControlCBF와 유사)
   * @param state 현재 상태 (nx,)
   * @param u 제어 입력 (nu,)
   * @param model 동역학 모델
   * @param gamma_t 시간 감쇠된 gamma
   * @return 투영된 제어 (nu,)
   */
  Eigen::VectorXd projectStep(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& u,
    const MotionModel& model,
    double gamma_t) const;

  /** @brief 단일 상태 동역학: f(x, u) -> x_dot */
  Eigen::VectorXd computeXdot(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& u,
    const MotionModel& model) const;

  /** @brief 제어 입력 클리핑 */
  Eigen::VectorXd clipControl(const Eigen::VectorXd& u) const;

  BarrierFunctionSet* barrier_set_;
  double gamma_;
  double dt_;
  Eigen::VectorXd u_min_;
  Eigen::VectorXd u_max_;
  int max_iterations_{10};
  double horizon_decay_{1.0};  // gamma_t = gamma * decay^t (1.0 = 균일)
  double step_size_{0.1};      // projected gradient step size
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__PREDICTIVE_SAFETY_FILTER_HPP_
