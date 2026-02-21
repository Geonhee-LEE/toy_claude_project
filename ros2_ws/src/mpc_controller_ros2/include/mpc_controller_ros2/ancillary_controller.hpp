#ifndef MPC_CONTROLLER_ROS2__ANCILLARY_CONTROLLER_HPP_
#define MPC_CONTROLLER_ROS2__ANCILLARY_CONTROLLER_HPP_

#include <Eigen/Dense>
#include <cmath>

namespace mpc_controller_ros2
{

/**
 * @brief Tube-MPPI용 Body Frame 피드백 보정 컨트롤러
 *
 * 실제 상태와 nominal (외란 없는) 상태 간의 오차를 body frame에서 계산하고,
 * 선형 피드백을 통해 제어 입력을 보정합니다.
 *
 * 수식:
 *   u_applied = u_nominal + K_fb · e_body
 *
 * 여기서:
 *   e_body = [e_forward, e_lateral, e_angle]^T
 *   K_fb = [k_forward   0          0     ]  (2x3 행렬)
 *          [0           k_lateral  k_angle]
 *
 * 참조:
 *   - Tube MPC 논문: "Robust Model Predictive Control with Tubes"
 *   - Python M2 구현: mpc_controller/controllers/mppi/ancillary_controller.py
 */
class AncillaryController
{
public:
  /**
   * @brief 생성자
   * @param K_fb 피드백 게인 매트릭스 (nu x nx)
   */
  explicit AncillaryController(const Eigen::MatrixXd& K_fb);

  /**
   * @brief 기본 게인으로 생성 (DiffDrive 2x3)
   * @param k_forward 전진 오차 게인
   * @param k_lateral 측면 오차 게인
   * @param k_angle 각도 오차 게인
   */
  AncillaryController(
    double k_forward = 0.8,
    double k_lateral = 0.5,
    double k_angle = 1.0
  );

  /**
   * @brief Body frame 오차 계산
   * @param nominal_state Nominal 상태 (nx,)
   * @param actual_state 실제 상태 (nx,)
   * @return Body frame 오차 [e_forward, e_lateral, e_angle]
   */
  Eigen::VectorXd computeBodyFrameError(
    const Eigen::VectorXd& nominal_state,
    const Eigen::VectorXd& actual_state
  ) const;

  /**
   * @brief 피드백 보정 제어 입력 계산
   * @param nominal_control Nominal 제어 입력 (nu,)
   * @param nominal_state Nominal 상태 (nx,)
   * @param actual_state 실제 상태 (nx,)
   * @return 보정된 제어 입력 (nu,)
   */
  Eigen::VectorXd computeCorrectedControl(
    const Eigen::VectorXd& nominal_control,
    const Eigen::VectorXd& nominal_state,
    const Eigen::VectorXd& actual_state
  ) const;

  /**
   * @brief 피드백 게인 업데이트
   */
  void setGains(const Eigen::MatrixXd& K_fb);
  void setGains(double k_forward, double k_lateral, double k_angle);

  /**
   * @brief 현재 게인 반환
   */
  Eigen::MatrixXd getGains() const { return K_fb_; }

  /**
   * @brief 피드백 보정량만 계산 (디버깅용)
   */
  Eigen::VectorXd computeFeedbackCorrection(
    const Eigen::VectorXd& body_error
  ) const;

  /**
   * @brief 각도 정규화 (-π ~ π)
   */
  static double normalizeAngle(double angle);

private:
  Eigen::MatrixXd K_fb_;  // 피드백 게인 매트릭스 (nu x nx)
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__ANCILLARY_CONTROLLER_HPP_
