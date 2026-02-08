#include "mpc_controller_ros2/ancillary_controller.hpp"

namespace mpc_controller_ros2
{

AncillaryController::AncillaryController(const Eigen::Matrix<double, 2, 3>& K_fb)
: K_fb_(K_fb)
{
}

AncillaryController::AncillaryController(
  double k_forward,
  double k_lateral,
  double k_angle
)
{
  K_fb_ << k_forward, 0.0,       0.0,
           0.0,       k_lateral, k_angle;
}

Eigen::Vector3d AncillaryController::computeBodyFrameError(
  const Eigen::Vector3d& nominal_state,
  const Eigen::Vector3d& actual_state
) const
{
  // World frame 오차
  double dx = nominal_state(0) - actual_state(0);
  double dy = nominal_state(1) - actual_state(1);
  double dtheta = normalizeAngle(nominal_state(2) - actual_state(2));

  // 실제 로봇의 heading 기준으로 회전 변환 (World → Body)
  double theta = actual_state(2);
  double cos_theta = std::cos(theta);
  double sin_theta = std::sin(theta);

  // Body frame 오차
  // e_forward: 로봇 전진 방향 오차 (양수 = nominal이 앞에 있음)
  // e_lateral: 로봇 측면 방향 오차 (양수 = nominal이 왼쪽에 있음)
  double e_forward = cos_theta * dx + sin_theta * dy;
  double e_lateral = -sin_theta * dx + cos_theta * dy;
  double e_angle = dtheta;

  return Eigen::Vector3d(e_forward, e_lateral, e_angle);
}

Eigen::Vector2d AncillaryController::computeCorrectedControl(
  const Eigen::Vector2d& nominal_control,
  const Eigen::Vector3d& nominal_state,
  const Eigen::Vector3d& actual_state
) const
{
  // Body frame 오차 계산
  Eigen::Vector3d body_error = computeBodyFrameError(nominal_state, actual_state);

  // 피드백 보정
  Eigen::Vector2d correction = computeFeedbackCorrection(body_error);

  // 최종 제어 입력 = nominal + feedback
  return nominal_control + correction;
}

Eigen::Vector2d AncillaryController::computeFeedbackCorrection(
  const Eigen::Vector3d& body_error
) const
{
  // u_correction = K_fb * e_body
  // [dv]   = [k_forward   0          0     ] [e_forward]
  // [dω]     [0           k_lateral  k_angle] [e_lateral]
  //                                           [e_angle  ]
  return K_fb_ * body_error;
}

void AncillaryController::setGains(const Eigen::Matrix<double, 2, 3>& K_fb)
{
  K_fb_ = K_fb;
}

void AncillaryController::setGains(double k_forward, double k_lateral, double k_angle)
{
  K_fb_ << k_forward, 0.0,       0.0,
           0.0,       k_lateral, k_angle;
}

double AncillaryController::normalizeAngle(double angle)
{
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

}  // namespace mpc_controller_ros2
