#include "mpc_controller_ros2/ancillary_controller.hpp"

namespace mpc_controller_ros2
{

AncillaryController::AncillaryController(const Eigen::MatrixXd& K_fb)
: K_fb_(K_fb)
{
}

AncillaryController::AncillaryController(
  double k_forward,
  double k_lateral,
  double k_angle
)
{
  K_fb_ = Eigen::MatrixXd::Zero(2, 3);
  K_fb_(0, 0) = k_forward;
  K_fb_(1, 1) = k_lateral;
  K_fb_(1, 2) = k_angle;
}

Eigen::VectorXd AncillaryController::computeBodyFrameError(
  const Eigen::VectorXd& nominal_state,
  const Eigen::VectorXd& actual_state
) const
{
  // World frame 오차 (x, y, theta는 항상 인덱스 0, 1, 2)
  double dx = nominal_state(0) - actual_state(0);
  double dy = nominal_state(1) - actual_state(1);
  double dtheta = normalizeAngle(nominal_state(2) - actual_state(2));

  // 실제 로봇의 heading 기준으로 회전 변환 (World → Body)
  double theta = actual_state(2);
  double cos_theta = std::cos(theta);
  double sin_theta = std::sin(theta);

  // Body frame 오차
  double e_forward = cos_theta * dx + sin_theta * dy;
  double e_lateral = -sin_theta * dx + cos_theta * dy;
  double e_angle = dtheta;

  Eigen::VectorXd body_error(3);
  body_error << e_forward, e_lateral, e_angle;
  return body_error;
}

Eigen::VectorXd AncillaryController::computeCorrectedControl(
  const Eigen::VectorXd& nominal_control,
  const Eigen::VectorXd& nominal_state,
  const Eigen::VectorXd& actual_state
) const
{
  Eigen::VectorXd body_error = computeBodyFrameError(nominal_state, actual_state);
  Eigen::VectorXd correction = computeFeedbackCorrection(body_error);
  return nominal_control + correction;
}

Eigen::VectorXd AncillaryController::computeFeedbackCorrection(
  const Eigen::VectorXd& body_error
) const
{
  return K_fb_ * body_error;
}

void AncillaryController::setGains(const Eigen::MatrixXd& K_fb)
{
  K_fb_ = K_fb;
}

void AncillaryController::setGains(double k_forward, double k_lateral, double k_angle)
{
  K_fb_ = Eigen::MatrixXd::Zero(2, 3);
  K_fb_(0, 0) = k_forward;
  K_fb_(1, 1) = k_lateral;
  K_fb_(1, 2) = k_angle;
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
