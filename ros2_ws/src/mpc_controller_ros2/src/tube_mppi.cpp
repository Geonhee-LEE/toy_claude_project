#include "mpc_controller_ros2/tube_mppi.hpp"
#include <algorithm>

namespace mpc_controller_ros2
{

TubeMPPI::TubeMPPI(const MPPIParams& params)
: ancillary_(params.k_forward, params.k_lateral, params.k_angle),
  tube_width_(params.tube_width),
  max_tube_width_(params.tube_width * 2.0),
  min_tube_width_(params.tube_width * 0.2),
  tube_adaptation_rate_(0.1)
{
}

std::pair<Eigen::VectorXd, TubeMPPIInfo> TubeMPPI::computeCorrectedControl(
  const Eigen::VectorXd& nominal_control,
  const Eigen::MatrixXd& nominal_trajectory,
  const Eigen::VectorXd& actual_state
)
{
  TubeMPPIInfo info;
  info.nominal_control = nominal_control;
  info.tube_width = tube_width_;

  // Nominal 상태 = 궤적의 첫 번째 점 (현재 시점)
  if (nominal_trajectory.rows() > 0) {
    info.nominal_state = nominal_trajectory.row(0).transpose();
  } else {
    info.nominal_state = actual_state;  // 폴백
  }

  // Body frame 오차 계산
  info.body_error = ancillary_.computeBodyFrameError(info.nominal_state, actual_state);

  // 피드백 보정량 계산
  info.feedback_correction = ancillary_.computeFeedbackCorrection(info.body_error);

  // 최종 제어 입력 = nominal + feedback
  info.applied_control = nominal_control + info.feedback_correction;

  // Tube 경계점 계산 (시각화용)
  auto boundary_pairs = computeTubeBoundary(nominal_trajectory);
  for (const auto& pair : boundary_pairs) {
    info.tube_boundary.push_back(pair.first);   // 좌측
    info.tube_boundary.push_back(pair.second);  // 우측
  }

  return {info.applied_control, info};
}

void TubeMPPI::updateTubeWidth(double tracking_error)
{
  // 적응형 tube 폭 조정
  // 오차가 크면 tube 확장, 작으면 축소
  double target_width = tracking_error * 2.0;  // 휴리스틱

  tube_width_ += tube_adaptation_rate_ * (target_width - tube_width_);
  tube_width_ = std::clamp(tube_width_, min_tube_width_, max_tube_width_);
}

std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> TubeMPPI::computeTubeBoundary(
  const Eigen::MatrixXd& nominal_trajectory
) const
{
  std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> boundaries;

  int n_points = nominal_trajectory.rows();
  if (n_points < 2) {
    return boundaries;
  }

  // 각 waypoint에서 tube 경계점 계산
  for (int i = 0; i < n_points; ++i) {
    double x = nominal_trajectory(i, 0);
    double y = nominal_trajectory(i, 1);
    double theta = (nominal_trajectory.cols() >= 3) ? nominal_trajectory(i, 2) : 0.0;

    double perpendicular_x = -std::sin(theta);
    double perpendicular_y = std::cos(theta);

    Eigen::VectorXd left_point(3);
    left_point(0) = x + tube_width_ * perpendicular_x;
    left_point(1) = y + tube_width_ * perpendicular_y;
    left_point(2) = theta;

    Eigen::VectorXd right_point(3);
    right_point(0) = x - tube_width_ * perpendicular_x;
    right_point(1) = y - tube_width_ * perpendicular_y;
    right_point(2) = theta;

    boundaries.push_back({left_point, right_point});
  }

  return boundaries;
}

bool TubeMPPI::isInsideTube(
  const Eigen::VectorXd& nominal_state,
  const Eigen::VectorXd& actual_state
) const
{
  double dx = actual_state(0) - nominal_state(0);
  double dy = actual_state(1) - nominal_state(1);
  double position_error = std::sqrt(dx * dx + dy * dy);

  return position_error <= tube_width_;
}

void TubeMPPI::setFeedbackGains(double k_forward, double k_lateral, double k_angle)
{
  ancillary_.setGains(k_forward, k_lateral, k_angle);
}

void TubeMPPI::setTubeWidth(double width)
{
  tube_width_ = std::clamp(width, min_tube_width_, max_tube_width_);
}

void TubeMPPI::updateParams(const MPPIParams& params)
{
  ancillary_.setGains(params.k_forward, params.k_lateral, params.k_angle);
  tube_width_ = params.tube_width;
  max_tube_width_ = params.tube_width * 2.0;
  min_tube_width_ = params.tube_width * 0.2;
}

}  // namespace mpc_controller_ros2
