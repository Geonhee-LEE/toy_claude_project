// =============================================================================
// WheelLevel4DModel — 4D 바퀴 레벨 Swerve 동역학 모델
//
// MPPI-H (IROS 2024, arXiv:2409.08648) 논문의 4D 샘플링 공간.
// 대각 바퀴 쌍 (FL, RR)의 속도와 조향각을 직접 제어.
//
// nx=3 [x,y,θ], nu=4 [V_fl, V_rr, δ_fl, δ_rr]
// =============================================================================

#include "mpc_controller_ros2/wheel_level_4d_model.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <cmath>

namespace mpc_controller_ros2
{

WheelLevel4DModel::WheelLevel4DModel(
  double lf, double lr, double dl, double dr,
  double v_wheel_max, double delta_max)
: lf_(lf), lr_(lr), dl_(dl), dr_(dr),
  v_wheel_max_(v_wheel_max), delta_max_(delta_max)
{
}

// =============================================================================
// Static FK: 4D → body velocity
// =============================================================================

Eigen::Vector3d WheelLevel4DModel::forwardKinematics(
  const Eigen::Vector4d& u4d, double dl, double dr)
{
  double V_fl = u4d(0);
  double V_rr = u4d(1);
  double delta_fl = u4d(2);
  double delta_rr = u4d(3);

  double cos_fl = std::cos(delta_fl);
  double sin_fl = std::sin(delta_fl);
  double cos_rr = std::cos(delta_rr);
  double sin_rr = std::sin(delta_rr);

  double Vx = (V_fl * cos_fl + V_rr * cos_rr) / 2.0;
  double Vy = (V_fl * sin_fl + V_rr * sin_rr) / 2.0;
  double omega = (V_rr * cos_rr - V_fl * cos_fl) / (dl + dr);

  return Eigen::Vector3d(Vx, Vy, omega);
}

// =============================================================================
// Static IK: body velocity → 4D
// =============================================================================

Eigen::Vector4d WheelLevel4DModel::inverseKinematics(
  const Eigen::Vector3d& body_vel, double lf, double lr, double dl, double dr)
{
  double Vx = body_vel(0);
  double Vy = body_vel(1);
  double omega = body_vel(2);

  // FL wheel (front-left diagonal)
  double fl_x = Vx - omega * dl;
  double fl_y = Vy + omega * lf;
  double delta_fl = std::atan2(fl_y, fl_x);
  double V_fl = std::sqrt(fl_x * fl_x + fl_y * fl_y);

  // RR wheel (rear-right diagonal)
  double rr_x = Vx + omega * dr;
  double rr_y = Vy - omega * lr;
  double delta_rr = std::atan2(rr_y, rr_x);
  double V_rr = std::sqrt(rr_x * rr_x + rr_y * rr_y);

  return Eigen::Vector4d(V_fl, V_rr, delta_fl, delta_rr);
}

// =============================================================================
// Batch dynamics: x_dot = f(x, u)
// =============================================================================

Eigen::MatrixXd WheelLevel4DModel::dynamicsBatch(
  const Eigen::MatrixXd& states,
  const Eigen::MatrixXd& controls) const
{
  int M = states.rows();
  Eigen::MatrixXd state_dot(M, 3);

  Eigen::ArrayXd theta = states.col(2).array();
  Eigen::ArrayXd V_fl = controls.col(0).array();
  Eigen::ArrayXd V_rr = controls.col(1).array();
  Eigen::ArrayXd delta_fl = controls.col(2).array();
  Eigen::ArrayXd delta_rr = controls.col(3).array();

  Eigen::ArrayXd cos_fl = delta_fl.cos();
  Eigen::ArrayXd sin_fl = delta_fl.sin();
  Eigen::ArrayXd cos_rr = delta_rr.cos();
  Eigen::ArrayXd sin_rr = delta_rr.sin();

  // FK: 4D → body velocity
  Eigen::ArrayXd Vx = (V_fl * cos_fl + V_rr * cos_rr) / 2.0;
  Eigen::ArrayXd Vy = (V_fl * sin_fl + V_rr * sin_rr) / 2.0;
  Eigen::ArrayXd omega = (V_rr * cos_rr - V_fl * cos_fl) / (dl_ + dr_);

  // Body → World frame 변환
  Eigen::ArrayXd cos_theta = theta.cos();
  Eigen::ArrayXd sin_theta = theta.sin();

  state_dot.col(0) = Vx * cos_theta - Vy * sin_theta;  // x_dot
  state_dot.col(1) = Vx * sin_theta + Vy * cos_theta;  // y_dot
  state_dot.col(2) = omega;                              // theta_dot

  return state_dot;
}

// =============================================================================
// Control clipping
// =============================================================================

Eigen::MatrixXd WheelLevel4DModel::clipControls(
  const Eigen::MatrixXd& controls) const
{
  Eigen::MatrixXd clipped = controls;
  // V_fl, V_rr: [-v_wheel_max, v_wheel_max]
  clipped.col(0) = clipped.col(0).cwiseMax(-v_wheel_max_).cwiseMin(v_wheel_max_);
  clipped.col(1) = clipped.col(1).cwiseMax(-v_wheel_max_).cwiseMin(v_wheel_max_);
  // delta_fl, delta_rr: [-delta_max, delta_max]
  clipped.col(2) = clipped.col(2).cwiseMax(-delta_max_).cwiseMin(delta_max_);
  clipped.col(3) = clipped.col(3).cwiseMax(-delta_max_).cwiseMin(delta_max_);
  return clipped;
}

// =============================================================================
// State normalization
// =============================================================================

void WheelLevel4DModel::normalizeStates(Eigen::MatrixXd& states) const
{
  states.col(2) = normalizeAngleBatch(states.col(2));
}

// =============================================================================
// Control ↔ Twist 변환
// =============================================================================

geometry_msgs::msg::Twist WheelLevel4DModel::controlToTwist(
  const Eigen::VectorXd& control) const
{
  // 4D → body velocity via FK → Twist
  Eigen::Vector4d u4d = control;
  Eigen::Vector3d body_vel = forwardKinematics(u4d, dl_, dr_);

  geometry_msgs::msg::Twist twist;
  twist.linear.x = body_vel(0);
  twist.linear.y = body_vel(1);
  twist.angular.z = body_vel(2);
  return twist;
}

Eigen::VectorXd WheelLevel4DModel::twistToControl(
  const geometry_msgs::msg::Twist& twist) const
{
  // Twist → body velocity → 4D via IK
  Eigen::Vector3d body_vel(twist.linear.x, twist.linear.y, twist.angular.z);
  Eigen::Vector4d u4d = inverseKinematics(body_vel, lf_, lr_, dl_, dr_);
  return u4d;
}

}  // namespace mpc_controller_ros2
