#include "mpc_controller_ros2/non_coaxial_swerve_model.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

NonCoaxialSwerveModel::NonCoaxialSwerveModel(
  double v_min, double v_max, double omega_max, double max_steering_rate,
  double max_steering_angle)
: v_min_(v_min), v_max_(v_max), omega_max_(omega_max),
  max_steering_rate_(max_steering_rate),
  max_steering_angle_(max_steering_angle)
{
}

Eigen::MatrixXd NonCoaxialSwerveModel::dynamicsBatch(
  const Eigen::MatrixXd& states,
  const Eigen::MatrixXd& controls) const
{
  int M = states.rows();
  Eigen::MatrixXd state_dot(M, 4);

  Eigen::VectorXd theta = states.col(2);
  Eigen::VectorXd delta = states.col(3);
  Eigen::VectorXd v = controls.col(0);
  Eigen::VectorXd omega = controls.col(1);
  Eigen::VectorXd delta_dot = controls.col(2);

  Eigen::ArrayXd cos_theta = theta.array().cos();
  Eigen::ArrayXd sin_theta = theta.array().sin();
  Eigen::ArrayXd cos_delta = delta.array().cos();
  Eigen::ArrayXd sin_delta = delta.array().sin();

  // Body frame velocities (steering angle determines direction)
  Eigen::ArrayXd vx_body = v.array() * cos_delta;
  Eigen::ArrayXd vy_body = v.array() * sin_delta;

  // World frame velocities
  state_dot.col(0) = vx_body * cos_theta - vy_body * sin_theta;  // x_dot
  state_dot.col(1) = vx_body * sin_theta + vy_body * cos_theta;  // y_dot
  state_dot.col(2) = omega;                                       // theta_dot
  state_dot.col(3) = delta_dot;                                   // delta_dot

  return state_dot;
}

Eigen::MatrixXd NonCoaxialSwerveModel::clipControls(
  const Eigen::MatrixXd& controls) const
{
  Eigen::MatrixXd clipped = controls;
  clipped.col(0) = clipped.col(0).cwiseMax(v_min_).cwiseMin(v_max_);
  clipped.col(1) = clipped.col(1).cwiseMax(-omega_max_).cwiseMin(omega_max_);
  clipped.col(2) = clipped.col(2).cwiseMax(-max_steering_rate_).cwiseMin(max_steering_rate_);
  return clipped;
}

void NonCoaxialSwerveModel::normalizeStates(Eigen::MatrixXd& states) const
{
  // theta 각도 정규화
  states.col(2) = normalizeAngleBatch(states.col(2));
  // delta (steering angle) clamp to limits
  states.col(3) = states.col(3).cwiseMax(-max_steering_angle_).cwiseMin(max_steering_angle_);
}

Eigen::MatrixXd NonCoaxialSwerveModel::propagateBatch(
  const Eigen::MatrixXd& states,
  const Eigen::MatrixXd& controls,
  double dt) const
{
  // RK4 integration (동일)
  Eigen::MatrixXd k1 = dynamicsBatch(states, controls);
  Eigen::MatrixXd k2 = dynamicsBatch(states + dt / 2.0 * k1, controls);
  Eigen::MatrixXd k3 = dynamicsBatch(states + dt / 2.0 * k2, controls);
  Eigen::MatrixXd k4 = dynamicsBatch(states + dt * k3, controls);

  Eigen::MatrixXd states_next = states + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

  // Normalize (angle wrapping + delta clamp)
  normalizeStates(states_next);

  return states_next;
}

geometry_msgs::msg::Twist NonCoaxialSwerveModel::controlToTwist(
  const Eigen::VectorXd& control) const
{
  geometry_msgs::msg::Twist twist;
  // v, omega → Twist (delta_dot은 Twist에 매핑하기 어려움)
  // body frame velocity: vx_body = v * cos(delta) — 하지만 delta는 상태이므로
  // 여기서는 단순히 v → linear.x, omega → angular.z 매핑
  twist.linear.x = control(0);
  twist.angular.z = control(1);
  // delta_dot은 별도 토픽으로 publish하거나 JointCommand 사용 필요
  return twist;
}

Eigen::VectorXd NonCoaxialSwerveModel::twistToControl(
  const geometry_msgs::msg::Twist& twist) const
{
  Eigen::VectorXd control(3);
  control(0) = twist.linear.x;
  control(1) = twist.angular.z;
  control(2) = 0.0;  // delta_dot은 Twist에서 직접 추출 불가
  return control;
}

}  // namespace mpc_controller_ros2
