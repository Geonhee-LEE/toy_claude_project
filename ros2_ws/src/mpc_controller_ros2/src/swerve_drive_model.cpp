#include "mpc_controller_ros2/swerve_drive_model.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

SwerveDriveModel::SwerveDriveModel(
  double vx_max, double vy_max, double omega_max)
: vx_max_(vx_max), vy_max_(vy_max), omega_max_(omega_max)
{
}

Eigen::MatrixXd SwerveDriveModel::dynamicsBatch(
  const Eigen::MatrixXd& states,
  const Eigen::MatrixXd& controls) const
{
  int M = states.rows();
  Eigen::MatrixXd state_dot(M, 3);

  Eigen::VectorXd theta = states.col(2);
  Eigen::VectorXd vx = controls.col(0);
  Eigen::VectorXd vy = controls.col(1);
  Eigen::VectorXd omega = controls.col(2);

  Eigen::ArrayXd cos_theta = theta.array().cos();
  Eigen::ArrayXd sin_theta = theta.array().sin();

  // Body → World frame 변환
  state_dot.col(0) = vx.array() * cos_theta - vy.array() * sin_theta;  // x_dot
  state_dot.col(1) = vx.array() * sin_theta + vy.array() * cos_theta;  // y_dot
  state_dot.col(2) = omega;                                             // theta_dot

  return state_dot;
}

Eigen::MatrixXd SwerveDriveModel::clipControls(
  const Eigen::MatrixXd& controls) const
{
  Eigen::MatrixXd clipped = controls;
  clipped.col(0) = clipped.col(0).cwiseMax(-vx_max_).cwiseMin(vx_max_);
  clipped.col(1) = clipped.col(1).cwiseMax(-vy_max_).cwiseMin(vy_max_);
  clipped.col(2) = clipped.col(2).cwiseMax(-omega_max_).cwiseMin(omega_max_);
  return clipped;
}

void SwerveDriveModel::normalizeStates(Eigen::MatrixXd& states) const
{
  states.col(2) = normalizeAngleBatch(states.col(2));
}

geometry_msgs::msg::Twist SwerveDriveModel::controlToTwist(
  const Eigen::VectorXd& control) const
{
  geometry_msgs::msg::Twist twist;
  twist.linear.x = control(0);
  twist.linear.y = control(1);
  twist.angular.z = control(2);
  return twist;
}

Eigen::VectorXd SwerveDriveModel::twistToControl(
  const geometry_msgs::msg::Twist& twist) const
{
  Eigen::VectorXd control(3);
  control(0) = twist.linear.x;
  control(1) = twist.linear.y;
  control(2) = twist.angular.z;
  return control;
}

}  // namespace mpc_controller_ros2
