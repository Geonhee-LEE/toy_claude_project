#include "mpc_controller_ros2/diff_drive_model.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

DiffDriveModel::DiffDriveModel(
  double v_min, double v_max, double omega_min, double omega_max)
: v_min_(v_min), v_max_(v_max), omega_min_(omega_min), omega_max_(omega_max)
{
}

Eigen::MatrixXd DiffDriveModel::dynamicsBatch(
  const Eigen::MatrixXd& states,
  const Eigen::MatrixXd& controls) const
{
  int M = states.rows();
  Eigen::MatrixXd state_dot(M, 3);

  Eigen::VectorXd theta = states.col(2);
  Eigen::VectorXd v = controls.col(0);
  Eigen::VectorXd omega = controls.col(1);

  state_dot.col(0) = v.array() * theta.array().cos();  // x_dot
  state_dot.col(1) = v.array() * theta.array().sin();  // y_dot
  state_dot.col(2) = omega;                             // theta_dot

  return state_dot;
}

Eigen::MatrixXd DiffDriveModel::clipControls(
  const Eigen::MatrixXd& controls) const
{
  Eigen::MatrixXd clipped = controls;
  clipped.col(0) = clipped.col(0).cwiseMax(v_min_).cwiseMin(v_max_);
  clipped.col(1) = clipped.col(1).cwiseMax(omega_min_).cwiseMin(omega_max_);
  return clipped;
}

void DiffDriveModel::normalizeStates(Eigen::MatrixXd& states) const
{
  states.col(2) = normalizeAngleBatch(states.col(2));
}

geometry_msgs::msg::Twist DiffDriveModel::controlToTwist(
  const Eigen::VectorXd& control) const
{
  geometry_msgs::msg::Twist twist;
  twist.linear.x = control(0);
  twist.angular.z = control(1);
  return twist;
}

Eigen::VectorXd DiffDriveModel::twistToControl(
  const geometry_msgs::msg::Twist& twist) const
{
  Eigen::VectorXd control(2);
  control(0) = twist.linear.x;
  control(1) = twist.angular.z;
  return control;
}

Linearization DiffDriveModel::getLinearization(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& control,
  double dt) const
{
  double theta = state(2);
  double v = control(0);

  double cos_t = std::cos(theta);
  double sin_t = std::sin(theta);

  // A = I + dt * df/dx
  Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
  A(0, 2) += dt * (-v * sin_t);
  A(1, 2) += dt * (v * cos_t);

  // B = dt * df/du
  Eigen::MatrixXd B(3, 2);
  B(0, 0) = dt * cos_t;   B(0, 1) = 0.0;
  B(1, 0) = dt * sin_t;   B(1, 1) = 0.0;
  B(2, 0) = 0.0;          B(2, 1) = dt;

  return {A, B};
}

}  // namespace mpc_controller_ros2
