#include "mpc_controller_ros2/ackermann_model.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

AckermannModel::AckermannModel(
  double v_min, double v_max, double max_steering_rate,
  double max_steering_angle, double wheelbase)
: v_min_(v_min), v_max_(v_max),
  max_steering_rate_(max_steering_rate),
  max_steering_angle_(max_steering_angle),
  wheelbase_(wheelbase)
{
}

Eigen::MatrixXd AckermannModel::dynamicsBatch(
  const Eigen::MatrixXd& states,
  const Eigen::MatrixXd& controls) const
{
  int M = states.rows();
  Eigen::MatrixXd state_dot(M, 4);

  Eigen::VectorXd theta = states.col(2);
  Eigen::VectorXd delta = states.col(3);
  Eigen::VectorXd v = controls.col(0);
  Eigen::VectorXd delta_dot = controls.col(1);

  Eigen::ArrayXd cos_theta = theta.array().cos();
  Eigen::ArrayXd sin_theta = theta.array().sin();
  Eigen::ArrayXd tan_delta = delta.array().tan();

  // Bicycle model dynamics
  state_dot.col(0) = v.array() * cos_theta;                    // x_dot
  state_dot.col(1) = v.array() * sin_theta;                    // y_dot
  state_dot.col(2) = v.array() * tan_delta / wheelbase_;       // theta_dot = v*tan(δ)/L
  state_dot.col(3) = delta_dot;                                 // delta_dot

  return state_dot;
}

Eigen::MatrixXd AckermannModel::clipControls(
  const Eigen::MatrixXd& controls) const
{
  Eigen::MatrixXd clipped = controls;
  clipped.col(0) = clipped.col(0).cwiseMax(v_min_).cwiseMin(v_max_);
  clipped.col(1) = clipped.col(1).cwiseMax(-max_steering_rate_).cwiseMin(max_steering_rate_);
  return clipped;
}

void AckermannModel::normalizeStates(Eigen::MatrixXd& states) const
{
  // theta 각도 정규화
  states.col(2) = normalizeAngleBatch(states.col(2));
  // delta (steering angle) clamp to limits
  states.col(3) = states.col(3).cwiseMax(-max_steering_angle_).cwiseMin(max_steering_angle_);
}

Eigen::MatrixXd AckermannModel::propagateBatch(
  const Eigen::MatrixXd& states,
  const Eigen::MatrixXd& controls,
  double dt) const
{
  // RK4 integration
  Eigen::MatrixXd k1 = dynamicsBatch(states, controls);
  Eigen::MatrixXd k2 = dynamicsBatch(states + dt / 2.0 * k1, controls);
  Eigen::MatrixXd k3 = dynamicsBatch(states + dt / 2.0 * k2, controls);
  Eigen::MatrixXd k4 = dynamicsBatch(states + dt * k3, controls);

  Eigen::MatrixXd states_next = states + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

  // Normalize (angle wrapping + delta clamp)
  normalizeStates(states_next);

  return states_next;
}

geometry_msgs::msg::Twist AckermannModel::controlToTwist(
  const Eigen::VectorXd& control) const
{
  geometry_msgs::msg::Twist twist;
  double v = control(0);
  // Ackermann: linear.x=v, angular.z = v*tan(δ)/L
  twist.linear.x = v;
  twist.angular.z = v * std::tan(last_delta_) / wheelbase_;
  return twist;
}

Eigen::VectorXd AckermannModel::twistToControl(
  const geometry_msgs::msg::Twist& twist) const
{
  Eigen::VectorXd control(2);
  control(0) = twist.linear.x;
  control(1) = 0.0;  // delta_dot은 Twist에서 직접 추출 불가
  return control;
}

Linearization AckermannModel::getLinearization(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& control,
  double dt) const
{
  double theta = state(2);
  double delta = state(3);
  double v = control(0);

  double cos_t = std::cos(theta);
  double sin_t = std::sin(theta);
  double tan_d = std::tan(delta);
  double sec2_d = 1.0 / (std::cos(delta) * std::cos(delta));

  // A = I + dt * df/dx
  Eigen::Matrix4d A = Eigen::Matrix4d::Identity();
  A(0, 2) += dt * (-v * sin_t);
  A(1, 2) += dt * (v * cos_t);
  A(2, 3) += dt * (v * sec2_d / wheelbase_);

  // B = dt * df/du
  Eigen::MatrixXd B(4, 2);
  B(0, 0) = dt * cos_t;              B(0, 1) = 0.0;
  B(1, 0) = dt * sin_t;              B(1, 1) = 0.0;
  B(2, 0) = dt * tan_d / wheelbase_; B(2, 1) = 0.0;
  B(3, 0) = 0.0;                     B(3, 1) = dt;

  return {A, B};
}

}  // namespace mpc_controller_ros2
