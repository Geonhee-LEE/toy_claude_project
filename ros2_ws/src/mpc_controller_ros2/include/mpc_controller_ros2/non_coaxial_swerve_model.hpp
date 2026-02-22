#ifndef MPC_CONTROLLER_ROS2__NON_COAXIAL_SWERVE_MODEL_HPP_
#define MPC_CONTROLLER_ROS2__NON_COAXIAL_SWERVE_MODEL_HPP_

#include "mpc_controller_ros2/motion_model.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Non-Coaxial Swerve Drive 동역학 모델
 *
 * 스티어링 각도가 상태에 포함된 비동축 스워브 드라이브.
 * 스티어링 각도 제한으로 완전한 holonomic 이동이 불가.
 *
 * State:   [x, y, theta, delta]     (nx=4)
 * Control: [v, omega, delta_dot]    (nu=3)
 *
 * 연속 동역학:
 *   vx_body   = v * cos(delta)
 *   vy_body   = v * sin(delta)
 *   x_dot     = vx_body * cos(theta) - vy_body * sin(theta)
 *   y_dot     = vx_body * sin(theta) + vy_body * cos(theta)
 *   theta_dot = omega
 *   delta_dot = delta_dot (제어 입력)
 *
 * Python 대응: mpc_controller/models/non_coaxial_swerve.py
 */
class NonCoaxialSwerveModel : public MotionModel
{
public:
  NonCoaxialSwerveModel(
    double v_min, double v_max, double omega_max, double max_steering_rate,
    double max_steering_angle = M_PI / 2.0);

  int stateDim() const override { return 4; }
  int controlDim() const override { return 3; }
  bool isHolonomic() const override { return false; }
  std::string name() const override { return "non_coaxial_swerve"; }

  Eigen::MatrixXd dynamicsBatch(
    const Eigen::MatrixXd& states,
    const Eigen::MatrixXd& controls) const override;

  Eigen::MatrixXd clipControls(
    const Eigen::MatrixXd& controls) const override;

  void normalizeStates(Eigen::MatrixXd& states) const override;

  geometry_msgs::msg::Twist controlToTwist(
    const Eigen::VectorXd& control) const override;

  Eigen::VectorXd twistToControl(
    const geometry_msgs::msg::Twist& twist) const override;

  std::vector<int> angleIndices() const override { return {2}; }

  // NonCoaxial 전용: propagateBatch에서 delta clamp 추가
  Eigen::MatrixXd propagateBatch(
    const Eigen::MatrixXd& states,
    const Eigen::MatrixXd& controls,
    double dt) const override;

private:
  double v_min_, v_max_, omega_max_, max_steering_rate_, max_steering_angle_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__NON_COAXIAL_SWERVE_MODEL_HPP_
