#ifndef MPC_CONTROLLER_ROS2__SWERVE_DRIVE_MODEL_HPP_
#define MPC_CONTROLLER_ROS2__SWERVE_DRIVE_MODEL_HPP_

#include "mpc_controller_ros2/motion_model.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Swerve Drive (Holonomic) 동역학 모델
 *
 * State:   [x, y, theta]       (nx=3)
 * Control: [vx, vy, omega]     (nu=3)
 *
 * 연속 동역학 (Body → World frame 변환):
 *   x_dot     = vx * cos(theta) - vy * sin(theta)
 *   y_dot     = vx * sin(theta) + vy * cos(theta)
 *   theta_dot = omega
 *
 * Python 대응: mpc_controller/models/swerve_drive.py
 */
class SwerveDriveModel : public MotionModel
{
public:
  SwerveDriveModel(double vx_max, double vy_max, double omega_max);

  int stateDim() const override { return 3; }
  int controlDim() const override { return 3; }
  bool isHolonomic() const override { return true; }
  std::string name() const override { return "swerve"; }

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

private:
  double vx_max_, vy_max_, omega_max_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__SWERVE_DRIVE_MODEL_HPP_
