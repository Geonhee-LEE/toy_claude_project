#ifndef MPC_CONTROLLER_ROS2__DIFF_DRIVE_MODEL_HPP_
#define MPC_CONTROLLER_ROS2__DIFF_DRIVE_MODEL_HPP_

#include "mpc_controller_ros2/motion_model.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Differential Drive 동역학 모델
 *
 * State:   [x, y, theta]       (nx=3)
 * Control: [v, omega]          (nu=2)
 *
 * 연속 동역학:
 *   x_dot     = v * cos(theta)
 *   y_dot     = v * sin(theta)
 *   theta_dot = omega
 *
 * Python 대응: mpc_controller/models/differential_drive.py
 */
class DiffDriveModel : public MotionModel
{
public:
  DiffDriveModel(double v_min, double v_max, double omega_min, double omega_max);

  int stateDim() const override { return 3; }
  int controlDim() const override { return 2; }
  bool isHolonomic() const override { return false; }
  std::string name() const override { return "diff_drive"; }

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
  double v_min_, v_max_, omega_min_, omega_max_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__DIFF_DRIVE_MODEL_HPP_
