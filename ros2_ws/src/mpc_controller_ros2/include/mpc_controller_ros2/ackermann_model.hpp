#ifndef MPC_CONTROLLER_ROS2__ACKERMANN_MODEL_HPP_
#define MPC_CONTROLLER_ROS2__ACKERMANN_MODEL_HPP_

#include "mpc_controller_ros2/motion_model.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Ackermann (Bicycle model) 동역학 모델
 *
 * 전륜 조향 차량의 Bicycle model 근사.
 * θ̇ = v·tan(δ)/L 구속 조건으로 NonCoaxialSwerve(omega 독립)와 구분.
 *
 * State:   [x, y, theta, delta]     (nx=4)
 * Control: [v, delta_dot]           (nu=2)
 *
 * 연속 동역학:
 *   x_dot     = v * cos(theta)
 *   y_dot     = v * sin(theta)
 *   theta_dot = v * tan(delta) / wheelbase   ← Ackermann 핵심
 *   delta_dot = delta_dot (제어 입력)
 */
class AckermannModel : public MotionModel
{
public:
  AckermannModel(
    double v_min, double v_max, double max_steering_rate,
    double max_steering_angle, double wheelbase);

  int stateDim() const override { return 4; }
  int controlDim() const override { return 2; }
  bool isHolonomic() const override { return false; }
  std::string name() const override { return "ackermann"; }

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

  // delta clamp 추가된 RK4 propagation
  Eigen::MatrixXd propagateBatch(
    const Eigen::MatrixXd& states,
    const Eigen::MatrixXd& controls,
    double dt) const override;

  // controlToTwist용 last_delta_ 추적 (플러그인에서 갱신)
  void setLastDelta(double delta) { last_delta_ = delta; }
  double getLastDelta() const { return last_delta_; }

  double wheelbase() const { return wheelbase_; }

private:
  double v_min_, v_max_, max_steering_rate_, max_steering_angle_;
  double wheelbase_;
  double last_delta_{0.0};
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__ACKERMANN_MODEL_HPP_
