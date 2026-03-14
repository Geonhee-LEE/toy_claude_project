#ifndef MPC_CONTROLLER_ROS2__WHEEL_LEVEL_4D_MODEL_HPP_
#define MPC_CONTROLLER_ROS2__WHEEL_LEVEL_4D_MODEL_HPP_

#include "mpc_controller_ros2/motion_model.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Wheel-Level 4D Swerve 동역학 모델
 *
 * MPPI-H (IROS 2024, arXiv:2409.08648) 논문의 4D 바퀴 레벨 모델.
 * 4WIDS 대각 바퀴 쌍 (FL, RR)을 직접 제어.
 *
 * State:   [x, y, theta]                    (nx=3)
 * Control: [V_fl, V_rr, delta_fl, delta_rr] (nu=4)
 *
 * FK (4D → body velocity):
 *   Vx = (V_fl·cos(δ_fl) + V_rr·cos(δ_rr)) / 2
 *   Vy = (V_fl·sin(δ_fl) + V_rr·sin(δ_rr)) / 2
 *   ω  = (V_rr·cos(δ_rr) - V_fl·cos(δ_fl)) / (dl + dr)
 *
 * 연속 동역학 (Body → World frame 변환):
 *   x_dot     = Vx·cos(θ) - Vy·sin(θ)
 *   y_dot     = Vx·sin(θ) + Vy·cos(θ)
 *   theta_dot = ω
 *
 * 기하 파라미터:
 *   lf, lr: 전/후 바퀴 중심까지의 종방향 거리 (m)
 *   dl, dr: 좌/우 바퀴 중심까지의 횡방향 거리 (m)
 */
class WheelLevel4DModel : public MotionModel
{
public:
  /**
   * @param lf 전방 바퀴 종방향 거리 (m)
   * @param lr 후방 바퀴 종방향 거리 (m)
   * @param dl 좌측 바퀴 횡방향 거리 (m)
   * @param dr 우측 바퀴 횡방향 거리 (m)
   * @param v_wheel_max 최대 바퀴 속도 (m/s)
   * @param delta_max 최대 바퀴 조향각 (rad)
   */
  WheelLevel4DModel(
    double lf, double lr, double dl, double dr,
    double v_wheel_max, double delta_max);

  int stateDim() const override { return 3; }
  int controlDim() const override { return 4; }
  bool isHolonomic() const override { return true; }
  std::string name() const override { return "wheel_level_4d"; }

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

  // ─── Static FK/IK (Hybrid 전환에서 사용) ───

  /**
   * @brief FK: 4D 바퀴 제어 → body velocity [Vx, Vy, ω]
   * @param u4d [V_fl, V_rr, delta_fl, delta_rr]
   * @param dl 좌측 횡방향 거리
   * @param dr 우측 횡방향 거리
   */
  static Eigen::Vector3d forwardKinematics(
    const Eigen::Vector4d& u4d, double dl, double dr);

  /**
   * @brief IK: body velocity [Vx, Vy, ω] → 4D 바퀴 제어
   * @param body_vel [Vx, Vy, ω]
   * @param lf 전방 종방향 거리
   * @param lr 후방 종방향 거리
   * @param dl 좌측 횡방향 거리
   * @param dr 우측 횡방향 거리
   */
  static Eigen::Vector4d inverseKinematics(
    const Eigen::Vector3d& body_vel, double lf, double lr, double dl, double dr);

  // Getters for geometry parameters
  double lf() const { return lf_; }
  double lr() const { return lr_; }
  double dl() const { return dl_; }
  double dr() const { return dr_; }

private:
  double lf_, lr_, dl_, dr_;
  double v_wheel_max_, delta_max_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__WHEEL_LEVEL_4D_MODEL_HPP_
