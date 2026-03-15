#ifndef MPC_CONTROLLER_ROS2__C3BF_BARRIER_HPP_
#define MPC_CONTROLLER_ROS2__C3BF_BARRIER_HPP_

#include <Eigen/Dense>
#include <vector>

namespace mpc_controller_ros2
{

/**
 * @brief Collision Cone CBF (C3BF)
 *
 * 장애물 속도를 고려하여 접근 방향에서만 페널티를 부여합니다.
 * 정적 장애물에도 로봇 속도 기반으로 동작합니다.
 *
 * h(x) = (p_rel · v_rel) + ||p_rel|| · ||v_rel|| · cos(α_safe)
 *
 * - 양수: 안전 (이격 또는 충돌 콘 밖)
 * - 음수: 위험 (접근 중, 충돌 콘 내부)
 *
 * v_rel = v_obs - v_robot (로봇 시점에서의 상대 속도)
 */
class C3BFBarrier
{
public:
  /**
   * @param obs_x 장애물 x 위치
   * @param obs_y 장애물 y 위치
   * @param obs_radius 장애물 반지름
   * @param robot_radius 로봇 반지름
   * @param safety_margin 추가 안전 마진
   * @param alpha_safe 안전 콘 반각 (rad, default π/4)
   */
  C3BFBarrier(double obs_x, double obs_y, double obs_radius,
              double robot_radius, double safety_margin,
              double alpha_safe = 0.7854);

  /**
   * @brief h(x) 평가
   * @param state 로봇 상태 (최소 [x, y, ...])
   * @param robot_vx 로봇 x 속도
   * @param robot_vy 로봇 y 속도
   * @return h 값 (양수=안전, 음수=위험)
   */
  double evaluate(const Eigen::VectorXd& state,
                  double robot_vx, double robot_vy) const;

  /**
   * @brief h(x) 배치 평가
   * @param states (M, nx)
   * @param robot_vx (M,) 로봇 x 속도
   * @param robot_vy (M,) 로봇 y 속도
   * @return (M,) h 값
   */
  Eigen::VectorXd evaluateBatch(const Eigen::MatrixXd& states,
                                 const Eigen::VectorXd& robot_vx,
                                 const Eigen::VectorXd& robot_vy) const;

  /**
   * @brief ∇h(x) gradient (state에 대해)
   * @return (nx,) gradient
   */
  Eigen::VectorXd gradient(const Eigen::VectorXd& state,
                            double robot_vx, double robot_vy) const;

  /** @brief 장애물 속도 업데이트 */
  void updateObstacleVelocity(double vx, double vy);

  double obsX() const { return obs_x_; }
  double obsY() const { return obs_y_; }
  double obsVx() const { return obs_vx_; }
  double obsVy() const { return obs_vy_; }
  double safeDistance() const { return d_safe_; }

private:
  double obs_x_, obs_y_;
  double obs_vx_{0.0}, obs_vy_{0.0};
  double d_safe_;
  double cos_alpha_safe_;
  double robot_radius_;
  double safety_margin_;
  double alpha_safe_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__C3BF_BARRIER_HPP_
