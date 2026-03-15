#include "mpc_controller_ros2/c3bf_barrier.hpp"
#include <cmath>

namespace mpc_controller_ros2
{

C3BFBarrier::C3BFBarrier(double obs_x, double obs_y, double obs_radius,
                          double robot_radius, double safety_margin,
                          double alpha_safe)
: obs_x_(obs_x), obs_y_(obs_y),
  d_safe_(obs_radius + robot_radius + safety_margin),
  cos_alpha_safe_(std::cos(alpha_safe)),
  robot_radius_(robot_radius),
  safety_margin_(safety_margin),
  alpha_safe_(alpha_safe)
{
}

double C3BFBarrier::evaluate(const Eigen::VectorXd& state,
                              double robot_vx, double robot_vy) const
{
  // 상대 위치: p_rel = p_obs - p_robot
  double px = obs_x_ - state(0);
  double py = obs_y_ - state(1);
  double p_norm = std::sqrt(px * px + py * py);

  if (p_norm < 1e-6) {
    return -d_safe_ * d_safe_;  // 장애물 위에 있음 → 매우 위험
  }

  // 상대 속도: v_rel = v_obs - v_robot (장애물 쪽으로의 접근 속도)
  double vx = obs_vx_ - robot_vx;
  double vy = obs_vy_ - robot_vy;
  double v_norm = std::sqrt(vx * vx + vy * vy);

  // p_rel · v_rel (음수 = 접근 중)
  double p_dot_v = px * vx + py * vy;

  // ||p_rel|| · ||v_rel|| · cos(α_safe)
  double cone_term = p_norm * v_norm * cos_alpha_safe_;

  // h = (p · v) + ||p|| · ||v|| · cos(α)
  // 접근 중(p·v < 0)이고 충돌 콘 내부이면 h < 0
  return p_dot_v + cone_term;
}

Eigen::VectorXd C3BFBarrier::evaluateBatch(
  const Eigen::MatrixXd& states,
  const Eigen::VectorXd& robot_vx,
  const Eigen::VectorXd& robot_vy) const
{
  int M = states.rows();
  Eigen::VectorXd h(M);
  for (int i = 0; i < M; ++i) {
    Eigen::VectorXd s = states.row(i).transpose();
    h(i) = evaluate(s, robot_vx(i), robot_vy(i));
  }
  return h;
}

Eigen::VectorXd C3BFBarrier::gradient(const Eigen::VectorXd& state,
                                        double robot_vx, double robot_vy) const
{
  int nx = state.size();
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(nx);

  double px = obs_x_ - state(0);
  double py = obs_y_ - state(1);
  double p_norm = std::sqrt(px * px + py * py);

  double vx = obs_vx_ - robot_vx;
  double vy = obs_vy_ - robot_vy;
  double v_norm = std::sqrt(vx * vx + vy * vy);

  // ∂h/∂x = ∂(p·v)/∂x + ∂(||p||·||v||·cos(α))/∂x
  // ∂(p·v)/∂x_robot = -v_rel (p_rel = p_obs - p_robot, ∂p_rel/∂x_robot = -I)
  grad(0) = -vx;
  grad(1) = -vy;

  // ∂(||p||)/∂x_robot = -(p/||p||)
  if (p_norm > 1e-6) {
    grad(0) += -px / p_norm * v_norm * cos_alpha_safe_;
    grad(1) += -py / p_norm * v_norm * cos_alpha_safe_;
  }

  return grad;
}

void C3BFBarrier::updateObstacleVelocity(double vx, double vy)
{
  obs_vx_ = vx;
  obs_vy_ = vy;
}

}  // namespace mpc_controller_ros2
