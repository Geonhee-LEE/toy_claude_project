#include "mpc_controller_ros2/clf_function.hpp"
#include <cmath>
#include <stdexcept>

namespace mpc_controller_ros2
{

CLFFunction::CLFFunction(const Eigen::MatrixXd& P, double c,
                         const std::vector<int>& angle_indices)
: P_(P), c_(c), nx_(P.rows()), angle_indices_(angle_indices)
{
  if (P.rows() != P.cols()) {
    throw std::invalid_argument("CLFFunction: P must be square");
  }
  if (c <= 0.0) {
    throw std::invalid_argument("CLFFunction: c must be positive");
  }
}

Eigen::VectorXd CLFFunction::stateError(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& x_des) const
{
  Eigen::VectorXd err = state - x_des;
  // Angle wrapping for specified indices
  for (int idx : angle_indices_) {
    if (idx < err.size()) {
      double a = err(idx);
      // Wrap to [-π, π]
      a = std::fmod(a + M_PI, 2.0 * M_PI);
      if (a < 0) a += 2.0 * M_PI;
      err(idx) = a - M_PI;
    }
  }
  return err;
}

double CLFFunction::evaluate(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& x_des) const
{
  Eigen::VectorXd err = stateError(state, x_des);
  return err.transpose() * P_ * err;
}

Eigen::VectorXd CLFFunction::gradient(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& x_des) const
{
  Eigen::VectorXd err = stateError(state, x_des);
  return 2.0 * P_ * err;
}

std::pair<double, Eigen::VectorXd> CLFFunction::lieDerivatives(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& x_des,
  const Eigen::VectorXd& x_dot,
  const Eigen::MatrixXd& B) const
{
  Eigen::VectorXd grad_V = gradient(state, x_des);

  // L_f V = ∇V · f(x, u=0) ≈ ∇V · (x_dot - B·u)
  // 하지만 실제로는 V̇ = ∇V · x_dot 이고,
  // ∂ḣ/∂u = ∇V · B (control-affine 근사)
  // L_f V = ∇V · x_dot (drift component = 전체 x_dot at current u)
  double L_f_V = grad_V.dot(x_dot);

  // L_g V = ∇V · B (control sensitivity)
  // shape: (1, nx) * (nx, nu) = (1, nu) → (nu,)
  Eigen::VectorXd L_g_V = B.transpose() * grad_V;

  return {L_f_V, L_g_V};
}

}  // namespace mpc_controller_ros2
