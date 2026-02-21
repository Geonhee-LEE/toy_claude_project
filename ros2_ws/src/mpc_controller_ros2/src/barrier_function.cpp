#include "mpc_controller_ros2/barrier_function.hpp"
#include <cmath>

namespace mpc_controller_ros2
{

// ============================================================================
// CircleBarrier
// ============================================================================

CircleBarrier::CircleBarrier(double obs_x, double obs_y, double obs_radius,
                             double robot_radius, double safety_margin)
: obs_x_(obs_x), obs_y_(obs_y)
{
  d_safe_ = obs_radius + robot_radius + safety_margin;
  d_safe_sq_ = d_safe_ * d_safe_;
}

double CircleBarrier::evaluate(const Eigen::VectorXd& state) const
{
  double dx = state(0) - obs_x_;
  double dy = state(1) - obs_y_;
  return dx * dx + dy * dy - d_safe_sq_;
}

Eigen::VectorXd CircleBarrier::evaluateBatch(const Eigen::MatrixXd& states) const
{
  int M = states.rows();
  Eigen::VectorXd h(M);
  for (int i = 0; i < M; ++i) {
    double dx = states(i, 0) - obs_x_;
    double dy = states(i, 1) - obs_y_;
    h(i) = dx * dx + dy * dy - d_safe_sq_;
  }
  return h;
}

Eigen::VectorXd CircleBarrier::gradient(const Eigen::VectorXd& state) const
{
  int nx = state.size();
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(nx);
  grad(0) = 2.0 * (state(0) - obs_x_);
  grad(1) = 2.0 * (state(1) - obs_y_);
  return grad;
}

// ============================================================================
// BarrierFunctionSet
// ============================================================================

BarrierFunctionSet::BarrierFunctionSet(double robot_radius,
                                       double safety_margin,
                                       double activation_distance)
: robot_radius_(robot_radius),
  safety_margin_(safety_margin),
  activation_distance_(activation_distance)
{
}

void BarrierFunctionSet::setObstacles(const std::vector<Eigen::Vector3d>& obstacles)
{
  barriers_.clear();
  barriers_.reserve(obstacles.size());
  for (const auto& obs : obstacles) {
    barriers_.emplace_back(obs(0), obs(1), obs(2), robot_radius_, safety_margin_);
  }
}

std::vector<const CircleBarrier*> BarrierFunctionSet::getActiveBarriers(
  const Eigen::VectorXd& state) const
{
  std::vector<const CircleBarrier*> active;
  for (const auto& barrier : barriers_) {
    double dx = state(0) - barrier.obsX();
    double dy = state(1) - barrier.obsY();
    double dist = std::sqrt(dx * dx + dy * dy);
    if (dist <= activation_distance_) {
      active.push_back(&barrier);
    }
  }
  return active;
}

Eigen::VectorXd BarrierFunctionSet::evaluateAll(const Eigen::VectorXd& state) const
{
  int n = barriers_.size();
  Eigen::VectorXd values(n);
  for (int i = 0; i < n; ++i) {
    values(i) = barriers_[i].evaluate(state);
  }
  return values;
}

}  // namespace mpc_controller_ros2
