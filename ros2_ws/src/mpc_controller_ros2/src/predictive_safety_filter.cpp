#include "mpc_controller_ros2/predictive_safety_filter.hpp"
#include <algorithm>
#include <cmath>

namespace mpc_controller_ros2
{

PredictiveSafetyFilter::PredictiveSafetyFilter(
  BarrierFunctionSet* barrier_set,
  double gamma, double dt,
  const Eigen::VectorXd& u_min,
  const Eigen::VectorXd& u_max)
: barrier_set_(barrier_set), gamma_(gamma), dt_(dt),
  u_min_(u_min), u_max_(u_max)
{
}

Eigen::VectorXd PredictiveSafetyFilter::computeXdot(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& u,
  const MotionModel& model) const
{
  // MotionModel::dynamics は (x, u) → x_dot
  Eigen::MatrixXd states(1, state.size());
  states.row(0) = state.transpose();
  Eigen::MatrixXd controls(1, u.size());
  controls.row(0) = u.transpose();
  // dynamicsBatch returns (1 x nx)
  Eigen::MatrixXd xdot_batch = model.dynamicsBatch(states, controls);
  return xdot_batch.row(0).transpose();
}

Eigen::VectorXd PredictiveSafetyFilter::clipControl(
  const Eigen::VectorXd& u) const
{
  Eigen::VectorXd clipped = u;
  for (int i = 0; i < u.size(); ++i) {
    clipped(i) = std::clamp(clipped(i), u_min_(i), u_max_(i));
  }
  return clipped;
}

Eigen::VectorXd PredictiveSafetyFilter::projectStep(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& u,
  const MotionModel& model,
  double gamma_t) const
{
  Eigen::VectorXd u_proj = u;

  auto active = barrier_set_->getActiveBarriers(state);
  if (active.empty()) {
    return clipControl(u_proj);
  }

  for (int iter = 0; iter < max_iterations_; ++iter) {
    Eigen::VectorXd xdot = computeXdot(state, u_proj, model);

    bool all_satisfied = true;
    for (const auto* barrier : active) {
      double h = barrier->evaluate(state);
      Eigen::VectorXd grad_h = barrier->gradient(state);
      double h_dot = grad_h.dot(xdot);
      double margin = h_dot + gamma_t * h;

      if (margin < -1e-6) {
        all_satisfied = false;
        // Projected gradient: u += step * (∂ḣ/∂u) * violation
        // ∂ḣ/∂u ≈ ∇h^T · ∂f/∂u (finite difference)
        int nu = u_proj.size();
        Eigen::VectorXd dh_du(nu);
        constexpr double delta = 1e-4;
        for (int j = 0; j < nu; ++j) {
          Eigen::VectorXd u_plus = u_proj;
          u_plus(j) += delta;
          Eigen::VectorXd xdot_plus = computeXdot(state, u_plus, model);
          double h_dot_plus = grad_h.dot(xdot_plus);
          dh_du(j) = (h_dot_plus - h_dot) / delta;
        }

        double norm_sq = dh_du.squaredNorm();
        if (norm_sq > 1e-10) {
          double correction = (-margin) / norm_sq;
          u_proj = u_proj + correction * dh_du;
          u_proj = clipControl(u_proj);
        }
      }
    }

    if (all_satisfied) {
      break;
    }
  }

  return clipControl(u_proj);
}

PredictiveSafetyResult PredictiveSafetyFilter::filter(
  const Eigen::VectorXd& x0,
  const Eigen::MatrixXd& control_sequence,
  const MotionModel& model) const
{
  int N = control_sequence.rows();
  int nu = control_sequence.cols();
  int nx = x0.size();

  PredictiveSafetyResult result;
  result.u_safe_sequence = control_sequence;  // (N x nu) 복사
  result.safe_trajectory = Eigen::MatrixXd::Zero(N + 1, nx);
  result.safe_trajectory.row(0) = x0.transpose();
  result.feasible = true;
  result.num_corrected_steps = 0;
  result.min_barrier_values.resize(N + 1, std::numeric_limits<double>::infinity());

  Eigen::VectorXd x = x0;

  for (int t = 0; t < N; ++t) {
    // 시간 감쇠 gamma
    double gamma_t = gamma_ * std::pow(horizon_decay_, t);

    // 최소 barrier 값 기록
    auto active = barrier_set_->getActiveBarriers(x);
    if (!active.empty()) {
      double min_h = std::numeric_limits<double>::infinity();
      for (const auto* b : active) {
        min_h = std::min(min_h, b->evaluate(x));
      }
      result.min_barrier_values[t] = min_h;
    }

    // CBF 투영
    Eigen::VectorXd u_orig = control_sequence.row(t).transpose();
    Eigen::VectorXd u_safe = projectStep(x, u_orig, model, gamma_t);

    if ((u_safe - u_orig).norm() > 1e-6) {
      result.num_corrected_steps++;
    }

    result.u_safe_sequence.row(t) = u_safe.transpose();

    // Forward rollout with corrected control
    Eigen::VectorXd xdot = computeXdot(x, u_safe, model);
    x = x + dt_ * xdot;

    // Normalize angles if needed
    // normalizeStates expects MatrixXd reference
    Eigen::MatrixXd x_mat(1, x.size());
    x_mat.row(0) = x.transpose();
    model.normalizeStates(x_mat);
    x = x_mat.row(0).transpose();

    result.safe_trajectory.row(t + 1) = x.transpose();
  }

  // 마지막 스텝 barrier 값
  auto final_active = barrier_set_->getActiveBarriers(x);
  if (!final_active.empty()) {
    double min_h = std::numeric_limits<double>::infinity();
    for (const auto* b : final_active) {
      min_h = std::min(min_h, b->evaluate(x));
    }
    result.min_barrier_values[N] = min_h;
    if (min_h < 0) {
      result.feasible = false;
    }
  }

  return result;
}

bool PredictiveSafetyFilter::verifyTrajectory(
  const Eigen::VectorXd& x0,
  const Eigen::MatrixXd& control_sequence,
  const MotionModel& model) const
{
  int N = control_sequence.rows();
  Eigen::VectorXd x = x0;

  for (int t = 0; t < N; ++t) {
    double gamma_t = gamma_ * std::pow(horizon_decay_, t);
    Eigen::VectorXd u = control_sequence.row(t).transpose();
    Eigen::VectorXd xdot = computeXdot(x, u, model);

    auto active = barrier_set_->getActiveBarriers(x);
    for (const auto* barrier : active) {
      double h = barrier->evaluate(x);
      double h_dot = barrier->gradient(x).dot(xdot);
      if (h_dot + gamma_t * h < -1e-6) {
        return false;
      }
    }

    x = x + dt_ * xdot;
    // normalizeStates expects MatrixXd reference
    Eigen::MatrixXd x_mat(1, x.size());
    x_mat.row(0) = x.transpose();
    model.normalizeStates(x_mat);
    x = x_mat.row(0).transpose();
  }

  return true;
}

}  // namespace mpc_controller_ros2
