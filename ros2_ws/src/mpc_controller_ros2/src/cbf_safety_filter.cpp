#include "mpc_controller_ros2/cbf_safety_filter.hpp"
#include <algorithm>
#include <cmath>

namespace mpc_controller_ros2
{

CBFSafetyFilter::CBFSafetyFilter(BarrierFunctionSet* barrier_set,
                                 double gamma, double dt,
                                 const Eigen::VectorXd& u_min,
                                 const Eigen::VectorXd& u_max)
: barrier_set_(barrier_set), gamma_(gamma), dt_(dt),
  u_min_(u_min), u_max_(u_max)
{
}

Eigen::VectorXd CBFSafetyFilter::computeXdot(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& u,
  const BatchDynamicsWrapper& dynamics) const
{
  // dynamicsBatch는 (M x nx), (M x nu) → (M x nx)
  // M=1로 단일 상태 처리
  Eigen::MatrixXd states(1, state.size());
  states.row(0) = state.transpose();
  Eigen::MatrixXd controls(1, u.size());
  controls.row(0) = u.transpose();
  Eigen::MatrixXd xdot = dynamics.dynamicsBatch(states, controls);
  return xdot.row(0).transpose();
}

Eigen::VectorXd CBFSafetyFilter::clipToBounds(const Eigen::VectorXd& u) const
{
  Eigen::VectorXd clipped = u;
  for (int i = 0; i < u.size(); ++i) {
    clipped(i) = std::clamp(clipped(i), u_min_(i), u_max_(i));
  }
  return clipped;
}

bool CBFSafetyFilter::isSafe(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& u,
  const std::vector<const CircleBarrier*>& active_barriers,
  const BatchDynamicsWrapper& dynamics) const
{
  Eigen::VectorXd xdot = computeXdot(state, u, dynamics);

  for (const auto* barrier : active_barriers) {
    double h = barrier->evaluate(state);
    Eigen::VectorXd grad_h = barrier->gradient(state);
    double h_dot = grad_h.dot(xdot);
    double margin = h_dot + gamma_ * h;
    if (margin < -kTolerance) {
      return false;
    }
  }
  return true;
}

std::pair<Eigen::VectorXd, CBFFilterInfo> CBFSafetyFilter::filter(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& u_mppi,
  const BatchDynamicsWrapper& dynamics) const
{
  CBFFilterInfo info;

  // 1. 활성 장애물 필터링
  auto active_barriers = barrier_set_->getActiveBarriers(state);
  info.num_active_barriers = active_barriers.size();

  // barrier values 기록
  for (const auto* barrier : active_barriers) {
    info.barrier_values.push_back(barrier->evaluate(state));
  }

  // 활성 장애물이 없으면 u_mppi 그대로 반환
  if (active_barriers.empty()) {
    info.filter_applied = false;
    info.qp_success = true;
    return {u_mppi, info};
  }

  // 2. u_mppi가 이미 안전한지 빠른 확인
  if (isSafe(state, u_mppi, active_barriers, dynamics)) {
    // 마진 기록
    Eigen::VectorXd xdot = computeXdot(state, u_mppi, dynamics);
    for (const auto* barrier : active_barriers) {
      double h = barrier->evaluate(state);
      Eigen::VectorXd grad_h = barrier->gradient(state);
      double h_dot = grad_h.dot(xdot);
      info.constraint_margins.push_back(h_dot + gamma_ * h);
    }
    info.filter_applied = false;
    info.qp_success = true;
    return {u_mppi, info};
  }

  // 3. Projected gradient descent로 QP 해결
  //    min ||u - u_mppi||²
  //    s.t. ḣ_i + γ·h_i ≥ 0, u_min ≤ u ≤ u_max
  info.filter_applied = true;
  int nu = u_mppi.size();
  Eigen::VectorXd u = u_mppi;

  for (int iter = 0; iter < kMaxIterations; ++iter) {
    // Objective gradient: ∇f = u - u_mppi
    Eigen::VectorXd grad_obj = u - u_mppi;

    // 제약 위반 확인 및 correction
    Eigen::VectorXd xdot = computeXdot(state, u, dynamics);
    bool all_satisfied = true;
    Eigen::VectorXd constraint_correction = Eigen::VectorXd::Zero(nu);

    for (const auto* barrier : active_barriers) {
      double h = barrier->evaluate(state);
      Eigen::VectorXd grad_h = barrier->gradient(state);
      double h_dot = grad_h.dot(xdot);
      double margin = h_dot + gamma_ * h;

      if (margin < 0.0) {
        all_satisfied = false;
        // ḣ = ∇h · f(x,u) → ∂ḣ/∂u = ∇h · ∂f/∂u
        // Lie derivative의 u에 대한 기울기 근사:
        // DiffDrive: f = [v·cos(θ), v·sin(θ), ω]
        // ∂f/∂u ≈ 유한차분으로 수치적 계산
        double delta = 1e-4;
        Eigen::VectorXd dmargin_du(nu);
        for (int j = 0; j < nu; ++j) {
          Eigen::VectorXd u_plus = u;
          u_plus(j) += delta;
          Eigen::VectorXd xdot_plus = computeXdot(state, u_plus, dynamics);
          double h_dot_plus = grad_h.dot(xdot_plus);
          double margin_plus = h_dot_plus + gamma_ * h;
          dmargin_du(j) = (margin_plus - margin) / delta;
        }

        // 제약 만족 방향으로 보정: margin ≥ 0이 되도록
        double dmargin_norm_sq = dmargin_du.squaredNorm();
        if (dmargin_norm_sq > 1e-10) {
          // 제약 위반량만큼 gradient projection
          double correction_scale = (-margin) / dmargin_norm_sq;
          constraint_correction += correction_scale * dmargin_du;
        }
      }
    }

    if (all_satisfied) {
      // 목적함수 gradient step (제약 만족 상태에서 u_mppi에 가깝게)
      double obj_step = kStepSize;
      Eigen::VectorXd u_candidate = u - obj_step * grad_obj;
      u_candidate = clipToBounds(u_candidate);

      // 목적함수 step 후에도 제약 만족하는지 확인
      if (isSafe(state, u_candidate, active_barriers, dynamics)) {
        u = u_candidate;
      }

      // 수렴: 모든 제약 만족
      break;
    } else {
      // 제약 보정 적용
      u = u + constraint_correction;
      u = clipToBounds(u);
    }
  }

  // 최종 안전성 확인
  if (!isSafe(state, u, active_barriers, dynamics)) {
    // 최후 수단: 0 제어 시도
    Eigen::VectorXd u_zero = Eigen::VectorXd::Zero(nu);
    if (isSafe(state, u_zero, active_barriers, dynamics)) {
      u = u_zero;
      info.qp_success = true;
    } else {
      // 완전 실패 — u_mppi fallback
      u = u_mppi;
      info.qp_success = false;
    }
  } else {
    info.qp_success = true;
  }

  // 최종 마진 기록
  Eigen::VectorXd final_xdot = computeXdot(state, u, dynamics);
  for (const auto* barrier : active_barriers) {
    double h = barrier->evaluate(state);
    Eigen::VectorXd grad_h = barrier->gradient(state);
    double h_dot = grad_h.dot(final_xdot);
    info.constraint_margins.push_back(h_dot + gamma_ * h);
  }

  return {u, info};
}

}  // namespace mpc_controller_ros2
