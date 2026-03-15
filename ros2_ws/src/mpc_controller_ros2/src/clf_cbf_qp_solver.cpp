#include "mpc_controller_ros2/clf_cbf_qp_solver.hpp"
#include <algorithm>
#include <cmath>

namespace mpc_controller_ros2
{

CLFCBFQPSolver::CLFCBFQPSolver(
  const CLFFunction* clf,
  BarrierFunctionSet* barrier_set,
  double gamma,
  double slack_penalty,
  const Eigen::VectorXd& u_min,
  const Eigen::VectorXd& u_max)
: clf_(clf), barrier_set_(barrier_set), gamma_(gamma),
  slack_penalty_(slack_penalty), u_min_(u_min), u_max_(u_max)
{
}

Eigen::VectorXd CLFCBFQPSolver::computeXdot(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& u,
  const BatchDynamicsWrapper& dynamics) const
{
  Eigen::MatrixXd states(1, state.size());
  states.row(0) = state.transpose();
  Eigen::MatrixXd controls(1, u.size());
  controls.row(0) = u.transpose();
  return dynamics.dynamicsBatch(states, controls).row(0).transpose();
}

Eigen::MatrixXd CLFCBFQPSolver::computeB(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& u,
  const Eigen::VectorXd& x_dot,
  const BatchDynamicsWrapper& dynamics) const
{
  int nx = state.size();
  int nu = u.size();
  Eigen::MatrixXd B(nx, nu);

  for (int j = 0; j < nu; ++j) {
    Eigen::VectorXd u_plus = u;
    u_plus(j) += kFiniteDiffDelta;
    Eigen::VectorXd xdot_plus = computeXdot(state, u_plus, dynamics);
    B.col(j) = (xdot_plus - x_dot) / kFiniteDiffDelta;
  }
  return B;
}

Eigen::VectorXd CLFCBFQPSolver::clipToBounds(const Eigen::VectorXd& u) const
{
  Eigen::VectorXd clipped = u;
  for (int i = 0; i < u.size(); ++i) {
    clipped(i) = std::clamp(clipped(i), u_min_(i), u_max_(i));
  }
  return clipped;
}

CLFCBFQPResult CLFCBFQPSolver::solve(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& x_des,
  const Eigen::VectorXd& u_ref,
  const BatchDynamicsWrapper& dynamics) const
{
  CLFCBFQPResult result;
  int nu = u_ref.size();

  // CLF 값 계산
  result.clf_value = clf_->evaluate(state, x_des);

  // 활성 barrier 필터링
  auto active_barriers = barrier_set_->getActiveBarriers(state);

  // 초기 해: u_ref
  Eigen::VectorXd u = clipToBounds(u_ref);
  double delta = 0.0;  // CLF slack

  for (int iter = 0; iter < kMaxIterations; ++iter) {
    result.iterations = iter + 1;

    // 동역학 계산
    Eigen::VectorXd x_dot = computeXdot(state, u, dynamics);
    Eigen::MatrixXd B = computeB(state, u, x_dot, dynamics);

    // === CLF 제약: L_f V + L_g V·u + c·V ≤ δ ===
    auto [L_f_V, L_g_V] = clf_->lieDerivatives(state, x_des, x_dot, B);
    double clf_margin = L_f_V + clf_->c() * result.clf_value - delta;
    // clf_margin ≤ 0 이면 CLF 만족
    // 주의: L_f_V는 이미 현재 u에서의 V̇를 포함하므로
    // 실제 제약: V̇ + c·V ≤ δ → grad_V · x_dot + c·V ≤ δ
    Eigen::VectorXd grad_V = clf_->gradient(state, x_des);
    double V_dot = grad_V.dot(x_dot);
    clf_margin = V_dot + clf_->c() * result.clf_value - delta;
    // L_g V for sensitivity
    Eigen::VectorXd dVdot_du = B.transpose() * grad_V;

    // === CBF 제약: ḣ_i + γ·h_i ≥ 0 ===
    bool all_cbf_satisfied = true;
    Eigen::VectorXd cbf_correction = Eigen::VectorXd::Zero(nu);

    result.cbf_margins.clear();
    for (const auto* barrier : active_barriers) {
      double h = barrier->evaluate(state);
      Eigen::VectorXd grad_h = barrier->gradient(state);
      double h_dot = grad_h.dot(x_dot);
      double cbf_margin_i = h_dot + gamma_ * h;
      result.cbf_margins.push_back(cbf_margin_i);

      if (cbf_margin_i < 0.0) {
        all_cbf_satisfied = false;
        // ∂(ḣ)/∂u = ∇h · B
        Eigen::VectorXd dhdot_du = B.transpose() * grad_h.head(std::min((int)grad_h.size(), (int)B.rows()));
        double norm_sq = dhdot_du.squaredNorm();
        if (norm_sq > 1e-10) {
          double correction_scale = (-cbf_margin_i) / norm_sq;
          cbf_correction += correction_scale * dhdot_du;
        }
      }
    }

    // === CLF correction ===
    bool clf_satisfied = (clf_margin <= kTolerance);
    Eigen::VectorXd clf_correction = Eigen::VectorXd::Zero(nu);
    double delta_update = 0.0;

    if (!clf_satisfied) {
      // CLF 위반: u를 V̇ 감소 방향으로 보정 또는 slack 증가
      double dVdot_du_norm_sq = dVdot_du.squaredNorm();
      if (dVdot_du_norm_sq > 1e-10) {
        // u 보정량 계산 (CLF 만족 방향)
        double clf_correction_scale = clf_margin / dVdot_du_norm_sq;
        clf_correction = -clf_correction_scale * dVdot_du;
      }
      // slack 업데이트: CLF-CBF 충돌 시 slack 허용
      if (!all_cbf_satisfied) {
        // CBF가 위반 중이면 CLF slack을 증가시켜 안전 우선
        delta_update = std::max(0.0, clf_margin);
      }
    }

    // === 통합 업데이트 ===
    if (all_cbf_satisfied && clf_satisfied) {
      // 모든 제약 만족 — 목적함수(u_ref에 가깝게) 방향 step
      Eigen::VectorXd grad_obj = u - u_ref;
      double obj_step = kStepSize;
      Eigen::VectorXd u_candidate = u - obj_step * grad_obj;
      // slack 감소 방향
      double delta_candidate = delta * (1.0 - kStepSize);
      u_candidate = clipToBounds(u_candidate);

      // 제약 유지 확인
      Eigen::VectorXd x_dot_cand = computeXdot(state, u_candidate, dynamics);
      double V_dot_cand = grad_V.dot(x_dot_cand);
      bool still_clf_ok = (V_dot_cand + clf_->c() * result.clf_value - delta_candidate <= kTolerance);

      bool still_cbf_ok = true;
      for (const auto* barrier : active_barriers) {
        double h = barrier->evaluate(state);
        Eigen::VectorXd grad_h_b = barrier->gradient(state);
        double h_dot_cand = grad_h_b.dot(x_dot_cand);
        if (h_dot_cand + gamma_ * h < -kTolerance) {
          still_cbf_ok = false;
          break;
        }
      }

      if (still_clf_ok && still_cbf_ok) {
        u = u_candidate;
        delta = delta_candidate;
      }
      break;  // 수렴
    }

    // CBF 보정 우선 (안전 > 수렴)
    if (!all_cbf_satisfied) {
      u = u + cbf_correction;
      u = clipToBounds(u);
      delta += delta_update;
    } else if (!clf_satisfied) {
      // CBF 만족, CLF만 위반 → CLF 보정
      u = u + clf_correction;
      u = clipToBounds(u);
    }
  }

  // 최종 결과
  result.u_safe = u;
  result.slack = delta;
  result.feasible = true;

  // 최종 제약 확인
  Eigen::VectorXd final_xdot = computeXdot(state, u, dynamics);
  Eigen::VectorXd final_grad_V = clf_->gradient(state, x_des);
  double final_V_dot = final_grad_V.dot(final_xdot);
  result.clf_constraint = final_V_dot + clf_->c() * result.clf_value - delta;

  // CBF 마진 최종 계산
  result.cbf_margins.clear();
  for (const auto* barrier : active_barriers) {
    double h = barrier->evaluate(state);
    Eigen::VectorXd grad_h = barrier->gradient(state);
    double h_dot = grad_h.dot(final_xdot);
    result.cbf_margins.push_back(h_dot + gamma_ * h);
  }

  // feasibility 확인
  for (double m : result.cbf_margins) {
    if (m < -kTolerance * 10) {
      result.feasible = false;
      break;
    }
  }

  return result;
}

CLFCBFQPResult CLFCBFQPSolver::solveCLFOnly(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& x_des,
  const Eigen::VectorXd& u_ref,
  const BatchDynamicsWrapper& dynamics) const
{
  CLFCBFQPResult result;
  int nu = u_ref.size();

  result.clf_value = clf_->evaluate(state, x_des);

  Eigen::VectorXd u = clipToBounds(u_ref);
  double delta = 0.0;

  for (int iter = 0; iter < kMaxIterations; ++iter) {
    result.iterations = iter + 1;

    Eigen::VectorXd x_dot = computeXdot(state, u, dynamics);
    Eigen::MatrixXd B = computeB(state, u, x_dot, dynamics);
    Eigen::VectorXd grad_V = clf_->gradient(state, x_des);
    double V_dot = grad_V.dot(x_dot);
    double clf_margin = V_dot + clf_->c() * result.clf_value - delta;

    if (clf_margin <= kTolerance) {
      // CLF 만족 — u_ref 방향으로 step
      Eigen::VectorXd grad_obj = u - u_ref;
      Eigen::VectorXd u_candidate = u - kStepSize * grad_obj;
      u_candidate = clipToBounds(u_candidate);
      double delta_candidate = delta * (1.0 - kStepSize);

      Eigen::VectorXd x_dot_c = computeXdot(state, u_candidate, dynamics);
      double V_dot_c = grad_V.dot(x_dot_c);
      if (V_dot_c + clf_->c() * result.clf_value - delta_candidate <= kTolerance) {
        u = u_candidate;
        delta = delta_candidate;
      }
      break;
    }

    // CLF 보정
    Eigen::VectorXd dVdot_du = B.transpose() * grad_V;
    double norm_sq = dVdot_du.squaredNorm();
    if (norm_sq > 1e-10) {
      double scale = clf_margin / norm_sq;
      u = u - scale * dVdot_du;
      u = clipToBounds(u);
    } else {
      // gradient 0 → slack 증가
      delta += std::abs(clf_margin);
      break;
    }
  }

  result.u_safe = u;
  result.slack = delta;
  result.feasible = true;

  Eigen::VectorXd final_xdot = computeXdot(state, u, dynamics);
  Eigen::VectorXd final_grad_V = clf_->gradient(state, x_des);
  result.clf_constraint = final_grad_V.dot(final_xdot) +
    clf_->c() * result.clf_value - delta;

  return result;
}

}  // namespace mpc_controller_ros2
