#include "mpc_controller_ros2/ilqr_solver.hpp"
#include <cmath>
#include <algorithm>

namespace mpc_controller_ros2
{

ILQRSolver::ILQRSolver(const ILQRParams& params, int nx, int nu)
: params_(params), nx_(nx), nu_(nu)
{
  V_x_ = Eigen::VectorXd::Zero(nx);
  V_xx_ = Eigen::MatrixXd::Zero(nx, nx);
}

double ILQRSolver::solve(
  const Eigen::VectorXd& x0,
  Eigen::MatrixXd& control_sequence,
  const Eigen::MatrixXd& reference,
  const MotionModel& model,
  const Eigen::MatrixXd& Q,
  const Eigen::MatrixXd& Qf,
  const Eigen::MatrixXd& R,
  double dt)
{
  int N = control_sequence.rows();

  // 가중치 행렬 저장
  Q_ = Q;
  Qf_ = Qf;
  R_ = R;

  // 사전 할당 (첫 호출 시 또는 N 변경 시)
  if (static_cast<int>(k_.size()) != N) {
    k_.resize(N);
    K_.resize(N);
    for (int t = 0; t < N; ++t) {
      k_[t] = Eigen::VectorXd::Zero(nu_);
      K_[t] = Eigen::MatrixXd::Zero(nu_, nx_);
    }
    X_new_.resize(N + 1, nx_);
    U_new_.resize(N, nu_);
  }

  // 현재 nominal 궤적 생성
  Eigen::MatrixXd X_bar(N + 1, nx_);
  rolloutNominal(x0, control_sequence, model, dt, X_bar);

  double current_cost = computeTrajectoryCost(X_bar, control_sequence, reference);

  for (int iter = 0; iter < params_.max_iterations; ++iter) {
    // Backward pass
    backwardPass(X_bar, control_sequence, reference, model, dt);

    // Forward pass with line search
    bool accepted = false;
    static constexpr double alphas[] = {1.0, 0.5, 0.25, 0.1};
    int n_alphas = std::min(params_.line_search_steps, 4);

    for (int i = 0; i < n_alphas; ++i) {
      double new_cost = forwardPass(x0, X_bar, control_sequence, model, dt, alphas[i]);
      if (new_cost >= 0.0 && new_cost < current_cost) {
        // 수락: X_new_, U_new_ → X_bar, control_sequence
        X_bar = X_new_;
        control_sequence = U_new_;
        double improvement = (current_cost - new_cost) / (std::abs(current_cost) + 1e-10);
        current_cost = new_cost;
        accepted = true;

        // 수렴 판정
        if (improvement < params_.cost_tolerance) {
          return current_cost;
        }
        break;
      }
    }

    if (!accepted) {
      // 모든 alpha에서 개선 실패 → 현재 해 유지
      break;
    }
  }

  return current_cost;
}

void ILQRSolver::rolloutNominal(
  const Eigen::VectorXd& x0,
  const Eigen::MatrixXd& U,
  const MotionModel& model,
  double dt,
  Eigen::MatrixXd& X_out) const
{
  int N = U.rows();
  X_out.row(0) = x0.transpose();

  for (int t = 0; t < N; ++t) {
    Eigen::MatrixXd s(1, nx_);
    s.row(0) = X_out.row(t);
    Eigen::MatrixXd c(1, nu_);
    c.row(0) = U.row(t);
    X_out.row(t + 1) = model.propagateBatch(s, c, dt).row(0);
  }
}

double ILQRSolver::backwardPass(
  const Eigen::MatrixXd& X_bar,
  const Eigen::MatrixXd& U_bar,
  const Eigen::MatrixXd& ref,
  const MotionModel& model,
  double dt)
{
  int N = U_bar.rows();

  // 터미널 비용 gradient/Hessian
  Eigen::VectorXd x_err = X_bar.row(N).transpose() - ref.row(std::min(N, static_cast<int>(ref.rows()) - 1)).transpose();
  // 각도 정규화 (angle indices)
  auto angle_idx = model.angleIndices();
  for (int idx : angle_idx) {
    if (idx < x_err.size()) {
      x_err(idx) = std::atan2(std::sin(x_err(idx)), std::cos(x_err(idx)));
    }
  }

  V_x_ = Qf_ * x_err;
  V_xx_ = Qf_;

  double dV = 0.0;

  for (int t = N - 1; t >= 0; --t) {
    Eigen::VectorXd x_t = X_bar.row(t).transpose();
    Eigen::VectorXd u_t = U_bar.row(t).transpose();

    // 선형화
    Linearization lin = model.getLinearization(x_t, u_t, dt);
    const Eigen::MatrixXd& A_t = lin.A;
    const Eigen::MatrixXd& B_t = lin.B;

    // 스테이지 비용 gradient/Hessian
    int ref_idx = std::min(t, static_cast<int>(ref.rows()) - 1);
    Eigen::VectorXd dx = x_t - ref.row(ref_idx).transpose();
    for (int idx : angle_idx) {
      if (idx < dx.size()) {
        dx(idx) = std::atan2(std::sin(dx(idx)), std::cos(dx(idx)));
      }
    }

    Eigen::VectorXd l_x = Q_ * dx;
    Eigen::VectorXd l_u = R_ * u_t;
    // l_xx = Q_, l_uu = R_, l_ux = 0

    // Q-function 확장
    Eigen::VectorXd Q_x = l_x + A_t.transpose() * V_x_;
    Eigen::VectorXd Q_u = l_u + B_t.transpose() * V_x_;
    Eigen::MatrixXd Q_xx = Q_ + A_t.transpose() * V_xx_ * A_t;
    Eigen::MatrixXd Q_ux = B_t.transpose() * V_xx_ * A_t;
    Eigen::MatrixXd Q_uu = R_ + B_t.transpose() * V_xx_ * B_t;

    // 정규화
    Q_uu.diagonal().array() += params_.regularization;

    // nu=2 → 2x2 직접 역행렬 (성능), 그 외 LDLT
    Eigen::MatrixXd Q_uu_inv;
    if (nu_ == 2) {
      double a = Q_uu(0, 0), b = Q_uu(0, 1);
      double c = Q_uu(1, 0), d = Q_uu(1, 1);
      double det = a * d - b * c;
      Q_uu_inv.resize(2, 2);
      Q_uu_inv(0, 0) = d / det;
      Q_uu_inv(0, 1) = -b / det;
      Q_uu_inv(1, 0) = -c / det;
      Q_uu_inv(1, 1) = a / det;
    } else {
      Q_uu_inv = Q_uu.ldlt().solve(Eigen::MatrixXd::Identity(nu_, nu_));
    }

    // 피드포워드 및 피드백 게인
    k_[t] = -Q_uu_inv * Q_u;
    K_[t] = -Q_uu_inv * Q_ux;

    // 값 함수 업데이트
    V_x_ = Q_x + K_[t].transpose() * Q_uu * k_[t]
          + K_[t].transpose() * Q_u + Q_ux.transpose() * k_[t];
    V_xx_ = Q_xx + K_[t].transpose() * Q_uu * K_[t]
           + K_[t].transpose() * Q_ux + Q_ux.transpose() * K_[t];
    // 대칭화 (수치 안정성)
    V_xx_ = 0.5 * (V_xx_ + V_xx_.transpose());

    dV += k_[t].dot(Q_u);
  }

  return dV;
}

double ILQRSolver::forwardPass(
  const Eigen::VectorXd& x0,
  const Eigen::MatrixXd& X_bar,
  const Eigen::MatrixXd& U_bar,
  const MotionModel& model,
  double dt,
  double alpha)
{
  int N = U_bar.rows();

  X_new_.row(0) = x0.transpose();

  for (int t = 0; t < N; ++t) {
    Eigen::VectorXd dx = X_new_.row(t).transpose() - X_bar.row(t).transpose();

    // 각도 정규화
    auto angle_idx = model.angleIndices();
    for (int idx : angle_idx) {
      if (idx < dx.size()) {
        dx(idx) = std::atan2(std::sin(dx(idx)), std::cos(dx(idx)));
      }
    }

    // u_new = u_bar + alpha*k + K*dx
    U_new_.row(t) = (U_bar.row(t).transpose() + alpha * k_[t] + K_[t] * dx).transpose();

    // 제어 클리핑
    Eigen::MatrixXd u_mat(1, nu_);
    u_mat.row(0) = U_new_.row(t);
    U_new_.row(t) = model.clipControls(u_mat).row(0);

    // 상태 전파
    Eigen::MatrixXd s(1, nx_);
    s.row(0) = X_new_.row(t);
    Eigen::MatrixXd c(1, nu_);
    c.row(0) = U_new_.row(t);
    X_new_.row(t + 1) = model.propagateBatch(s, c, dt).row(0);
  }

  return computeTrajectoryCost(X_new_, U_new_, X_bar);  // ref = X_bar의 참조 사용
}

double ILQRSolver::computeTrajectoryCost(
  const Eigen::MatrixXd& X,
  const Eigen::MatrixXd& U,
  const Eigen::MatrixXd& ref) const
{
  int N = U.rows();
  double cost = 0.0;

  for (int t = 0; t < N; ++t) {
    int ref_idx = std::min(t, static_cast<int>(ref.rows()) - 1);
    Eigen::VectorXd dx = X.row(t).transpose() - ref.row(ref_idx).transpose();
    Eigen::VectorXd u = U.row(t).transpose();
    cost += 0.5 * dx.dot(Q_ * dx) + 0.5 * u.dot(R_ * u);
  }

  // 터미널 비용
  int ref_idx = std::min(N, static_cast<int>(ref.rows()) - 1);
  Eigen::VectorXd dx_f = X.row(N).transpose() - ref.row(ref_idx).transpose();
  cost += 0.5 * dx_f.dot(Qf_ * dx_f);

  return cost;
}

}  // namespace mpc_controller_ros2
