#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <memory>

#include "mpc_controller_ros2/ilqr_solver.hpp"
#include "mpc_controller_ros2/diff_drive_model.hpp"
#include "mpc_controller_ros2/swerve_drive_model.hpp"
#include "mpc_controller_ros2/non_coaxial_swerve_model.hpp"
#include "mpc_controller_ros2/ackermann_model.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"

using namespace mpc_controller_ros2;

// =============================================================================
// Helper
// =============================================================================

static Eigen::MatrixXd createStraightReference(int N, int nx, double speed = 0.5, double dt = 0.1)
{
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int t = 0; t <= N; ++t) {
    ref(t, 0) = speed * t * dt;  // x
    // y, theta 등 나머지는 0
  }
  return ref;
}

// =============================================================================
// DiffDrive Jacobian 테스트
// =============================================================================

TEST(ILQRSolver, DiffDriveJacobian)
{
  DiffDriveModel model(0.0, 1.0, -1.0, 1.0);

  Eigen::VectorXd state(3);
  state << 1.0, 2.0, 0.5;
  Eigen::VectorXd control(2);
  control << 0.3, 0.2;
  double dt = 0.1;

  // 해석적 Jacobian
  auto lin = model.getLinearization(state, control, dt);

  // 유한차분 비교
  constexpr double eps = 1e-5;
  Eigen::MatrixXd s0(1, 3), c0(1, 2);
  s0.row(0) = state.transpose();
  c0.row(0) = control.transpose();
  Eigen::VectorXd f0 = model.propagateBatch(s0, c0, dt).row(0).transpose();

  Eigen::MatrixXd A_fd(3, 3);
  for (int j = 0; j < 3; ++j) {
    Eigen::MatrixXd s_p = s0;
    s_p(0, j) += eps;
    Eigen::VectorXd f_p = model.propagateBatch(s_p, c0, dt).row(0).transpose();
    A_fd.col(j) = (f_p - f0) / eps;
  }

  Eigen::MatrixXd B_fd(3, 2);
  for (int j = 0; j < 2; ++j) {
    Eigen::MatrixXd c_p = c0;
    c_p(0, j) += eps;
    Eigen::VectorXd f_p = model.propagateBatch(s0, c_p, dt).row(0).transpose();
    B_fd.col(j) = (f_p - f0) / eps;
  }

  // Euler vs RK4 mismatch → O(dt²) 오차 허용
  EXPECT_LT((lin.A - A_fd).norm(), 0.01)
    << "A diff:\n" << (lin.A - A_fd);
  EXPECT_LT((lin.B - B_fd).norm(), 0.01)
    << "B diff:\n" << (lin.B - B_fd);
}

// =============================================================================
// Ackermann Jacobian 테스트
// =============================================================================

TEST(ILQRSolver, AckermannJacobian)
{
  AckermannModel model(0.0, 1.0, 2.0, M_PI / 4.0, 0.5);

  Eigen::VectorXd state(4);
  state << 1.0, 2.0, 0.3, 0.2;
  Eigen::VectorXd control(2);
  control << 0.4, 0.1;
  double dt = 0.1;

  auto lin = model.getLinearization(state, control, dt);

  // 유한차분 비교
  constexpr double eps = 1e-5;
  Eigen::MatrixXd s0(1, 4), c0(1, 2);
  s0.row(0) = state.transpose();
  c0.row(0) = control.transpose();
  Eigen::VectorXd f0 = model.propagateBatch(s0, c0, dt).row(0).transpose();

  Eigen::MatrixXd A_fd(4, 4);
  for (int j = 0; j < 4; ++j) {
    Eigen::MatrixXd s_p = s0;
    s_p(0, j) += eps;
    Eigen::VectorXd f_p = model.propagateBatch(s_p, c0, dt).row(0).transpose();
    A_fd.col(j) = (f_p - f0) / eps;
  }

  Eigen::MatrixXd B_fd(4, 2);
  for (int j = 0; j < 2; ++j) {
    Eigen::MatrixXd c_p = c0;
    c_p(0, j) += eps;
    Eigen::VectorXd f_p = model.propagateBatch(s0, c_p, dt).row(0).transpose();
    B_fd.col(j) = (f_p - f0) / eps;
  }

  // Euler vs RK4 mismatch → O(dt²) 오차 허용
  EXPECT_LT((lin.A - A_fd).norm(), 0.01)
    << "A diff:\n" << (lin.A - A_fd);
  EXPECT_LT((lin.B - B_fd).norm(), 0.01)
    << "B diff:\n" << (lin.B - B_fd);
}

// =============================================================================
// BackwardPass 차원 검증
// =============================================================================

TEST(ILQRSolver, BackwardPassDims)
{
  DiffDriveModel model(0.0, 1.0, -1.0, 1.0);
  int nx = 3, nu = 2, N = 10;

  ILQRParams params;
  params.max_iterations = 1;
  ILQRSolver solver(params, nx, nu);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(N, nu);
  Eigen::MatrixXd ref = createStraightReference(N, nx);

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nx, nx) * 10.0;
  Eigen::MatrixXd Qf = Q * 2.0;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(nu, nu) * 0.1;

  double cost = solver.solve(x0, U, ref, model, Q, Qf, R, 0.1);

  // 비용이 유한해야 함
  EXPECT_TRUE(std::isfinite(cost));
  // 제어 시퀀스 차원 유지
  EXPECT_EQ(U.rows(), N);
  EXPECT_EQ(U.cols(), nu);
}

// =============================================================================
// ForwardPass 비용 개선
// =============================================================================

TEST(ILQRSolver, ForwardPassImproves)
{
  DiffDriveModel model(0.0, 1.0, -1.0, 1.0);
  int nx = 3, nu = 2, N = 20;

  ILQRParams params;
  params.max_iterations = 2;
  ILQRSolver solver(params, nx, nu);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(N, nu);
  Eigen::MatrixXd ref = createStraightReference(N, nx);

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nx, nx) * 10.0;
  Eigen::MatrixXd Qf = Q * 2.0;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(nu, nu) * 0.1;

  // 초기 비용 (u=0)
  Eigen::MatrixXd X0(N + 1, nx);
  X0.row(0) = x0.transpose();
  for (int t = 0; t < N; ++t) {
    Eigen::MatrixXd s(1, nx); s.row(0) = X0.row(t);
    Eigen::MatrixXd c(1, nu); c.row(0) = U.row(t);
    X0.row(t + 1) = model.propagateBatch(s, c, 0.1).row(0);
  }
  double cost0 = 0.0;
  for (int t = 0; t < N; ++t) {
    Eigen::VectorXd dx = X0.row(t).transpose() - ref.row(t).transpose();
    cost0 += 0.5 * dx.dot(Q * dx);
  }
  Eigen::VectorXd dxf = X0.row(N).transpose() - ref.row(N).transpose();
  cost0 += 0.5 * dxf.dot(Qf * dxf);

  double cost_after = solver.solve(x0, U, ref, model, Q, Qf, R, 0.1);

  EXPECT_LT(cost_after, cost0)
    << "iLQR should reduce cost. Before: " << cost0 << " After: " << cost_after;
}

// =============================================================================
// LineSearch 수렴
// =============================================================================

TEST(ILQRSolver, LineSearchAccepts)
{
  DiffDriveModel model(-0.5, 1.0, -1.0, 1.0);
  int nx = 3, nu = 2, N = 15;

  ILQRParams params;
  params.max_iterations = 3;
  params.line_search_steps = 4;
  ILQRSolver solver(params, nx, nu);

  Eigen::VectorXd x0(3);
  x0 << 0.0, 0.5, 0.0;  // 약간 off-track
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(N, nu);
  Eigen::MatrixXd ref = createStraightReference(N, nx);

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nx, nx) * 10.0;
  Eigen::MatrixXd Qf = Q * 2.0;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(nu, nu) * 0.1;

  double cost = solver.solve(x0, U, ref, model, Q, Qf, R, 0.1);
  EXPECT_TRUE(std::isfinite(cost));
  EXPECT_GT(cost, 0.0);
}

// =============================================================================
// WarmStart 수렴 (2회 반복 충분)
// =============================================================================

TEST(ILQRSolver, WarmStartConverges)
{
  DiffDriveModel model(0.0, 1.0, -1.0, 1.0);
  int nx = 3, nu = 2, N = 20;

  // 먼저 5회로 풀어서 "좋은 해" 확보
  ILQRParams params5;
  params5.max_iterations = 5;
  ILQRSolver solver5(params5, nx, nu);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd U5 = Eigen::MatrixXd::Zero(N, nu);
  Eigen::MatrixXd ref = createStraightReference(N, nx);

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nx, nx) * 10.0;
  Eigen::MatrixXd Qf = Q * 2.0;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(nu, nu) * 0.1;

  double cost5 = solver5.solve(x0, U5, ref, model, Q, Qf, R, 0.1);

  // warm-start: U5를 shift한 후 2회 반복
  Eigen::MatrixXd U_ws = Eigen::MatrixXd::Zero(N, nu);
  for (int t = 0; t < N - 1; ++t) {
    U_ws.row(t) = U5.row(t + 1);
  }

  ILQRParams params2;
  params2.max_iterations = 2;
  ILQRSolver solver2(params2, nx, nu);

  double cost2 = solver2.solve(x0, U_ws, ref, model, Q, Qf, R, 0.1);

  // 2회 반복으로도 5회와 비슷한 비용 달성
  EXPECT_LT(cost2, cost5 * 1.5)
    << "Warm-start 2-iter cost should be close to 5-iter. cost2=" << cost2 << " cost5=" << cost5;
}

// =============================================================================
// 정규화 효과
// =============================================================================

TEST(ILQRSolver, RegularizationEffect)
{
  DiffDriveModel model(0.0, 1.0, -1.0, 1.0);
  int nx = 3, nu = 2, N = 15;

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd ref = createStraightReference(N, nx);

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nx, nx) * 10.0;
  Eigen::MatrixXd Qf = Q * 2.0;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(nu, nu) * 0.1;

  // 낮은 정규화
  ILQRParams params_lo;
  params_lo.max_iterations = 1;  // 1회만 (업데이트 크기 비교에 최적)
  params_lo.regularization = 1e-6;
  ILQRSolver solver_lo(params_lo, nx, nu);
  Eigen::MatrixXd U_lo = Eigen::MatrixXd::Zero(N, nu);
  solver_lo.solve(x0, U_lo, ref, model, Q, Qf, R, 0.1);

  // 높은 정규화
  ILQRParams params_hi;
  params_hi.max_iterations = 1;
  params_hi.regularization = 100.0;
  ILQRSolver solver_hi(params_hi, nx, nu);
  Eigen::MatrixXd U_hi = Eigen::MatrixXd::Zero(N, nu);
  solver_hi.solve(x0, U_hi, ref, model, Q, Qf, R, 0.1);

  // 높은 정규화 → 더 보수적(작은) 제어 업데이트
  double norm_lo = U_lo.norm();
  double norm_hi = U_hi.norm();
  EXPECT_LE(norm_hi, norm_lo * 1.01)
    << "High reg should give smaller update. norm_hi=" << norm_hi << " norm_lo=" << norm_lo;
}

// =============================================================================
// 제어 한계 클리핑
// =============================================================================

TEST(ILQRSolver, ClipRespected)
{
  DiffDriveModel model(-0.2, 0.5, -0.8, 0.8);
  int nx = 3, nu = 2, N = 15;

  ILQRParams params;
  params.max_iterations = 3;
  ILQRSolver solver(params, nx, nu);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(N, nu);
  Eigen::MatrixXd ref = createStraightReference(N, nx, 1.0);  // 빠른 참조 → 큰 제어

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nx, nx) * 50.0;
  Eigen::MatrixXd Qf = Q * 2.0;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(nu, nu) * 0.01;

  solver.solve(x0, U, ref, model, Q, Qf, R, 0.1);

  for (int t = 0; t < N; ++t) {
    EXPECT_GE(U(t, 0), -0.2 - 1e-6) << "v below min at t=" << t;
    EXPECT_LE(U(t, 0), 0.5 + 1e-6) << "v above max at t=" << t;
    EXPECT_GE(U(t, 1), -0.8 - 1e-6) << "omega below min at t=" << t;
    EXPECT_LE(U(t, 1), 0.8 + 1e-6) << "omega above max at t=" << t;
  }
}

// =============================================================================
// 초기 제어 0에서 시작
// =============================================================================

TEST(ILQRSolver, ZeroInitialControl)
{
  DiffDriveModel model(0.0, 1.0, -1.0, 1.0);
  int nx = 3, nu = 2, N = 20;

  ILQRParams params;
  params.max_iterations = 3;
  ILQRSolver solver(params, nx, nu);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(N, nu);
  Eigen::MatrixXd ref = createStraightReference(N, nx);

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nx, nx) * 10.0;
  Eigen::MatrixXd Qf = Q * 2.0;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(nu, nu) * 0.1;

  double cost = solver.solve(x0, U, ref, model, Q, Qf, R, 0.1);
  EXPECT_TRUE(std::isfinite(cost));

  // iLQR 이후 v > 0 (전진 참조를 따라가야 함)
  bool has_positive_v = false;
  for (int t = 0; t < N; ++t) {
    if (U(t, 0) > 0.01) {
      has_positive_v = true;
      break;
    }
  }
  EXPECT_TRUE(has_positive_v) << "iLQR should produce non-zero forward velocity";
}

// =============================================================================
// 모든 모델 호환
// =============================================================================

TEST(ILQRSolver, AllModels)
{
  MPPIParams p;

  // DiffDrive
  {
    auto model = MotionModelFactory::create("diff_drive", p);
    int nx = model->stateDim(), nu = model->controlDim();
    ILQRSolver solver(ILQRParams{}, nx, nu);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(10, nu);
    Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(11, nx);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nx, nx);
    double cost = solver.solve(x0, U, ref, *model, Q, Q * 2, Eigen::MatrixXd::Identity(nu, nu) * 0.1, 0.1);
    EXPECT_TRUE(std::isfinite(cost)) << "DiffDrive failed";
  }

  // Swerve
  {
    auto model = MotionModelFactory::create("swerve", p);
    int nx = model->stateDim(), nu = model->controlDim();
    ILQRSolver solver(ILQRParams{}, nx, nu);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(10, nu);
    Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(11, nx);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nx, nx);
    double cost = solver.solve(x0, U, ref, *model, Q, Q * 2, Eigen::MatrixXd::Identity(nu, nu) * 0.1, 0.1);
    EXPECT_TRUE(std::isfinite(cost)) << "Swerve failed";
  }

  // NonCoaxialSwerve
  {
    auto model = MotionModelFactory::create("non_coaxial_swerve", p);
    int nx = model->stateDim(), nu = model->controlDim();
    ILQRSolver solver(ILQRParams{}, nx, nu);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(10, nu);
    Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(11, nx);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nx, nx);
    double cost = solver.solve(x0, U, ref, *model, Q, Q * 2, Eigen::MatrixXd::Identity(nu, nu) * 0.1, 0.1);
    EXPECT_TRUE(std::isfinite(cost)) << "NonCoaxialSwerve failed";
  }

  // Ackermann
  {
    auto model = MotionModelFactory::create("ackermann", p);
    int nx = model->stateDim(), nu = model->controlDim();
    ILQRSolver solver(ILQRParams{}, nx, nu);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(10, nu);
    Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(11, nx);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nx, nx);
    double cost = solver.solve(x0, U, ref, *model, Q, Q * 2, Eigen::MatrixXd::Identity(nu, nu) * 0.1, 0.1);
    EXPECT_TRUE(std::isfinite(cost)) << "Ackermann failed";
  }
}

// =============================================================================
// 성능 예산
// =============================================================================

TEST(ILQRSolver, PerfBudget)
{
  DiffDriveModel model(0.0, 1.0, -1.0, 1.0);
  int nx = 3, nu = 2, N = 30;

  ILQRParams params;
  params.max_iterations = 2;
  ILQRSolver solver(params, nx, nu);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(N, nu);
  Eigen::MatrixXd ref = createStraightReference(N, nx);

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nx, nx) * 10.0;
  Eigen::MatrixXd Qf = Q * 2.0;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(nu, nu) * 0.1;

  // 워밍업
  solver.solve(x0, U, ref, model, Q, Qf, R, 0.1);

  // 벤치마크
  int n_runs = 100;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n_runs; ++i) {
    U = Eigen::MatrixXd::Zero(N, nu);
    solver.solve(x0, U, ref, model, Q, Qf, R, 0.1);
  }
  auto end = std::chrono::high_resolution_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / n_runs;

  std::cout << "[PerfBudget] iLQR (2 iter, N=30): " << elapsed_ms << " ms/call" << std::endl;

  // Release 빌드에서 0.1ms 미만 (Debug에서는 느릴 수 있으므로 넉넉하게)
  EXPECT_LT(elapsed_ms, 5.0) << "iLQR too slow: " << elapsed_ms << " ms";
}

// =============================================================================
// Ackermann 전용: steering 동역학
// =============================================================================

TEST(ILQRSolver, AckermannSteering)
{
  AckermannModel model(0.0, 1.0, 2.0, M_PI / 4.0, 0.5);
  int nx = 4, nu = 2, N = 20;

  ILQRParams params;
  params.max_iterations = 3;
  ILQRSolver solver(params, nx, nu);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(N, nu);

  // 곡선 참조
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int t = 0; t <= N; ++t) {
    double s = 0.1 * t;
    ref(t, 0) = 2.0 * std::sin(0.3 * s);
    ref(t, 1) = s;
  }

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nx, nx) * 10.0;
  Q(3, 3) = 0.1;  // delta 추적은 약하게
  Eigen::MatrixXd Qf = Q * 2.0;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(nu, nu) * 0.1;

  double cost = solver.solve(x0, U, ref, model, Q, Qf, R, 0.1);
  EXPECT_TRUE(std::isfinite(cost));
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
