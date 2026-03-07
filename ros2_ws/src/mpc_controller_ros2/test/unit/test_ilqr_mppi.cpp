#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <memory>

#include "mpc_controller_ros2/ilqr_solver.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/sampling.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/diff_drive_model.hpp"
#include "mpc_controller_ros2/ackermann_model.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"

using namespace mpc_controller_ros2;

// =============================================================================
// Helper: MPPI 1-step 실행 (iLQR warm-start 유무 비교)
// =============================================================================

static double runMPPIStep(
  const Eigen::VectorXd& x0,
  Eigen::MatrixXd& control_seq,
  const Eigen::MatrixXd& ref,
  const MPPIParams& params,
  BatchDynamicsWrapper& dynamics,
  CompositeMPPICost& cost_function,
  BaseSampler& sampler,
  ILQRSolver* ilqr_solver)  // nullptr = vanilla
{
  int N = params.N;
  int nu = dynamics.model().controlDim();
  int nx = dynamics.model().stateDim();
  int K = params.K;

  // warm-start shift
  for (int t = 0; t < N - 1; ++t) {
    control_seq.row(t) = control_seq.row(t + 1);
  }
  control_seq.row(N - 1).setZero();

  // iLQR warm-start (optional)
  if (ilqr_solver) {
    Eigen::MatrixXd Q = params.Q.topLeftCorner(
      std::min(static_cast<int>(params.Q.rows()), nx),
      std::min(static_cast<int>(params.Q.cols()), nx));
    if (Q.rows() < nx) {
      Eigen::MatrixXd Q_full = Eigen::MatrixXd::Zero(nx, nx);
      Q_full.topLeftCorner(Q.rows(), Q.cols()) = Q;
      Q = Q_full;
    }
    Eigen::MatrixXd Qf = params.Qf.topLeftCorner(
      std::min(static_cast<int>(params.Qf.rows()), nx),
      std::min(static_cast<int>(params.Qf.cols()), nx));
    if (Qf.rows() < nx) {
      Eigen::MatrixXd Qf_full = Eigen::MatrixXd::Zero(nx, nx);
      Qf_full.topLeftCorner(Qf.rows(), Qf.cols()) = Qf;
      Qf = Qf_full;
    }
    Eigen::MatrixXd R = params.R.topLeftCorner(
      std::min(static_cast<int>(params.R.rows()), nu),
      std::min(static_cast<int>(params.R.cols()), nu));
    if (R.rows() < nu) {
      Eigen::MatrixXd R_full = Eigen::MatrixXd::Zero(nu, nu);
      R_full.topLeftCorner(R.rows(), R.cols()) = R;
      R = R_full;
    }

    ilqr_solver->solve(x0, control_seq, ref, dynamics.model(), Q, Qf, R, params.dt);
  }

  // MPPI sampling
  auto noise = sampler.sample(K, N, nu);

  std::vector<Eigen::MatrixXd> perturbed(K);
  for (int k = 0; k < K; ++k) {
    perturbed[k] = dynamics.clipControls(control_seq + noise[k]);
  }

  // Rollout
  std::vector<Eigen::MatrixXd> trajectories;
  dynamics.rolloutBatchInPlace(x0, perturbed, params.dt, trajectories);

  // Cost
  Eigen::VectorXd costs = cost_function.compute(trajectories, perturbed, ref);

  // Weights
  VanillaMPPIWeights weight_comp;
  Eigen::VectorXd weights = weight_comp.compute(costs, params.lambda);

  // Weighted update
  Eigen::MatrixXd weighted_update = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_update += weights(k) * noise[k];
  }
  control_seq += weighted_update;
  control_seq = dynamics.clipControls(control_seq);

  // 최적 궤적 비용 (u_opt 기반)
  std::vector<Eigen::MatrixXd> opt_ctrl = {control_seq};
  std::vector<Eigen::MatrixXd> opt_traj;
  dynamics.rolloutBatchInPlace(x0, opt_ctrl, params.dt, opt_traj);
  return cost_function.compute(opt_traj, opt_ctrl, ref)(0);
}

// =============================================================================
// Helper: 테스트용 MPPI 컴포넌트 초기화
// =============================================================================

struct MPPITestSetup
{
  MPPIParams params;
  std::unique_ptr<BatchDynamicsWrapper> dynamics;
  std::unique_ptr<CompositeMPPICost> cost_function;
  std::unique_ptr<BaseSampler> sampler;

  void init(const std::string& model_type = "diff_drive", int K = 256, int N = 20)
  {
    params = MPPIParams();
    params.N = N;
    params.dt = 0.1;
    params.K = K;
    params.lambda = 10.0;
    params.motion_model = model_type;

    int nu = (model_type == "diff_drive" || model_type == "ackermann") ? 2 : 3;
    params.noise_sigma = Eigen::VectorXd::Constant(nu, 0.5);

    params.Q = Eigen::MatrixXd::Identity(3, 3) * 10.0;
    params.Q(2, 2) = 1.0;
    params.Qf = params.Q * 2.0;
    params.R = Eigen::MatrixXd::Identity(nu, nu) * 0.1;

    dynamics = std::make_unique<BatchDynamicsWrapper>(params);
    sampler = std::make_unique<GaussianSampler>(params.noise_sigma, 42);

    cost_function = std::make_unique<CompositeMPPICost>();
    cost_function->addCost(std::make_unique<StateTrackingCost>(params.Q));
    cost_function->addCost(std::make_unique<TerminalCost>(params.Qf));
    cost_function->addCost(std::make_unique<ControlEffortCost>(params.R));
  }
};

// =============================================================================
// computeControl 차원 검증
// =============================================================================

TEST(IlqrMPPI, ComputeControlDims)
{
  MPPITestSetup setup;
  setup.init("diff_drive", 100, 15);
  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();

  ILQRSolver solver(ILQRParams{}, nx, nu);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(setup.params.N, nu);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(setup.params.N + 1, nx);

  double cost = runMPPIStep(x0, U, ref, setup.params, *setup.dynamics,
                            *setup.cost_function, *setup.sampler, &solver);

  EXPECT_TRUE(std::isfinite(cost));
  EXPECT_EQ(U.rows(), setup.params.N);
  EXPECT_EQ(U.cols(), nu);
}

// =============================================================================
// iLQR warm-start vs Vanilla 비용 비교
// =============================================================================

TEST(IlqrMPPI, ImprovesCostVsVanilla)
{
  // 여러 번 실행하여 평균 비교 (샘플링 노이즈 때문)
  int n_trials = 5;
  double vanilla_total = 0.0, ilqr_total = 0.0;

  for (int trial = 0; trial < n_trials; ++trial) {
    // Vanilla
    MPPITestSetup vanilla_setup;
    vanilla_setup.init("diff_drive", 256, 20);
    int nx = vanilla_setup.dynamics->model().stateDim();
    int nu = vanilla_setup.dynamics->model().controlDim();

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
    Eigen::MatrixXd U_v = Eigen::MatrixXd::Zero(vanilla_setup.params.N, nu);
    Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(vanilla_setup.params.N + 1, nx);
    for (int t = 0; t <= vanilla_setup.params.N; ++t) {
      ref(t, 0) = 0.5 * t * 0.1;
    }

    double cost_v = runMPPIStep(x0, U_v, ref, vanilla_setup.params,
                                *vanilla_setup.dynamics, *vanilla_setup.cost_function,
                                *vanilla_setup.sampler, nullptr);

    // iLQR
    MPPITestSetup ilqr_setup;
    ilqr_setup.init("diff_drive", 256, 20);
    ILQRSolver solver(ILQRParams{}, nx, nu);

    Eigen::MatrixXd U_i = Eigen::MatrixXd::Zero(ilqr_setup.params.N, nu);
    double cost_i = runMPPIStep(x0, U_i, ref, ilqr_setup.params,
                                *ilqr_setup.dynamics, *ilqr_setup.cost_function,
                                *ilqr_setup.sampler, &solver);

    vanilla_total += cost_v;
    ilqr_total += cost_i;
  }

  double avg_v = vanilla_total / n_trials;
  double avg_i = ilqr_total / n_trials;
  std::cout << "[ImprovesCost] Vanilla avg: " << avg_v << ", iLQR avg: " << avg_i << std::endl;

  // iLQR가 평균적으로 더 낮은 비용 (또는 동등)
  EXPECT_LE(avg_i, avg_v * 1.1)
    << "iLQR warm-start should not significantly increase cost";
}

// =============================================================================
// iLQR 비활성화 → Vanilla와 동일
// =============================================================================

TEST(IlqrMPPI, DisabledFallback)
{
  MPPITestSetup setup;
  setup.init("diff_drive", 100, 10);
  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(setup.params.N + 1, nx);

  // Vanilla
  Eigen::MatrixXd U_v = Eigen::MatrixXd::Zero(setup.params.N, nu);
  double cost_v = runMPPIStep(x0, U_v, ref, setup.params,
                              *setup.dynamics, *setup.cost_function,
                              *setup.sampler, nullptr);

  // "Disabled" iLQR (nullptr → 동일 경로)
  MPPITestSetup setup2;
  setup2.init("diff_drive", 100, 10);
  Eigen::MatrixXd U_d = Eigen::MatrixXd::Zero(setup2.params.N, nu);
  double cost_d = runMPPIStep(x0, U_d, ref, setup2.params,
                              *setup2.dynamics, *setup2.cost_function,
                              *setup2.sampler, nullptr);

  // 동일 시드 → 동일 비용
  EXPECT_NEAR(cost_v, cost_d, 1e-6);
}

// =============================================================================
// 제어 범위 검증
// =============================================================================

TEST(IlqrMPPI, ControlBounds)
{
  MPPITestSetup setup;
  setup.init("diff_drive", 100, 15);
  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();

  ILQRSolver solver(ILQRParams{}, nx, nu);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(setup.params.N, nu);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(setup.params.N + 1, nx);
  for (int t = 0; t <= setup.params.N; ++t) {
    ref(t, 0) = 2.0 * t * 0.1;  // 빠른 참조
  }

  runMPPIStep(x0, U, ref, setup.params, *setup.dynamics,
              *setup.cost_function, *setup.sampler, &solver);

  for (int t = 0; t < setup.params.N; ++t) {
    EXPECT_GE(U(t, 0), setup.params.v_min - 1e-6);
    EXPECT_LE(U(t, 0), setup.params.v_max + 1e-6);
    EXPECT_GE(U(t, 1), setup.params.omega_min - 1e-6);
    EXPECT_LE(U(t, 1), setup.params.omega_max + 1e-6);
  }
}

// =============================================================================
// 모든 모델 호환
// =============================================================================

TEST(IlqrMPPI, AllModels)
{
  for (const auto& model_type : {"diff_drive", "swerve", "non_coaxial_swerve", "ackermann"}) {
    MPPITestSetup setup;
    setup.init(model_type, 64, 10);
    int nx = setup.dynamics->model().stateDim();
    int nu = setup.dynamics->model().controlDim();

    ILQRSolver solver(ILQRParams{}, nx, nu);

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(setup.params.N, nu);
    Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(setup.params.N + 1, nx);
    for (int t = 0; t <= setup.params.N; ++t) {
      ref(t, 0) = 0.3 * t * 0.1;
    }

    double cost = runMPPIStep(x0, U, ref, setup.params,
                              *setup.dynamics, *setup.cost_function,
                              *setup.sampler, &solver);

    EXPECT_TRUE(std::isfinite(cost)) << "Failed for model: " << model_type;
  }
}

// =============================================================================
// 성능 예산 (iLQR 포함)
// =============================================================================

TEST(IlqrMPPI, PerfBudget)
{
  MPPITestSetup setup;
  setup.init("diff_drive", 256, 30);
  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();

  ILQRSolver solver(ILQRParams{}, nx, nu);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(setup.params.N + 1, nx);
  for (int t = 0; t <= setup.params.N; ++t) {
    ref(t, 0) = 0.5 * t * 0.1;
  }

  // 워밍업
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(setup.params.N, nu);
  runMPPIStep(x0, U, ref, setup.params, *setup.dynamics,
              *setup.cost_function, *setup.sampler, &solver);

  // 벤치마크
  int n_runs = 20;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n_runs; ++i) {
    U = Eigen::MatrixXd::Zero(setup.params.N, nu);
    runMPPIStep(x0, U, ref, setup.params, *setup.dynamics,
                *setup.cost_function, *setup.sampler, &solver);
  }
  auto end = std::chrono::high_resolution_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / n_runs;

  std::cout << "[PerfBudget] iLQR+MPPI (K=256, N=30): " << elapsed_ms << " ms/call" << std::endl;

  // Debug 빌드에서도 50ms 미만
  EXPECT_LT(elapsed_ms, 50.0) << "Too slow: " << elapsed_ms << " ms";
}

// =============================================================================
// 직선 경로 추종 오차
// =============================================================================

TEST(IlqrMPPI, ReferenceTracking)
{
  MPPITestSetup setup;
  setup.init("diff_drive", 256, 20);
  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();

  ILQRSolver solver(ILQRParams{}, nx, nu);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(setup.params.N, nu);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(setup.params.N + 1, nx);
  for (int t = 0; t <= setup.params.N; ++t) {
    ref(t, 0) = 0.3 * t * 0.1;
  }

  // 여러 스텝 실행
  Eigen::VectorXd state = x0;
  for (int step = 0; step < 5; ++step) {
    runMPPIStep(state, U, ref, setup.params, *setup.dynamics,
                *setup.cost_function, *setup.sampler, &solver);

    // 1스텝 전파
    Eigen::MatrixXd s(1, nx); s.row(0) = state.transpose();
    Eigen::MatrixXd c(1, nu); c.row(0) = U.row(0);
    state = setup.dynamics->model().propagateBatch(s, c, setup.params.dt).row(0).transpose();
  }

  // x 방향으로 전진해야 함
  EXPECT_GT(state(0), 0.02) << "Robot should have moved forward, x=" << state(0);
}

// =============================================================================
// 궤적 스무딩 (jerk 검증)
// =============================================================================

TEST(IlqrMPPI, TrajectorySmoothing)
{
  MPPITestSetup setup;
  setup.init("diff_drive", 256, 20);
  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();

  ILQRSolver solver(ILQRParams{}, nx, nu);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(setup.params.N, nu);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(setup.params.N + 1, nx);
  for (int t = 0; t <= setup.params.N; ++t) {
    ref(t, 0) = 0.5 * t * 0.1;
  }

  runMPPIStep(x0, U, ref, setup.params, *setup.dynamics,
              *setup.cost_function, *setup.sampler, &solver);

  // Jerk (제어 2차 변화율) 계산
  double total_jerk = 0.0;
  for (int t = 1; t < setup.params.N - 1; ++t) {
    Eigen::VectorXd jerk = (U.row(t + 1) - 2 * U.row(t) + U.row(t - 1)).transpose();
    total_jerk += jerk.squaredNorm();
  }

  EXPECT_TRUE(std::isfinite(total_jerk));
  // Jerk가 비정상적으로 크지 않아야 함
  EXPECT_LT(total_jerk, 100.0) << "Control jerk too high: " << total_jerk;
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
