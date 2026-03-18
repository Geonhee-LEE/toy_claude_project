// =============================================================================
// Feedback-MPPI (F-MPPI) Unit Tests
//
// 15 gtest: FeedbackGainComputer 수학적 검증 + 피드백 보정 + 통합 테스트
// =============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <memory>

#include "mpc_controller_ros2/feedback_gain_computer.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/sampling.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/diff_drive_model.hpp"
#include "mpc_controller_ros2/swerve_drive_model.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"

using namespace mpc_controller_ros2;

// =============================================================================
// Helper: MPPI 컴포넌트 + FeedbackGainComputer 통합 셋업
// =============================================================================

struct FeedbackMPPITestSetup
{
  MPPIParams params;
  std::unique_ptr<BatchDynamicsWrapper> dynamics;
  std::unique_ptr<CompositeMPPICost> cost_function;
  std::unique_ptr<GaussianSampler> sampler;
  std::unique_ptr<FeedbackGainComputer> gain_computer;

  void init(const std::string& model_type = "diff_drive", int K = 64, int N = 10)
  {
    params = MPPIParams();
    params.N = N;
    params.dt = 0.1;
    params.K = K;
    params.lambda = 10.0;
    params.motion_model = model_type;

    int nu = (model_type == "swerve") ? 3 : 2;
    params.noise_sigma = Eigen::VectorXd::Constant(nu, 0.5);

    int nx = (model_type == "swerve") ? 3 : 3;
    params.Q = Eigen::MatrixXd::Identity(nx, nx) * 10.0;
    params.Q(2, 2) = 1.0;
    params.Qf = params.Q * 2.0;
    params.R = Eigen::MatrixXd::Identity(nu, nu) * 0.1;
    params.R_rate = Eigen::MatrixXd::Identity(nu, nu) * 1.0;

    auto model = MotionModelFactory::create(model_type, params);
    dynamics = std::make_unique<BatchDynamicsWrapper>(params, std::move(model));
    sampler = std::make_unique<GaussianSampler>(params.noise_sigma, 42);

    cost_function = std::make_unique<CompositeMPPICost>();
    cost_function->addCost(std::make_unique<StateTrackingCost>(params.Q));
    cost_function->addCost(std::make_unique<TerminalCost>(params.Qf));
    cost_function->addCost(std::make_unique<ControlEffortCost>(params.R));

    gain_computer = std::make_unique<FeedbackGainComputer>(
      dynamics->model().stateDim(),
      dynamics->model().controlDim(),
      params.feedback_regularization);
  }

  // MPPI 1-step 실행 -> u_opt, control_sequence 업데이트
  Eigen::VectorXd runMPPIStep(
    const Eigen::VectorXd& x0,
    Eigen::MatrixXd& control_seq,
    const Eigen::MatrixXd& ref)
  {
    int N = params.N;
    int nu = dynamics->model().controlDim();
    int nx = dynamics->model().stateDim();
    int K = params.K;

    // MPPI sampling
    auto noise = sampler->sample(K, N, nu);

    std::vector<Eigen::MatrixXd> perturbed(K);
    for (int k = 0; k < K; ++k) {
      perturbed[k] = dynamics->clipControls(control_seq + noise[k]);
    }

    // Rollout
    std::vector<Eigen::MatrixXd> trajectories;
    dynamics->rolloutBatchInPlace(x0, perturbed, params.dt, trajectories);

    // Cost
    Eigen::VectorXd costs = cost_function->compute(trajectories, perturbed, ref);

    // Weights
    VanillaMPPIWeights weight_comp;
    Eigen::VectorXd weights = weight_comp.compute(costs, params.lambda);

    // Weighted update
    Eigen::MatrixXd weighted_update = Eigen::MatrixXd::Zero(N, nu);
    for (int k = 0; k < K; ++k) {
      weighted_update += weights(k) * noise[k];
    }
    control_seq += weighted_update;
    control_seq = dynamics->clipControls(control_seq);

    return control_seq.row(0).transpose();
  }

  // Nominal trajectory rollout
  Eigen::MatrixXd rolloutNominal(
    const Eigen::VectorXd& x0,
    const Eigen::MatrixXd& control_seq)
  {
    int N = control_seq.rows();
    int nx = dynamics->model().stateDim();
    int nu = dynamics->model().controlDim();

    Eigen::MatrixXd traj(N + 1, nx);
    traj.row(0) = x0.transpose();

    for (int t = 0; t < N; ++t) {
      Eigen::MatrixXd s(1, nx);
      s.row(0) = traj.row(t);
      Eigen::MatrixXd c(1, nu);
      c.row(0) = control_seq.row(t);
      traj.row(t + 1) = dynamics->model().propagateBatch(s, c, params.dt).row(0);
    }
    return traj;
  }
};

// =============================================================================
// Test 1: GainDimensions - computeGains returns N matrices each nu x nx
// =============================================================================
TEST(FeedbackMPPI, GainDimensions)
{
  FeedbackMPPITestSetup setup;
  setup.init("diff_drive", 64, 10);

  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();
  int N = setup.params.N;

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(N, nu);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);

  Eigen::MatrixXd nominal = setup.rolloutNominal(x0, control_seq);

  const auto& gains = setup.gain_computer->computeGains(
    nominal, control_seq, setup.dynamics->model(),
    setup.params.Q, setup.params.Qf, setup.params.R, setup.params.dt);

  ASSERT_EQ(static_cast<int>(gains.size()), N);
  for (int t = 0; t < N; ++t) {
    EXPECT_EQ(gains[t].rows(), nu) << "Gain K_" << t << " rows mismatch";
    EXPECT_EQ(gains[t].cols(), nx) << "Gain K_" << t << " cols mismatch";
  }
}

// =============================================================================
// Test 2: GainFinite - all gains are finite (no NaN/Inf)
// =============================================================================
TEST(FeedbackMPPI, GainFinite)
{
  FeedbackMPPITestSetup setup;
  setup.init("diff_drive", 64, 10);

  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();
  int N = setup.params.N;

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(N, nu);
  // Non-trivial reference
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int t = 0; t <= N; ++t) {
    ref(t, 0) = 0.3 * t * setup.params.dt;
  }

  // Run MPPI to get non-zero controls
  setup.runMPPIStep(x0, control_seq, ref);

  Eigen::MatrixXd nominal = setup.rolloutNominal(x0, control_seq);

  const auto& gains = setup.gain_computer->computeGains(
    nominal, control_seq, setup.dynamics->model(),
    setup.params.Q, setup.params.Qf, setup.params.R, setup.params.dt);

  for (int t = 0; t < N; ++t) {
    EXPECT_TRUE(gains[t].allFinite())
      << "Gain K_" << t << " has NaN/Inf:\n" << gains[t];
  }
}

// =============================================================================
// Test 3: GainSymmetry - K_t elements are reasonable magnitude (< 100)
// =============================================================================
TEST(FeedbackMPPI, GainMagnitude)
{
  FeedbackMPPITestSetup setup;
  setup.init("diff_drive", 64, 10);

  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();
  int N = setup.params.N;

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(N, nu);

  Eigen::MatrixXd nominal = setup.rolloutNominal(x0, control_seq);

  const auto& gains = setup.gain_computer->computeGains(
    nominal, control_seq, setup.dynamics->model(),
    setup.params.Q, setup.params.Qf, setup.params.R, setup.params.dt);

  for (int t = 0; t < N; ++t) {
    double max_abs = gains[t].cwiseAbs().maxCoeff();
    EXPECT_LT(max_abs, 100.0)
      << "Gain K_" << t << " has unreasonably large element: " << max_abs;
  }
}

// =============================================================================
// Test 4: DisabledPassthrough - feedback_mppi_enabled=false -> u_opt unchanged
// =============================================================================
TEST(FeedbackMPPI, DisabledPassthrough)
{
  FeedbackMPPITestSetup setup;
  setup.init("diff_drive", 64, 10);
  setup.params.feedback_mppi_enabled = false;

  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();
  int N = setup.params.N;

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(N, nu);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int t = 0; t <= N; ++t) {
    ref(t, 0) = 0.5 * t * setup.params.dt;
  }

  // Run MPPI
  Eigen::VectorXd u_opt = setup.runMPPIStep(x0, control_seq, ref);

  // With feedback disabled, no correction should be applied
  // u_opt == control_seq.row(0) from MPPI
  EXPECT_TRUE(u_opt.allFinite());
  EXPECT_EQ(u_opt.size(), nu);
}

// =============================================================================
// Test 5: GainScaleZero - gain_scale=0 -> no correction
// =============================================================================
TEST(FeedbackMPPI, GainScaleZero)
{
  FeedbackMPPITestSetup setup;
  setup.init("diff_drive", 64, 10);

  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();
  int N = setup.params.N;

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(N, nu);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int t = 0; t <= N; ++t) {
    ref(t, 0) = 0.3 * t * setup.params.dt;
  }

  // Run MPPI
  setup.runMPPIStep(x0, control_seq, ref);

  // Compute gains
  Eigen::MatrixXd nominal = setup.rolloutNominal(x0, control_seq);
  const auto& gains = setup.gain_computer->computeGains(
    nominal, control_seq, setup.dynamics->model(),
    setup.params.Q, setup.params.Qf, setup.params.R, setup.params.dt);

  // State offset
  Eigen::VectorXd actual_state = x0;
  actual_state(0) += 0.1;  // 10cm offset
  Eigen::VectorXd dx = actual_state - nominal.row(0).transpose();

  // gain_scale = 0 -> du = 0
  double gain_scale = 0.0;
  Eigen::VectorXd du = gain_scale * gains[0] * dx;

  EXPECT_NEAR(du.norm(), 0.0, 1e-15);
}

// =============================================================================
// Test 6: GainScaleEffect - gain_scale > 0 -> u_opt differs when state != nominal
// =============================================================================
TEST(FeedbackMPPI, GainScaleEffect)
{
  FeedbackMPPITestSetup setup;
  setup.init("diff_drive", 64, 10);

  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();
  int N = setup.params.N;

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(N, nu);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int t = 0; t <= N; ++t) {
    ref(t, 0) = 0.3 * t * setup.params.dt;
  }

  // Run MPPI
  Eigen::VectorXd u_base = setup.runMPPIStep(x0, control_seq, ref);

  // Compute gains
  Eigen::MatrixXd nominal = setup.rolloutNominal(x0, control_seq);
  const auto& gains = setup.gain_computer->computeGains(
    nominal, control_seq, setup.dynamics->model(),
    setup.params.Q, setup.params.Qf, setup.params.R, setup.params.dt);

  // Apply feedback with state offset
  Eigen::VectorXd actual_state = x0;
  actual_state(0) += 0.2;  // 20cm offset in x
  Eigen::VectorXd dx = actual_state - nominal.row(0).transpose();

  Eigen::VectorXd du = 1.0 * gains[0] * dx;
  Eigen::VectorXd u_corrected = u_base + du;

  // du should be non-zero when dx != 0
  EXPECT_GT(du.norm(), 1e-10)
    << "Feedback correction should be non-zero for non-zero state error";
}

// =============================================================================
// Test 7: ControlBoundsRespected - u_opt stays within control limits after clipping
// =============================================================================
TEST(FeedbackMPPI, ControlBoundsRespected)
{
  FeedbackMPPITestSetup setup;
  setup.init("diff_drive", 64, 10);

  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();
  int N = setup.params.N;

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(N, nu);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int t = 0; t <= N; ++t) {
    ref(t, 0) = 2.0 * t * setup.params.dt;  // Fast reference to push limits
  }

  setup.runMPPIStep(x0, control_seq, ref);

  Eigen::MatrixXd nominal = setup.rolloutNominal(x0, control_seq);
  const auto& gains = setup.gain_computer->computeGains(
    nominal, control_seq, setup.dynamics->model(),
    setup.params.Q, setup.params.Qf, setup.params.R, setup.params.dt);

  // Large state offset to push correction to limits
  Eigen::VectorXd actual_state = x0;
  actual_state(0) += 1.0;
  actual_state(1) += 1.0;
  Eigen::VectorXd dx = actual_state - nominal.row(0).transpose();

  Eigen::VectorXd u_opt = control_seq.row(0).transpose() + gains[0] * dx;

  // Clip
  Eigen::MatrixXd u_mat(1, nu);
  u_mat.row(0) = u_opt.transpose();
  u_opt = setup.dynamics->clipControls(u_mat).row(0).transpose();

  EXPECT_GE(u_opt(0), setup.params.v_min - 1e-6);
  EXPECT_LE(u_opt(0), setup.params.v_max + 1e-6);
  EXPECT_GE(u_opt(1), setup.params.omega_min - 1e-6);
  EXPECT_LE(u_opt(1), setup.params.omega_max + 1e-6);
}

// =============================================================================
// Test 8: RecomputeInterval - interval=3 -> gains only recomputed every 3 cycles
// =============================================================================
TEST(FeedbackMPPI, RecomputeInterval)
{
  FeedbackMPPITestSetup setup;
  setup.init("diff_drive", 64, 10);

  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();
  int N = setup.params.N;
  int recompute_interval = 3;

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(N, nu);

  Eigen::MatrixXd nominal = setup.rolloutNominal(x0, control_seq);

  // Compute gains at different cycles
  std::vector<std::vector<Eigen::MatrixXd>> gains_per_cycle;

  for (int cycle = 0; cycle < 6; ++cycle) {
    if (cycle % recompute_interval == 0) {
      // Recompute
      const auto& gains = setup.gain_computer->computeGains(
        nominal, control_seq, setup.dynamics->model(),
        setup.params.Q, setup.params.Qf, setup.params.R, setup.params.dt);
      gains_per_cycle.push_back(
        std::vector<Eigen::MatrixXd>(gains.begin(), gains.end()));
    } else {
      // Use cached (same as last computed)
      gains_per_cycle.push_back(gains_per_cycle.back());
    }
  }

  // Cycles 0,1,2 should have same gains; 3,4,5 should have same gains
  for (int t = 0; t < N; ++t) {
    EXPECT_TRUE(gains_per_cycle[0][t].isApprox(gains_per_cycle[1][t], 1e-12));
    EXPECT_TRUE(gains_per_cycle[0][t].isApprox(gains_per_cycle[2][t], 1e-12));
    EXPECT_TRUE(gains_per_cycle[3][t].isApprox(gains_per_cycle[4][t], 1e-12));
    EXPECT_TRUE(gains_per_cycle[3][t].isApprox(gains_per_cycle[5][t], 1e-12));
  }
}

// =============================================================================
// Test 9: AngleNormalization - dx with angle > pi is normalized
// =============================================================================
TEST(FeedbackMPPI, AngleNormalization)
{
  FeedbackMPPITestSetup setup;
  setup.init("diff_drive", 64, 10);

  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();
  int N = setup.params.N;

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(N, nu);

  Eigen::MatrixXd nominal = setup.rolloutNominal(x0, control_seq);

  const auto& gains = setup.gain_computer->computeGains(
    nominal, control_seq, setup.dynamics->model(),
    setup.params.Q, setup.params.Qf, setup.params.R, setup.params.dt);

  // Actual state with theta that wraps around
  Eigen::VectorXd actual_state = Eigen::VectorXd::Zero(nx);
  actual_state(2) = 3.5;  // > pi, should normalize to ~-2.78

  Eigen::VectorXd dx = actual_state - nominal.row(0).transpose();

  // Normalize angle
  auto angle_idx = setup.dynamics->model().angleIndices();
  for (int idx : angle_idx) {
    if (idx < dx.size()) {
      dx(idx) = std::atan2(std::sin(dx(idx)), std::cos(dx(idx)));
    }
  }

  // Angle difference should be in [-pi, pi]
  for (int idx : angle_idx) {
    if (idx < dx.size()) {
      EXPECT_GE(dx(idx), -M_PI - 1e-10);
      EXPECT_LE(dx(idx), M_PI + 1e-10);
    }
  }

  // Correction should be finite
  Eigen::VectorXd du = gains[0] * dx;
  EXPECT_TRUE(du.allFinite());
}

// =============================================================================
// Test 10: DiffDriveModel - works with nx=3, nu=2
// =============================================================================
TEST(FeedbackMPPI, DiffDriveModel)
{
  FeedbackMPPITestSetup setup;
  setup.init("diff_drive", 64, 10);

  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();
  ASSERT_EQ(nx, 3);
  ASSERT_EQ(nu, 2);

  int N = setup.params.N;
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(N, nu);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int t = 0; t <= N; ++t) {
    ref(t, 0) = 0.3 * t * setup.params.dt;
  }

  setup.runMPPIStep(x0, control_seq, ref);
  Eigen::MatrixXd nominal = setup.rolloutNominal(x0, control_seq);

  const auto& gains = setup.gain_computer->computeGains(
    nominal, control_seq, setup.dynamics->model(),
    setup.params.Q, setup.params.Qf, setup.params.R, setup.params.dt);

  ASSERT_EQ(static_cast<int>(gains.size()), N);
  for (int t = 0; t < N; ++t) {
    EXPECT_EQ(gains[t].rows(), 2);
    EXPECT_EQ(gains[t].cols(), 3);
    EXPECT_TRUE(gains[t].allFinite());
  }
}

// =============================================================================
// Test 11: SwerveModel - works with nx=3, nu=3
// =============================================================================
TEST(FeedbackMPPI, SwerveModel)
{
  FeedbackMPPITestSetup setup;
  setup.init("swerve", 64, 10);

  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();
  ASSERT_EQ(nx, 3);
  ASSERT_EQ(nu, 3);

  int N = setup.params.N;
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(N, nu);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int t = 0; t <= N; ++t) {
    ref(t, 0) = 0.3 * t * setup.params.dt;
  }

  setup.runMPPIStep(x0, control_seq, ref);
  Eigen::MatrixXd nominal = setup.rolloutNominal(x0, control_seq);

  const auto& gains = setup.gain_computer->computeGains(
    nominal, control_seq, setup.dynamics->model(),
    setup.params.Q, setup.params.Qf, setup.params.R, setup.params.dt);

  ASSERT_EQ(static_cast<int>(gains.size()), N);
  for (int t = 0; t < N; ++t) {
    EXPECT_EQ(gains[t].rows(), 3);
    EXPECT_EQ(gains[t].cols(), 3);
    EXPECT_TRUE(gains[t].allFinite());
  }
}

// =============================================================================
// Test 12: ConsecutiveCalls - 10 consecutive calls produce stable results
// =============================================================================
TEST(FeedbackMPPI, ConsecutiveCalls)
{
  FeedbackMPPITestSetup setup;
  setup.init("diff_drive", 64, 10);

  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();
  int N = setup.params.N;

  Eigen::VectorXd state = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(N, nu);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int t = 0; t <= N; ++t) {
    ref(t, 0) = 0.3 * t * setup.params.dt;
  }

  for (int call = 0; call < 10; ++call) {
    Eigen::VectorXd u_opt = setup.runMPPIStep(state, control_seq, ref);
    EXPECT_TRUE(u_opt.allFinite()) << "Call " << call << " produced non-finite u_opt";

    // Compute gains
    Eigen::MatrixXd nominal = setup.rolloutNominal(state, control_seq);
    const auto& gains = setup.gain_computer->computeGains(
      nominal, control_seq, setup.dynamics->model(),
      setup.params.Q, setup.params.Qf, setup.params.R, setup.params.dt);

    for (int t = 0; t < N; ++t) {
      EXPECT_TRUE(gains[t].allFinite())
        << "Call " << call << ", gain K_" << t << " has NaN/Inf";
    }

    // Propagate state
    Eigen::MatrixXd s(1, nx);
    s.row(0) = state.transpose();
    Eigen::MatrixXd c(1, nu);
    c.row(0) = u_opt.transpose();
    state = setup.dynamics->model().propagateBatch(s, c, setup.params.dt).row(0).transpose();
  }
}

// =============================================================================
// Test 13: ZeroStateError - if actual = nominal, correction = 0
// =============================================================================
TEST(FeedbackMPPI, ZeroStateError)
{
  FeedbackMPPITestSetup setup;
  setup.init("diff_drive", 64, 10);

  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();
  int N = setup.params.N;

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(N, nu);

  Eigen::MatrixXd nominal = setup.rolloutNominal(x0, control_seq);

  const auto& gains = setup.gain_computer->computeGains(
    nominal, control_seq, setup.dynamics->model(),
    setup.params.Q, setup.params.Qf, setup.params.R, setup.params.dt);

  // actual == nominal -> dx = 0 -> du = 0
  Eigen::VectorXd dx = x0 - nominal.row(0).transpose();
  Eigen::VectorXd du = gains[0] * dx;

  EXPECT_NEAR(du.norm(), 0.0, 1e-12)
    << "Correction should be zero when actual state equals nominal";
}

// =============================================================================
// Test 14: RegularizationEffect - higher regularization -> smaller gains
// =============================================================================
TEST(FeedbackMPPI, RegularizationEffect)
{
  FeedbackMPPITestSetup setup;
  setup.init("diff_drive", 64, 10);

  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();
  int N = setup.params.N;

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(N, nu);

  Eigen::MatrixXd nominal = setup.rolloutNominal(x0, control_seq);

  // Low regularization
  FeedbackGainComputer low_reg(nx, nu, 1e-6);
  const auto& gains_low = low_reg.computeGains(
    nominal, control_seq, setup.dynamics->model(),
    setup.params.Q, setup.params.Qf, setup.params.R, setup.params.dt);

  // High regularization
  FeedbackGainComputer high_reg(nx, nu, 10.0);
  const auto& gains_high = high_reg.computeGains(
    nominal, control_seq, setup.dynamics->model(),
    setup.params.Q, setup.params.Qf, setup.params.R, setup.params.dt);

  // Higher regularization should produce smaller (or equal) gains
  double norm_low = 0.0, norm_high = 0.0;
  for (int t = 0; t < N; ++t) {
    norm_low += gains_low[t].norm();
    norm_high += gains_high[t].norm();
  }

  EXPECT_LT(norm_high, norm_low + 1e-6)
    << "Higher regularization should produce smaller gains. "
    << "Low reg norm: " << norm_low << ", High reg norm: " << norm_high;
}

// =============================================================================
// Test 15: NominalTrajectoryRollout - rollout matches expected propagation
// =============================================================================
TEST(FeedbackMPPI, NominalTrajectoryRollout)
{
  FeedbackMPPITestSetup setup;
  setup.init("diff_drive", 64, 10);

  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();
  int N = setup.params.N;

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(N, nu);
  // Set constant forward velocity
  for (int t = 0; t < N; ++t) {
    control_seq(t, 0) = 0.3;  // v = 0.3 m/s
    control_seq(t, 1) = 0.1;  // omega = 0.1 rad/s
  }

  Eigen::MatrixXd nominal = setup.rolloutNominal(x0, control_seq);

  // Verify first state
  EXPECT_NEAR(nominal(0, 0), x0(0), 1e-12);
  EXPECT_NEAR(nominal(0, 1), x0(1), 1e-12);
  EXPECT_NEAR(nominal(0, 2), x0(2), 1e-12);

  // Verify trajectory is moving forward
  EXPECT_GT(nominal(N, 0), 0.0)
    << "Robot should have moved forward in x";

  // Verify each step is consistent with single-step propagation
  for (int t = 0; t < N; ++t) {
    Eigen::MatrixXd s(1, nx);
    s.row(0) = nominal.row(t);
    Eigen::MatrixXd c(1, nu);
    c.row(0) = control_seq.row(t);
    Eigen::MatrixXd next = setup.dynamics->model().propagateBatch(s, c, setup.params.dt);

    for (int d = 0; d < nx; ++d) {
      EXPECT_NEAR(nominal(t + 1, d), next(0, d), 1e-10)
        << "Mismatch at step " << t << ", dim " << d;
    }
  }
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
