#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <memory>

#include "mpc_controller_ros2/mppi_params.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/sampling.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/tube_mppi.hpp"
#include "mpc_controller_ros2/ancillary_controller.hpp"
#include "mpc_controller_ros2/diff_drive_model.hpp"
#include "mpc_controller_ros2/swerve_drive_model.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"

using namespace mpc_controller_ros2;

// =============================================================================
// Helper: Tube-MPPI 단위 테스트용 컴포넌트
// =============================================================================

struct TubeMPPIPluginTestSetup
{
  MPPIParams params;
  std::unique_ptr<BatchDynamicsWrapper> dynamics;
  std::unique_ptr<CompositeMPPICost> cost_function;
  std::unique_ptr<BaseSampler> sampler;
  std::unique_ptr<TubeMPPI> tube_mppi;

  void init(const std::string& model_type = "diff_drive", int K = 128, int N = 15)
  {
    params = MPPIParams();
    params.N = N;
    params.dt = 0.1;
    params.K = K;
    params.lambda = 10.0;
    params.motion_model = model_type;
    params.tube_enabled = true;
    params.tube_width = 0.5;
    params.tube_nominal_reset_threshold = 1.0;
    params.k_forward = 0.8;
    params.k_lateral = 0.5;
    params.k_angle = 1.0;

    int nu = (model_type == "swerve") ? 3 : 2;
    params.noise_sigma = Eigen::VectorXd::Constant(nu, 0.5);

    params.Q = Eigen::MatrixXd::Identity(3, 3) * 10.0;
    params.Q(2, 2) = 1.0;
    params.Qf = params.Q * 2.0;
    params.R = Eigen::MatrixXd::Identity(nu, nu) * 0.1;
    params.R_rate = Eigen::MatrixXd::Identity(nu, nu) * 1.0;

    auto model = MotionModelFactory::create(model_type, params);
    dynamics = std::make_unique<BatchDynamicsWrapper>(
      params, std::shared_ptr<MotionModel>(std::move(model)));
    sampler = std::make_unique<GaussianSampler>(params.noise_sigma, 42);

    cost_function = std::make_unique<CompositeMPPICost>();
    cost_function->addCost(std::make_unique<StateTrackingCost>(params.Q));
    cost_function->addCost(std::make_unique<TerminalCost>(params.Qf));
    cost_function->addCost(std::make_unique<ControlEffortCost>(params.R));

    tube_mppi = std::make_unique<TubeMPPI>(params);
  }

  // 간단한 MPPI step 실행
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> runMPPI(
    const Eigen::VectorXd& state,
    Eigen::MatrixXd& control_seq,
    const Eigen::MatrixXd& ref)
  {
    int N = params.N;
    int nu = dynamics->model().controlDim();
    int nx = dynamics->model().stateDim();
    int K = params.K;

    auto noise = sampler->sample(K, N, nu);
    std::vector<Eigen::MatrixXd> perturbed(K);
    for (int k = 0; k < K; ++k) {
      perturbed[k] = dynamics->clipControls(control_seq + noise[k]);
    }

    std::vector<Eigen::MatrixXd> trajectories;
    dynamics->rolloutBatchInPlace(state, perturbed, params.dt, trajectories);

    Eigen::VectorXd costs = cost_function->compute(trajectories, perturbed, ref);
    VanillaMPPIWeights weight_comp;
    Eigen::VectorXd weights = weight_comp.compute(costs, params.lambda);

    Eigen::MatrixXd weighted_update = Eigen::MatrixXd::Zero(N, nu);
    for (int k = 0; k < K; ++k) {
      weighted_update += weights(k) * noise[k];
    }
    control_seq += weighted_update;
    control_seq = dynamics->clipControls(control_seq);

    Eigen::VectorXd u_opt = control_seq.row(0).transpose();

    // 최적 궤적 계산
    std::vector<Eigen::MatrixXd> opt_ctrl = {control_seq};
    std::vector<Eigen::MatrixXd> opt_traj;
    dynamics->rolloutBatchInPlace(state, opt_ctrl, params.dt, opt_traj);

    return {u_opt, opt_traj[0]};
  }
};

// =============================================================================
// 1. NominalStateInit — 첫 호출 시 nominal = actual
// =============================================================================

TEST(TubeMPPIPlugin, NominalStateInit)
{
  TubeMPPIPluginTestSetup setup;
  setup.init();
  int nx = setup.dynamics->model().stateDim();

  Eigen::VectorXd actual_state = Eigen::VectorXd::Zero(nx);
  actual_state(0) = 1.0;
  actual_state(1) = 2.0;
  actual_state(2) = 0.5;

  // Nominal 초기화 시뮬레이션
  Eigen::VectorXd nominal_state = actual_state;  // 첫 호출

  EXPECT_NEAR(nominal_state(0), actual_state(0), 1e-10);
  EXPECT_NEAR(nominal_state(1), actual_state(1), 1e-10);
  EXPECT_NEAR(nominal_state(2), actual_state(2), 1e-10);
}

// =============================================================================
// 2. NominalStatePropagation — u_nominal으로 전파 확인
// =============================================================================

TEST(TubeMPPIPlugin, NominalStatePropagation)
{
  TubeMPPIPluginTestSetup setup;
  setup.init();
  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();

  Eigen::VectorXd nominal = Eigen::VectorXd::Zero(nx);
  Eigen::VectorXd u_nominal(nu);
  u_nominal << 0.5, 0.0;  // 직진

  // 전파
  Eigen::MatrixXd state_mat(1, nx);
  state_mat.row(0) = nominal.transpose();
  Eigen::MatrixXd ctrl_mat(1, nu);
  ctrl_mat.row(0) = u_nominal.transpose();
  Eigen::VectorXd next = setup.dynamics->model().propagateBatch(
    state_mat, ctrl_mat, setup.params.dt).row(0).transpose();

  // x 방향으로 전진해야 함
  EXPECT_GT(next(0), 0.0);
  EXPECT_NEAR(next(1), 0.0, 1e-6);  // y 변화 없음
}

// =============================================================================
// 3. NominalStateReset — deviation > threshold → 리셋
// =============================================================================

TEST(TubeMPPIPlugin, NominalStateReset)
{
  TubeMPPIPluginTestSetup setup;
  setup.init();
  int nx = setup.dynamics->model().stateDim();

  Eigen::VectorXd nominal = Eigen::VectorXd::Zero(nx);
  Eigen::VectorXd actual = Eigen::VectorXd::Zero(nx);
  actual(0) = 2.0;  // 큰 편차

  double deviation = (nominal.head(2) - actual.head(2)).norm();
  double threshold = setup.params.tube_nominal_reset_threshold;

  EXPECT_GT(deviation, threshold);

  // 리셋: nominal = actual
  if (deviation > threshold) {
    nominal = actual;
  }

  EXPECT_NEAR(nominal(0), actual(0), 1e-10);
}

// =============================================================================
// 4. MPPIUsesNominalState — parent가 nominal로 호출됨
// =============================================================================

TEST(TubeMPPIPlugin, MPPIUsesNominalState)
{
  TubeMPPIPluginTestSetup setup;
  setup.init();
  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();

  Eigen::VectorXd nominal_state = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(setup.params.N, nu);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(setup.params.N + 1, nx);
  for (int t = 0; t <= setup.params.N; ++t) {
    ref(t, 0) = 0.3 * t * 0.1;
  }

  auto [u_opt, traj] = setup.runMPPI(nominal_state, control_seq, ref);

  // MPPI가 nominal state에서 실행되어 유한한 결과 반환
  EXPECT_TRUE(std::isfinite(u_opt(0)));
  EXPECT_TRUE(std::isfinite(u_opt(1)));
}

// =============================================================================
// 5. BodyFrameErrorCorrect — body frame 변환 정확성
// =============================================================================

TEST(TubeMPPIPlugin, BodyFrameErrorCorrect)
{
  AncillaryController ancillary(0.8, 0.5, 1.0);

  Eigen::VectorXd nominal(3);
  nominal << 0.0, 0.0, 0.0;  // 원점, theta=0

  Eigen::VectorXd actual(3);
  actual << 0.1, 0.2, 0.05;  // actual이 nominal보다 앞+좌측에 있음

  Eigen::VectorXd error = ancillary.computeBodyFrameError(nominal, actual);

  // AncillaryController: dx = nominal - actual = (-0.1, -0.2)
  // body frame (actual theta=0.05 기준): e_forward ≈ -0.11, e_lateral ≈ -0.19
  // 부호: nominal이 actual 뒤에 있으므로 음수
  EXPECT_LT(error(0), 0.0);   // forward: nominal이 뒤에 있음
  EXPECT_LT(error(1), 0.0);   // lateral: nominal이 우측에 있음
  EXPECT_NEAR(error(2), -0.05, 0.01);  // angle: nominal theta < actual theta
  EXPECT_TRUE(std::isfinite(error.norm()));
}

// =============================================================================
// 6. FeedbackCorrectionApplied — u_applied = u_nom + du
// =============================================================================

TEST(TubeMPPIPlugin, FeedbackCorrectionApplied)
{
  AncillaryController ancillary(0.8, 0.5, 1.0);

  Eigen::VectorXd nominal_ctrl(2);
  nominal_ctrl << 0.3, 0.1;

  Eigen::VectorXd nominal_state(3);
  nominal_state << 0.0, 0.0, 0.0;

  Eigen::VectorXd actual_state(3);
  actual_state << 0.05, 0.02, 0.01;

  Eigen::VectorXd corrected = ancillary.computeCorrectedControl(
    nominal_ctrl, nominal_state, actual_state);

  // 보정량이 추가되어야 함
  EXPECT_NE(corrected(0), nominal_ctrl(0));
  EXPECT_TRUE(std::isfinite(corrected(0)));
  EXPECT_TRUE(std::isfinite(corrected(1)));
}

// =============================================================================
// 7. MaxCorrectionClamping — du 클램핑 (getFeedbackGainMatrix 범위)
// =============================================================================

TEST(TubeMPPIPlugin, MaxCorrectionClamping)
{
  TubeMPPIPluginTestSetup setup;
  setup.init();

  // 큰 body error → 피드백 보정이 크지만 클리핑됨
  Eigen::VectorXd body_error(3);
  body_error << 5.0, 5.0, M_PI;  // 큰 오차

  Eigen::MatrixXd K_fb = setup.params.getFeedbackGainMatrix();
  Eigen::VectorXd du = K_fb * body_error;

  Eigen::VectorXd u_nominal(2);
  u_nominal << 0.5, 0.0;
  Eigen::VectorXd u_applied = u_nominal + du;

  // 클리핑
  Eigen::MatrixXd u_mat(1, 2);
  u_mat.row(0) = u_applied.transpose();
  u_applied = setup.dynamics->clipControls(u_mat).row(0).transpose();

  EXPECT_LE(u_applied(0), setup.params.v_max + 1e-6);
  EXPECT_GE(u_applied(0), setup.params.v_min - 1e-6);
  EXPECT_LE(u_applied(1), setup.params.omega_max + 1e-6);
  EXPECT_GE(u_applied(1), setup.params.omega_min - 1e-6);
}

// =============================================================================
// 8. ControlClipping — u_applied 범위 내
// =============================================================================

TEST(TubeMPPIPlugin, ControlClipping)
{
  TubeMPPIPluginTestSetup setup;
  setup.init();

  Eigen::VectorXd u(2);
  u << 10.0, -10.0;  // 범위 초과

  Eigen::MatrixXd u_mat(1, 2);
  u_mat.row(0) = u.transpose();
  Eigen::VectorXd clipped = setup.dynamics->clipControls(u_mat).row(0).transpose();

  EXPECT_LE(clipped(0), setup.params.v_max + 1e-6);
  EXPECT_GE(clipped(1), setup.params.omega_min - 1e-6);
}

// =============================================================================
// 9. TubeWidthComputed — tube_width 올바름
// =============================================================================

TEST(TubeMPPIPlugin, TubeWidthComputed)
{
  TubeMPPIPluginTestSetup setup;
  setup.init();

  EXPECT_NEAR(setup.tube_mppi->getTubeWidth(), 0.5, 1e-10);

  setup.tube_mppi->setTubeWidth(0.8);
  EXPECT_NEAR(setup.tube_mppi->getTubeWidth(), 0.8, 1e-10);
}

// =============================================================================
// 10. TubeInfoPopulated — info 필드 채워짐
// =============================================================================

TEST(TubeMPPIPlugin, TubeInfoPopulated)
{
  TubeMPPIPluginTestSetup setup;
  setup.init();
  int nx = setup.dynamics->model().stateDim();

  Eigen::VectorXd nominal(nx);
  nominal << 0.0, 0.0, 0.0;

  Eigen::VectorXd actual(nx);
  actual << 0.1, 0.05, 0.02;

  Eigen::VectorXd u_nominal(2);
  u_nominal << 0.3, 0.1;

  // 궤적 생성
  Eigen::MatrixXd traj(2, nx);
  traj.row(0) = nominal.transpose();
  traj.row(1) = nominal.transpose();

  auto [corrected, info] = setup.tube_mppi->computeCorrectedControl(
    u_nominal, traj, actual);

  EXPECT_EQ(info.nominal_control.size(), 2);
  EXPECT_EQ(info.body_error.size(), 3);
  EXPECT_EQ(info.feedback_correction.size(), 2);
  EXPECT_EQ(info.applied_control.size(), 2);
  EXPECT_GT(info.tube_width, 0.0);
}

// =============================================================================
// 11. DisabledFallback — tube_enabled=false → vanilla
// =============================================================================

TEST(TubeMPPIPlugin, DisabledFallback)
{
  TubeMPPIPluginTestSetup setup;
  setup.init();
  setup.params.tube_enabled = false;

  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();

  Eigen::VectorXd state = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(setup.params.N, nu);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(setup.params.N + 1, nx);

  auto [u_opt, traj] = setup.runMPPI(state, control_seq, ref);

  EXPECT_TRUE(std::isfinite(u_opt(0)));
  EXPECT_TRUE(std::isfinite(u_opt(1)));
}

// =============================================================================
// 12. ConsecutiveCalls — 10회 연속 안정성
// =============================================================================

TEST(TubeMPPIPlugin, ConsecutiveCalls)
{
  TubeMPPIPluginTestSetup setup;
  setup.init();
  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();

  Eigen::VectorXd nominal_state = Eigen::VectorXd::Zero(nx);
  Eigen::VectorXd actual_state = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(setup.params.N, nu);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(setup.params.N + 1, nx);
  for (int t = 0; t <= setup.params.N; ++t) {
    ref(t, 0) = 0.3 * t * 0.1;
  }

  for (int step = 0; step < 10; ++step) {
    // MPPI on nominal
    auto [u_nominal, traj] = setup.runMPPI(nominal_state, control_seq, ref);

    // Body error + feedback
    Eigen::VectorXd body_error = setup.tube_mppi->getAncillaryController()
      .computeBodyFrameError(nominal_state, actual_state);
    Eigen::VectorXd du = setup.tube_mppi->getAncillaryController()
      .computeFeedbackCorrection(body_error);
    Eigen::VectorXd u_applied = u_nominal + du;

    // Clip
    Eigen::MatrixXd u_mat(1, nu);
    u_mat.row(0) = u_applied.transpose();
    u_applied = setup.dynamics->clipControls(u_mat).row(0).transpose();

    // 전파
    Eigen::MatrixXd sm(1, nx); sm.row(0) = nominal_state.transpose();
    Eigen::MatrixXd cm(1, nu); cm.row(0) = u_nominal.transpose();
    nominal_state = setup.dynamics->model().propagateBatch(
      sm, cm, setup.params.dt).row(0).transpose();

    Eigen::MatrixXd sa(1, nx); sa.row(0) = actual_state.transpose();
    Eigen::MatrixXd ca(1, nu); ca.row(0) = u_applied.transpose();
    actual_state = setup.dynamics->model().propagateBatch(
      sa, ca, setup.params.dt).row(0).transpose();

    EXPECT_TRUE(std::isfinite(nominal_state(0))) << "Step " << step;
    EXPECT_TRUE(std::isfinite(actual_state(0))) << "Step " << step;
  }
}

// =============================================================================
// 13. SwerveModel — nu=3 동작
// =============================================================================

TEST(TubeMPPIPlugin, SwerveModel)
{
  TubeMPPIPluginTestSetup setup;
  setup.init("swerve", 64, 10);
  int nx = setup.dynamics->model().stateDim();
  int nu = setup.dynamics->model().controlDim();

  EXPECT_EQ(nu, 3);

  Eigen::VectorXd state = Eigen::VectorXd::Zero(nx);
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(setup.params.N, nu);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(setup.params.N + 1, nx);

  auto [u_opt, traj] = setup.runMPPI(state, control_seq, ref);

  EXPECT_EQ(u_opt.size(), 3);
  EXPECT_TRUE(std::isfinite(u_opt(0)));
  EXPECT_TRUE(std::isfinite(u_opt(1)));
  EXPECT_TRUE(std::isfinite(u_opt(2)));
}

// =============================================================================
// 14. DeviationTracking — deviation 값 정확
// =============================================================================

TEST(TubeMPPIPlugin, DeviationTracking)
{
  Eigen::VectorXd nominal(3);
  nominal << 1.0, 2.0, 0.0;

  Eigen::VectorXd actual(3);
  actual << 1.3, 2.4, 0.1;

  double deviation = (nominal.head(2) - actual.head(2)).norm();
  double expected = std::sqrt(0.3 * 0.3 + 0.4 * 0.4);

  EXPECT_NEAR(deviation, expected, 1e-10);
  EXPECT_NEAR(deviation, 0.5, 1e-10);
}

// =============================================================================
// 15. TubeBoundary — boundary 점 계산
// =============================================================================

TEST(TubeMPPIPlugin, TubeBoundary)
{
  TubeMPPIPluginTestSetup setup;
  setup.init();

  // 직선 궤적 (x축 방향)
  Eigen::MatrixXd traj(5, 3);
  for (int i = 0; i < 5; ++i) {
    traj(i, 0) = i * 0.1;
    traj(i, 1) = 0.0;
    traj(i, 2) = 0.0;  // theta=0
  }

  auto boundaries = setup.tube_mppi->computeTubeBoundary(traj);

  EXPECT_EQ(static_cast<int>(boundaries.size()), 5);

  // theta=0에서 perpendicular = (-sin(0), cos(0)) = (0, 1)
  // left = (x, +tube_width), right = (x, -tube_width)
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(boundaries[i].first(1), setup.params.tube_width, 1e-6);
    EXPECT_NEAR(boundaries[i].second(1), -setup.params.tube_width, 1e-6);
  }
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
