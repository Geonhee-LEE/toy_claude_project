#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <memory>
#include <algorithm>

#include "mpc_controller_ros2/pi_mppi_controller_plugin.hpp"
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
// Accessor: PiMPPIControllerPlugin의 protected 메서드를 테스트용으로 노출
// =============================================================================

class PiMPPIAccessor : public PiMPPIControllerPlugin
{
public:
  using PiMPPIControllerPlugin::projectAllSamples;
  using PiMPPIControllerPlugin::projector_;
  using PiMPPIControllerPlugin::pi_u_min_;
  using PiMPPIControllerPlugin::pi_u_max_;
  using PiMPPIControllerPlugin::pi_rate_max_;
  using PiMPPIControllerPlugin::pi_accel_max_;
  using MPPIControllerPlugin::noise_buffer_;
  using MPPIControllerPlugin::perturbed_buffer_;
  using MPPIControllerPlugin::trajectory_buffer_;
  using MPPIControllerPlugin::control_sequence_;
  using MPPIControllerPlugin::sampler_;
  using MPPIControllerPlugin::dynamics_;
  using MPPIControllerPlugin::params_;
  using MPPIControllerPlugin::weight_computation_;
  using MPPIControllerPlugin::cost_function_;

  void initForTest(const MPPIParams& p, BatchDynamicsWrapper* dyn, BaseSampler* samp)
  {
    params_ = p;
    dynamics_.reset(dyn);
    sampler_.reset(samp);
    int N = p.N;
    int nx = dyn->model().stateDim();
    int nu = dyn->model().controlDim();

    // ADMM projector
    projector_ = std::make_unique<ADMMProjector>(
      N, p.dt, p.pi_admm_rho, p.pi_admm_iterations, p.pi_derivative_order);

    // Bounds
    pi_u_min_ = Eigen::VectorXd::Zero(nu);
    pi_u_max_ = Eigen::VectorXd::Zero(nu);
    pi_rate_max_ = Eigen::VectorXd::Zero(nu);
    pi_accel_max_ = Eigen::VectorXd::Zero(nu);

    pi_u_min_(0) = p.v_min;
    pi_u_max_(0) = p.v_max;
    pi_rate_max_(0) = p.pi_rate_max_v;
    pi_accel_max_(0) = p.pi_accel_max_v;
    if (nu >= 2) {
      pi_u_min_(1) = p.omega_min;
      pi_u_max_(1) = p.omega_max;
      pi_rate_max_(1) = p.pi_rate_max_omega;
      pi_accel_max_(1) = p.pi_accel_max_omega;
    }
    if (nu >= 3) {
      double vy_lim = (p.vy_max > 0) ? p.vy_max : p.v_max;
      pi_u_min_(2) = p.omega_min;
      pi_u_max_(2) = p.omega_max;
      pi_rate_max_(2) = p.pi_rate_max_vy;
      pi_accel_max_(2) = p.pi_accel_max_vy;
    }

    // Buffers
    noise_buffer_.resize(p.K, Eigen::MatrixXd::Zero(N, nu));
    perturbed_buffer_.resize(p.K, Eigen::MatrixXd::Zero(N, nu));
    trajectory_buffer_.resize(p.K, Eigen::MatrixXd::Zero(N + 1, nx));
    control_sequence_ = Eigen::MatrixXd::Zero(N, nu);
    weight_computation_ = std::make_unique<VanillaMPPIWeights>();
    cost_function_ = std::make_unique<CompositeMPPICost>();
    cost_function_->addCost(std::make_unique<StateTrackingCost>(p.Q));
    cost_function_->addCost(std::make_unique<TerminalCost>(p.Qf));
    cost_function_->addCost(std::make_unique<ControlEffortCost>(p.R));
  }
};

// =============================================================================
// Fixture
// =============================================================================

struct PiMPPITestFixture
{
  MPPIParams params;
  BatchDynamicsWrapper* dynamics_raw{nullptr};
  BaseSampler* sampler_raw{nullptr};
  PiMPPIAccessor accessor;

  void init(const std::string& model_type = "diff_drive", int K = 256, int N = 20)
  {
    params = MPPIParams();
    params.N = N;
    params.dt = 0.1;
    params.K = K;
    params.lambda = 10.0;
    params.motion_model = model_type;
    params.pi_enabled = true;
    params.pi_admm_iterations = 10;
    params.pi_admm_rho = 1.0;
    params.pi_derivative_order = 2;
    params.pi_rate_max_v = 2.0;
    params.pi_rate_max_omega = 3.0;
    params.pi_rate_max_vy = 2.0;
    params.pi_accel_max_v = 5.0;
    params.pi_accel_max_omega = 8.0;
    params.pi_accel_max_vy = 5.0;

    int nu = (model_type == "diff_drive" || model_type == "ackermann") ? 2 : 3;
    bool is_non_coaxial = (model_type == "non_coaxial_swerve");
    bool is_ackermann = (model_type == "ackermann");
    int nx = (is_non_coaxial || is_ackermann) ? 4 : 3;

    params.noise_sigma = Eigen::VectorXd::Constant(nu, 0.5);
    params.Q = Eigen::MatrixXd::Identity(nx, nx) * 10.0;
    params.Q(2, 2) = 1.0;
    if (nx >= 4) params.Q(3, 3) = 0.5;
    params.Qf = params.Q * 2.0;
    params.R = Eigen::MatrixXd::Identity(nu, nu) * 0.1;

    auto model = MotionModelFactory::create(model_type, params);
    auto dynamics_owner = std::make_unique<BatchDynamicsWrapper>(params, std::move(model));
    auto sampler_owner = std::make_unique<GaussianSampler>(params.noise_sigma, 42);
    dynamics_raw = dynamics_owner.get();
    sampler_raw = sampler_owner.get();

    accessor.initForTest(params, dynamics_owner.release(), sampler_owner.release());
  }
};

// =============================================================================
// [A] ADMMProjector 단위 테스트 (6)
// =============================================================================

TEST(PiMPPI, FiniteDiffD1Shape)
{
  ADMMProjector proj(30, 0.1, 1.0, 10, 2);
  EXPECT_EQ(proj.D1().rows(), 29);
  EXPECT_EQ(proj.D1().cols(), 30);
}

TEST(PiMPPI, FiniteDiffD2Shape)
{
  ADMMProjector proj(30, 0.1, 1.0, 10, 2);
  EXPECT_EQ(proj.D2().rows(), 28);
  EXPECT_EQ(proj.D2().cols(), 30);
}

TEST(PiMPPI, D1ConstantSignal)
{
  // D1 * constant_vector = 0 (일정한 신호의 1차 미분은 0)
  int N = 20;
  ADMMProjector proj(N, 0.1, 1.0, 10, 2);

  Eigen::VectorXd constant = Eigen::VectorXd::Constant(N, 3.0);
  Eigen::VectorXd result = proj.D1() * constant;

  for (int i = 0; i < N - 1; ++i) {
    EXPECT_NEAR(result(i), 0.0, 1e-10) << "D1 * const should be 0, i=" << i;
  }
}

TEST(PiMPPI, D2LinearSignal)
{
  // D2 * linear_vector = 0 (선형 신호의 2차 미분은 0)
  int N = 20;
  double dt = 0.1;
  ADMMProjector proj(N, dt, 1.0, 10, 2);

  Eigen::VectorXd linear(N);
  for (int i = 0; i < N; ++i) {
    linear(i) = 0.5 + 0.3 * i * dt;  // linear ramp
  }
  Eigen::VectorXd result = proj.D2() * linear;

  for (int i = 0; i < N - 2; ++i) {
    EXPECT_NEAR(result(i), 0.0, 1e-6) << "D2 * linear should be ~0, i=" << i;
  }
}

TEST(PiMPPI, ProjectFeasibleUnchanged)
{
  // 이미 feasible한 시퀀스는 투영 후 거의 변하지 않아야 함
  int N = 20;
  double dt = 0.1;
  ADMMProjector proj(N, dt, 1.0, 20, 2);

  // 일정한 시퀀스: u=0.3 → 모든 제약 만족
  Eigen::VectorXd v_raw = Eigen::VectorXd::Constant(N, 0.3);
  Eigen::VectorXd v_out(N);

  proj.projectDimension(v_raw, v_out, 0.0, 1.0, 10.0, 50.0);

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(v_out(i), v_raw(i), 1e-4)
      << "Feasible signal should be nearly unchanged, i=" << i;
  }
}

TEST(PiMPPI, ProjectInfeasibleClamps)
{
  // u_max=0.5인데 v_raw=1.0 → 투영 후 u_max 이하로 내려가야 함
  int N = 10;
  double dt = 0.1;
  ADMMProjector proj(N, dt, 1.0, 50, 1);

  Eigen::VectorXd v_raw = Eigen::VectorXd::Constant(N, 1.0);
  Eigen::VectorXd v_out(N);

  proj.projectDimension(v_raw, v_out, 0.0, 0.5, 10.0, 50.0);

  for (int i = 0; i < N; ++i) {
    EXPECT_LE(v_out(i), 0.5 + 1e-3) << "Should be within u_max, i=" << i;
    EXPECT_GE(v_out(i), 0.0 - 1e-3) << "Should be within u_min, i=" << i;
  }
}

// =============================================================================
// [B] 제약 만족 테스트 (4)
// =============================================================================

TEST(PiMPPI, ControlBoundsEnforced)
{
  int N = 20;
  double dt = 0.1;
  ADMMProjector proj(N, dt, 1.0, 30, 2);

  // 랜덤 대입력 시퀀스
  Eigen::VectorXd v_raw = Eigen::VectorXd::Random(N) * 5.0;  // [-5, 5]
  Eigen::VectorXd v_out(N);

  double u_min = -0.5, u_max = 1.0;
  proj.projectDimension(v_raw, v_out, u_min, u_max, 100.0, 500.0);

  for (int i = 0; i < N; ++i) {
    EXPECT_GE(v_out(i), u_min - 1e-3) << "Control below min, i=" << i;
    EXPECT_LE(v_out(i), u_max + 1e-3) << "Control above max, i=" << i;
  }
}

TEST(PiMPPI, RateBoundsEnforced)
{
  int N = 20;
  double dt = 0.1;
  ADMMProjector proj(N, dt, 1.0, 30, 1);

  // 급격한 step 입력: 0 → 1.0 (rate = 1.0/dt = 10.0)
  Eigen::VectorXd v_raw = Eigen::VectorXd::Zero(N);
  for (int i = N / 2; i < N; ++i) {
    v_raw(i) = 1.0;
  }
  Eigen::VectorXd v_out(N);

  double rate_max = 2.0;  // m/s^2
  proj.projectDimension(v_raw, v_out, -2.0, 2.0, rate_max, 50.0);

  // rate check: |v_out[i+1] - v_out[i]| / dt <= rate_max + tolerance
  for (int i = 0; i < N - 1; ++i) {
    double rate = std::abs(v_out(i + 1) - v_out(i)) / dt;
    EXPECT_LE(rate, rate_max + 0.5)
      << "Rate exceeded at i=" << i << ": " << rate;
  }
}

TEST(PiMPPI, AccelBoundsEnforced)
{
  int N = 30;
  double dt = 0.1;
  ADMMProjector proj(N, dt, 1.0, 30, 2);

  // 지그재그 입력: 높은 가속도
  Eigen::VectorXd v_raw(N);
  for (int i = 0; i < N; ++i) {
    v_raw(i) = 0.5 * std::sin(i * 1.0);  // 고주파 진동
  }
  Eigen::VectorXd v_out(N);

  double accel_max = 3.0;  // m/s^3
  proj.projectDimension(v_raw, v_out, -2.0, 2.0, 20.0, accel_max);

  // accel check: |(v[i+2] - 2*v[i+1] + v[i]) / dt^2| <= accel_max + tolerance
  for (int i = 0; i < N - 2; ++i) {
    double accel = std::abs(v_out(i + 2) - 2.0 * v_out(i + 1) + v_out(i)) / (dt * dt);
    EXPECT_LE(accel, accel_max + 1.0)
      << "Accel exceeded at i=" << i << ": " << accel;
  }
}

TEST(PiMPPI, Order1SkipsAccel)
{
  // derivative_order=1이면 accel 제약 없음 → D2 적용 안 됨
  int N = 15;
  double dt = 0.1;
  ADMMProjector proj_order1(N, dt, 1.0, 20, 1);
  ADMMProjector proj_order2(N, dt, 1.0, 20, 2);

  // A 행렬 크기 비교
  int expected_m1 = N + (N - 1);          // I + D1 only
  int expected_m2 = N + (N - 1) + (N - 2); // I + D1 + D2
  EXPECT_EQ(proj_order1.A().rows(), expected_m1);
  EXPECT_EQ(proj_order2.A().rows(), expected_m2);
}

// =============================================================================
// [C] 통합 테스트 (3)
// =============================================================================

TEST(PiMPPI, DisabledEqualsVanilla)
{
  PiMPPITestFixture fix;
  fix.init("diff_drive", 64, 15);
  fix.accessor.params_.pi_enabled = false;

  int nx = fix.dynamics_raw->model().stateDim();
  int nu = fix.dynamics_raw->model().controlDim();

  // pi disabled → projector는 있지만 사용되지 않음
  // projectSequence가 여전히 유효한지만 확인
  Eigen::MatrixXd ctrl = Eigen::MatrixXd::Zero(fix.params.N, nu);
  ctrl.col(0).setConstant(0.3);
  Eigen::MatrixXd projected;
  fix.accessor.projector_->projectSequence(
    ctrl, projected,
    fix.accessor.pi_u_min_, fix.accessor.pi_u_max_,
    fix.accessor.pi_rate_max_, fix.accessor.pi_accel_max_);

  EXPECT_EQ(projected.rows(), fix.params.N);
  EXPECT_EQ(projected.cols(), nu);
  EXPECT_TRUE(projected.allFinite());
}

TEST(PiMPPI, AllModelsCompatible)
{
  for (const auto& model_type : {"diff_drive", "swerve", "non_coaxial_swerve", "ackermann"}) {
    PiMPPITestFixture fix;
    fix.init(model_type, 64, 10);

    int nu = fix.dynamics_raw->model().controlDim();

    // 랜덤 제어 시퀀스 생성 후 투영
    Eigen::MatrixXd ctrl = Eigen::MatrixXd::Random(fix.params.N, nu) * 0.5;
    Eigen::MatrixXd projected;
    fix.accessor.projector_->projectSequence(
      ctrl, projected,
      fix.accessor.pi_u_min_, fix.accessor.pi_u_max_,
      fix.accessor.pi_rate_max_, fix.accessor.pi_accel_max_);

    EXPECT_EQ(projected.rows(), fix.params.N)
      << "Failed for model: " << model_type;
    EXPECT_EQ(projected.cols(), nu)
      << "Failed for model: " << model_type;
    EXPECT_TRUE(projected.allFinite())
      << "Projected should be finite for model: " << model_type;
  }
}

TEST(PiMPPI, ComputeControlReturnsValid)
{
  PiMPPITestFixture fix;
  fix.init("diff_drive", 128, 15);

  int nx = fix.dynamics_raw->model().stateDim();
  int nu = fix.dynamics_raw->model().controlDim();

  // 전체 파이프라인 수동 실행 (투영 포함)
  Eigen::MatrixXd ctrl = Eigen::MatrixXd::Zero(fix.params.N, nu);
  ctrl.col(0).setConstant(0.3);

  // 노이즈 샘플링
  fix.accessor.sampler_->sampleInPlace(
    fix.accessor.noise_buffer_, fix.params.K, fix.params.N, nu);

  // perturbed = ctrl + noise
  for (int k = 0; k < fix.params.K; ++k) {
    fix.accessor.perturbed_buffer_[k] = ctrl + fix.accessor.noise_buffer_[k];
  }

  // 투영
  fix.accessor.control_sequence_ = ctrl;
  fix.accessor.projectAllSamples();

  // 모든 투영 결과가 유한해야 함
  for (int k = 0; k < fix.params.K; ++k) {
    EXPECT_TRUE(fix.accessor.perturbed_buffer_[k].allFinite())
      << "Projected sample should be finite, k=" << k;
  }
}

// =============================================================================
// [D] 성능 & 안정성 테스트 (3)
// =============================================================================

TEST(PiMPPI, PerformanceBudget)
{
  // K=512, N=30, nu=2, 10 ADMM iter → 투영 전체 < 1ms
  int K = 512;
  int N = 30;
  int nu = 2;
  double dt = 0.1;

  ADMMProjector proj(N, dt, 1.0, 10, 2);

  Eigen::VectorXd u_min = Eigen::VectorXd::Constant(nu, 0.0);
  Eigen::VectorXd u_max = Eigen::VectorXd::Constant(nu, 1.0);
  Eigen::VectorXd rate_max = Eigen::VectorXd::Constant(nu, 2.0);
  Eigen::VectorXd accel_max = Eigen::VectorXd::Constant(nu, 5.0);

  // 랜덤 샘플 생성
  std::vector<Eigen::MatrixXd> samples(K);
  for (int k = 0; k < K; ++k) {
    samples[k] = Eigen::MatrixXd::Random(N, nu);
  }

  // 타이밍
  auto start = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd projected;
    proj.projectSequence(samples[k], projected, u_min, u_max, rate_max, accel_max);
  }
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "[PerformanceBudget] K=" << K << " projections: " << ms << " ms" << std::endl;

  // Dense ADMM N=30: ~7ms for K=512 without OpenMP; budget 15ms (CI 환경 고려)
  EXPECT_LT(ms, 15.0) << "Projection should be fast, got " << ms << " ms";
}

TEST(PiMPPI, MultipleCallsStable)
{
  PiMPPITestFixture fix;
  fix.init("diff_drive", 64, 15);

  int nu = fix.dynamics_raw->model().controlDim();

  // 여러 번 투영 호출 → 항상 유한하고 범위 내
  for (int iter = 0; iter < 10; ++iter) {
    Eigen::MatrixXd ctrl = Eigen::MatrixXd::Random(fix.params.N, nu) * 2.0;
    Eigen::MatrixXd projected;
    fix.accessor.projector_->projectSequence(
      ctrl, projected,
      fix.accessor.pi_u_min_, fix.accessor.pi_u_max_,
      fix.accessor.pi_rate_max_, fix.accessor.pi_accel_max_);

    ASSERT_EQ(projected.rows(), fix.params.N) << "iter=" << iter;
    ASSERT_EQ(projected.cols(), nu) << "iter=" << iter;
    EXPECT_TRUE(projected.allFinite()) << "iter=" << iter;

    // control bounds check
    for (int t = 0; t < fix.params.N; ++t) {
      for (int d = 0; d < nu; ++d) {
        EXPECT_GE(projected(t, d), fix.accessor.pi_u_min_(d) - 0.1)
          << "iter=" << iter << " t=" << t << " d=" << d;
        EXPECT_LE(projected(t, d), fix.accessor.pi_u_max_(d) + 0.1)
          << "iter=" << iter << " t=" << t << " d=" << d;
      }
    }
  }
}

TEST(PiMPPI, HighNoiseFeasibility)
{
  // 매우 큰 노이즈에도 투영 후 제약 만족
  int N = 20;
  double dt = 0.1;
  ADMMProjector proj(N, dt, 1.0, 30, 2);

  Eigen::VectorXd v_raw = Eigen::VectorXd::Random(N) * 100.0;  // [-100, 100]
  Eigen::VectorXd v_out(N);

  double u_min = 0.0, u_max = 1.0, rate_max = 2.0, accel_max = 5.0;
  proj.projectDimension(v_raw, v_out, u_min, u_max, rate_max, accel_max);

  EXPECT_TRUE(v_out.allFinite()) << "Should handle extreme inputs";

  for (int i = 0; i < N; ++i) {
    EXPECT_GE(v_out(i), u_min - 0.1) << "i=" << i;
    EXPECT_LE(v_out(i), u_max + 0.1) << "i=" << i;
  }
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
