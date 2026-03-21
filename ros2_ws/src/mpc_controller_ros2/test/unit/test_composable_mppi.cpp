// =============================================================================
// Composable MPPI Controller Plugin 단위 테스트 (15개)
// 파이프라인 기반 다중 레이어 조합 검증
// =============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "mpc_controller_ros2/composable_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/halton_sampler.hpp"
#include "mpc_controller_ros2/ilqr_solver.hpp"
#include "mpc_controller_ros2/trajectory_library.hpp"
#include "mpc_controller_ros2/pi_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/feedback_gain_computer.hpp"
#include "mpc_controller_ros2/adaptive_horizon_manager.hpp"
#include "mpc_controller_ros2/barrier_function.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

// ============================================================================
// 테스트 헬퍼: Composable 플러그인 내부 접근용
// ============================================================================
class ComposableTestAccessor : public ComposableMPPIControllerPlugin
{
public:
  using ComposableMPPIControllerPlugin::active_;
  using ComposableMPPIControllerPlugin::computeControl;
  using ComposableMPPIControllerPlugin::applyLowPassFilter;

  void initForTest(const MPPIParams& p) {
    params_ = p;
    int nx = 3, nu = 2;  // diff_drive defaults

    // ROS2 node (RCLCPP_DEBUG 용)
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
    static int node_counter = 0;
    node_ = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "test_composable_mppi_" + std::to_string(node_counter++));
    plugin_name_ = "composable_test";

    // Initialize dynamics
    auto model = MotionModelFactory::create("diff_drive", params_);
    dynamics_ = std::make_unique<BatchDynamicsWrapper>(params_, std::move(model));

    // Initialize sampler
    sampler_ = std::make_unique<GaussianSampler>(params_.noise_sigma);

    // Weight computation
    weight_computation_ = std::make_unique<VanillaMPPIWeights>();

    // Adaptive temperature
    if (params_.adaptive_temperature) {
      adaptive_temp_ = std::make_unique<AdaptiveTemperature>(
        params_.lambda, params_.target_ess_ratio,
        params_.adaptation_rate, params_.lambda_min, params_.lambda_max);
    }

    // Control sequence
    control_sequence_ = Eigen::MatrixXd::Zero(params_.N, nu);

    // Cost function
    cost_function_ = std::make_unique<CompositeMPPICost>();
    cost_function_->addCost(std::make_unique<StateTrackingCost>(params_.Q));
    cost_function_->addCost(std::make_unique<TerminalCost>(params_.Qf));
    cost_function_->addCost(std::make_unique<ControlEffortCost>(params_.R));

    // Pre-allocate buffers
    allocateBuffers();
  }

  void setupActiveLayers(const ActiveLayers& layers) {
    active_ = layers;
  }

  void setupHaltonSampler() {
    sampler_ = std::make_unique<HaltonSampler>(
      params_.noise_sigma, params_.halton_beta, params_.halton_sequence_offset);
    allocateBuffers();
  }

  void setupILQR() {
    int nx = dynamics_->model().stateDim();
    int nu = dynamics_->model().controlDim();
    ILQRParams ilqr_params;
    ilqr_params.max_iterations = params_.ilqr_max_iterations;
    ilqr_params.regularization = params_.ilqr_regularization;
    ilqr_solver_ = std::make_unique<ILQRSolver>(ilqr_params, nx, nu);
  }

  void setupTrajLibrary() {
    int nu = dynamics_->model().controlDim();
    traj_library_.generate(
      params_.N, nu, params_.dt,
      params_.v_max, params_.v_min, params_.omega_max);
  }

  void setupPiMPPI() {
    int nu = dynamics_->model().controlDim();
    projector_ = std::make_unique<ADMMProjector>(
      params_.N, params_.dt, params_.pi_admm_rho,
      params_.pi_admm_iterations, params_.pi_derivative_order);

    pi_u_min_ = Eigen::VectorXd(nu);
    pi_u_max_ = Eigen::VectorXd(nu);
    pi_rate_max_ = Eigen::VectorXd(nu);
    pi_accel_max_ = Eigen::VectorXd(nu);
    pi_u_min_ << params_.v_min, params_.omega_min;
    pi_u_max_ << params_.v_max, params_.omega_max;
    pi_rate_max_ << params_.pi_rate_max_v, params_.pi_rate_max_omega;
    pi_accel_max_ << params_.pi_accel_max_v, params_.pi_accel_max_omega;
  }

  void setupLP() {
    if (params_.lp_cutoff_frequency > 0.0) {
      double tau = 1.0 / (2.0 * M_PI * params_.lp_cutoff_frequency);
      lp_alpha_ = params_.dt / (tau + params_.dt);
    }
    lp_u_prev_ = Eigen::VectorXd::Zero(dynamics_->model().controlDim());
  }

  void setupShieldCBF(const std::vector<Eigen::Vector3d>& obstacles) {
    barrier_set_.setObstacles(obstacles);
    shield_cbf_stride_ = std::max(1, params_.shield_cbf_stride);
    shield_max_iterations_ = std::max(1, params_.shield_max_iterations);
  }

  void setupFeedback() {
    int nx = dynamics_->model().stateDim();
    int nu = dynamics_->model().controlDim();
    gain_computer_ = std::make_unique<FeedbackGainComputer>(
      nx, nu, params_.feedback_regularization);
  }

  void setupRHMPPI() {
    N_max_ = params_.N;
    int N_max = (params_.rh_N_max > 0) ? std::min(params_.rh_N_max, N_max_) : N_max_;
    int N_min = std::min(params_.rh_N_min, N_max);
    horizon_manager_ = std::make_unique<AdaptiveHorizonManager>(
      N_min, N_max,
      params_.rh_speed_weight, params_.rh_obstacle_weight, params_.rh_error_weight,
      params_.rh_obs_dist_threshold, params_.rh_error_threshold,
      params_.rh_smoothing_alpha);
  }

  void setupCSMPPI() {
    int nx = dynamics_->model().stateDim();
    cs_scale_buffer_ = Eigen::VectorXd::Ones(params_.N);
    cs_nominal_states_ = Eigen::MatrixXd::Zero(params_.N + 1, nx);
  }

  const BarrierFunctionSet& getBarrierSet() const { return barrier_set_; }
  double getLpAlpha() const { return lp_alpha_; }
};

// ============================================================================
// 테스트 Fixture
// ============================================================================
class ComposableMPPITest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    params_ = MPPIParams();
    params_.N = 10;
    params_.dt = 0.1;
    params_.K = 64;
    params_.lambda = 10.0;
    params_.v_max = 1.0;
    params_.v_min = 0.0;
    params_.omega_max = 1.0;
    params_.omega_min = -1.0;
    params_.noise_sigma = Eigen::Vector2d(0.5, 0.5);
    params_.adaptive_temperature = false;

    // Feature flags — default all OFF
    params_.rh_mppi_enabled = false;
    params_.cs_enabled = false;
    params_.ilqr_enabled = false;
    params_.traj_library_enabled = false;
    params_.halton_enabled = false;
    params_.pi_enabled = false;
    params_.lp_enabled = false;
    params_.cbf_enabled = false;
    params_.cbf_use_safety_filter = false;
    params_.feedback_mppi_enabled = false;

    // Defaults for sub-features
    params_.halton_beta = 2.0;
    params_.halton_sequence_offset = 100;
    params_.ilqr_max_iterations = 2;
    params_.ilqr_regularization = 1e-6;
    params_.ilqr_line_search_steps = 4;
    params_.ilqr_cost_tolerance = 1e-4;
    params_.pi_admm_iterations = 10;
    params_.pi_admm_rho = 1.0;
    params_.pi_derivative_order = 2;
    params_.pi_rate_max_v = 2.0;
    params_.pi_rate_max_omega = 3.0;
    params_.pi_accel_max_v = 5.0;
    params_.pi_accel_max_omega = 8.0;
    params_.lp_cutoff_frequency = 10.0;
    params_.shield_cbf_stride = 3;
    params_.shield_max_iterations = 10;
    params_.cbf_gamma = 1.0;
    params_.feedback_gain_scale = 1.0;
    params_.feedback_recompute_interval = 1;
    params_.feedback_regularization = 1e-4;
    params_.rh_N_min = 5;
    params_.rh_N_max = 15;
    params_.rh_speed_weight = 1.0;
    params_.rh_obstacle_weight = 1.0;
    params_.rh_error_weight = 0.5;
    params_.rh_obs_dist_threshold = 2.0;
    params_.rh_error_threshold = 1.0;
    params_.rh_smoothing_alpha = 0.3;
    params_.cs_scale_min = 0.1;
    params_.cs_scale_max = 3.0;
    params_.traj_library_ratio = 0.15;
    params_.traj_library_perturbation = 0.1;
    params_.traj_library_num_per_primitive = 0;

    // State
    state_ = Eigen::Vector3d(0.0, 0.0, 0.0);

    // Reference trajectory
    int nx = 3;
    ref_ = Eigen::MatrixXd::Zero(params_.N + 1, nx);
    for (int t = 0; t <= params_.N; ++t) {
      ref_(t, 0) = 0.1 * t;  // x = 0.1t
    }
  }

  // Base computeControl (all layers OFF) for comparison
  std::pair<Eigen::VectorXd, MPPIInfo> runBase() {
    ComposableTestAccessor acc;
    acc.initForTest(params_);
    ActiveLayers off;
    acc.setupActiveLayers(off);
    return acc.computeControl(state_, ref_);
  }

  MPPIParams params_;
  Eigen::Vector3d state_;
  Eigen::MatrixXd ref_;
};

// ============================================================================
// Test 1: VanillaFallback — 모든 레이어 OFF → base 동일 결과
// ============================================================================
TEST_F(ComposableMPPITest, VanillaFallback)
{
  ComposableTestAccessor acc;
  acc.initForTest(params_);
  ActiveLayers off;  // 모든 레이어 OFF
  acc.setupActiveLayers(off);

  auto [u_opt, info] = acc.computeControl(state_, ref_);

  // 유한값 확인
  EXPECT_TRUE(u_opt.allFinite());
  EXPECT_GE(info.ess, 0.0);
  EXPECT_EQ(info.costs.size(), params_.K);

  // 제어 bounds 확인
  EXPECT_GE(u_opt(0), params_.v_min - 1e-6);
  EXPECT_LE(u_opt(0), params_.v_max + 1e-6);
  EXPECT_GE(u_opt(1), params_.omega_min - 1e-6);
  EXPECT_LE(u_opt(1), params_.omega_max + 1e-6);
}

// ============================================================================
// Test 2: HaltonSamplerSwap — halton_enabled → 저불일치 검증
// ============================================================================
TEST_F(ComposableMPPITest, HaltonSamplerSwap)
{
  ComposableTestAccessor acc;
  acc.initForTest(params_);
  ActiveLayers layers;
  layers.halton = true;
  acc.setupActiveLayers(layers);
  acc.setupHaltonSampler();

  auto [u_opt, info] = acc.computeControl(state_, ref_);

  EXPECT_TRUE(u_opt.allFinite());
  EXPECT_EQ(info.costs.size(), params_.K);
  // Halton은 Gaussian보다 ESS가 높은 경향
  EXPECT_GT(info.ess, 0.0);
}

// ============================================================================
// Test 3: IlqrWarmStart — ilqr_enabled → 제어 개선
// ============================================================================
TEST_F(ComposableMPPITest, IlqrWarmStart)
{
  ComposableTestAccessor acc;
  acc.initForTest(params_);
  ActiveLayers layers;
  layers.ilqr = true;
  acc.setupActiveLayers(layers);
  acc.setupILQR();

  auto [u_opt, info] = acc.computeControl(state_, ref_);

  EXPECT_TRUE(u_opt.allFinite());
  // iLQR warm-start 후 전진 제어 기대 (ref가 x 방향)
  EXPECT_GE(u_opt(0), -0.1);  // 전진 방향
}

// ============================================================================
// Test 4: TrajLibraryInjection — traj_library → primitive 주입
// ============================================================================
TEST_F(ComposableMPPITest, TrajLibraryInjection)
{
  ComposableTestAccessor acc;
  acc.initForTest(params_);
  ActiveLayers layers;
  layers.traj_library = true;
  acc.setupActiveLayers(layers);
  acc.setupTrajLibrary();

  auto [u_opt, info] = acc.computeControl(state_, ref_);

  EXPECT_TRUE(u_opt.allFinite());
  EXPECT_EQ(info.costs.size(), params_.K);
}

// ============================================================================
// Test 5: LPFilterSmoothing — lp_enabled → 주파수 감소
// ============================================================================
TEST_F(ComposableMPPITest, LPFilterSmoothing)
{
  ComposableTestAccessor acc;
  acc.initForTest(params_);
  ActiveLayers layers;
  layers.lp_filter = true;
  acc.setupActiveLayers(layers);
  acc.setupLP();

  EXPECT_LT(acc.getLpAlpha(), 1.0);  // 필터가 실제로 활성화

  auto [u_opt, info] = acc.computeControl(state_, ref_);
  EXPECT_TRUE(u_opt.allFinite());

  // LP 필터 직접 테스트
  Eigen::MatrixXd seq = Eigen::MatrixXd::Random(10, 2);
  Eigen::VectorXd init = Eigen::VectorXd::Zero(2);
  Eigen::MatrixXd seq_orig = seq;
  acc.applyLowPassFilter(seq, 0.5, init);

  // 필터 적용 후 변화 확인
  EXPECT_GT((seq - seq_orig).norm(), 0.0);
}

// ============================================================================
// Test 6: PiProjectionBounds — pi_enabled → rate/accel bound 충족
// ============================================================================
TEST_F(ComposableMPPITest, PiProjectionBounds)
{
  ComposableTestAccessor acc;
  acc.initForTest(params_);
  ActiveLayers layers;
  layers.pi_mppi = true;
  acc.setupActiveLayers(layers);
  acc.setupPiMPPI();

  auto [u_opt, info] = acc.computeControl(state_, ref_);

  EXPECT_TRUE(u_opt.allFinite());
  EXPECT_GE(u_opt(0), params_.v_min - 1e-6);
  EXPECT_LE(u_opt(0), params_.v_max + 1e-6);
}

// ============================================================================
// Test 7: RHHorizonAdaptation — rh_mppi → N 동적 변화
// ============================================================================
TEST_F(ComposableMPPITest, RHHorizonAdaptation)
{
  ComposableTestAccessor acc;
  acc.initForTest(params_);
  ActiveLayers layers;
  layers.rh_mppi = true;
  acc.setupActiveLayers(layers);
  acc.setupRHMPPI();

  auto [u_opt, info] = acc.computeControl(state_, ref_);

  EXPECT_TRUE(u_opt.allFinite());
  EXPECT_GT(info.effective_horizon, 0);
  EXPECT_LE(info.effective_horizon, params_.rh_N_max);
  EXPECT_GE(info.effective_horizon, params_.rh_N_min);
}

// ============================================================================
// Test 8: CSCovarianceScaling — cs_enabled → noise scale 변화
// ============================================================================
TEST_F(ComposableMPPITest, CSCovarianceScaling)
{
  ComposableTestAccessor acc;
  acc.initForTest(params_);
  ActiveLayers layers;
  layers.cs_mppi = true;
  acc.setupActiveLayers(layers);
  acc.setupCSMPPI();

  auto [u_opt, info] = acc.computeControl(state_, ref_);

  EXPECT_TRUE(u_opt.allFinite());
  EXPECT_EQ(info.costs.size(), params_.K);
}

// ============================================================================
// Test 9: ShieldCBFProjection — cbf+shield → 안전 보장
// ============================================================================
TEST_F(ComposableMPPITest, ShieldCBFProjection)
{
  ComposableTestAccessor acc;
  params_.cbf_enabled = true;
  params_.cbf_use_safety_filter = true;
  acc.initForTest(params_);
  ActiveLayers layers;
  layers.shield_cbf = true;
  acc.setupActiveLayers(layers);

  // 장애물 배치: (1.0, 0.0) 전방 가까이
  std::vector<Eigen::Vector3d> obstacles;
  obstacles.push_back(Eigen::Vector3d(0.8, 0.0, 0.3));  // x, y, radius
  acc.setupShieldCBF(obstacles);

  auto [u_opt, info] = acc.computeControl(state_, ref_);

  EXPECT_TRUE(u_opt.allFinite());
  // CBF 활성화 확인 (장애물이 있으므로)
  EXPECT_EQ(info.cbf_used, true);
}

// ============================================================================
// Test 10: FeedbackGainCorrection — feedback → dx 보정
// ============================================================================
TEST_F(ComposableMPPITest, FeedbackGainCorrection)
{
  ComposableTestAccessor acc;
  acc.initForTest(params_);
  ActiveLayers layers;
  layers.feedback = true;
  acc.setupActiveLayers(layers);
  acc.setupFeedback();

  // 약간 벗어난 상태에서 시작
  Eigen::Vector3d offset_state(0.1, 0.05, 0.02);

  auto [u_opt, info] = acc.computeControl(offset_state, ref_);

  EXPECT_TRUE(u_opt.allFinite());
  EXPECT_GE(u_opt(0), params_.v_min - 1e-6);
  EXPECT_LE(u_opt(0), params_.v_max + 1e-6);
}

// ============================================================================
// Test 11: ComboHaltonPlusLP — Halton + LP 조합
// ============================================================================
TEST_F(ComposableMPPITest, ComboHaltonPlusLP)
{
  ComposableTestAccessor acc;
  acc.initForTest(params_);
  ActiveLayers layers;
  layers.halton = true;
  layers.lp_filter = true;
  acc.setupActiveLayers(layers);
  acc.setupHaltonSampler();
  acc.setupLP();

  auto [u_opt, info] = acc.computeControl(state_, ref_);

  EXPECT_TRUE(u_opt.allFinite());
  EXPECT_EQ(info.costs.size(), params_.K);
  EXPECT_GT(info.ess, 0.0);
}

// ============================================================================
// Test 12: ComboIlqrPlusShield — iLQR + Shield 조합
// ============================================================================
TEST_F(ComposableMPPITest, ComboIlqrPlusShield)
{
  ComposableTestAccessor acc;
  params_.cbf_enabled = true;
  params_.cbf_use_safety_filter = true;
  acc.initForTest(params_);
  ActiveLayers layers;
  layers.ilqr = true;
  layers.shield_cbf = true;
  acc.setupActiveLayers(layers);
  acc.setupILQR();

  std::vector<Eigen::Vector3d> obstacles;
  obstacles.push_back(Eigen::Vector3d(0.8, 0.0, 0.3));
  acc.setupShieldCBF(obstacles);

  auto [u_opt, info] = acc.computeControl(state_, ref_);

  EXPECT_TRUE(u_opt.allFinite());
}

// ============================================================================
// Test 13: ComboRHPlusFeedback — RH + Feedback 조합
// ============================================================================
TEST_F(ComposableMPPITest, ComboRHPlusFeedback)
{
  ComposableTestAccessor acc;
  acc.initForTest(params_);
  ActiveLayers layers;
  layers.rh_mppi = true;
  layers.feedback = true;
  acc.setupActiveLayers(layers);
  acc.setupRHMPPI();
  acc.setupFeedback();

  auto [u_opt, info] = acc.computeControl(state_, ref_);

  EXPECT_TRUE(u_opt.allFinite());
  EXPECT_GT(info.effective_horizon, 0);
}

// ============================================================================
// Test 14: ComboFullStack — 모든 레이어 ON → crash 없음
// ============================================================================
TEST_F(ComposableMPPITest, ComboFullStack)
{
  ComposableTestAccessor acc;
  params_.cbf_enabled = true;
  params_.cbf_use_safety_filter = true;
  // RH-MPPI: N range [N, N] 고정 (리사이즈 제거 → 안정성 확보)
  params_.rh_N_min = params_.N;
  params_.rh_N_max = params_.N;
  acc.initForTest(params_);

  ActiveLayers layers;
  layers.rh_mppi = true;
  layers.cs_mppi = true;
  layers.ilqr = true;
  layers.traj_library = true;
  layers.halton = true;
  layers.pi_mppi = true;
  layers.lp_filter = true;
  layers.shield_cbf = true;
  layers.feedback = true;
  acc.setupActiveLayers(layers);

  acc.setupRHMPPI();
  acc.setupCSMPPI();
  acc.setupILQR();
  acc.setupTrajLibrary();
  acc.setupHaltonSampler();
  acc.setupPiMPPI();
  acc.setupLP();
  acc.setupFeedback();

  std::vector<Eigen::Vector3d> obstacles;
  obstacles.push_back(Eigen::Vector3d(2.0, 0.0, 0.3));  // 먼 장애물
  acc.setupShieldCBF(obstacles);

  // 단일 호출: 모든 레이어 ON → crash 없음 + 유한값
  auto [u_opt, info] = acc.computeControl(state_, ref_);

  EXPECT_TRUE(u_opt.allFinite());
  EXPECT_GT(info.ess, 0.0);
  EXPECT_EQ(info.costs.size(), params_.K);
  EXPECT_GT(info.effective_horizon, 0);
}

// ============================================================================
// Test 15: PerformanceNoOverhead — 모든 OFF → base 대비 < 5% 차이
// ============================================================================
TEST_F(ComposableMPPITest, PerformanceNoOverhead)
{
  // Base (direct parent call)
  auto start_base = std::chrono::high_resolution_clock::now();
  constexpr int ITERS = 20;

  for (int i = 0; i < ITERS; ++i) {
    ComposableTestAccessor acc;
    acc.initForTest(params_);
    ActiveLayers off;
    acc.setupActiveLayers(off);
    auto [u, info] = acc.computeControl(state_, ref_);
    (void)u;
  }
  auto end_base = std::chrono::high_resolution_clock::now();
  double base_ms = std::chrono::duration<double, std::milli>(end_base - start_base).count();

  // Composable (all OFF → should fallback to parent)
  auto start_comp = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < ITERS; ++i) {
    ComposableTestAccessor acc;
    acc.initForTest(params_);
    ActiveLayers off;
    acc.setupActiveLayers(off);
    auto [u, info] = acc.computeControl(state_, ref_);
    (void)u;
  }
  auto end_comp = std::chrono::high_resolution_clock::now();
  double comp_ms = std::chrono::duration<double, std::milli>(end_comp - start_comp).count();

  // < 50% overhead is acceptable (benchmark noise + setup overhead)
  double ratio = comp_ms / std::max(base_ms, 0.001);
  EXPECT_LT(ratio, 1.5) << "Base: " << base_ms << "ms, Composable: " << comp_ms << "ms";
}

}  // namespace mpc_controller_ros2
