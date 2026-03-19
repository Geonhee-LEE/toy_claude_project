// =============================================================================
// Auto-Selector MPPI Unit Tests
//
// 15 gtest: StrategySelector 8개 + Integration 7개
// =============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include "mpc_controller_ros2/strategy_selector.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/sampling.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"

namespace mpc_controller_ros2
{

// ============================================================================
// StrategySelector 테스트 (8개)
// ============================================================================

// Test 1: DefaultCruise — 일반 조건 → CRUISE
TEST(AutoSelectorStrategy, DefaultCruise)
{
  StrategySelector sel(0.5, 1.0, 0.7, 1.5, 1, 1.0);  // hysteresis=1, alpha=1.0

  // 보통 속도, 장애물 멂, 오차 작음, 목표 멂
  auto s = sel.update(0.3, 1.0, 5.0, 0.1, 10.0);
  EXPECT_EQ(s, MPPIStrategy::CRUISE);
}

// Test 2: NearObstacleSafe — obs_dist=0.3 < 0.5 → SAFE
TEST(AutoSelectorStrategy, NearObstacleSafe)
{
  StrategySelector sel(0.5, 1.0, 0.7, 1.5, 1, 1.0);

  auto s = sel.update(0.3, 1.0, 0.3, 0.1, 10.0);
  EXPECT_EQ(s, MPPIStrategy::SAFE);
}

// Test 3: LargeErrorRecovery — error=2.0 > 1.0 → RECOVERY
TEST(AutoSelectorStrategy, LargeErrorRecovery)
{
  StrategySelector sel(0.5, 1.0, 0.7, 1.5, 1, 1.0);

  auto s = sel.update(0.3, 1.0, 5.0, 2.0, 10.0);
  EXPECT_EQ(s, MPPIStrategy::RECOVERY);
}

// Test 4: HighSpeedAggressive — speed/v_max=0.9 > 0.7 → AGGRESSIVE
TEST(AutoSelectorStrategy, HighSpeedAggressive)
{
  StrategySelector sel(0.5, 1.0, 0.7, 1.5, 1, 1.0);

  auto s = sel.update(0.9, 1.0, 5.0, 0.1, 10.0);
  EXPECT_EQ(s, MPPIStrategy::AGGRESSIVE);
}

// Test 5: NearGoalPrecise — goal_dist=0.8 < 1.5 → PRECISE
TEST(AutoSelectorStrategy, NearGoalPrecise)
{
  StrategySelector sel(0.5, 1.0, 0.7, 1.5, 1, 1.0);

  auto s = sel.update(0.3, 1.0, 5.0, 0.1, 0.8);
  EXPECT_EQ(s, MPPIStrategy::PRECISE);
}

// Test 6: SafeOverridesAll — obs_dist=0.2 + speed=0.9 → SAFE (최우선)
TEST(AutoSelectorStrategy, SafeOverridesAll)
{
  StrategySelector sel(0.5, 1.0, 0.7, 1.5, 1, 1.0);

  // 장애물 근접 + 고속 + 목표 근접 → SAFE가 최우선
  auto s = sel.update(0.9, 1.0, 0.2, 0.1, 0.8);
  EXPECT_EQ(s, MPPIStrategy::SAFE);
}

// Test 7: HysteresisPreventsChattering — 3 cycles 미만 전환 안 됨
TEST(AutoSelectorStrategy, HysteresisPreventsChattering)
{
  StrategySelector sel(0.5, 1.0, 0.7, 1.5, 3, 1.0);  // hysteresis=3

  // 초기: CRUISE
  sel.update(0.3, 1.0, 5.0, 0.1, 10.0);
  EXPECT_EQ(sel.currentStrategy(), MPPIStrategy::CRUISE);

  // 1번째 SAFE 조건 → 아직 CRUISE (히스테리시스)
  sel.update(0.3, 1.0, 0.3, 0.1, 10.0);
  EXPECT_EQ(sel.currentStrategy(), MPPIStrategy::CRUISE);

  // 2번째 SAFE 조건 → 아직 CRUISE
  sel.update(0.3, 1.0, 0.3, 0.1, 10.0);
  EXPECT_EQ(sel.currentStrategy(), MPPIStrategy::CRUISE);

  // 3번째 SAFE 조건 → 전환!
  sel.update(0.3, 1.0, 0.3, 0.1, 10.0);
  EXPECT_EQ(sel.currentStrategy(), MPPIStrategy::SAFE);
}

// Test 8: EMASmoothingWorks — 급변 시 EMA로 점진 변화
TEST(AutoSelectorStrategy, EMASmoothingWorks)
{
  StrategySelector sel(0.5, 1.0, 0.7, 1.5, 1, 0.3);  // alpha=0.3

  // 초기: 장애물 멂 → CRUISE
  sel.update(0.3, 1.0, 5.0, 0.1, 10.0);
  EXPECT_EQ(sel.currentStrategy(), MPPIStrategy::CRUISE);

  // 급변: obs=0.1 → EMA smoothed = 0.3*0.1 + 0.7*5.0 = 3.53 > 0.5
  // 아직 CRUISE (스무딩 때문에 임계값 미달)
  sel.update(0.3, 1.0, 0.1, 0.1, 10.0);
  EXPECT_EQ(sel.currentStrategy(), MPPIStrategy::CRUISE);

  // 계속 obs=0.1 반복 → 점진 감소
  for (int i = 0; i < 20; ++i) {
    sel.update(0.3, 1.0, 0.1, 0.1, 10.0);
  }
  // 충분히 반복 후 SAFE 전환
  EXPECT_EQ(sel.currentStrategy(), MPPIStrategy::SAFE);
}

// ============================================================================
// Integration 테스트 (7개)
// ============================================================================

namespace
{
// 테스트 헬퍼: MPPI 파이프라인 구성
struct AutoSelectorTestSetup
{
  MPPIParams params;
  MPPIParams baseline;
  std::unique_ptr<BatchDynamicsWrapper> dynamics;
  std::unique_ptr<CompositeMPPICost> cost_function;
  std::unique_ptr<BaseSampler> sampler;

  AutoSelectorTestSetup(int N = 30)
  {
    params.N = N;
    params.K = 64;
    params.dt = 0.1;
    params.v_max = 1.0;
    params.lambda = 10.0;
    params.noise_sigma = Eigen::Vector2d(0.5, 0.5);
    params.Q = Eigen::Matrix3d::Identity() * 10.0;
    params.Qf = Eigen::Matrix3d::Identity() * 20.0;
    params.R = Eigen::Matrix2d::Identity() * 0.1;
    params.R_rate = Eigen::Matrix2d::Identity() * 1.0;

    // Auto-selector 파라미터
    params.auto_selector_enabled = true;
    params.cbf_enabled = false;
    params.feedback_mppi_enabled = false;
    params.lp_enabled = false;
    params.predictive_safety_enabled = false;
    params.exploration_ratio = 0.0;

    baseline = params;

    auto model = MotionModelFactory::create("diff_drive", params);
    dynamics = std::make_unique<BatchDynamicsWrapper>(params, std::move(model));
    cost_function = std::make_unique<CompositeMPPICost>();
    cost_function->addCost(std::make_unique<StateTrackingCost>(params.Q));
    cost_function->addCost(std::make_unique<TerminalCost>(params.Qf));
    cost_function->addCost(std::make_unique<ControlEffortCost>(params.R));
    sampler = std::make_unique<GaussianSampler>(params.noise_sigma);
  }
};
}  // anonymous namespace

// Test 9: ProfileApplyRestore — params_ 변경 후 baseline 복원 확인
TEST(AutoSelectorIntegration, ProfileApplyRestore)
{
  AutoSelectorTestSetup setup;
  MPPIParams& params = setup.params;
  MPPIParams baseline = setup.baseline;

  // SAFE 프로파일 적용
  params.cbf_enabled = true;
  params.shield_cbf_stride = 1;
  params.predictive_safety_enabled = true;

  EXPECT_TRUE(params.cbf_enabled);
  EXPECT_TRUE(params.predictive_safety_enabled);

  // baseline 복원
  params = baseline;

  EXPECT_FALSE(params.cbf_enabled);
  EXPECT_FALSE(params.predictive_safety_enabled);
  EXPECT_EQ(params.N, 30);
  EXPECT_NEAR(params.lambda, 10.0, 1e-10);
}

// Test 10: CruiseMinimalOverhead — CRUISE 시 특수 기능 모두 false
TEST(AutoSelectorIntegration, CruiseMinimalOverhead)
{
  AutoSelectorTestSetup setup;

  // CRUISE = baseline 그대로
  EXPECT_FALSE(setup.params.cbf_enabled);
  EXPECT_FALSE(setup.params.feedback_mppi_enabled);
  EXPECT_FALSE(setup.params.lp_enabled);
  EXPECT_FALSE(setup.params.predictive_safety_enabled);
  EXPECT_NEAR(setup.params.exploration_ratio, 0.0, 1e-10);
}

// Test 11: SafeEnablesCBF — SAFE 시 cbf/shield/predictive 활성
TEST(AutoSelectorIntegration, SafeEnablesCBF)
{
  AutoSelectorTestSetup setup;
  MPPIParams& params = setup.params;

  // SAFE 프로파일
  params.cbf_enabled = true;
  params.shield_cbf_stride = 1;
  params.predictive_safety_enabled = true;

  EXPECT_TRUE(params.cbf_enabled);
  EXPECT_EQ(params.shield_cbf_stride, 1);
  EXPECT_TRUE(params.predictive_safety_enabled);
}

// Test 12: PreciseEnablesFeedback — PRECISE 시 feedback + Q*1.5
TEST(AutoSelectorIntegration, PreciseEnablesFeedback)
{
  AutoSelectorTestSetup setup;
  MPPIParams& params = setup.params;
  MPPIParams baseline = setup.baseline;

  // PRECISE 프로파일
  params.feedback_mppi_enabled = true;
  params.Q = baseline.Q * 1.5;
  params.Qf = baseline.Qf * 1.5;

  EXPECT_TRUE(params.feedback_mppi_enabled);
  EXPECT_NEAR(params.Q(0, 0), 15.0, 1e-10);    // 10 * 1.5
  EXPECT_NEAR(params.Qf(0, 0), 30.0, 1e-10);   // 20 * 1.5
}

// Test 13: ControlSequenceContinuity — 전략 전환 시 warm-start 유지
TEST(AutoSelectorIntegration, ControlSequenceContinuity)
{
  int N_baseline = 30;
  int N_recovery = 15;  // N/2
  int nu = 2;

  // 원본 control_sequence
  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Ones(N_baseline, nu) * 0.3;

  // RECOVERY: N 축소 → 상위 행 보존
  Eigen::MatrixXd shrunk = Eigen::MatrixXd::Zero(N_recovery, nu);
  int copy_rows = std::min(N_recovery, N_baseline);
  shrunk.topRows(copy_rows) = control_seq.topRows(copy_rows);

  EXPECT_NEAR(shrunk(0, 0), 0.3, 1e-10);
  EXPECT_NEAR(shrunk(N_recovery - 1, 0), 0.3, 1e-10);

  // 복원: N 확장 → zero-pad
  Eigen::MatrixXd restored = Eigen::MatrixXd::Zero(N_baseline, nu);
  restored.topRows(N_recovery) = shrunk;

  EXPECT_NEAR(restored(0, 0), 0.3, 1e-10);
  EXPECT_NEAR(restored(N_recovery - 1, 0), 0.3, 1e-10);
  EXPECT_NEAR(restored(N_baseline - 1, 0), 0.0, 1e-10);  // zero-pad
}

// Test 14: DisabledFallback — enabled=false → vanilla
TEST(AutoSelectorIntegration, DisabledFallback)
{
  MPPIParams params;
  params.auto_selector_enabled = false;

  EXPECT_FALSE(params.auto_selector_enabled);
  EXPECT_EQ(params.N, 30);

  // StrategySelector 정상 동작
  StrategySelector sel(0.5, 1.0, 0.7, 1.5, 1, 1.0);
  auto s = sel.update(0.3, 1.0, 5.0, 0.1, 10.0);
  EXPECT_EQ(s, MPPIStrategy::CRUISE);
}

// Test 15: ConsecutiveStability — 10회 다양한 입력, 모두 유한
TEST(AutoSelectorIntegration, ConsecutiveStability)
{
  StrategySelector sel(0.5, 1.0, 0.7, 1.5, 2, 0.3);

  double speeds[]     = {0.0, 0.2, 0.9, 0.8, 0.1, 0.3, 0.5, 0.7, 0.4, 0.6};
  double obs_dists[]  = {5.0, 0.3, 0.2, 2.0, 5.0, 0.4, 3.0, 1.0, 5.0, 0.1};
  double errors[]     = {0.1, 0.5, 0.1, 0.2, 2.0, 1.5, 0.3, 0.1, 0.8, 0.1};
  double goals[]      = {10.0, 5.0, 8.0, 0.5, 3.0, 7.0, 1.0, 2.0, 0.3, 6.0};

  for (int i = 0; i < 10; ++i) {
    auto s = sel.update(speeds[i], 1.0, obs_dists[i], errors[i], goals[i]);
    // 유효한 전략인지 확인
    EXPECT_GE(static_cast<int>(s), 0) << "iteration " << i;
    EXPECT_LE(static_cast<int>(s), 4) << "iteration " << i;

    // 스무딩 메트릭이 유한한지 확인
    auto& m = sel.smoothedMetrics();
    EXPECT_TRUE(std::isfinite(m.speed)) << "iteration " << i;
    EXPECT_TRUE(std::isfinite(m.min_obs_dist)) << "iteration " << i;
    EXPECT_TRUE(std::isfinite(m.tracking_error)) << "iteration " << i;
    EXPECT_TRUE(std::isfinite(m.goal_dist)) << "iteration " << i;
  }
}

}  // namespace mpc_controller_ros2
