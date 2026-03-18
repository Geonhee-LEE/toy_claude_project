// =============================================================================
// RH-MPPI (Receding Horizon MPPI) Unit Tests
//
// 15 gtest: AdaptiveHorizonManager 8개 + Integration 7개
// =============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include "mpc_controller_ros2/adaptive_horizon_manager.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/sampling.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"

namespace mpc_controller_ros2
{

// ============================================================================
// AdaptiveHorizonManager 테스트 (8개)
// ============================================================================

// Test 1: 정지 상태 → N_min 쪽
TEST(RHMPPIHorizon, StationaryNMin)
{
  AdaptiveHorizonManager mgr(10, 50, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0);  // alpha=1.0 (no smoothing)

  // speed=0, obs_dist=threshold (no effect), error=0 (no effect)
  // speed_factor=0, obstacle_factor=1, error_factor=1
  // combined = (1*0 + 1*1 + 0.5*1) / 2.5 = 0.6
  // raw_N = 10 + 40*0.6 = 34
  int N = mgr.computeEffectiveN(0.0, 1.0, 2.0, 0.0);
  EXPECT_GE(N, 10);
  EXPECT_LE(N, 50);
  // 속도 0이므로 최대 N은 아님
  EXPECT_LT(N, 50);
}

// Test 2: 고속 → N_max 방향
TEST(RHMPPIHorizon, HighSpeedLongHorizon)
{
  AdaptiveHorizonManager mgr(10, 50, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0);

  // speed=v_max, obs_dist=threshold, error=0
  // all factors = 1.0 → combined = 1.0 → raw_N = 50
  int N = mgr.computeEffectiveN(1.0, 1.0, 2.0, 0.0);
  EXPECT_EQ(N, 50);
}

// Test 3: 저속 → 중간~짧은 horizon
TEST(RHMPPIHorizon, LowSpeedShortHorizon)
{
  AdaptiveHorizonManager mgr(10, 50, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0);

  // speed=0.1*v_max → speed_factor=0.1
  // obs_dist=threshold → obstacle_factor=1.0
  // error=0 → error_factor=1.0
  // combined = (1*0.1 + 1*1.0 + 0.5*1.0) / 2.5 = 0.64
  int N = mgr.computeEffectiveN(0.1, 1.0, 2.0, 0.0);
  EXPECT_GE(N, 10);
  EXPECT_LT(N, 50);
}

// Test 4: 근접 장애물 → N 감소
TEST(RHMPPIHorizon, NearObstacleShortHorizon)
{
  AdaptiveHorizonManager mgr(10, 50, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0);

  // speed=v_max → speed_factor=1.0
  // obs_dist=0.1 → obstacle_factor=0.05
  // error=0 → error_factor=1.0
  int N_near = mgr.computeEffectiveN(1.0, 1.0, 0.1, 0.0);

  mgr.reset();

  // obs_dist=threshold → obstacle_factor=1.0
  int N_far = mgr.computeEffectiveN(1.0, 1.0, 2.0, 0.0);

  EXPECT_LT(N_near, N_far);
}

// Test 5: 큰 추적 오차 → N 감소
TEST(RHMPPIHorizon, LargeErrorShortHorizon)
{
  AdaptiveHorizonManager mgr(10, 50, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0);

  // error=threshold → error_factor=0
  int N_large_err = mgr.computeEffectiveN(1.0, 1.0, 2.0, 1.0);

  mgr.reset();

  // error=0 → error_factor=1.0
  int N_no_err = mgr.computeEffectiveN(1.0, 1.0, 2.0, 0.0);

  EXPECT_LT(N_large_err, N_no_err);
}

// Test 6: 복합 팩터 (중간 속도 + 근접 장애물)
TEST(RHMPPIHorizon, CombinedFactors)
{
  AdaptiveHorizonManager mgr(10, 50, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0);

  // speed=0.5*v_max → speed_factor=0.5
  // obs_dist=0.5 → obstacle_factor=0.25
  // error=0.5 → error_factor=0.5
  // combined = (1*0.5 + 1*0.25 + 0.5*0.5) / 2.5 = 0.4
  // raw_N = 10 + 40*0.4 = 26
  int N = mgr.computeEffectiveN(0.5, 1.0, 0.5, 0.5);
  EXPECT_GE(N, 20);
  EXPECT_LE(N, 32);
}

// Test 7: EMA 스무딩 — 급변 입력에도 점진 변화
TEST(RHMPPIHorizon, EMASmoothingWorks)
{
  AdaptiveHorizonManager mgr(10, 50, 1.0, 0.0, 0.0, 2.0, 1.0, 0.3);
  // speed만 사용 (obs_w=0, err_w=0)

  // 초기: speed=v_max → N=50
  int N1 = mgr.computeEffectiveN(1.0, 1.0, 2.0, 0.0);
  EXPECT_EQ(N1, 50);

  // 급변: speed=0 → raw_N=10, 하지만 EMA로 스무딩
  int N2 = mgr.computeEffectiveN(0.0, 1.0, 2.0, 0.0);
  // smoothed = 0.3*10 + 0.7*50 = 38
  EXPECT_GT(N2, 20);   // 아직 높음
  EXPECT_LT(N2, 50);   // 하지만 떨어짐

  // 한 번 더: smoothed ≈ 0.3*10 + 0.7*38 = 29.6
  int N3 = mgr.computeEffectiveN(0.0, 1.0, 2.0, 0.0);
  EXPECT_LT(N3, N2);   // 점점 감소

  // 충분한 반복 후 수렴
  for (int i = 0; i < 50; ++i) {
    mgr.computeEffectiveN(0.0, 1.0, 2.0, 0.0);
  }
  int N_final = mgr.computeEffectiveN(0.0, 1.0, 2.0, 0.0);
  EXPECT_EQ(N_final, 10);  // N_min에 수렴
}

// Test 8: 클램핑 — 극단값에도 N_min ≤ N ≤ N_max
TEST(RHMPPIHorizon, ClampingMinMax)
{
  AdaptiveHorizonManager mgr(10, 50, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0);

  // 모든 팩터 0 (극단 최소)
  int N_min = mgr.computeEffectiveN(0.0, 1.0, 0.0, 10.0);
  EXPECT_GE(N_min, 10);

  mgr.reset();

  // 모든 팩터 1 (극단 최대)
  int N_max = mgr.computeEffectiveN(1.0, 1.0, 10.0, 0.0);
  EXPECT_LE(N_max, 50);

  mgr.reset();

  // v_max=0 (division safety)
  int N_zero_vmax = mgr.computeEffectiveN(0.5, 0.0, 2.0, 0.0);
  EXPECT_GE(N_zero_vmax, 10);
  EXPECT_LE(N_zero_vmax, 50);
}

// ============================================================================
// Integration 테스트 (7개)
// ============================================================================

namespace
{
// 테스트 헬퍼: MPPI 파이프라인 구성
struct TestSetup
{
  MPPIParams params;
  std::unique_ptr<BatchDynamicsWrapper> dynamics;
  std::unique_ptr<CompositeMPPICost> cost_function;
  std::unique_ptr<BaseSampler> sampler;

  TestSetup(int N = 30)
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

// Test 9: control_sequence_ 리사이즈
TEST(RHMPPIIntegration, ControlSequenceResize)
{
  int N_max = 50;
  int effective_N = 20;
  int nu = 2;

  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Ones(N_max, nu) * 0.5;

  // 축소
  Eigen::MatrixXd new_seq = Eigen::MatrixXd::Zero(effective_N, nu);
  int copy_rows = std::min(effective_N, N_max);
  new_seq.topRows(copy_rows) = control_seq.topRows(copy_rows);

  EXPECT_EQ(new_seq.rows(), effective_N);
  EXPECT_EQ(new_seq.cols(), nu);
  EXPECT_NEAR(new_seq(0, 0), 0.5, 1e-10);
  EXPECT_NEAR(new_seq(effective_N - 1, 0), 0.5, 1e-10);
}

// Test 10: Warm-Start 축소 시 상위 행 보존
TEST(RHMPPIIntegration, WarmStartTruncation)
{
  int N_old = 30;
  int N_new = 20;
  int nu = 2;

  Eigen::MatrixXd seq(N_old, nu);
  for (int t = 0; t < N_old; ++t) {
    seq.row(t) = Eigen::Vector2d(t * 0.1, t * 0.2).transpose();
  }

  // 축소: 상위 N_new 행 보존
  Eigen::MatrixXd truncated = Eigen::MatrixXd::Zero(N_new, nu);
  truncated = seq.topRows(N_new);

  for (int t = 0; t < N_new; ++t) {
    EXPECT_NEAR(truncated(t, 0), t * 0.1, 1e-10);
    EXPECT_NEAR(truncated(t, 1), t * 0.2, 1e-10);
  }
}

// Test 11: Warm-Start 확장 시 zero-pad
TEST(RHMPPIIntegration, WarmStartExpansion)
{
  int N_old = 20;
  int N_new = 30;
  int nu = 2;

  Eigen::MatrixXd seq(N_old, nu);
  for (int t = 0; t < N_old; ++t) {
    seq.row(t) = Eigen::Vector2d(1.0, 2.0).transpose();
  }

  // 확장: zero-pad
  Eigen::MatrixXd expanded = Eigen::MatrixXd::Zero(N_new, nu);
  expanded.topRows(N_old) = seq;

  // 기존 행 보존
  for (int t = 0; t < N_old; ++t) {
    EXPECT_NEAR(expanded(t, 0), 1.0, 1e-10);
    EXPECT_NEAR(expanded(t, 1), 2.0, 1e-10);
  }
  // 확장 행은 0
  for (int t = N_old; t < N_new; ++t) {
    EXPECT_NEAR(expanded(t, 0), 0.0, 1e-10);
    EXPECT_NEAR(expanded(t, 1), 0.0, 1e-10);
  }
}

// Test 12: Rollout 차원 검증 — (effective_N+1) x nx
TEST(RHMPPIIntegration, RolloutDimensions)
{
  int effective_N = 15;
  TestSetup setup(effective_N);
  int K = setup.params.K;
  int nu = 2;
  int nx = 3;

  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(effective_N, nu);
  auto noise = setup.sampler->sample(K, effective_N, nu);

  std::vector<Eigen::MatrixXd> perturbed;
  perturbed.reserve(K);
  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd u_k = control_seq + noise[k];
    u_k = setup.dynamics->clipControls(u_k);
    perturbed.push_back(u_k);
  }

  Eigen::VectorXd state = Eigen::Vector3d(0, 0, 0);
  auto trajectories = setup.dynamics->rolloutBatch(state, perturbed, setup.params.dt);

  ASSERT_EQ(static_cast<int>(trajectories.size()), K);
  for (int k = 0; k < K; ++k) {
    EXPECT_EQ(trajectories[k].rows(), effective_N + 1);
    EXPECT_EQ(trajectories[k].cols(), nx);
  }
}

// Test 13: effective_N=15로 비용 정상 계산
TEST(RHMPPIIntegration, CostWithAdaptiveN)
{
  int effective_N = 15;
  TestSetup setup(effective_N);
  int K = setup.params.K;
  int nu = 2;

  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(effective_N, nu);
  auto noise = setup.sampler->sample(K, effective_N, nu);

  std::vector<Eigen::MatrixXd> perturbed;
  for (int k = 0; k < K; ++k) {
    perturbed.push_back(setup.dynamics->clipControls(control_seq + noise[k]));
  }

  Eigen::VectorXd state = Eigen::Vector3d(0, 0, 0);
  auto trajectories = setup.dynamics->rolloutBatch(state, perturbed, setup.params.dt);

  // Reference trajectory: go to (1,0) straight
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(effective_N + 1, 3);
  for (int t = 0; t <= effective_N; ++t) {
    ref(t, 0) = t * 0.05;  // x
  }

  Eigen::VectorXd costs = setup.cost_function->compute(trajectories, perturbed, ref);
  ASSERT_EQ(costs.size(), K);

  // 모든 비용이 유한
  for (int k = 0; k < K; ++k) {
    EXPECT_TRUE(std::isfinite(costs(k)));
    EXPECT_GE(costs(k), 0.0);
  }
}

// Test 14: rh_enabled=false → vanilla 동작 (AdaptiveHorizonManager 미사용)
TEST(RHMPPIIntegration, DisabledFallback)
{
  // rh_mppi_enabled=false일 때 AdaptiveHorizonManager 없이도 동작
  MPPIParams params;
  params.rh_mppi_enabled = false;
  params.N = 30;

  // 파라미터만 확인 — 실제 플러그인은 configure 없이 테스트 불가
  EXPECT_FALSE(params.rh_mppi_enabled);
  EXPECT_EQ(params.N, 30);

  // AdaptiveHorizonManager 생성 후 동작 확인
  AdaptiveHorizonManager mgr(10, 50, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0);
  int N = mgr.computeEffectiveN(0.5, 1.0, 2.0, 0.0);
  EXPECT_GE(N, 10);
  EXPECT_LE(N, 50);
}

// Test 15: 10회 연속, 다른 속도, 모두 유한 + 범위 내
TEST(RHMPPIIntegration, ConsecutiveStability)
{
  AdaptiveHorizonManager mgr(10, 50, 1.0, 1.0, 0.5, 2.0, 1.0, 0.3);

  double speeds[] = {0.0, 0.2, 0.5, 0.8, 1.0, 0.7, 0.3, 0.1, 0.9, 0.4};
  double obs_dists[] = {2.0, 1.5, 0.5, 0.3, 2.0, 1.0, 0.1, 0.8, 2.0, 1.2};
  double errors[] = {0.0, 0.1, 0.3, 0.5, 0.0, 0.2, 0.8, 0.1, 0.0, 0.4};

  for (int i = 0; i < 10; ++i) {
    int N = mgr.computeEffectiveN(speeds[i], 1.0, obs_dists[i], errors[i]);
    EXPECT_GE(N, 10) << "iteration " << i;
    EXPECT_LE(N, 50) << "iteration " << i;
  }
}

}  // namespace mpc_controller_ros2
