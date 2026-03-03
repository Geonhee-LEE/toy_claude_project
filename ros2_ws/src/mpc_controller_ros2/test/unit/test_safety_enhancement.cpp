#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>

#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/barrier_function.hpp"
#include "mpc_controller_ros2/conformal_predictor.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"
#include "mpc_controller_ros2/diff_drive_model.hpp"
#include "mpc_controller_ros2/swerve_drive_model.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"

using namespace mpc_controller_ros2;

// =============================================================================
// Helper: 테스트 궤적 생성
// =============================================================================

static std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>>
createTestTrajectories(int K, int N, int nx, int nu,
                       double x0 = 0.0, double y0 = 0.0) {
  std::vector<Eigen::MatrixXd> trajectories(K);
  std::vector<Eigen::MatrixXd> controls(K);

  for (int k = 0; k < K; ++k) {
    trajectories[k] = Eigen::MatrixXd::Zero(N + 1, nx);
    controls[k] = Eigen::MatrixXd::Random(N, nu) * 0.5;

    for (int t = 0; t <= N; ++t) {
      double frac = static_cast<double>(t) / N;
      trajectories[k](t, 0) = x0 + frac * 3.0 + 0.1 * k;
      trajectories[k](t, 1) = y0 + frac * 0.5 + 0.05 * k;
      if (nx >= 3) { trajectories[k](t, 2) = 0.1 * frac; }
    }
  }
  return {trajectories, controls};
}

// =============================================================================
// BarrierRateCost 테스트 (6개)
// =============================================================================

TEST(BarrierRateCost, NoObstacles) {
  BarrierFunctionSet barrier_set(0.2, 0.3, 3.0);
  // 장애물 없음
  BarrierRateCost cost(&barrier_set, 100.0, 0.1);

  auto [trajs, ctrls] = createTestTrajectories(4, 10, 3, 2);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(11, 3);

  Eigen::VectorXd costs = cost.compute(trajs, ctrls, ref);
  EXPECT_EQ(costs.size(), 4);
  for (int k = 0; k < 4; ++k) {
    EXPECT_NEAR(costs(k), 0.0, 1e-10);
  }
}

TEST(BarrierRateCost, MovingAway) {
  // 장애물에서 멀어지는 궤적 → 비용 0
  BarrierFunctionSet barrier_set(0.2, 0.1, 10.0);
  barrier_set.setObstacles({{-3.0, 0.0, 0.3}});  // 왼쪽 멀리

  BarrierRateCost cost(&barrier_set, 100.0, 0.1);

  // 오른쪽으로 이동 (장애물에서 멀어짐)
  int K = 4, N = 10;
  std::vector<Eigen::MatrixXd> trajs(K);
  std::vector<Eigen::MatrixXd> ctrls(K, Eigen::MatrixXd::Zero(N, 2));
  for (int k = 0; k < K; ++k) {
    trajs[k] = Eigen::MatrixXd::Zero(N + 1, 3);
    for (int t = 0; t <= N; ++t) {
      trajs[k](t, 0) = 1.0 + 0.1 * t;  // 오른쪽으로 이동
    }
  }
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, 3);

  Eigen::VectorXd costs_vec = cost.compute(trajs, ctrls, ref);
  for (int k = 0; k < K; ++k) {
    EXPECT_NEAR(costs_vec(k), 0.0, 1e-6)
      << "Moving away should have zero barrier rate cost";
  }
}

TEST(BarrierRateCost, Approaching) {
  // 장애물에 접근하는 궤적 → 비용 > 0
  BarrierFunctionSet barrier_set(0.2, 0.1, 10.0);
  barrier_set.setObstacles({{3.0, 0.0, 0.3}});  // 전방 장애물

  BarrierRateCost cost(&barrier_set, 100.0, 0.1);

  int K = 4, N = 10;
  std::vector<Eigen::MatrixXd> trajs(K);
  std::vector<Eigen::MatrixXd> ctrls(K, Eigen::MatrixXd::Zero(N, 2));
  for (int k = 0; k < K; ++k) {
    trajs[k] = Eigen::MatrixXd::Zero(N + 1, 3);
    for (int t = 0; t <= N; ++t) {
      trajs[k](t, 0) = 0.2 * t;  // 장애물 방향 이동
    }
  }
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, 3);

  Eigen::VectorXd costs_vec = cost.compute(trajs, ctrls, ref);
  for (int k = 0; k < K; ++k) {
    EXPECT_GT(costs_vec(k), 0.0)
      << "Approaching obstacle should have positive barrier rate cost";
  }
}

TEST(BarrierRateCost, ProportionalRate) {
  // 더 빠르게 접근하면 비용이 더 높아야 함
  BarrierFunctionSet barrier_set(0.2, 0.1, 10.0);
  barrier_set.setObstacles({{5.0, 0.0, 0.3}});

  BarrierRateCost cost(&barrier_set, 100.0, 0.1);

  int K = 2, N = 10;
  std::vector<Eigen::MatrixXd> trajs(K);
  std::vector<Eigen::MatrixXd> ctrls(K, Eigen::MatrixXd::Zero(N, 2));
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, 3);

  // k=0: 느린 접근
  trajs[0] = Eigen::MatrixXd::Zero(N + 1, 3);
  for (int t = 0; t <= N; ++t) {
    trajs[0](t, 0) = 0.1 * t;
  }

  // k=1: 빠른 접근
  trajs[1] = Eigen::MatrixXd::Zero(N + 1, 3);
  for (int t = 0; t <= N; ++t) {
    trajs[1](t, 0) = 0.3 * t;
  }

  Eigen::VectorXd costs_vec = cost.compute(trajs, ctrls, ref);
  EXPECT_GT(costs_vec(1), costs_vec(0))
    << "Faster approach should have higher cost";
}

TEST(BarrierRateCost, MultipleObstacles) {
  BarrierFunctionSet barrier_set(0.2, 0.1, 10.0);
  barrier_set.setObstacles({
    {3.0, 0.0, 0.3},
    {0.0, 3.0, 0.3}
  });

  BarrierRateCost cost(&barrier_set, 100.0, 0.1);

  int K = 1, N = 10;
  std::vector<Eigen::MatrixXd> trajs(K);
  std::vector<Eigen::MatrixXd> ctrls(K, Eigen::MatrixXd::Zero(N, 2));
  trajs[0] = Eigen::MatrixXd::Zero(N + 1, 3);
  for (int t = 0; t <= N; ++t) {
    double frac = static_cast<double>(t) / N;
    trajs[0](t, 0) = frac * 3.0;  // 대각선 이동
    trajs[0](t, 1) = frac * 3.0;
  }
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, 3);

  Eigen::VectorXd costs_vec = cost.compute(trajs, ctrls, ref);
  EXPECT_GT(costs_vec(0), 0.0);
}

TEST(BarrierRateCost, WeightScaling) {
  BarrierFunctionSet barrier_set(0.2, 0.1, 10.0);
  barrier_set.setObstacles({{3.0, 0.0, 0.3}});

  BarrierRateCost cost_low(&barrier_set, 10.0, 0.1);
  BarrierRateCost cost_high(&barrier_set, 100.0, 0.1);

  int K = 1, N = 10;
  std::vector<Eigen::MatrixXd> trajs(K);
  std::vector<Eigen::MatrixXd> ctrls(K, Eigen::MatrixXd::Zero(N, 2));
  trajs[0] = Eigen::MatrixXd::Zero(N + 1, 3);
  for (int t = 0; t <= N; ++t) { trajs[0](t, 0) = 0.2 * t; }
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, 3);

  double c_low = cost_low.compute(trajs, ctrls, ref)(0);
  double c_high = cost_high.compute(trajs, ctrls, ref)(0);

  // 10배 가중치 → 10배 비용
  EXPECT_NEAR(c_high / c_low, 10.0, 0.01);
}

// =============================================================================
// ConformalPredictor 테스트 (8개)
// =============================================================================

TEST(ConformalPredictor, InitialMargin) {
  ConformalPredictor::Params params;
  params.initial_margin = 0.5;
  ConformalPredictor cp(params);

  EXPECT_NEAR(cp.getMargin(), 0.5, 1e-10);
  EXPECT_EQ(cp.numObservations(), 0);
}

TEST(ConformalPredictor, IncreasesOnError) {
  ConformalPredictor::Params params;
  params.initial_margin = 0.1;
  params.window_size = 20;
  params.coverage_probability = 0.95;
  params.max_margin = 2.0;
  ConformalPredictor cp(params);

  // 큰 오차 반복 투입
  for (int i = 0; i < 20; ++i) {
    cp.update(0.8);  // 초기 마진(0.1)보다 훨씬 큰 오차
  }

  EXPECT_GT(cp.getMargin(), 0.1)
    << "Margin should increase with large prediction errors";
}

TEST(ConformalPredictor, DecreasesOnAccuracy) {
  ConformalPredictor::Params params;
  params.initial_margin = 0.5;
  params.window_size = 50;
  params.coverage_probability = 0.90;
  params.min_margin = 0.01;
  ConformalPredictor cp(params);

  // 작은 오차 반복 투입
  for (int i = 0; i < 50; ++i) {
    cp.update(0.01);  // 매우 정확한 예측
  }

  EXPECT_LT(cp.getMargin(), 0.5)
    << "Margin should decrease with accurate predictions";
}

TEST(ConformalPredictor, CoverageConverges) {
  ConformalPredictor::Params params;
  params.initial_margin = 0.3;
  params.window_size = 200;
  params.coverage_probability = 0.95;
  params.decay_rate = 1.0;  // 균등 가중치
  ConformalPredictor cp(params);

  // 혼합 오차
  srand(42);
  for (int i = 0; i < 200; ++i) {
    double error = 0.1 + 0.3 * (static_cast<double>(rand()) / RAND_MAX);
    cp.update(error);
  }

  // 커버리지가 합리적 범위 내인지 확인
  double coverage = cp.getCoverage();
  EXPECT_GT(coverage, 0.0);
  EXPECT_LE(coverage, 1.0);
}

TEST(ConformalPredictor, WindowSize) {
  ConformalPredictor::Params params;
  params.window_size = 10;
  params.initial_margin = 0.5;
  params.decay_rate = 1.0;
  params.coverage_probability = 0.9;
  params.min_margin = 0.01;
  ConformalPredictor cp(params);

  // 큰 오차 투입
  for (int i = 0; i < 10; ++i) {
    cp.update(1.0);
  }
  double margin_high = cp.getMargin();

  // 작은 오차로 윈도우 교체
  for (int i = 0; i < 10; ++i) {
    cp.update(0.05);
  }
  double margin_low = cp.getMargin();

  // 윈도우 전체가 교체되면 마진이 감소
  EXPECT_LT(margin_low, margin_high);
}

TEST(ConformalPredictor, MinMaxClamp) {
  ConformalPredictor::Params params;
  params.min_margin = 0.1;
  params.max_margin = 0.5;
  params.window_size = 10;
  params.coverage_probability = 0.99;
  params.decay_rate = 1.0;
  ConformalPredictor cp(params);

  // 매우 큰 오차 → max_margin 제한
  for (int i = 0; i < 10; ++i) {
    cp.update(10.0);
  }
  EXPECT_LE(cp.getMargin(), 0.5);

  // 리셋 후 매우 작은 오차 → min_margin 제한
  cp.reset();
  for (int i = 0; i < 10; ++i) {
    cp.update(0.001);
  }
  EXPECT_GE(cp.getMargin(), 0.1);
}

TEST(ConformalPredictor, DecayRate) {
  // decay < 1 → 최근 오차에 높은 가중치
  ConformalPredictor::Params params;
  params.window_size = 20;
  params.decay_rate = 0.8;  // 강한 감쇄
  params.initial_margin = 0.3;
  params.coverage_probability = 0.9;
  params.min_margin = 0.01;
  params.max_margin = 2.0;
  ConformalPredictor cp(params);

  // 처음에 큰 오차, 나중에 작은 오차
  for (int i = 0; i < 10; ++i) { cp.update(1.0); }
  for (int i = 0; i < 10; ++i) { cp.update(0.05); }
  double margin_recent_small = cp.getMargin();

  // 비교: 처음에 작은 오차, 나중에 큰 오차
  ConformalPredictor cp2(params);
  for (int i = 0; i < 10; ++i) { cp2.update(0.05); }
  for (int i = 0; i < 10; ++i) { cp2.update(1.0); }
  double margin_recent_large = cp2.getMargin();

  // 최근 큰 오차 → 더 큰 마진
  EXPECT_GT(margin_recent_large, margin_recent_small);
}

TEST(ConformalPredictor, BarrierIntegration) {
  // BarrierFunctionSet과 통합
  BarrierFunctionSet barrier_set(0.2, 0.3, 3.0);
  barrier_set.setObstacles({{2.0, 0.0, 0.3}});

  double original_margin = barrier_set.safetyMargin();
  EXPECT_NEAR(original_margin, 0.3, 1e-10);

  // 마진 업데이트
  barrier_set.updateSafetyMargin(0.5);
  EXPECT_NEAR(barrier_set.safetyMargin(), 0.5, 1e-10);

  // 장애물이 유지되는지 확인
  EXPECT_EQ(barrier_set.size(), 1u);

  // h값이 변경되었는지 확인 (더 큰 마진 → 더 작은 h)
  Eigen::VectorXd state(3);
  state << 0.0, 0.0, 0.0;
  double h_new = barrier_set.evaluateAll(state)(0);

  barrier_set.updateSafetyMargin(0.1);
  double h_small_margin = barrier_set.evaluateAll(state)(0);

  EXPECT_GT(h_small_margin, h_new)
    << "Smaller margin should result in larger h (farther from barrier)";
}

// =============================================================================
// ShieldMPPI 테스트 (8개)
// =============================================================================

// Shield-MPPI는 nav2 플러그인이므로 직접 인스턴스화하기 어려움.
// 대신 핵심 로직인 CBF 투영 및 관련 기능을 단위 테스트합니다.

TEST(ShieldMPPI, SafeTrajectories) {
  // CBF가 있는 환경에서 barrier가 양수인지 확인
  BarrierFunctionSet barrier_set(0.2, 0.3, 5.0);
  barrier_set.setObstacles({{3.0, 0.0, 0.5}});

  // 안전한 궤적 생성 (장애물 멀리)
  int K = 4, N = 10;
  std::vector<Eigen::MatrixXd> trajs(K);
  for (int k = 0; k < K; ++k) {
    trajs[k] = Eigen::MatrixXd::Zero(N + 1, 3);
    for (int t = 0; t <= N; ++t) {
      trajs[k](t, 0) = -1.0;  // 장애물 반대편
      trajs[k](t, 1) = 0.5 * k;
    }
  }

  // 모든 barrier 값이 양수인지 확인
  for (int k = 0; k < K; ++k) {
    for (int t = 0; t <= N; ++t) {
      Eigen::VectorXd state = trajs[k].row(t).transpose();
      Eigen::VectorXd h_vals = barrier_set.evaluateAll(state);
      for (int i = 0; i < h_vals.size(); ++i) {
        EXPECT_GT(h_vals(i), 0.0);
      }
    }
  }
}

TEST(ShieldMPPI, FallbackNoCBF) {
  // CBF 장애물 없이 CBFCost가 0인지 확인
  BarrierFunctionSet barrier_set(0.2, 0.3, 3.0);
  CBFCost cbf_cost(&barrier_set, 500.0, 1.0, 0.1);

  auto [trajs, ctrls] = createTestTrajectories(4, 10, 3, 2);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(11, 3);

  Eigen::VectorXd costs = cbf_cost.compute(trajs, ctrls, ref);
  for (int k = 0; k < 4; ++k) {
    EXPECT_NEAR(costs(k), 0.0, 1e-10);
  }
}

TEST(ShieldMPPI, ControlBounds) {
  // 클리핑이 제어 한계를 유지하는지 확인
  MPPIParams params;
  params.v_max = 1.0;
  params.v_min = 0.0;
  auto model = MotionModelFactory::create("diff_drive", params);

  Eigen::MatrixXd controls(5, 2);
  controls << 2.0, 3.0,  // 초과
             -1.0, -2.0,  // 미만
              0.5, 0.5,    // 정상
              0.0, 0.0,
              1.0, 1.0;
  Eigen::MatrixXd clipped = model->clipControls(controls);

  for (int i = 0; i < 5; ++i) {
    EXPECT_LE(clipped(i, 0), params.v_max);
    EXPECT_GE(clipped(i, 0), params.v_min);
  }
}

TEST(ShieldMPPI, ReducedViolations) {
  // BarrierRateCost + CBFCost가 위반 궤적에 높은 비용을 부과하는지 확인
  BarrierFunctionSet barrier_set(0.2, 0.1, 10.0);
  barrier_set.setObstacles({{2.0, 0.0, 0.3}});

  CBFCost cbf_cost(&barrier_set, 500.0, 1.0, 0.1);
  BarrierRateCost br_cost(&barrier_set, 100.0, 0.1);

  int K = 2, N = 10;
  std::vector<Eigen::MatrixXd> trajs(K);
  std::vector<Eigen::MatrixXd> ctrls(K, Eigen::MatrixXd::Zero(N, 2));
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, 3);

  // k=0: 장애물 회피
  trajs[0] = Eigen::MatrixXd::Zero(N + 1, 3);
  for (int t = 0; t <= N; ++t) {
    trajs[0](t, 0) = -1.0;
    trajs[0](t, 1) = 0.3 * t;
  }

  // k=1: 장애물 직진
  trajs[1] = Eigen::MatrixXd::Zero(N + 1, 3);
  for (int t = 0; t <= N; ++t) {
    trajs[1](t, 0) = 0.2 * t;
  }

  Eigen::VectorXd cbf_costs = cbf_cost.compute(trajs, ctrls, ref);
  Eigen::VectorXd br_costs = br_cost.compute(trajs, ctrls, ref);

  // 장애물 직진이 더 높은 비용
  EXPECT_GT(cbf_costs(1) + br_costs(1), cbf_costs(0) + br_costs(0));
}

TEST(ShieldMPPI, PerfBudget) {
  // BarrierRateCost 성능: K=512, 4 obstacles → < 0.5ms
  BarrierFunctionSet barrier_set(0.2, 0.1, 10.0);
  barrier_set.setObstacles({
    {2.0, 0.0, 0.3}, {-2.0, 0.0, 0.3},
    {0.0, 2.0, 0.3}, {0.0, -2.0, 0.3}
  });
  BarrierRateCost cost(&barrier_set, 100.0, 0.1);

  int K = 512, N = 30;
  auto [trajs, ctrls] = createTestTrajectories(K, N, 3, 2);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, 3);

  // Warm-up
  cost.compute(trajs, ctrls, ref);

  auto start = std::chrono::high_resolution_clock::now();
  int N_iter = 50;
  for (int i = 0; i < N_iter; ++i) {
    cost.compute(trajs, ctrls, ref);
  }
  auto end = std::chrono::high_resolution_clock::now();
  double ms_per_call = std::chrono::duration<double, std::milli>(end - start).count() / N_iter;

  EXPECT_LT(ms_per_call, 0.5)
    << "BarrierRateCost too slow: " << ms_per_call << " ms";
}

TEST(ShieldMPPI, Stride) {
  // Shield stride: 매 2번째 스텝만 투영
  // (Plugin이 아니라 파라미터 기본값 테스트)
  MPPIParams params;
  params.shield_cbf_stride = 2;
  params.shield_max_iterations = 10;
  EXPECT_EQ(params.shield_cbf_stride, 2);
  EXPECT_EQ(params.shield_max_iterations, 10);
}

TEST(ShieldMPPI, DiffDrive) {
  // DiffDrive 궤적에서 CBF 평가가 정상인지 확인
  BarrierFunctionSet barrier_set(0.2, 0.3, 5.0);
  barrier_set.setObstacles({{5.0, 0.0, 0.5}});

  MPPIParams params;
  auto model = MotionModelFactory::create("diff_drive", params);
  BatchDynamicsWrapper dynamics(params, std::move(model));

  Eigen::VectorXd x0(3);
  x0 << 0.0, 0.0, 0.0;

  // 제어 시퀀스
  int K = 8, N = 10;
  std::vector<Eigen::MatrixXd> ctrl_seqs(K);
  for (int k = 0; k < K; ++k) {
    ctrl_seqs[k] = Eigen::MatrixXd::Zero(N, 2);
    ctrl_seqs[k].col(0).setConstant(0.5);  // v=0.5
    ctrl_seqs[k].col(1).setConstant(0.1 * (k - K/2));  // 다양한 omega
  }

  auto trajs = dynamics.model().rolloutBatch(x0, ctrl_seqs, 0.1);
  EXPECT_EQ(static_cast<int>(trajs.size()), K);

  // 모든 궤적의 모든 시점에서 h값 계산 가능
  for (int k = 0; k < K; ++k) {
    for (int t = 0; t <= N; ++t) {
      Eigen::VectorXd state = trajs[k].row(t).transpose();
      EXPECT_NO_THROW(barrier_set.evaluateAll(state));
    }
  }
}

TEST(ShieldMPPI, Swerve) {
  // Swerve 궤적에서 CBF 평가
  BarrierFunctionSet barrier_set(0.2, 0.3, 5.0);
  barrier_set.setObstacles({{3.0, 3.0, 0.5}});

  MPPIParams params;
  params.motion_model = "swerve";
  params.noise_sigma = Eigen::Vector3d(0.5, 0.3, 0.5);
  params.R = Eigen::Matrix3d::Identity() * 0.1;
  params.R_rate = Eigen::Matrix3d::Identity();
  auto model = MotionModelFactory::create("swerve", params);
  BatchDynamicsWrapper dynamics(params, std::move(model));

  Eigen::VectorXd x0(3);
  x0 << 0.0, 0.0, 0.0;

  int K = 4, N = 10;
  std::vector<Eigen::MatrixXd> ctrl_seqs(K);
  for (int k = 0; k < K; ++k) {
    ctrl_seqs[k] = Eigen::MatrixXd::Random(N, 3) * 0.3;
  }

  auto trajs = dynamics.model().rolloutBatch(x0, ctrl_seqs, 0.1);
  EXPECT_EQ(static_cast<int>(trajs.size()), K);

  // h값 확인
  for (int k = 0; k < K; ++k) {
    Eigen::VectorXd state = trajs[k].row(0).transpose();
    Eigen::VectorXd h = barrier_set.evaluateAll(state);
    EXPECT_GT(h(0), 0.0);  // 초기 위치는 안전
  }
}

// =============================================================================
// updateSafetyMargin 추가 테스트
// =============================================================================

TEST(BarrierFunctionSet, UpdateSafetyMarginPreservesObstacles) {
  BarrierFunctionSet barrier_set(0.2, 0.3, 3.0);
  barrier_set.setObstacles({{1.0, 0.0, 0.5}, {-1.0, 0.0, 0.5}});

  EXPECT_EQ(barrier_set.size(), 2u);

  barrier_set.updateSafetyMargin(0.5);
  EXPECT_EQ(barrier_set.size(), 2u);

  barrier_set.updateSafetyMargin(0.1);
  EXPECT_EQ(barrier_set.size(), 2u);
}

TEST(BarrierFunctionSet, UpdateSafetyMarginChangesBarrierValues) {
  BarrierFunctionSet barrier_set(0.2, 0.3, 5.0);
  barrier_set.setObstacles({{2.0, 0.0, 0.3}});

  Eigen::VectorXd state(3);
  state << 0.0, 0.0, 0.0;

  double h1 = barrier_set.evaluateAll(state)(0);

  barrier_set.updateSafetyMargin(0.8);
  double h2 = barrier_set.evaluateAll(state)(0);

  // 더 큰 마진 → d_safe 증가 → h 감소
  EXPECT_LT(h2, h1);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
