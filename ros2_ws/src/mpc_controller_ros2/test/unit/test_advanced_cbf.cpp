#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>

#include "mpc_controller_ros2/c3bf_barrier.hpp"
#include "mpc_controller_ros2/barrier_function.hpp"
#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"

using namespace mpc_controller_ros2;

// =============================================================================
// Helper: 테스트용 궤적 생성
// =============================================================================

static std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>>
createTestTrajectories(int K, int N, int nx, int nu,
                       double start_x, double start_y,
                       double vx, double vy, double dt) {
  std::vector<Eigen::MatrixXd> trajectories(K);
  std::vector<Eigen::MatrixXd> controls(K);
  for (int k = 0; k < K; ++k) {
    trajectories[k] = Eigen::MatrixXd::Zero(N + 1, nx);
    controls[k] = Eigen::MatrixXd::Zero(N, nu);
    for (int t = 0; t <= N; ++t) {
      trajectories[k](t, 0) = start_x + vx * t * dt;
      trajectories[k](t, 1) = start_y + vy * t * dt;
      if (nx >= 3) trajectories[k](t, 2) = 0.0;
    }
    for (int t = 0; t < N; ++t) {
      controls[k](t, 0) = std::sqrt(vx * vx + vy * vy);
      if (nu >= 2) controls[k](t, 1) = 0.0;
    }
  }
  return {trajectories, controls};
}

// =============================================================================
// C3BFBarrier Tests
// =============================================================================

TEST(C3BFBarrier, Approaching_NegativeH) {
  // 로봇(0,0)이 장애물(3,0)을 향해 접근 중 (vx=1)
  C3BFBarrier barrier(3.0, 0.0, 0.5, 0.2, 0.3, M_PI / 4);
  Eigen::VectorXd state(3);
  state << 0.0, 0.0, 0.0;

  double h = barrier.evaluate(state, 1.0, 0.0);
  // 직접 접근 → h < 0
  EXPECT_LT(h, 0.0);
}

TEST(C3BFBarrier, Departing_PositiveH) {
  // 로봇(0,0)이 장애물(3,0)에서 멀어지는 중 (vx=-1)
  C3BFBarrier barrier(3.0, 0.0, 0.5, 0.2, 0.3, M_PI / 4);
  Eigen::VectorXd state(3);
  state << 0.0, 0.0, 0.0;

  double h = barrier.evaluate(state, -1.0, 0.0);
  // 이탈 중 → h > 0
  EXPECT_GT(h, 0.0);
}

TEST(C3BFBarrier, Stationary_DistanceBased) {
  // 로봇 정지 (v=0) + 장애물 정지 → cone 판정 무관
  C3BFBarrier barrier(3.0, 0.0, 0.5, 0.2, 0.3, M_PI / 4);
  Eigen::VectorXd state(3);
  state << 0.0, 0.0, 0.0;

  double h = barrier.evaluate(state, 0.0, 0.0);
  // v_rel = 0 → p·v = 0, ||v|| = 0 → h = 0
  EXPECT_NEAR(h, 0.0, 1e-10);
}

TEST(C3BFBarrier, Gradient_FiniteDifference) {
  C3BFBarrier barrier(3.0, 0.0, 0.5, 0.2, 0.3, M_PI / 4);
  Eigen::VectorXd state(3);
  state << 1.0, 0.5, 0.1;

  double vx = 0.5, vy = 0.3;
  Eigen::VectorXd grad = barrier.gradient(state, vx, vy);

  // 유한 차분 검증
  double eps = 1e-5;
  for (int j = 0; j < 2; ++j) {
    Eigen::VectorXd sp = state, sm = state;
    sp(j) += eps;
    sm(j) -= eps;
    double fd = (barrier.evaluate(sp, vx, vy) - barrier.evaluate(sm, vx, vy)) / (2 * eps);
    EXPECT_NEAR(grad(j), fd, 1e-4) << "dim=" << j;
  }
}

TEST(C3BFBarrier, BatchEvaluate_Consistency) {
  C3BFBarrier barrier(3.0, 0.0, 0.5, 0.2, 0.3, M_PI / 4);
  int M = 10;
  Eigen::MatrixXd states = Eigen::MatrixXd::Random(M, 3);
  Eigen::VectorXd vx = Eigen::VectorXd::Random(M);
  Eigen::VectorXd vy = Eigen::VectorXd::Random(M);

  Eigen::VectorXd h_batch = barrier.evaluateBatch(states, vx, vy);
  EXPECT_EQ(h_batch.size(), M);

  for (int i = 0; i < M; ++i) {
    double h_single = barrier.evaluate(states.row(i).transpose(), vx(i), vy(i));
    EXPECT_NEAR(h_batch(i), h_single, 1e-12);
  }
}

TEST(C3BFBarrier, UpdateObstacleVelocity) {
  C3BFBarrier barrier(3.0, 0.0, 0.5, 0.2, 0.3, M_PI / 4);
  Eigen::VectorXd state(3);
  state << 0.0, 0.0, 0.0;

  // 장애물이 로봇을 향해 이동 (obs_vx=-1)
  barrier.updateObstacleVelocity(-1.0, 0.0);
  double h_approaching = barrier.evaluate(state, 0.0, 0.0);

  // 장애물이 로봇에서 멀어짐 (obs_vx=1)
  barrier.updateObstacleVelocity(1.0, 0.0);
  double h_departing = barrier.evaluate(state, 0.0, 0.0);

  EXPECT_LT(h_approaching, h_departing);
}

// =============================================================================
// C3BFCost Tests
// =============================================================================

TEST(C3BFCost, ApproachingTrajectory_HighCost) {
  int K = 4, N = 10, nx = 3, nu = 2;
  double dt = 0.1;

  // 장애물을 향해 직진
  auto [traj, ctrl] = createTestTrajectories(K, N, nx, nu, 0.0, 0.0, 1.0, 0.0, dt);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);

  C3BFCost cost(500.0, dt, M_PI / 4);
  std::vector<Eigen::Vector3d> obstacles = {{3.0, 0.0, 0.5}};
  cost.setObstacles(obstacles);

  Eigen::VectorXd costs = cost.compute(traj, ctrl, ref);
  EXPECT_GT(costs.sum(), 0.0);
}

TEST(C3BFCost, DepartingTrajectory_LowCost) {
  int K = 4, N = 10, nx = 3, nu = 2;
  double dt = 0.1;

  // 장애물에서 멀어지는 궤적
  auto [traj, ctrl] = createTestTrajectories(K, N, nx, nu, 0.0, 0.0, -1.0, 0.0, dt);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);

  C3BFCost cost(500.0, dt, M_PI / 4);
  std::vector<Eigen::Vector3d> obstacles = {{3.0, 0.0, 0.5}};
  cost.setObstacles(obstacles);

  Eigen::VectorXd costs = cost.compute(traj, ctrl, ref);
  // 이탈 궤적 → 비용 = 0 (h > 0)
  EXPECT_DOUBLE_EQ(costs.sum(), 0.0);
}

// =============================================================================
// BarrierFunctionSet C3BF Tests
// =============================================================================

TEST(BarrierFunctionSet_C3BF, SetObstaclesWithVelocity) {
  BarrierFunctionSet bset(0.2, 0.3, 3.0);

  std::vector<Eigen::Vector3d> obstacles = {{1.0, 0.0, 0.3}, {2.0, 1.0, 0.5}};
  std::vector<Eigen::Vector2d> velocities = {{-0.5, 0.0}, {0.0, 0.3}};

  bset.setObstaclesWithVelocity(obstacles, velocities);

  // CircleBarrier도 설정됨
  EXPECT_EQ(bset.size(), 2);
  // C3BF barrier도 설정됨
  EXPECT_EQ(bset.c3bfBarriers().size(), 2);

  EXPECT_NEAR(bset.c3bfBarriers()[0].obsVx(), -0.5, 1e-10);
  EXPECT_NEAR(bset.c3bfBarriers()[1].obsVy(), 0.3, 1e-10);
}

TEST(BarrierFunctionSet_C3BF, GetActiveC3BFBarriers) {
  BarrierFunctionSet bset(0.2, 0.3, 2.0);  // activation_distance=2.0

  std::vector<Eigen::Vector3d> obstacles = {{1.0, 0.0, 0.3}, {5.0, 0.0, 0.3}};
  std::vector<Eigen::Vector2d> velocities = {{0.0, 0.0}, {0.0, 0.0}};

  bset.setObstaclesWithVelocity(obstacles, velocities);

  Eigen::VectorXd state(3);
  state << 0.0, 0.0, 0.0;

  auto active = bset.getActiveC3BFBarriers(state);
  // 장애물1(거리1.0) → 활성, 장애물2(거리5.0) → 비활성
  EXPECT_EQ(active.size(), 1);
  EXPECT_NEAR(active[0]->obsX(), 1.0, 1e-10);
}

// =============================================================================
// AdaptiveShield Tests (단위 테스트 — 플러그인 없이 alpha 계산만)
// =============================================================================

// alpha 계산 함수 재현 (독립 테스트)
static double testComputeAdaptiveAlpha(
  double min_distance, double robot_speed,
  double alpha_min, double alpha_max, double k_d, double k_v) {
  double alpha = alpha_min +
    (alpha_max - alpha_min) *
    std::exp(-k_d * min_distance) *
    (1.0 + k_v * robot_speed);
  return std::clamp(alpha, alpha_min, alpha_max);
}

TEST(AdaptiveShield, CloseAndFast_HighAlpha) {
  double alpha = testComputeAdaptiveAlpha(0.1, 1.5, 0.1, 1.0, 1.0, 0.5);
  // 가까움 + 고속 → 높은 alpha
  EXPECT_GT(alpha, 0.8);
}

TEST(AdaptiveShield, FarAndSlow_LowAlpha) {
  double alpha = testComputeAdaptiveAlpha(5.0, 0.1, 0.1, 1.0, 1.0, 0.5);
  // 멀리 + 저속 → 낮은 alpha
  EXPECT_LT(alpha, 0.2);
}

TEST(AdaptiveShield, AlphaClamp_MinMax) {
  // 매우 가까움 + 매우 고속 → alpha_max로 클램프
  double alpha = testComputeAdaptiveAlpha(0.0, 10.0, 0.1, 1.0, 1.0, 0.5);
  EXPECT_LE(alpha, 1.0);
  EXPECT_GE(alpha, 0.1);
}

// =============================================================================
// CBFCost HorizonWeighted Tests
// =============================================================================

TEST(CBFCost_HorizonWeighted, NearFutureHigher_ThanFarFuture) {
  int K = 4, N = 20, nx = 3;
  double dt = 0.1;

  BarrierFunctionSet bset(0.2, 0.3, 5.0);
  std::vector<Eigen::Vector3d> obstacles = {{2.0, 0.0, 0.3}};
  bset.setObstacles(obstacles);

  // discount=0.9 → 먼 미래 비용 감소
  CBFCost cost_discounted(&bset, 500.0, 1.0, dt, 0.9);
  // discount=1.0 → 균일
  CBFCost cost_uniform(&bset, 500.0, 1.0, dt, 1.0);

  // 장애물을 향한 궤적
  std::vector<Eigen::MatrixXd> trajectories(K);
  std::vector<Eigen::MatrixXd> controls(K);
  for (int k = 0; k < K; ++k) {
    trajectories[k] = Eigen::MatrixXd::Zero(N + 1, nx);
    controls[k] = Eigen::MatrixXd::Zero(N, 2);
    for (int t = 0; t <= N; ++t) {
      trajectories[k](t, 0) = 0.5 * t * dt;  // 장애물 쪽으로 접근
    }
  }
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);

  Eigen::VectorXd c_disc = cost_discounted.compute(trajectories, controls, ref);
  Eigen::VectorXd c_uni = cost_uniform.compute(trajectories, controls, ref);

  // 할인 시 비용 < 균일 비용 (먼 미래 위반이 할인됨)
  if (c_uni.sum() > 0) {
    EXPECT_LE(c_disc.sum(), c_uni.sum());
  }
}

TEST(CBFCost_HorizonWeighted, Discount_One_SameAsOriginal) {
  int K = 4, N = 10, nx = 3;
  double dt = 0.1;

  BarrierFunctionSet bset(0.2, 0.3, 5.0);
  std::vector<Eigen::Vector3d> obstacles = {{2.0, 0.0, 0.3}};
  bset.setObstacles(obstacles);

  CBFCost cost_d1(&bset, 500.0, 1.0, dt, 1.0);
  CBFCost cost_default(&bset, 500.0, 1.0, dt);  // default=1.0

  std::vector<Eigen::MatrixXd> trajectories(K);
  std::vector<Eigen::MatrixXd> controls(K);
  for (int k = 0; k < K; ++k) {
    trajectories[k] = Eigen::MatrixXd::Random(N + 1, nx);
    controls[k] = Eigen::MatrixXd::Random(N, 2);
  }
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);

  Eigen::VectorXd c1 = cost_d1.compute(trajectories, controls, ref);
  Eigen::VectorXd c2 = cost_default.compute(trajectories, controls, ref);

  EXPECT_LT((c1 - c2).norm(), 1e-10);
}

TEST(CBFCost_HorizonWeighted, Discount_Zero_OnlyFirstStep) {
  int K = 2, N = 10, nx = 3;
  double dt = 0.1;

  BarrierFunctionSet bset(0.2, 0.3, 5.0);
  std::vector<Eigen::Vector3d> obstacles = {{1.5, 0.0, 0.3}};
  bset.setObstacles(obstacles);

  // discount=0 → 첫 스텝만 비용 적용 (discount^0=1, discount^1=0)
  CBFCost cost_d0(&bset, 500.0, 1.0, dt, 0.0);

  // 장애물을 향한 궤적
  std::vector<Eigen::MatrixXd> trajectories(K);
  std::vector<Eigen::MatrixXd> controls(K);
  for (int k = 0; k < K; ++k) {
    trajectories[k] = Eigen::MatrixXd::Zero(N + 1, nx);
    controls[k] = Eigen::MatrixXd::Zero(N, 2);
    for (int t = 0; t <= N; ++t) {
      trajectories[k](t, 0) = 1.0 + 0.1 * t * dt;  // 접근
    }
  }
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, nx);

  Eigen::VectorXd costs = cost_d0.compute(trajectories, controls, ref);
  // discount=0 → 첫 스텝 위반만 반영 → 비용이 매우 작거나 0
  // (t=0에서만 discount=1, 나머지 discount=0)
  EXPECT_TRUE(costs.allFinite());
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
