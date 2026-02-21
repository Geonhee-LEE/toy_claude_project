#include <gtest/gtest.h>
#include <cmath>
#include "mpc_controller_ros2/barrier_function.hpp"
#include "mpc_controller_ros2/cbf_safety_filter.hpp"
#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"

namespace mpc_controller_ros2
{

// ============================================================================
// TestCircleBarrier
// ============================================================================

class CircleBarrierTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // 장애물: (2, 0), radius=0.3, robot_radius=0.2, safety_margin=0.1
    // d_safe = 0.3 + 0.2 + 0.1 = 0.6
    barrier_ = std::make_unique<CircleBarrier>(2.0, 0.0, 0.3, 0.2, 0.1);
  }

  std::unique_ptr<CircleBarrier> barrier_;
};

TEST_F(CircleBarrierTest, EvaluateInsideNegative)
{
  // 로봇이 장애물 내부: (2.1, 0) → dist=0.1 < d_safe=0.6
  Eigen::VectorXd state(3);
  state << 2.1, 0.0, 0.0;
  double h = barrier_->evaluate(state);
  EXPECT_LT(h, 0.0) << "장애물 내부에서 h < 0이어야 함";
}

TEST_F(CircleBarrierTest, EvaluateOutsidePositive)
{
  // 로봇이 장애물 밖: (0, 0) → dist=2.0 > d_safe=0.6
  Eigen::VectorXd state(3);
  state << 0.0, 0.0, 0.0;
  double h = barrier_->evaluate(state);
  EXPECT_GT(h, 0.0) << "장애물 밖에서 h > 0이어야 함";
}

TEST_F(CircleBarrierTest, EvaluateOnBoundaryZero)
{
  // 로봇이 barrier 경계: dist = d_safe = 0.6
  Eigen::VectorXd state(3);
  state << 2.0 + 0.6, 0.0, 0.0;  // (2.6, 0)
  double h = barrier_->evaluate(state);
  EXPECT_NEAR(h, 0.0, 1e-10) << "경계에서 h ≈ 0이어야 함";
}

TEST_F(CircleBarrierTest, GradientDirection)
{
  // (0, 0)에서 장애물 (2, 0) 방향의 gradient
  Eigen::VectorXd state(3);
  state << 0.0, 0.0, 0.0;
  Eigen::VectorXd grad = barrier_->gradient(state);

  // ∂h/∂x = 2*(0-2) = -4, ∂h/∂y = 0
  EXPECT_DOUBLE_EQ(grad(0), -4.0);
  EXPECT_DOUBLE_EQ(grad(1), 0.0);
  EXPECT_DOUBLE_EQ(grad(2), 0.0);  // theta 성분 없음
}

TEST_F(CircleBarrierTest, BatchConsistency)
{
  // 배치 평가가 개별 평가와 동일한지 검증
  Eigen::MatrixXd states(3, 3);
  states << 0.0, 0.0, 0.0,
            2.1, 0.0, 0.0,
            2.6, 0.0, 0.0;

  Eigen::VectorXd batch_h = barrier_->evaluateBatch(states);
  EXPECT_EQ(batch_h.size(), 3);

  for (int i = 0; i < 3; ++i) {
    double single_h = barrier_->evaluate(states.row(i).transpose());
    EXPECT_DOUBLE_EQ(batch_h(i), single_h);
  }
}

// ============================================================================
// TestBarrierFunctionSet
// ============================================================================

class BarrierFunctionSetTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    barrier_set_ = std::make_unique<BarrierFunctionSet>(0.2, 0.1, 3.0);

    // 2개 장애물: (1, 0, r=0.3), (10, 0, r=0.5)
    obstacles_.push_back(Eigen::Vector3d(1.0, 0.0, 0.3));
    obstacles_.push_back(Eigen::Vector3d(10.0, 0.0, 0.5));
    barrier_set_->setObstacles(obstacles_);
  }

  std::unique_ptr<BarrierFunctionSet> barrier_set_;
  std::vector<Eigen::Vector3d> obstacles_;
};

TEST_F(BarrierFunctionSetTest, ActiveBarriersFiltering)
{
  // 로봇 위치 (0, 0): 장애물 (1,0)은 활성(dist=1.0 ≤ 3.0), (10,0)은 비활성
  Eigen::VectorXd state(3);
  state << 0.0, 0.0, 0.0;
  auto active = barrier_set_->getActiveBarriers(state);
  EXPECT_EQ(active.size(), 1u) << "활성화 거리 내 장애물만 선택";
}

TEST_F(BarrierFunctionSetTest, EvaluateAll)
{
  Eigen::VectorXd state(3);
  state << 0.0, 0.0, 0.0;
  Eigen::VectorXd values = barrier_set_->evaluateAll(state);
  EXPECT_EQ(values.size(), 2);
  // 첫 번째 장애물 (1,0): dist²=1.0, d_safe=0.6, h = 1.0 - 0.36 = 0.64
  EXPECT_NEAR(values(0), 1.0 - 0.36, 1e-10);
  // 두 번째 장애물 (10,0): dist²=100, d_safe=0.8, h = 100 - 0.64
  EXPECT_NEAR(values(1), 100.0 - 0.64, 1e-10);
}

TEST_F(BarrierFunctionSetTest, SetObstaclesUpdates)
{
  // 새 장애물로 교체
  std::vector<Eigen::Vector3d> new_obs;
  new_obs.push_back(Eigen::Vector3d(5.0, 5.0, 0.1));
  barrier_set_->setObstacles(new_obs);
  EXPECT_EQ(barrier_set_->size(), 1u);
}

TEST_F(BarrierFunctionSetTest, EmptyObstacles)
{
  BarrierFunctionSet empty_set(0.2, 0.1, 3.0);
  Eigen::VectorXd state(3);
  state << 0.0, 0.0, 0.0;
  auto active = empty_set.getActiveBarriers(state);
  EXPECT_TRUE(active.empty());
  EXPECT_TRUE(empty_set.empty());
}

// ============================================================================
// TestCBFSafetyFilter
// ============================================================================

class CBFSafetyFilterTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    params_.N = 10;
    params_.dt = 0.1;
    params_.v_max = 1.0;
    params_.v_min = -0.5;
    params_.omega_max = 1.0;
    params_.omega_min = -1.0;

    dynamics_ = std::make_unique<BatchDynamicsWrapper>(params_);

    barrier_set_ = std::make_unique<BarrierFunctionSet>(0.2, 0.1, 5.0);

    Eigen::VectorXd u_min(2), u_max(2);
    u_min << -0.5, -1.0;
    u_max << 1.0, 1.0;
    filter_ = std::make_unique<CBFSafetyFilter>(
      barrier_set_.get(), 1.0, 0.1, u_min, u_max);
  }

  MPPIParams params_;
  std::unique_ptr<BatchDynamicsWrapper> dynamics_;
  std::unique_ptr<BarrierFunctionSet> barrier_set_;
  std::unique_ptr<CBFSafetyFilter> filter_;
};

TEST_F(CBFSafetyFilterTest, NoActiveBarriersPassthrough)
{
  // 장애물 없음 → u_mppi 그대로 반환
  Eigen::VectorXd state(3);
  state << 0.0, 0.0, 0.0;
  Eigen::VectorXd u_mppi(2);
  u_mppi << 0.5, 0.1;

  auto [u_safe, info] = filter_->filter(state, u_mppi, *dynamics_);
  EXPECT_EQ(info.num_active_barriers, 0);
  EXPECT_FALSE(info.filter_applied);
  EXPECT_TRUE(info.qp_success);
  EXPECT_DOUBLE_EQ(u_safe(0), u_mppi(0));
  EXPECT_DOUBLE_EQ(u_safe(1), u_mppi(1));
}

TEST_F(CBFSafetyFilterTest, FilterPreservesSafeControl)
{
  // 장애물에서 멀리 떨어진 상태 → u_mppi가 이미 안전
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(3.0, 0.0, 0.1));  // 장애물 3m 거리
  barrier_set_->setObstacles(obs);

  Eigen::VectorXd state(3);
  state << 0.0, 0.0, 0.0;
  Eigen::VectorXd u_mppi(2);
  u_mppi << 0.5, 0.0;  // 직진

  auto [u_safe, info] = filter_->filter(state, u_mppi, *dynamics_);
  EXPECT_EQ(info.num_active_barriers, 1);
  EXPECT_FALSE(info.filter_applied) << "이미 안전한 제어는 필터 적용 불필요";
  EXPECT_NEAR(u_safe(0), u_mppi(0), 1e-6);
  EXPECT_NEAR(u_safe(1), u_mppi(1), 1e-6);
}

TEST_F(CBFSafetyFilterTest, FilterModifiesUnsafeControl)
{
  // 장애물 바로 앞에서 직진하는 위험한 제어 → 수정 필요
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(0.5, 0.0, 0.1));  // 0.5m 앞 장애물
  barrier_set_->setObstacles(obs);

  Eigen::VectorXd state(3);
  state << 0.0, 0.0, 0.0;  // θ=0 (x 방향)
  Eigen::VectorXd u_mppi(2);
  u_mppi << 1.0, 0.0;  // 최대 속도 직진 → 장애물 충돌

  auto [u_safe, info] = filter_->filter(state, u_mppi, *dynamics_);
  EXPECT_EQ(info.num_active_barriers, 1);
  EXPECT_TRUE(info.filter_applied) << "위험한 제어는 필터가 수정해야 함";

  // u_safe는 u_mppi와 다르거나, 적어도 안전한 방향으로 수정
  // (정확한 값은 QP solver에 의존하므로 방향만 확인)
  if (info.qp_success) {
    EXPECT_LE(u_safe(0), u_mppi(0)) << "안전 필터는 전진 속도를 줄여야 함";
  }
}

TEST_F(CBFSafetyFilterTest, ControlBoundsRespected)
{
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(0.5, 0.0, 0.1));
  barrier_set_->setObstacles(obs);

  Eigen::VectorXd state(3);
  state << 0.0, 0.0, 0.0;
  Eigen::VectorXd u_mppi(2);
  u_mppi << 1.0, 0.0;

  auto [u_safe, info] = filter_->filter(state, u_mppi, *dynamics_);

  // bounds 확인: v ∈ [-0.5, 1.0], omega ∈ [-1.0, 1.0]
  EXPECT_GE(u_safe(0), -0.5 - 1e-6);
  EXPECT_LE(u_safe(0), 1.0 + 1e-6);
  EXPECT_GE(u_safe(1), -1.0 - 1e-6);
  EXPECT_LE(u_safe(1), 1.0 + 1e-6);
}

TEST_F(CBFSafetyFilterTest, MultipleConstraints)
{
  // 여러 장애물 동시 활성
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(0.6, 0.3, 0.1));
  obs.push_back(Eigen::Vector3d(0.6, -0.3, 0.1));
  barrier_set_->setObstacles(obs);

  Eigen::VectorXd state(3);
  state << 0.0, 0.0, 0.0;
  Eigen::VectorXd u_mppi(2);
  u_mppi << 1.0, 0.0;

  auto [u_safe, info] = filter_->filter(state, u_mppi, *dynamics_);
  EXPECT_EQ(info.num_active_barriers, 2);
}

TEST_F(CBFSafetyFilterTest, FallbackOnFailure)
{
  // QP 실패 시에도 반환값이 valid한지 확인
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(0.2, 0.0, 0.1));  // 매우 가까운 장애물
  barrier_set_->setObstacles(obs);

  Eigen::VectorXd state(3);
  state << 0.0, 0.0, 0.0;
  Eigen::VectorXd u_mppi(2);
  u_mppi << 1.0, 0.0;

  auto [u_safe, info] = filter_->filter(state, u_mppi, *dynamics_);
  // QP 성공/실패 상관없이 유효한 크기 반환
  EXPECT_EQ(u_safe.size(), 2);
}

// ============================================================================
// TestCBFCost
// ============================================================================

class CBFCostTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    K_ = 2;
    N_ = 5;

    barrier_set_ = std::make_unique<BarrierFunctionSet>(0.2, 0.1, 5.0);

    // 참조 궤적 (사용 안 됨, placeholder)
    reference_ = Eigen::MatrixXd::Zero(N_ + 1, 3);

    // 제어 시퀀스 (사용 안 됨)
    controls_.resize(K_, Eigen::MatrixXd::Zero(N_, 2));
  }

  int K_;
  int N_;
  std::unique_ptr<BarrierFunctionSet> barrier_set_;
  Eigen::MatrixXd reference_;
  std::vector<Eigen::MatrixXd> controls_;
};

TEST_F(CBFCostTest, ZeroCostWhenSafe)
{
  // 장애물에서 멀리 떨어진 궤적 → 비용 0
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(10.0, 10.0, 0.1));
  barrier_set_->setObstacles(obs);

  CBFCost cost(barrier_set_.get(), 500.0, 1.0, 0.1);

  std::vector<Eigen::MatrixXd> trajectories;
  for (int k = 0; k < K_; ++k) {
    Eigen::MatrixXd traj = Eigen::MatrixXd::Zero(N_ + 1, 3);
    for (int t = 0; t <= N_; ++t) {
      traj(t, 0) = t * 0.1;  // x 전진
    }
    trajectories.push_back(traj);
  }

  Eigen::VectorXd costs = cost.compute(trajectories, controls_, reference_);
  EXPECT_EQ(costs.size(), K_);
  for (int k = 0; k < K_; ++k) {
    EXPECT_NEAR(costs(k), 0.0, 1e-6) << "안전 궤적은 CBF 비용 0";
  }
}

TEST_F(CBFCostTest, PositiveCostWhenViolating)
{
  // 장애물 쪽으로 접근하는 궤적 → 비용 > 0
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(1.0, 0.0, 0.1));  // d_safe = 0.4
  barrier_set_->setObstacles(obs);

  CBFCost cost(barrier_set_.get(), 500.0, 1.0, 0.1);

  std::vector<Eigen::MatrixXd> trajectories;
  // 궤적: (0,0) → (0.3,0) → (0.6,0) → (0.9,0) → ... 장애물 쪽으로 빠르게 접근
  Eigen::MatrixXd traj = Eigen::MatrixXd::Zero(N_ + 1, 3);
  for (int t = 0; t <= N_; ++t) {
    traj(t, 0) = t * 0.3;  // 빠른 접근
  }
  trajectories.push_back(traj);
  trajectories.push_back(traj);

  Eigen::VectorXd costs = cost.compute(trajectories, controls_, reference_);
  // 장애물에 급격히 접근하므로 DCBF 위반 발생
  EXPECT_GT(costs(0), 0.0) << "DCBF 위반 궤적은 양의 비용";
}

TEST_F(CBFCostTest, EmptyObstaclesZeroCost)
{
  // 장애물 없으면 비용 0
  CBFCost cost(barrier_set_.get(), 500.0, 1.0, 0.1);

  std::vector<Eigen::MatrixXd> trajectories;
  for (int k = 0; k < K_; ++k) {
    trajectories.push_back(Eigen::MatrixXd::Zero(N_ + 1, 3));
  }

  Eigen::VectorXd costs = cost.compute(trajectories, controls_, reference_);
  for (int k = 0; k < K_; ++k) {
    EXPECT_DOUBLE_EQ(costs(k), 0.0);
  }
}

TEST_F(CBFCostTest, BatchDimensionConsistency)
{
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(1.0, 0.0, 0.1));
  barrier_set_->setObstacles(obs);

  CBFCost cost(barrier_set_.get(), 500.0, 1.0, 0.1);

  // K=4, N=10 테스트
  int K = 4, N = 10;
  std::vector<Eigen::MatrixXd> trajectories;
  std::vector<Eigen::MatrixXd> controls;
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, 3);

  for (int k = 0; k < K; ++k) {
    trajectories.push_back(Eigen::MatrixXd::Random(N + 1, 3));
    controls.push_back(Eigen::MatrixXd::Zero(N, 2));
  }

  Eigen::VectorXd costs_out = cost.compute(trajectories, controls, ref);
  EXPECT_EQ(costs_out.size(), K);
  for (int k = 0; k < K; ++k) {
    EXPECT_GE(costs_out(k), 0.0) << "비용은 항상 ≥ 0";
  }
}

// ============================================================================
// Integration: CBFCost + CompositeMPPICost
// ============================================================================

TEST(CBFIntegrationTest, CBFCostInComposite)
{
  auto barrier_set = std::make_unique<BarrierFunctionSet>(0.2, 0.1, 5.0);
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(1.0, 0.0, 0.1));
  barrier_set->setObstacles(obs);

  CompositeMPPICost composite;
  composite.addCost(std::make_unique<CBFCost>(
    barrier_set.get(), 500.0, 1.0, 0.1));

  // 안전 궤적
  int K = 2, N = 5;
  std::vector<Eigen::MatrixXd> trajectories;
  std::vector<Eigen::MatrixXd> controls;
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, 3);

  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd traj = Eigen::MatrixXd::Zero(N + 1, 3);
    for (int t = 0; t <= N; ++t) {
      traj(t, 0) = -2.0 + t * 0.01;  // 멀리서 천천히 접근
    }
    trajectories.push_back(traj);
    controls.push_back(Eigen::MatrixXd::Zero(N, 2));
  }

  Eigen::VectorXd costs = composite.compute(trajectories, controls, ref);
  EXPECT_EQ(costs.size(), K);
}

}  // namespace mpc_controller_ros2
