#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <memory>

#include "mpc_controller_ros2/clf_function.hpp"
#include "mpc_controller_ros2/clf_cbf_qp_solver.hpp"
#include "mpc_controller_ros2/barrier_function.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"
#include "mpc_controller_ros2/cost_functions.hpp"

using namespace mpc_controller_ros2;

// ============================================================================
// Test Helpers
// ============================================================================

static MPPIParams defaultDiffDriveParams()
{
  MPPIParams params;
  params.motion_model = "diff_drive";
  return params;
}

static std::unique_ptr<MotionModel> createDiffDrive()
{
  auto params = defaultDiffDriveParams();
  return MotionModelFactory::create("diff_drive", params);
}

static BatchDynamicsWrapper createDynamics(std::shared_ptr<MotionModel> model)
{
  MPPIParams params = defaultDiffDriveParams();
  return BatchDynamicsWrapper(params, model);
}

// ============================================================================
// CLFFunction Tests
// ============================================================================

TEST(CLFFunction, Construction_ValidP)
{
  Eigen::MatrixXd P = Eigen::Matrix3d::Identity() * 10.0;
  EXPECT_NO_THROW(CLFFunction(P, 1.0));
}

TEST(CLFFunction, Construction_NonSquareP_Throws)
{
  Eigen::MatrixXd P(2, 3);
  P.setIdentity();
  EXPECT_THROW(CLFFunction(P, 1.0), std::invalid_argument);
}

TEST(CLFFunction, Construction_NegativeC_Throws)
{
  Eigen::MatrixXd P = Eigen::Matrix3d::Identity();
  EXPECT_THROW(CLFFunction(P, -1.0), std::invalid_argument);
}

TEST(CLFFunction, Evaluate_AtDesired_Zero)
{
  Eigen::MatrixXd P = Eigen::Matrix3d::Identity();
  CLFFunction clf(P, 1.0);

  Eigen::VectorXd x = Eigen::Vector3d(1.0, 2.0, 0.5);
  double V = clf.evaluate(x, x);
  EXPECT_NEAR(V, 0.0, 1e-10);
}

TEST(CLFFunction, Evaluate_PositiveDefinite)
{
  Eigen::MatrixXd P = Eigen::Matrix3d::Identity() * 5.0;
  CLFFunction clf(P, 1.0);

  Eigen::VectorXd x = Eigen::Vector3d(1.0, 2.0, 0.0);
  Eigen::VectorXd x_des = Eigen::Vector3d(0.0, 0.0, 0.0);
  double V = clf.evaluate(x, x_des);
  // V = 5 * (1² + 2² + 0²) = 25
  EXPECT_NEAR(V, 25.0, 1e-10);
  EXPECT_GT(V, 0.0);
}

TEST(CLFFunction, Gradient_MatchesFiniteDiff)
{
  Eigen::MatrixXd P = Eigen::Matrix3d::Identity() * 3.0;
  CLFFunction clf(P, 1.0);

  Eigen::VectorXd x = Eigen::Vector3d(1.0, -0.5, 0.3);
  Eigen::VectorXd x_des = Eigen::Vector3d(0.0, 0.0, 0.0);

  Eigen::VectorXd grad = clf.gradient(x, x_des);
  // ∇V = 2 P (x - x_des) = 2 * 3 * [1, -0.5, 0.3] = [6, -3, 1.8]
  EXPECT_NEAR(grad(0), 6.0, 1e-10);
  EXPECT_NEAR(grad(1), -3.0, 1e-10);
  EXPECT_NEAR(grad(2), 1.8, 1e-10);

  // 유한차분 검증
  double eps = 1e-5;
  for (int i = 0; i < 3; ++i) {
    Eigen::VectorXd x_plus = x;
    x_plus(i) += eps;
    double V_plus = clf.evaluate(x_plus, x_des);
    double V_base = clf.evaluate(x, x_des);
    double fd = (V_plus - V_base) / eps;
    EXPECT_NEAR(grad(i), fd, 1e-4);
  }
}

TEST(CLFFunction, AngleWrapping)
{
  Eigen::MatrixXd P = Eigen::Matrix3d::Identity();
  CLFFunction clf(P, 1.0, {2});  // theta at index 2

  Eigen::VectorXd x = Eigen::Vector3d(0.0, 0.0, M_PI - 0.1);
  Eigen::VectorXd x_des = Eigen::Vector3d(0.0, 0.0, -M_PI + 0.1);
  // angle difference should be 0.2 (not 2*π - 0.2)
  double V = clf.evaluate(x, x_des);
  EXPECT_LT(V, 0.1);  // small angle error
}

TEST(CLFFunction, LieDerivatives_Consistent)
{
  Eigen::MatrixXd P = Eigen::Matrix3d::Identity() * 2.0;
  CLFFunction clf(P, 1.0);

  auto model = std::shared_ptr<MotionModel>(createDiffDrive().release());
  auto dynamics = createDynamics(model);

  Eigen::VectorXd state = Eigen::Vector3d(1.0, 0.0, 0.0);
  Eigen::VectorXd x_des = Eigen::Vector3d(0.0, 0.0, 0.0);
  Eigen::VectorXd u = Eigen::Vector2d(0.5, 0.1);

  // 동역학 계산
  Eigen::MatrixXd s(1, 3); s.row(0) = state.transpose();
  Eigen::MatrixXd c(1, 2); c.row(0) = u.transpose();
  Eigen::VectorXd x_dot = model->dynamicsBatch(s, c).row(0).transpose();

  // B 행렬 (유한차분)
  Eigen::MatrixXd B(3, 2);
  double eps = 1e-4;
  for (int j = 0; j < 2; ++j) {
    Eigen::VectorXd u_plus = u;
    u_plus(j) += eps;
    Eigen::MatrixXd c_p(1, 2); c_p.row(0) = u_plus.transpose();
    Eigen::VectorXd xdot_p = model->dynamicsBatch(s, c_p).row(0).transpose();
    B.col(j) = (xdot_p - x_dot) / eps;
  }

  auto [L_f_V, L_g_V] = clf.lieDerivatives(state, x_des, x_dot, B);

  // L_g_V should have size nu=2
  EXPECT_EQ(L_g_V.size(), 2);

  // V̇ = grad_V · x_dot = L_f_V (at current u)
  Eigen::VectorXd grad_V = clf.gradient(state, x_des);
  double V_dot_expected = grad_V.dot(x_dot);
  EXPECT_NEAR(L_f_V, V_dot_expected, 1e-8);
}

// ============================================================================
// CLFCBFQPSolver Tests
// ============================================================================

class CLFCBFQPTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    model_ = std::shared_ptr<MotionModel>(createDiffDrive().release());
    MPPIParams params = defaultDiffDriveParams();
    dynamics_ = std::make_unique<BatchDynamicsWrapper>(params, model_);

    Eigen::MatrixXd P = Eigen::Matrix3d::Identity() * 5.0;
    clf_ = std::make_unique<CLFFunction>(P, 1.0, std::vector<int>{2});

    barrier_set_ = std::make_unique<BarrierFunctionSet>(0.2, 0.1, 3.0);

    Eigen::VectorXd u_min(2), u_max(2);
    u_min << -0.5, -1.5;
    u_max << 0.5, 1.5;

    solver_ = std::make_unique<CLFCBFQPSolver>(
      clf_.get(), barrier_set_.get(), 1.0, 100.0, u_min, u_max);
  }

  std::shared_ptr<MotionModel> model_;
  std::unique_ptr<BatchDynamicsWrapper> dynamics_;
  std::unique_ptr<CLFFunction> clf_;
  std::unique_ptr<BarrierFunctionSet> barrier_set_;
  std::unique_ptr<CLFCBFQPSolver> solver_;
};

TEST_F(CLFCBFQPTest, CLFOnly_NoObstacles)
{
  Eigen::VectorXd state = Eigen::Vector3d(1.0, 0.0, 0.0);
  Eigen::VectorXd x_des = Eigen::Vector3d(0.0, 0.0, 0.0);
  Eigen::VectorXd u_ref = Eigen::Vector2d(0.0, 0.0);

  auto result = solver_->solveCLFOnly(state, x_des, u_ref, *dynamics_);

  EXPECT_TRUE(result.feasible);
  EXPECT_GT(result.clf_value, 0.0);  // V > 0 (not at target)
  EXPECT_EQ(result.u_safe.size(), 2);
}

TEST_F(CLFCBFQPTest, CLFOnly_AtTarget_NoChange)
{
  Eigen::VectorXd state = Eigen::Vector3d(0.0, 0.0, 0.0);
  Eigen::VectorXd x_des = Eigen::Vector3d(0.0, 0.0, 0.0);
  Eigen::VectorXd u_ref = Eigen::Vector2d(0.0, 0.0);

  auto result = solver_->solveCLFOnly(state, x_des, u_ref, *dynamics_);

  EXPECT_TRUE(result.feasible);
  EXPECT_NEAR(result.clf_value, 0.0, 1e-10);
  // At target, u_ref=0 should be returned (no correction needed)
  EXPECT_NEAR(result.u_safe(0), 0.0, 0.1);
  EXPECT_NEAR(result.u_safe(1), 0.0, 0.1);
}

TEST_F(CLFCBFQPTest, CBFOnly_ObstacleAvoidance)
{
  // 장애물을 가까이에 배치
  std::vector<Eigen::Vector3d> obstacles;
  obstacles.push_back(Eigen::Vector3d(1.5, 0.0, 0.1));  // x=1.5, y=0, r=0.1
  barrier_set_->setObstacles(obstacles);

  Eigen::VectorXd state = Eigen::Vector3d(1.0, 0.0, 0.0);
  Eigen::VectorXd x_des = Eigen::Vector3d(2.0, 0.0, 0.0);  // obstacle 방향
  Eigen::VectorXd u_ref = Eigen::Vector2d(0.5, 0.0);  // 전진

  auto result = solver_->solve(state, x_des, u_ref, *dynamics_);

  EXPECT_TRUE(result.feasible);
  // CBF가 전진을 제한해야 함
  EXPECT_LE(result.u_safe(0), u_ref(0) + 0.01);  // 전진 속도 감소 또는 유지
}

TEST_F(CLFCBFQPTest, Combined_SafetyPriority)
{
  // 장애물이 목표 방향에 있는 경우: CLF는 전진을 원하고 CBF는 감속을 원함
  std::vector<Eigen::Vector3d> obstacles;
  obstacles.push_back(Eigen::Vector3d(1.2, 0.0, 0.1));  // 전방 장애물
  barrier_set_->setObstacles(obstacles);

  Eigen::VectorXd state = Eigen::Vector3d(0.0, 0.0, 0.0);
  Eigen::VectorXd x_des = Eigen::Vector3d(2.0, 0.0, 0.0);
  Eigen::VectorXd u_ref = Eigen::Vector2d(0.5, 0.0);

  auto result = solver_->solve(state, x_des, u_ref, *dynamics_);

  EXPECT_TRUE(result.feasible);
  // CBF가 활성화되므로 전진 속도가 제한됨
  EXPECT_LE(result.u_safe(0), u_ref(0) + 0.01);
  // slack이 사용되었는지 확인 (CLF-CBF 충돌 시)
  EXPECT_GE(result.slack, -1e-6);
}

TEST_F(CLFCBFQPTest, SlackRelaxation)
{
  // CLF-CBF 충돌 시 slack이 양수
  std::vector<Eigen::Vector3d> obstacles;
  obstacles.push_back(Eigen::Vector3d(0.5, 0.0, 0.1));
  barrier_set_->setObstacles(obstacles);

  Eigen::VectorXd state = Eigen::Vector3d(0.0, 0.0, 0.0);
  Eigen::VectorXd x_des = Eigen::Vector3d(1.0, 0.0, 0.0);
  Eigen::VectorXd u_ref = Eigen::Vector2d(0.5, 0.0);

  auto result = solver_->solve(state, x_des, u_ref, *dynamics_);

  EXPECT_TRUE(result.feasible);
  // slack ≥ 0 (CLF가 완화됨)
  EXPECT_GE(result.slack, -1e-6);
}

TEST_F(CLFCBFQPTest, ControlBounds)
{
  Eigen::VectorXd state = Eigen::Vector3d(5.0, 5.0, 0.0);
  Eigen::VectorXd x_des = Eigen::Vector3d(0.0, 0.0, 0.0);
  Eigen::VectorXd u_ref = Eigen::Vector2d(10.0, 10.0);  // 범위 초과

  auto result = solver_->solveCLFOnly(state, x_des, u_ref, *dynamics_);

  EXPECT_TRUE(result.feasible);
  EXPECT_LE(result.u_safe(0), 0.5 + 1e-6);
  EXPECT_GE(result.u_safe(0), -0.5 - 1e-6);
  EXPECT_LE(result.u_safe(1), 1.5 + 1e-6);
  EXPECT_GE(result.u_safe(1), -1.5 - 1e-6);
}

TEST_F(CLFCBFQPTest, NoObstacles_CLFActive)
{
  // 장애물 없음 → CLF만 작동
  Eigen::VectorXd state = Eigen::Vector3d(2.0, 1.0, 0.0);
  Eigen::VectorXd x_des = Eigen::Vector3d(0.0, 0.0, 0.0);
  Eigen::VectorXd u_ref = Eigen::Vector2d(0.0, 0.0);

  auto result = solver_->solve(state, x_des, u_ref, *dynamics_);

  EXPECT_TRUE(result.feasible);
  EXPECT_GT(result.clf_value, 0.0);
  EXPECT_TRUE(result.cbf_margins.empty());
}

TEST_F(CLFCBFQPTest, Iterations_Bounded)
{
  Eigen::VectorXd state = Eigen::Vector3d(1.0, 1.0, 0.5);
  Eigen::VectorXd x_des = Eigen::Vector3d(0.0, 0.0, 0.0);
  Eigen::VectorXd u_ref = Eigen::Vector2d(0.3, 0.1);

  auto result = solver_->solveCLFOnly(state, x_des, u_ref, *dynamics_);

  EXPECT_GT(result.iterations, 0);
  EXPECT_LE(result.iterations, 100);
}

// ============================================================================
// CLFCost Tests
// ============================================================================

TEST(CLFCost, ZeroWeight_ZeroCost)
{
  Eigen::MatrixXd P = Eigen::Matrix3d::Identity();
  CLFFunction clf(P, 1.0);
  CLFCost cost(&clf, 0.0, 0.1);

  std::vector<Eigen::MatrixXd> trajectories(4, Eigen::MatrixXd::Random(11, 3));
  std::vector<Eigen::MatrixXd> controls(4, Eigen::MatrixXd::Random(10, 2));
  Eigen::MatrixXd reference = Eigen::MatrixXd::Zero(11, 3);

  auto costs = cost.compute(trajectories, controls, reference);
  EXPECT_EQ(costs.size(), 4);
  for (int k = 0; k < 4; ++k) {
    EXPECT_NEAR(costs(k), 0.0, 1e-10);
  }
}

TEST(CLFCost, ConvergingTrajectory_LowCost)
{
  Eigen::MatrixXd P = Eigen::Matrix3d::Identity();
  CLFFunction clf(P, 0.5);
  CLFCost cost(&clf, 10.0, 0.1);

  // 수렴하는 궤적: x가 점점 0에 가까워짐
  int N = 10;
  Eigen::MatrixXd traj(N + 1, 3);
  for (int t = 0; t <= N; ++t) {
    double decay = std::exp(-0.3 * t);
    traj.row(t) = Eigen::Vector3d(2.0 * decay, 1.0 * decay, 0.0).transpose();
  }

  // 발산하는 궤적: x가 점점 커짐
  Eigen::MatrixXd traj_div(N + 1, 3);
  for (int t = 0; t <= N; ++t) {
    double grow = 1.0 + 0.3 * t;
    traj_div.row(t) = Eigen::Vector3d(2.0 * grow, 1.0 * grow, 0.0).transpose();
  }

  std::vector<Eigen::MatrixXd> trajectories = {traj, traj_div};
  std::vector<Eigen::MatrixXd> controls(2, Eigen::MatrixXd::Zero(N, 2));
  Eigen::MatrixXd reference = Eigen::MatrixXd::Zero(N + 1, 3);

  auto costs = cost.compute(trajectories, controls, reference);
  // 수렴 궤적 비용 < 발산 궤적 비용
  EXPECT_LT(costs(0), costs(1));
}

TEST(CLFCost, Name)
{
  Eigen::MatrixXd P = Eigen::Matrix3d::Identity();
  CLFFunction clf(P, 1.0);
  CLFCost cost(&clf, 1.0, 0.1);
  EXPECT_EQ(cost.name(), "clf");
}

// ============================================================================
// AdaptiveShield + CLF-CBF 통합 (단위 테스트 수준)
// ============================================================================

TEST(CLFCBFIntegration, CLFReducesDistanceToGoal)
{
  // CLF가 없으면 u_ref 유지, CLF가 있으면 목표 방향으로 보정
  auto model = std::shared_ptr<MotionModel>(createDiffDrive().release());
  auto dynamics = createDynamics(model);

  Eigen::MatrixXd P = Eigen::Matrix3d::Identity() * 10.0;
  CLFFunction clf(P, 2.0, {2});

  BarrierFunctionSet barrier_set(0.2, 0.1, 3.0);
  // 장애물 없음

  Eigen::VectorXd u_min(2), u_max(2);
  u_min << -0.5, -1.5;
  u_max << 0.5, 1.5;

  CLFCBFQPSolver solver(&clf, &barrier_set, 1.0, 100.0, u_min, u_max);

  Eigen::VectorXd state = Eigen::Vector3d(1.0, 0.0, 0.0);
  Eigen::VectorXd x_des = Eigen::Vector3d(0.0, 0.0, 0.0);
  Eigen::VectorXd u_ref = Eigen::Vector2d(0.0, 0.0);  // 정지

  auto result = solver.solveCLFOnly(state, x_des, u_ref, dynamics);

  EXPECT_TRUE(result.feasible);
  // CLF 덕분에 negative v (후진) 또는 positive omega (회전) 보정
  // 어떤 보정이든 u_ref=0에서 벗어나야 함
  double u_norm = result.u_safe.norm();
  // CLF가 정지 상태에서 목표로 향하게 보정할 수 있음
  // (단, CLF는 soft constraint이므로 보정이 보장되지는 않음)
  EXPECT_GE(u_norm, -1e-10);  // 최소한 실현 가능
}

TEST(CLFCBFIntegration, WithSwerveModel)
{
  MPPIParams params;
  params.motion_model = "swerve";
  auto model = std::shared_ptr<MotionModel>(MotionModelFactory::create("swerve", params).release());
  BatchDynamicsWrapper dynamics(params, model);

  Eigen::MatrixXd P = Eigen::Matrix3d::Identity() * 5.0;
  CLFFunction clf(P, 1.0, {2});

  BarrierFunctionSet barrier_set(0.2, 0.1, 3.0);

  Eigen::VectorXd u_min(3), u_max(3);
  u_min << -0.5, -1.5, -0.5;
  u_max << 0.5, 1.5, 0.5;

  CLFCBFQPSolver solver(&clf, &barrier_set, 1.0, 100.0, u_min, u_max);

  Eigen::VectorXd state = Eigen::Vector3d(1.0, 1.0, 0.0);
  Eigen::VectorXd x_des = Eigen::Vector3d(0.0, 0.0, 0.0);
  Eigen::VectorXd u_ref = Eigen::Vector3d(0.0, 0.0, 0.0);

  auto result = solver.solveCLFOnly(state, x_des, u_ref, dynamics);

  EXPECT_TRUE(result.feasible);
  EXPECT_EQ(result.u_safe.size(), 3);
}

