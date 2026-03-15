#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <limits>
#include <vector>

#include "mpc_controller_ros2/barrier_function.hpp"
#include "mpc_controller_ros2/clf_cbf_qp_solver.hpp"
#include "mpc_controller_ros2/clf_function.hpp"
#include "mpc_controller_ros2/motion_model.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"

using namespace mpc_controller_ros2;

// ============================================================================
// Test Helpers
// ============================================================================

static MPPIParams makeDefaultParams()
{
  MPPIParams params;
  params.motion_model = "diff_drive";
  params.dt = 0.05;
  return params;
}

static std::shared_ptr<MotionModel> makeDiffDriveModel()
{
  auto params = makeDefaultParams();
  return std::shared_ptr<MotionModel>(
    MotionModelFactory::create("diff_drive", params).release());
}

// ============================================================================
// CBFComposition_EvaluateComposite
// ============================================================================

class CBFCompositionEvaluateTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    barrier_set_ = std::make_unique<BarrierFunctionSet>(0.2, 0.3, 10.0);
    state_ = Eigen::Vector3d(0.0, 0.0, 0.0);
  }

  std::unique_ptr<BarrierFunctionSet> barrier_set_;
  Eigen::VectorXd state_;
};

// --- 1. SingleBarrier_EqualsOriginal ---
// 단일 barrier인 경우, 모든 합성 방법이 동일한 값을 반환해야 함
TEST_F(CBFCompositionEvaluateTest, SingleBarrier_EqualsOriginal)
{
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(2.0, 0.0, 0.3));  // d_safe = 0.3 + 0.2 + 0.3 = 0.8
  barrier_set_->setObstacles(obs);

  // 개별 barrier 값
  Eigen::VectorXd all_h = barrier_set_->evaluateAll(state_);
  ASSERT_EQ(all_h.size(), 1);
  double h_single = all_h(0);

  // 모든 합성 방법이 동일한 값 반환
  double h_min = barrier_set_->evaluateComposite(
    state_, CBFCompositionMethod::MIN);
  double h_smooth = barrier_set_->evaluateComposite(
    state_, CBFCompositionMethod::SMOOTH_MIN, 100.0);
  double h_lse = barrier_set_->evaluateComposite(
    state_, CBFCompositionMethod::LOG_SUM_EXP);
  double h_prod = barrier_set_->evaluateComposite(
    state_, CBFCompositionMethod::PRODUCT);

  EXPECT_NEAR(h_min, h_single, 1e-10)
    << "MIN: 단일 barrier와 동일해야 함";
  EXPECT_NEAR(h_smooth, h_single, 0.05)
    << "SMOOTH_MIN(alpha=100): 단일 barrier에 근사해야 함";
  EXPECT_NEAR(h_lse, h_single, 0.01)
    << "LOG_SUM_EXP: 단일 barrier에 근사해야 함";
  EXPECT_NEAR(h_prod, h_single, 1e-10)
    << "PRODUCT: 단일 barrier와 동일해야 함";
}

// --- 2. SmoothMin_ApproximatesMin ---
// alpha가 클수록 smooth min이 실제 min에 가까워져야 함
TEST_F(CBFCompositionEvaluateTest, SmoothMin_ApproximatesMin)
{
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(1.5, 0.0, 0.3));   // 가까운 장애물
  obs.push_back(Eigen::Vector3d(3.0, 1.0, 0.5));    // 먼 장애물
  obs.push_back(Eigen::Vector3d(2.0, -1.0, 0.2));   // 중간 장애물
  barrier_set_->setObstacles(obs);

  double h_min = barrier_set_->evaluateComposite(
    state_, CBFCompositionMethod::MIN);
  double h_smooth_100 = barrier_set_->evaluateComposite(
    state_, CBFCompositionMethod::SMOOTH_MIN, 100.0);

  // smooth min은 항상 min 이하 (하한)
  EXPECT_LE(h_smooth_100, h_min + 1e-6)
    << "SMOOTH_MIN <= MIN (smooth min은 하한 근사)";

  // alpha=100이면 min과 매우 가까워야 함
  EXPECT_NEAR(h_smooth_100, h_min, 0.1)
    << "alpha=100: SMOOTH_MIN이 MIN에 가까워야 함";

  // alpha가 작으면 min과 더 멀어짐
  double h_smooth_1 = barrier_set_->evaluateComposite(
    state_, CBFCompositionMethod::SMOOTH_MIN, 1.0);
  double diff_100 = std::abs(h_smooth_100 - h_min);
  double diff_1 = std::abs(h_smooth_1 - h_min);
  EXPECT_LE(diff_100, diff_1 + 1e-6)
    << "alpha가 클수록 min에 더 가까움";
}

// --- 3. LogSumExp_AlphaOneCase ---
// LOG_SUM_EXP는 alpha=1인 smooth min과 동일
TEST_F(CBFCompositionEvaluateTest, LogSumExp_AlphaOneCase)
{
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(2.0, 0.0, 0.3));
  obs.push_back(Eigen::Vector3d(3.0, 2.0, 0.4));
  barrier_set_->setObstacles(obs);

  double h_lse = barrier_set_->evaluateComposite(
    state_, CBFCompositionMethod::LOG_SUM_EXP);
  double h_smooth_alpha1 = barrier_set_->evaluateComposite(
    state_, CBFCompositionMethod::SMOOTH_MIN, 1.0);

  EXPECT_NEAR(h_lse, h_smooth_alpha1, 1e-8)
    << "LOG_SUM_EXP == SMOOTH_MIN(alpha=1)";
}

// --- 4. Product_AllPositive ---
// 모든 h_i > 0일 때, product = h_1 * h_2 * ...
TEST_F(CBFCompositionEvaluateTest, Product_AllPositive)
{
  // 모든 장애물이 멀어서 h_i > 0
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(3.0, 0.0, 0.3));   // 먼 장애물
  obs.push_back(Eigen::Vector3d(0.0, 4.0, 0.2));   // 먼 장애물
  barrier_set_->setObstacles(obs);

  Eigen::VectorXd all_h = barrier_set_->evaluateAll(state_);
  ASSERT_EQ(all_h.size(), 2);
  EXPECT_GT(all_h(0), 0.0);
  EXPECT_GT(all_h(1), 0.0);

  double expected_product = all_h(0) * all_h(1);
  double h_prod = barrier_set_->evaluateComposite(
    state_, CBFCompositionMethod::PRODUCT);

  EXPECT_NEAR(h_prod, expected_product, 1e-8)
    << "PRODUCT: h_1 * h_2";
  EXPECT_GT(h_prod, 0.0)
    << "모든 h > 0이면 product > 0";
}

// --- 5. Product_OneNegative_ZeroProduct ---
// h_i 중 하나가 음수이면 product가 0으로 클램프되거나 부호가 바뀜
TEST_F(CBFCompositionEvaluateTest, Product_OneNegative_ZeroProduct)
{
  // 장애물 내부에 위치 (h < 0)
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(0.3, 0.0, 0.3));   // 매우 가까움 → h < 0
  obs.push_back(Eigen::Vector3d(5.0, 0.0, 0.2));   // 멀리 → h > 0
  barrier_set_->setObstacles(obs);

  Eigen::VectorXd all_h = barrier_set_->evaluateAll(state_);
  ASSERT_EQ(all_h.size(), 2);
  EXPECT_LT(all_h(0), 0.0) << "첫 번째 barrier는 음수 (장애물 내부)";
  EXPECT_GT(all_h(1), 0.0) << "두 번째 barrier는 양수 (안전 영역)";

  double h_prod = barrier_set_->evaluateComposite(
    state_, CBFCompositionMethod::PRODUCT);

  // product = 음수 * 양수 → 음수 또는 0 클램프
  // 핵심: h < 0인 barrier가 있으면 product가 안전하지 않음을 반영
  EXPECT_LE(h_prod, 0.0)
    << "음수 barrier가 있으면 product <= 0";
}

// --- 6. NoActiveBarriers_ReturnsInfinity ---
// 활성 barrier가 없으면 무한대(또는 매우 큰 양수) 반환
TEST_F(CBFCompositionEvaluateTest, NoActiveBarriers_ReturnsInfinity)
{
  // 장애물 설정하지 않음 (empty)
  double h_min = barrier_set_->evaluateComposite(
    state_, CBFCompositionMethod::MIN);
  double h_smooth = barrier_set_->evaluateComposite(
    state_, CBFCompositionMethod::SMOOTH_MIN);
  double h_lse = barrier_set_->evaluateComposite(
    state_, CBFCompositionMethod::LOG_SUM_EXP);
  double h_prod = barrier_set_->evaluateComposite(
    state_, CBFCompositionMethod::PRODUCT);

  // 장애물 없음 → 무한대 또는 매우 큰 양수
  double large_val = 1e6;
  EXPECT_GT(h_min, large_val)
    << "barrier 없으면 infinity 또는 매우 큰 값";
  EXPECT_GT(h_smooth, large_val);
  EXPECT_GT(h_lse, large_val);
  EXPECT_GT(h_prod, large_val);
}

// ============================================================================
// CBFComposition_CompositeGradient
// ============================================================================

class CBFCompositionGradientTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    barrier_set_ = std::make_unique<BarrierFunctionSet>(0.2, 0.3, 10.0);
    state_ = Eigen::Vector3d(0.0, 0.0, 0.0);
  }

  std::unique_ptr<BarrierFunctionSet> barrier_set_;
  Eigen::VectorXd state_;
};

// --- 7. SingleBarrier_EqualsGradient ---
// 단일 barrier gradient와 합성 gradient가 동일
TEST_F(CBFCompositionGradientTest, SingleBarrier_EqualsGradient)
{
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(2.0, 1.0, 0.3));
  barrier_set_->setObstacles(obs);

  // 개별 gradient
  const auto& barriers = barrier_set_->barriers();
  ASSERT_EQ(barriers.size(), 1u);
  Eigen::VectorXd grad_single = barriers[0].gradient(state_);

  // 합성 gradient (모든 방법)
  Eigen::VectorXd grad_min = barrier_set_->compositeGradient(
    state_, CBFCompositionMethod::MIN);
  Eigen::VectorXd grad_prod = barrier_set_->compositeGradient(
    state_, CBFCompositionMethod::PRODUCT);

  ASSERT_EQ(grad_min.size(), state_.size());
  ASSERT_EQ(grad_prod.size(), state_.size());

  for (int i = 0; i < state_.size(); ++i) {
    EXPECT_NEAR(grad_min(i), grad_single(i), 1e-8)
      << "MIN gradient[" << i << "]";
    EXPECT_NEAR(grad_prod(i), grad_single(i), 1e-8)
      << "PRODUCT gradient[" << i << "]";
  }
}

// --- 8. SmoothMin_WeightedAverage ---
// smooth min gradient는 softmax 가중 평균이고, 가중치 합 = 1
TEST_F(CBFCompositionGradientTest, SmoothMin_WeightedAverage)
{
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(2.0, 0.0, 0.3));
  obs.push_back(Eigen::Vector3d(0.0, 3.0, 0.4));
  barrier_set_->setObstacles(obs);

  double alpha = 10.0;
  Eigen::VectorXd grad_smooth = barrier_set_->compositeGradient(
    state_, CBFCompositionMethod::SMOOTH_MIN, alpha);

  ASSERT_EQ(grad_smooth.size(), state_.size());

  // gradient는 0이 아니어야 함 (장애물이 있으므로)
  EXPECT_GT(grad_smooth.norm(), 1e-10)
    << "합성 gradient는 비영벡터여야 함";

  // 유한차분 검증: ∂h_c/∂x_i ≈ (h_c(x+eps*e_i) - h_c(x-eps*e_i)) / (2*eps)
  double eps = 1e-5;
  for (int i = 0; i < state_.size(); ++i) {
    Eigen::VectorXd state_plus = state_;
    Eigen::VectorXd state_minus = state_;
    state_plus(i) += eps;
    state_minus(i) -= eps;

    double h_plus = barrier_set_->evaluateComposite(
      state_plus, CBFCompositionMethod::SMOOTH_MIN, alpha);
    double h_minus = barrier_set_->evaluateComposite(
      state_minus, CBFCompositionMethod::SMOOTH_MIN, alpha);
    double fd = (h_plus - h_minus) / (2.0 * eps);

    EXPECT_NEAR(grad_smooth(i), fd, 1e-3)
      << "SMOOTH_MIN gradient[" << i << "] 유한차분 검증";
  }
}

// --- 9. Min_ArgminGradient ---
// MIN 방법의 gradient는 argmin barrier의 gradient와 동일
TEST_F(CBFCompositionGradientTest, Min_ArgminGradient)
{
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(1.5, 0.0, 0.3));   // 가까움 (h 작음)
  obs.push_back(Eigen::Vector3d(5.0, 0.0, 0.2));   // 멀리 (h 큼)
  barrier_set_->setObstacles(obs);

  Eigen::VectorXd all_h = barrier_set_->evaluateAll(state_);
  ASSERT_EQ(all_h.size(), 2);

  // argmin 찾기
  int argmin_idx = 0;
  if (all_h(1) < all_h(0)) argmin_idx = 1;

  Eigen::VectorXd grad_argmin = barrier_set_->barriers()[argmin_idx].gradient(state_);
  Eigen::VectorXd grad_min = barrier_set_->compositeGradient(
    state_, CBFCompositionMethod::MIN);

  ASSERT_EQ(grad_min.size(), state_.size());
  for (int i = 0; i < state_.size(); ++i) {
    EXPECT_NEAR(grad_min(i), grad_argmin(i), 1e-8)
      << "MIN gradient[" << i << "]는 argmin barrier의 gradient";
  }
}

// --- 10. Product_ChainRule ---
// product gradient = sum_i (prod_{j!=i} h_j) * grad(h_i) (chain rule)
TEST_F(CBFCompositionGradientTest, Product_ChainRule)
{
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(3.0, 0.0, 0.3));   // h_0 > 0
  obs.push_back(Eigen::Vector3d(0.0, 4.0, 0.2));   // h_1 > 0
  barrier_set_->setObstacles(obs);

  Eigen::VectorXd all_h = barrier_set_->evaluateAll(state_);
  ASSERT_EQ(all_h.size(), 2);
  EXPECT_GT(all_h(0), 0.0);
  EXPECT_GT(all_h(1), 0.0);

  const auto& barriers = barrier_set_->barriers();
  Eigen::VectorXd g0 = barriers[0].gradient(state_);
  Eigen::VectorXd g1 = barriers[1].gradient(state_);

  // chain rule: grad(h0*h1) = h1*grad(h0) + h0*grad(h1)
  Eigen::VectorXd expected_grad = all_h(1) * g0 + all_h(0) * g1;

  Eigen::VectorXd grad_prod = barrier_set_->compositeGradient(
    state_, CBFCompositionMethod::PRODUCT);

  ASSERT_EQ(grad_prod.size(), state_.size());
  for (int i = 0; i < state_.size(); ++i) {
    EXPECT_NEAR(grad_prod(i), expected_grad(i), 1e-6)
      << "PRODUCT gradient[" << i << "] chain rule 검증";
  }
}

// ============================================================================
// CBFComposition_SolveComposite
// ============================================================================

class CBFCompositionSolveTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    model_ = makeDiffDriveModel();
    MPPIParams params = makeDefaultParams();
    dynamics_ = std::make_unique<BatchDynamicsWrapper>(params, model_);

    Eigen::MatrixXd P = Eigen::Matrix3d::Identity() * 5.0;
    clf_ = std::make_unique<CLFFunction>(P, 1.0, std::vector<int>{2});

    barrier_set_ = std::make_unique<BarrierFunctionSet>(0.2, 0.3, 5.0);

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

// --- 11. NoBarriers_FallsBackToCLFOnly ---
// 장애물이 없으면 CLF-only와 동일한 결과
TEST_F(CBFCompositionSolveTest, NoBarriers_FallsBackToCLFOnly)
{
  Eigen::VectorXd state = Eigen::Vector3d(1.0, 0.5, 0.0);
  Eigen::VectorXd x_des = Eigen::Vector3d(0.0, 0.0, 0.0);
  Eigen::VectorXd u_ref = Eigen::Vector2d(0.0, 0.0);

  auto result_composite = solver_->solveComposite(
    state, x_des, u_ref, *dynamics_, CBFCompositionMethod::SMOOTH_MIN);
  auto result_clf = solver_->solveCLFOnly(
    state, x_des, u_ref, *dynamics_);

  EXPECT_TRUE(result_composite.feasible)
    << "장애물 없으면 실현 가능";
  EXPECT_TRUE(result_clf.feasible);

  // CLF 값은 동일해야 함
  EXPECT_NEAR(result_composite.clf_value, result_clf.clf_value, 1e-8);

  // 제어 출력도 매우 유사해야 함
  EXPECT_NEAR(result_composite.u_safe(0), result_clf.u_safe(0), 0.05)
    << "장애물 없으면 CLF-only와 유사한 결과";
  EXPECT_NEAR(result_composite.u_safe(1), result_clf.u_safe(1), 0.05);
}

// --- 12. SingleBarrier_SameAsSolve ---
// 단일 barrier인 경우, solveComposite가 solve와 유사한 결과
TEST_F(CBFCompositionSolveTest, SingleBarrier_SameAsSolve)
{
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(2.0, 0.0, 0.1));
  barrier_set_->setObstacles(obs);

  Eigen::VectorXd state = Eigen::Vector3d(0.0, 0.0, 0.0);
  Eigen::VectorXd x_des = Eigen::Vector3d(3.0, 0.0, 0.0);
  Eigen::VectorXd u_ref = Eigen::Vector2d(0.3, 0.0);

  auto result_solve = solver_->solve(
    state, x_des, u_ref, *dynamics_);
  auto result_composite = solver_->solveComposite(
    state, x_des, u_ref, *dynamics_, CBFCompositionMethod::SMOOTH_MIN, 100.0);

  // 둘 다 feasible이거나, 아니면 제어 크기가 유사
  if (result_solve.feasible && result_composite.feasible) {
    EXPECT_NEAR(result_composite.u_safe(0), result_solve.u_safe(0), 0.3)
      << "단일 barrier: solveComposite ~ solve";
  }
  // 최소한 하나는 결과를 반환
  EXPECT_EQ(result_composite.u_safe.size(), 2);
}

// --- 13. MultipleBarriers_Feasible ---
// 여러 장애물이 있어도 QP가 feasible
TEST_F(CBFCompositionSolveTest, MultipleBarriers_Feasible)
{
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(3.0, 1.0, 0.2));
  obs.push_back(Eigen::Vector3d(3.0, -1.0, 0.2));
  obs.push_back(Eigen::Vector3d(4.0, 0.0, 0.3));
  barrier_set_->setObstacles(obs);

  Eigen::VectorXd state = Eigen::Vector3d(0.0, 0.0, 0.0);
  Eigen::VectorXd x_des = Eigen::Vector3d(5.0, 0.0, 0.0);
  Eigen::VectorXd u_ref = Eigen::Vector2d(0.3, 0.0);

  for (auto method : {CBFCompositionMethod::MIN,
                      CBFCompositionMethod::SMOOTH_MIN,
                      CBFCompositionMethod::LOG_SUM_EXP,
                      CBFCompositionMethod::PRODUCT}) {
    auto result = solver_->solveComposite(
      state, x_des, u_ref, *dynamics_, method);

    // 원거리 장애물 → feasible해야 함
    EXPECT_TRUE(result.feasible)
      << "원거리 다중 장애물에서 QP는 feasible이어야 함";
    EXPECT_EQ(result.u_safe.size(), 2);

    // 제어 범위 확인
    EXPECT_GE(result.u_safe(0), -0.5 - 1e-6);
    EXPECT_LE(result.u_safe(0), 0.5 + 1e-6);
    EXPECT_GE(result.u_safe(1), -1.5 - 1e-6);
    EXPECT_LE(result.u_safe(1), 1.5 + 1e-6);
  }
}

// --- 14. MultipleBarriers_SafetyMaintained ---
// 합성 CBF margin이 안전 임계값 이상
TEST_F(CBFCompositionSolveTest, MultipleBarriers_SafetyMaintained)
{
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(2.5, 0.5, 0.1));
  obs.push_back(Eigen::Vector3d(2.5, -0.5, 0.1));
  barrier_set_->setObstacles(obs);

  Eigen::VectorXd state = Eigen::Vector3d(0.0, 0.0, 0.0);
  Eigen::VectorXd x_des = Eigen::Vector3d(4.0, 0.0, 0.0);
  Eigen::VectorXd u_ref = Eigen::Vector2d(0.3, 0.0);

  auto result = solver_->solveComposite(
    state, x_des, u_ref, *dynamics_, CBFCompositionMethod::SMOOTH_MIN);

  EXPECT_TRUE(result.feasible);

  // cbf_margins는 합성 CBF의 단일 마진
  double tolerance = 0.2;  // projected gradient 수렴 허용 오차
  if (!result.cbf_margins.empty()) {
    EXPECT_GE(result.cbf_margins[0], -tolerance)
      << "합성 CBF margin >= -tolerance: 안전 제약 만족";
  }
}

// --- 15. MethodComparison ---
// SMOOTH_MIN과 MIN이 유사한 결과 (alpha가 크면)
TEST_F(CBFCompositionSolveTest, MethodComparison)
{
  std::vector<Eigen::Vector3d> obs;
  obs.push_back(Eigen::Vector3d(1.5, 0.0, 0.2));
  obs.push_back(Eigen::Vector3d(0.0, 2.0, 0.3));
  barrier_set_->setObstacles(obs);

  Eigen::VectorXd state = Eigen::Vector3d(0.0, 0.0, 0.0);
  Eigen::VectorXd x_des = Eigen::Vector3d(2.0, 1.0, 0.0);
  Eigen::VectorXd u_ref = Eigen::Vector2d(0.3, 0.1);

  auto result_min = solver_->solveComposite(
    state, x_des, u_ref, *dynamics_, CBFCompositionMethod::MIN);
  auto result_smooth = solver_->solveComposite(
    state, x_des, u_ref, *dynamics_, CBFCompositionMethod::SMOOTH_MIN, 50.0);

  EXPECT_TRUE(result_min.feasible);
  EXPECT_TRUE(result_smooth.feasible);

  // alpha가 크면 SMOOTH_MIN ≈ MIN이므로 제어 출력이 유사
  double u_diff = (result_min.u_safe - result_smooth.u_safe).norm();
  EXPECT_LT(u_diff, 0.3)
    << "SMOOTH_MIN(alpha=50) vs MIN: 유사한 제어 출력";
}
