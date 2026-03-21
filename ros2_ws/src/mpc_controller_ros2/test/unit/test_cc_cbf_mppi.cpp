// =============================================================================
// CC-CBF-MPPI (Chance-Constrained CBF-MPPI) 단위 테스트 (15개)
// Blackmore et al. (JGCD 2011) + Ames et al. (2019)
// =============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

#include "mpc_controller_ros2/cc_cbf_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/barrier_function.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

// ============================================================================
// 테스트 헬퍼
// ============================================================================
class CCCBFTestAccessor : public CCCBFMPPIControllerPlugin
{
public:
  void setTestParams(const MPPIParams& params) { params_ = params; }

  void setBarriers(const std::vector<Eigen::Vector3d>& obstacles) {
    barrier_set_.setObstacles(obstacles);
  }

  const BarrierFunctionSet& getBarrierSet() const { return barrier_set_; }

  Eigen::MatrixXd callEvaluateSampleViolations(
    const std::vector<Eigen::MatrixXd>& ctrls,
    const std::vector<Eigen::MatrixXd>& trajs) const {
    return evaluateSampleViolations(ctrls, trajs);
  }

  Eigen::Vector4d callEstimateViolationProbabilities(
    const Eigen::MatrixXd& violations) const {
    return estimateViolationProbabilities(violations);
  }

  Eigen::Vector4d callAllocateRisk(
    const Eigen::Vector4d& violation_probs) const {
    return allocateRisk(violation_probs);
  }

  Eigen::VectorXd callComputeChanceConstrainedCosts(
    const Eigen::VectorXd& base_costs,
    const Eigen::MatrixXd& violations,
    const Eigen::Vector4d& allocated_risk) const {
    return computeChanceConstrainedCosts(base_costs, violations, allocated_risk);
  }

  double callEmpiricalQuantile(
    const Eigen::VectorXd& values, double quantile_level) const {
    return empiricalQuantile(values, quantile_level);
  }

  Eigen::Vector4d getSmoothedQuantiles() const { return smoothed_quantiles_; }
  void setSmoothedQuantiles(const Eigen::Vector4d& q) { smoothed_quantiles_ = q; }
};

// ============================================================================
// 테스트 Fixture
// ============================================================================
class CCCBFMPPITest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    params_ = MPPIParams();
    params_.N = 10;
    params_.dt = 0.1;
    params_.K = 100;
    params_.lambda = 10.0;
    params_.v_max = 1.0;
    params_.v_min = 0.0;
    params_.omega_max = 1.0;
    params_.omega_min = -1.0;
    params_.noise_sigma = Eigen::Vector2d(0.5, 0.5);
    params_.constrained_accel_max_v = 2.0;
    params_.constrained_accel_max_omega = 3.0;

    params_.cc_mppi_enabled = true;
    params_.cc_risk_budget = 0.05;
    params_.cc_penalty_weight = 10.0;
    params_.cc_adaptive_risk = false;
    params_.cc_tightening_rate = 1.5;
    params_.cc_quantile_smoothing = 0.1;
    params_.cc_cbf_projection_enabled = true;

    params_.cbf_enabled = true;
    params_.cbf_gamma = 1.0;

    accessor_.setTestParams(params_);
  }

  // 위반 있는 제어 시퀀스 (v > v_max)
  std::vector<Eigen::MatrixXd> makeViolatingControls(int K, int N, double fraction) {
    std::vector<Eigen::MatrixXd> ctrls(K, Eigen::MatrixXd::Zero(N, 2));
    int num_violating = static_cast<int>(K * fraction);
    for (int k = 0; k < num_violating; ++k) {
      for (int t = 0; t < N; ++t) {
        ctrls[k](t, 0) = params_.v_max + 0.5;
        ctrls[k](t, 1) = 0.3;
      }
    }
    for (int k = num_violating; k < K; ++k) {
      for (int t = 0; t < N; ++t) {
        ctrls[k](t, 0) = params_.v_max * 0.5;
        ctrls[k](t, 1) = 0.3;
      }
    }
    return ctrls;
  }

  // 더미 궤적 (장애물 없는 안전 영역)
  std::vector<Eigen::MatrixXd> makeSafeTrajectories(int K, int N) {
    std::vector<Eigen::MatrixXd> trajs(K, Eigen::MatrixXd::Zero(N + 1, 3));
    for (int k = 0; k < K; ++k) {
      for (int t = 0; t <= N; ++t) {
        trajs[k](t, 0) = 0.1 * t;  // x: 전진
        trajs[k](t, 1) = 0.0;
        trajs[k](t, 2) = 0.0;
      }
    }
    return trajs;
  }

  // 장애물 근처 궤적 (clearance 위반)
  std::vector<Eigen::MatrixXd> makeCollisionTrajectories(
    int K, int N, double fraction, double obs_x, double obs_y) {
    std::vector<Eigen::MatrixXd> trajs(K, Eigen::MatrixXd::Zero(N + 1, 3));
    int num_colliding = static_cast<int>(K * fraction);
    for (int k = 0; k < num_colliding; ++k) {
      for (int t = 0; t <= N; ++t) {
        trajs[k](t, 0) = obs_x;  // 장애물 위치
        trajs[k](t, 1) = obs_y;
        trajs[k](t, 2) = 0.0;
      }
    }
    for (int k = num_colliding; k < K; ++k) {
      for (int t = 0; t <= N; ++t) {
        trajs[k](t, 0) = obs_x + 10.0;  // 장애물에서 멀리
        trajs[k](t, 1) = obs_y + 10.0;
        trajs[k](t, 2) = 0.0;
      }
    }
    return trajs;
  }

  MPPIParams params_;
  CCCBFTestAccessor accessor_;
};

// ============================================================================
// 1. 4종 제약 violations 행렬 크기 검증
// ============================================================================
TEST_F(CCCBFMPPITest, ViolationsMatrixSize)
{
  int K = 50, N = params_.N;
  auto ctrls = makeViolatingControls(K, N, 0.5);
  auto trajs = makeSafeTrajectories(K, N);

  Eigen::MatrixXd violations = accessor_.callEvaluateSampleViolations(ctrls, trajs);

  EXPECT_EQ(violations.rows(), K);
  EXPECT_EQ(violations.cols(), 4);  // vel, accel, clearance, cbf_rate
}

// ============================================================================
// 2. Barrier 없으면 clearance/cbf_rate = 0
// ============================================================================
TEST_F(CCCBFMPPITest, NoClearanceWithoutBarriers)
{
  int K = 50, N = params_.N;
  auto ctrls = makeViolatingControls(K, N, 0.5);
  auto trajs = makeSafeTrajectories(K, N);

  // barrier 미설정 → clearance = 0
  Eigen::MatrixXd violations = accessor_.callEvaluateSampleViolations(ctrls, trajs);

  for (int k = 0; k < K; ++k) {
    EXPECT_NEAR(violations(k, 2), 0.0, 1e-6);  // clearance
    EXPECT_NEAR(violations(k, 3), 0.0, 1e-6);  // cbf_rate
  }
  // velocity 위반은 있어야 함
  Eigen::Vector4d p_hat = accessor_.callEstimateViolationProbabilities(violations);
  EXPECT_NEAR(p_hat(0), 0.5, 0.01);
}

// ============================================================================
// 3. Barrier 있을 때 clearance 위반 감지
// ============================================================================
TEST_F(CCCBFMPPITest, ClearanceViolationWithBarriers)
{
  double obs_x = 2.0, obs_y = 0.0, obs_r = 1.0;
  accessor_.setBarriers({{obs_x, obs_y, obs_r}});

  int K = 100, N = params_.N;
  auto ctrls = makeViolatingControls(K, N, 0.0);  // 속도 정상
  auto trajs = makeCollisionTrajectories(K, N, 0.5, obs_x, obs_y);

  Eigen::MatrixXd violations = accessor_.callEvaluateSampleViolations(ctrls, trajs);

  // 50% 샘플이 장애물 위치 → clearance 위반
  Eigen::Vector4d p_hat = accessor_.callEstimateViolationProbabilities(violations);
  EXPECT_NEAR(p_hat(2), 0.5, 0.05);  // clearance 위반 확률

  // 위반 샘플의 clearance > 0
  int num_colliding = static_cast<int>(K * 0.5);
  for (int k = 0; k < num_colliding; ++k) {
    EXPECT_GT(violations(k, 2), 0.0) << "k=" << k;
  }
}

// ============================================================================
// 4. CBF rate 위반 감지 (장애물에 접근하는 궤적)
// ============================================================================
TEST_F(CCCBFMPPITest, CBFRateViolationDetected)
{
  double obs_x = 2.0, obs_y = 0.0, obs_r = 0.5;
  accessor_.setBarriers({{obs_x, obs_y, obs_r}});

  int K = 50, N = params_.N;
  auto ctrls = makeViolatingControls(K, N, 0.0);

  // 장애물로 빠르게 접근하는 궤적
  std::vector<Eigen::MatrixXd> trajs(K, Eigen::MatrixXd::Zero(N + 1, 3));
  for (int k = 0; k < K; ++k) {
    for (int t = 0; t <= N; ++t) {
      double frac = static_cast<double>(t) / N;
      trajs[k](t, 0) = obs_x * frac;  // 점점 장애물에 가까워짐
      trajs[k](t, 1) = 0.0;
      trajs[k](t, 2) = 0.0;
    }
  }

  Eigen::MatrixXd violations = accessor_.callEvaluateSampleViolations(ctrls, trajs);
  Eigen::Vector4d p_hat = accessor_.callEstimateViolationProbabilities(violations);

  // CBF rate 위반이 있어야 함 (접근 → dh/dt < 0)
  // h가 감소하면서 dh/dt + γh < 0이 될 수 있음
  // 최소한 clearance 또는 cbf_rate 중 하나는 감지되어야 함
  EXPECT_TRUE(p_hat(2) > 0.0 || p_hat(3) > 0.0);
}

// ============================================================================
// 5. Bonferroni 분배: eps_i = eps / 4
// ============================================================================
TEST_F(CCCBFMPPITest, BonferroniAllocation4Constraints)
{
  params_.cc_adaptive_risk = false;
  params_.cc_risk_budget = 0.08;
  accessor_.setTestParams(params_);

  Eigen::Vector4d p_hat(0.01, 0.02, 0.0, 0.0);
  Eigen::Vector4d eps = accessor_.callAllocateRisk(p_hat);

  double expected = 0.08 / 4.0;
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(eps(i), expected, 1e-8);
  }
}

// ============================================================================
// 6. Adaptive 분배: sum(eps) <= budget
// ============================================================================
TEST_F(CCCBFMPPITest, AdaptiveRiskConserved)
{
  params_.cc_adaptive_risk = true;
  params_.cc_risk_budget = 0.05;
  accessor_.setTestParams(params_);

  Eigen::Vector4d p_hat(0.03, 0.01, 0.02, 0.0);
  Eigen::Vector4d eps = accessor_.callAllocateRisk(p_hat);

  EXPECT_LE(eps.sum(), params_.cc_risk_budget + 1e-10);
  for (int i = 0; i < 4; ++i) {
    EXPECT_GT(eps(i), 0.0);
  }
}

// ============================================================================
// 7. Adaptive: 위반 큰 제약에 더 많은 risk
// ============================================================================
TEST_F(CCCBFMPPITest, AdaptiveGivesMoreToViolated)
{
  params_.cc_adaptive_risk = true;
  params_.cc_risk_budget = 0.08;
  accessor_.setTestParams(params_);

  // 제약 0만 위반, 나머지 정상
  Eigen::Vector4d p_hat(0.03, 0.0, 0.0, 0.0);
  Eigen::Vector4d eps = accessor_.callAllocateRisk(p_hat);

  EXPECT_GT(eps(0), 0.08 / 4.0);
  EXPECT_LT(eps(1), 0.08 / 4.0);
}

// ============================================================================
// 8. Risk 초과 시 페널티 적용
// ============================================================================
TEST_F(CCCBFMPPITest, PenaltyAppliedWhenExceedsRisk)
{
  accessor_.setBarriers({{2.0, 0.0, 1.0}});

  int K = 100, N = params_.N;
  auto ctrls = makeViolatingControls(K, N, 0.5);
  auto trajs = makeCollisionTrajectories(K, N, 0.5, 2.0, 0.0);

  Eigen::MatrixXd violations = accessor_.callEvaluateSampleViolations(ctrls, trajs);
  Eigen::VectorXd base_costs = Eigen::VectorXd::Ones(K);
  Eigen::Vector4d eps_alloc = Eigen::Vector4d::Constant(0.02);

  Eigen::VectorXd aug_costs = accessor_.callComputeChanceConstrainedCosts(
    base_costs, violations, eps_alloc);

  EXPECT_GT(aug_costs.maxCoeff(), base_costs.maxCoeff());
}

// ============================================================================
// 9. 정상이면 페널티 없음
// ============================================================================
TEST_F(CCCBFMPPITest, NoPenaltyWhenFeasible)
{
  int K = 50, N = params_.N;
  std::vector<Eigen::MatrixXd> ctrls(K, Eigen::MatrixXd::Zero(N, 2));
  auto trajs = makeSafeTrajectories(K, N);

  for (int k = 0; k < K; ++k) {
    for (int t = 0; t < N; ++t) {
      ctrls[k](t, 0) = 0.3;
      ctrls[k](t, 1) = 0.2;
    }
  }

  Eigen::MatrixXd violations = accessor_.callEvaluateSampleViolations(ctrls, trajs);
  Eigen::VectorXd base_costs = Eigen::VectorXd::Ones(K) * 5.0;
  Eigen::Vector4d eps_alloc = Eigen::Vector4d::Constant(0.05);

  Eigen::VectorXd aug_costs = accessor_.callComputeChanceConstrainedCosts(
    base_costs, violations, eps_alloc);

  for (int k = 0; k < K; ++k) {
    EXPECT_NEAR(aug_costs(k), base_costs(k), 1e-6);
  }
}

// ============================================================================
// 10. Penalty weight 비례 (2x weight → 2x penalty)
// ============================================================================
TEST_F(CCCBFMPPITest, PenaltyScalesWithWeight)
{
  int K = 100, N = params_.N;
  auto ctrls = makeViolatingControls(K, N, 0.5);
  auto trajs = makeSafeTrajectories(K, N);

  Eigen::MatrixXd violations = accessor_.callEvaluateSampleViolations(ctrls, trajs);
  Eigen::VectorXd base_costs = Eigen::VectorXd::Ones(K);
  Eigen::Vector4d eps_alloc = Eigen::Vector4d::Constant(0.02);

  params_.cc_penalty_weight = 10.0;
  accessor_.setTestParams(params_);
  Eigen::VectorXd aug1 = accessor_.callComputeChanceConstrainedCosts(
    base_costs, violations, eps_alloc);

  params_.cc_penalty_weight = 20.0;
  accessor_.setTestParams(params_);
  Eigen::VectorXd aug2 = accessor_.callComputeChanceConstrainedCosts(
    base_costs, violations, eps_alloc);

  double penalty1 = (aug1 - base_costs).sum();
  double penalty2 = (aug2 - base_costs).sum();
  EXPECT_NEAR(penalty2 / penalty1, 2.0, 0.1);
}

// ============================================================================
// 11. Empirical quantile 정확도
// ============================================================================
TEST_F(CCCBFMPPITest, EmpiricalQuantileCorrect)
{
  Eigen::VectorXd values(100);
  for (int i = 0; i < 100; ++i) {
    values(i) = static_cast<double>(i);
  }

  double q95 = accessor_.callEmpiricalQuantile(values, 0.95);
  EXPECT_GE(q95, 93.0);
  EXPECT_LE(q95, 96.0);
}

// ============================================================================
// 12. 다수 장애물 clearance 평가
// ============================================================================
TEST_F(CCCBFMPPITest, MultipleBarriersClearance)
{
  // 3개 장애물 설정
  accessor_.setBarriers({
    {2.0, 0.0, 0.5},
    {0.0, 2.0, 0.5},
    {-2.0, 0.0, 0.5}
  });

  int K = 50, N = params_.N;
  auto ctrls = makeViolatingControls(K, N, 0.0);

  // 첫 번째 장애물 위치를 통과하는 궤적
  auto trajs = makeCollisionTrajectories(K, N, 1.0, 2.0, 0.0);

  Eigen::MatrixXd violations = accessor_.callEvaluateSampleViolations(ctrls, trajs);
  Eigen::Vector4d p_hat = accessor_.callEstimateViolationProbabilities(violations);

  // 모든 샘플이 첫 번째 장애물과 충돌
  EXPECT_NEAR(p_hat(2), 1.0, 0.01);
}

// ============================================================================
// 13. 안전 궤적에서 clearance 위반 없음
// ============================================================================
TEST_F(CCCBFMPPITest, SafeTrajectoriesNoClearanceViolation)
{
  accessor_.setBarriers({{10.0, 10.0, 0.5}});  // 멀리 있는 장애물

  int K = 50, N = params_.N;
  auto ctrls = makeViolatingControls(K, N, 0.0);
  auto trajs = makeSafeTrajectories(K, N);  // (0,0) 근처

  Eigen::MatrixXd violations = accessor_.callEvaluateSampleViolations(ctrls, trajs);
  Eigen::Vector4d p_hat = accessor_.callEstimateViolationProbabilities(violations);

  EXPECT_NEAR(p_hat(2), 0.0, 1e-6);  // clearance 위반 없음
  EXPECT_NEAR(p_hat(3), 0.0, 1e-6);  // CBF rate 위반 없음
}

// ============================================================================
// 14. 10회 반복 안정성: NaN/Inf 없음
// ============================================================================
TEST_F(CCCBFMPPITest, MultipleCallsStable)
{
  accessor_.setBarriers({{3.0, 0.0, 0.5}});

  int K = 100, N = params_.N;

  for (int iter = 0; iter < 10; ++iter) {
    double fraction = 0.1 * iter;
    if (fraction > 0.9) fraction = 0.9;

    auto ctrls = makeViolatingControls(K, N, fraction);
    auto trajs = makeCollisionTrajectories(K, N, fraction, 3.0, 0.0);

    Eigen::MatrixXd violations = accessor_.callEvaluateSampleViolations(ctrls, trajs);
    Eigen::Vector4d p_hat = accessor_.callEstimateViolationProbabilities(violations);
    Eigen::Vector4d eps = accessor_.callAllocateRisk(p_hat);

    Eigen::VectorXd base_costs = Eigen::VectorXd::Ones(K) * 5.0;
    Eigen::VectorXd aug = accessor_.callComputeChanceConstrainedCosts(
      base_costs, violations, eps);

    for (int k = 0; k < K; ++k) {
      EXPECT_TRUE(std::isfinite(aug(k))) << "iter=" << iter << " k=" << k;
    }
    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(std::isfinite(p_hat(i)));
      EXPECT_TRUE(std::isfinite(eps(i)));
    }
  }
}

// ============================================================================
// 15. MPPIInfo CC 메트릭 필드 존재
// ============================================================================
TEST_F(CCCBFMPPITest, InfoContainsCCMetrics)
{
  MPPIInfo info;
  info.cc_violation_probability = 0.15;
  info.cc_effective_risk = 0.05;
  info.cc_num_tightened = 3;

  EXPECT_NEAR(info.cc_violation_probability, 0.15, 1e-8);
  EXPECT_NEAR(info.cc_effective_risk, 0.05, 1e-8);
  EXPECT_EQ(info.cc_num_tightened, 3);
}

}  // namespace mpc_controller_ros2
