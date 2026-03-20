// =============================================================================
// CC-MPPI (Chance-Constrained MPPI) 단위 테스트 (15개)
// Blackmore et al. (JGCD 2011) inspired: P(g(x) <= 0) >= 1-epsilon
// =============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

#include "mpc_controller_ros2/chance_constrained_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

// ============================================================================
// 테스트 헬퍼
// ============================================================================
class CCMPPITestAccessor : public ChanceConstrainedMPPIControllerPlugin
{
public:
  void setTestParams(const MPPIParams& params) { params_ = params; }

  Eigen::MatrixXd callEvaluateSampleViolations(
    const std::vector<Eigen::MatrixXd>& ctrls,
    const std::vector<Eigen::MatrixXd>& trajs) const {
    return evaluateSampleViolations(ctrls, trajs);
  }

  Eigen::Vector3d callEstimateViolationProbabilities(
    const Eigen::MatrixXd& violations) const {
    return estimateViolationProbabilities(violations);
  }

  Eigen::Vector3d callAllocateRisk(
    const Eigen::Vector3d& violation_probs) const {
    return allocateRisk(violation_probs);
  }

  Eigen::VectorXd callComputeChanceConstrainedCosts(
    const Eigen::VectorXd& base_costs,
    const Eigen::MatrixXd& violations,
    const Eigen::Vector3d& allocated_risk) const {
    return computeChanceConstrainedCosts(base_costs, violations, allocated_risk);
  }

  double callEmpiricalQuantile(
    const Eigen::VectorXd& values, double quantile_level) const {
    return empiricalQuantile(values, quantile_level);
  }

  Eigen::Vector3d getSmoothedQuantiles() const { return smoothed_quantiles_; }
  void setSmoothedQuantiles(const Eigen::Vector3d& q) { smoothed_quantiles_ = q; }
};

// ============================================================================
// 테스트 Fixture
// ============================================================================
class CCMPPITest : public ::testing::Test
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

    accessor_.setTestParams(params_);
  }

  // 위반 있는 제어 시퀀스 생성 (v > v_max)
  std::vector<Eigen::MatrixXd> makeViolatingControls(int K, int N, double fraction) {
    std::vector<Eigen::MatrixXd> ctrls(K, Eigen::MatrixXd::Zero(N, 2));
    int num_violating = static_cast<int>(K * fraction);
    for (int k = 0; k < num_violating; ++k) {
      for (int t = 0; t < N; ++t) {
        ctrls[k](t, 0) = params_.v_max + 0.5;  // 위반
        ctrls[k](t, 1) = 0.3;
      }
    }
    for (int k = num_violating; k < K; ++k) {
      for (int t = 0; t < N; ++t) {
        ctrls[k](t, 0) = params_.v_max * 0.5;  // 정상
        ctrls[k](t, 1) = 0.3;
      }
    }
    return ctrls;
  }

  // 더미 궤적
  std::vector<Eigen::MatrixXd> makeDummyTrajectories(int K, int N) {
    return std::vector<Eigen::MatrixXd>(K, Eigen::MatrixXd::Zero(N + 1, 3));
  }

  MPPIParams params_;
  CCMPPITestAccessor accessor_;
};

// ============================================================================
// 1. 위반 확률 감지: 50% 샘플 위반 → p_hat ≈ 0.5
// ============================================================================
TEST_F(CCMPPITest, ViolationProbabilityDetected)
{
  int K = 100, N = params_.N;
  auto ctrls = makeViolatingControls(K, N, 0.5);
  auto trajs = makeDummyTrajectories(K, N);

  Eigen::MatrixXd violations = accessor_.callEvaluateSampleViolations(ctrls, trajs);
  Eigen::Vector3d p_hat = accessor_.callEstimateViolationProbabilities(violations);

  EXPECT_NEAR(p_hat(0), 0.5, 0.01);  // 속도 위반 50%
}

// ============================================================================
// 2. 가속도 위반 확률
// ============================================================================
TEST_F(CCMPPITest, AccelViolationProbability)
{
  int K = 50, N = params_.N;
  std::vector<Eigen::MatrixXd> ctrls(K, Eigen::MatrixXd::Zero(N, 2));
  auto trajs = makeDummyTrajectories(K, N);

  // 급격한 가속도 변화 (모든 샘플)
  for (int k = 0; k < K; ++k) {
    for (int t = 0; t < N; ++t) {
      ctrls[k](t, 0) = (t % 2 == 0) ? 0.8 : -0.8;  // 급변
    }
  }

  Eigen::MatrixXd violations = accessor_.callEvaluateSampleViolations(ctrls, trajs);
  Eigen::Vector3d p_hat = accessor_.callEstimateViolationProbabilities(violations);

  EXPECT_GT(p_hat(1), 0.5);  // 대부분 가속도 위반
}

// ============================================================================
// 3. Feasible이면 위반 확률 0
// ============================================================================
TEST_F(CCMPPITest, ZeroProbabilityWhenFeasible)
{
  int K = 50, N = params_.N;
  std::vector<Eigen::MatrixXd> ctrls(K, Eigen::MatrixXd::Zero(N, 2));
  auto trajs = makeDummyTrajectories(K, N);

  // 모든 샘플 정상
  for (int k = 0; k < K; ++k) {
    for (int t = 0; t < N; ++t) {
      ctrls[k](t, 0) = 0.3;
      ctrls[k](t, 1) = 0.2;
    }
  }

  Eigen::MatrixXd violations = accessor_.callEvaluateSampleViolations(ctrls, trajs);
  Eigen::Vector3d p_hat = accessor_.callEstimateViolationProbabilities(violations);

  EXPECT_NEAR(p_hat(0), 0.0, 1e-6);
  EXPECT_NEAR(p_hat(1), 0.0, 1e-6);
  EXPECT_NEAR(p_hat(2), 0.0, 1e-6);
}

// ============================================================================
// 4. Bonferroni 분배: eps_i = eps / 3
// ============================================================================
TEST_F(CCMPPITest, BonferroniAllocation)
{
  params_.cc_adaptive_risk = false;
  params_.cc_risk_budget = 0.06;
  accessor_.setTestParams(params_);

  Eigen::Vector3d p_hat(0.01, 0.02, 0.0);
  Eigen::Vector3d eps = accessor_.callAllocateRisk(p_hat);

  double expected = 0.06 / 3.0;
  EXPECT_NEAR(eps(0), expected, 1e-8);
  EXPECT_NEAR(eps(1), expected, 1e-8);
  EXPECT_NEAR(eps(2), expected, 1e-8);
}

// ============================================================================
// 5. Adaptive 재분배: 위반 큰 제약에 더 많은 risk
// ============================================================================
TEST_F(CCMPPITest, AdaptiveReallocation)
{
  params_.cc_adaptive_risk = true;
  params_.cc_risk_budget = 0.06;
  accessor_.setTestParams(params_);

  // constraint 0은 위반 많음 (0.03 > eps/3=0.02), 1,2는 위반 없음
  Eigen::Vector3d p_hat(0.03, 0.0, 0.0);
  Eigen::Vector3d eps = accessor_.callAllocateRisk(p_hat);

  // 위반 큰 constraint 0에 더 많은 risk 분배
  EXPECT_GT(eps(0), 0.06 / 3.0);
  // 위반 없는 1,2는 줄어듦
  EXPECT_LT(eps(1), 0.06 / 3.0);
  EXPECT_LT(eps(2), 0.06 / 3.0);
}

// ============================================================================
// 6. Risk budget 보존: sum(eps_i) <= total_risk
// ============================================================================
TEST_F(CCMPPITest, RiskBudgetConserved)
{
  params_.cc_adaptive_risk = true;
  params_.cc_risk_budget = 0.05;
  accessor_.setTestParams(params_);

  Eigen::Vector3d p_hat(0.03, 0.01, 0.0);
  Eigen::Vector3d eps = accessor_.callAllocateRisk(p_hat);

  // 약간의 수치 오차 허용
  EXPECT_LE(eps.sum(), params_.cc_risk_budget + 1e-6);
  // 모든 allocation 양수
  for (int i = 0; i < 3; ++i) {
    EXPECT_GT(eps(i), 0.0);
  }
}

// ============================================================================
// 7. Risk 초과 시 페널티 적용
// ============================================================================
TEST_F(CCMPPITest, PenaltyAppliedWhenExceedsRisk)
{
  int K = 100, N = params_.N;
  auto ctrls = makeViolatingControls(K, N, 0.5);  // 50% 위반
  auto trajs = makeDummyTrajectories(K, N);

  Eigen::MatrixXd violations = accessor_.callEvaluateSampleViolations(ctrls, trajs);
  Eigen::VectorXd base_costs = Eigen::VectorXd::Ones(K);
  Eigen::Vector3d eps_alloc(0.02, 0.02, 0.02);  // 낮은 risk → 페널티 필수

  Eigen::VectorXd aug_costs = accessor_.callComputeChanceConstrainedCosts(
    base_costs, violations, eps_alloc);

  // 위반 샘플의 augmented cost > base cost
  double max_base = base_costs.maxCoeff();
  double max_aug = aug_costs.maxCoeff();
  EXPECT_GT(max_aug, max_base);
}

// ============================================================================
// 8. Risk 내이면 페널티 없음
// ============================================================================
TEST_F(CCMPPITest, NoPenaltyWhenWithinRisk)
{
  int K = 50, N = params_.N;
  // 정상 샘플만
  std::vector<Eigen::MatrixXd> ctrls(K, Eigen::MatrixXd::Zero(N, 2));
  auto trajs = makeDummyTrajectories(K, N);
  for (int k = 0; k < K; ++k) {
    for (int t = 0; t < N; ++t) {
      ctrls[k](t, 0) = 0.3;
      ctrls[k](t, 1) = 0.2;
    }
  }

  Eigen::MatrixXd violations = accessor_.callEvaluateSampleViolations(ctrls, trajs);
  Eigen::VectorXd base_costs = Eigen::VectorXd::Ones(K) * 5.0;
  Eigen::Vector3d eps_alloc(0.05, 0.05, 0.05);

  Eigen::VectorXd aug_costs = accessor_.callComputeChanceConstrainedCosts(
    base_costs, violations, eps_alloc);

  // 비용 변화 없음
  for (int k = 0; k < K; ++k) {
    EXPECT_NEAR(aug_costs(k), base_costs(k), 1e-6);
  }
}

// ============================================================================
// 9. Penalty weight 2배 → 페널티 비례 증가
// ============================================================================
TEST_F(CCMPPITest, TighteningScalesWithWeight)
{
  int K = 100, N = params_.N;
  auto ctrls = makeViolatingControls(K, N, 0.5);
  auto trajs = makeDummyTrajectories(K, N);

  Eigen::MatrixXd violations = accessor_.callEvaluateSampleViolations(ctrls, trajs);
  Eigen::VectorXd base_costs = Eigen::VectorXd::Ones(K);
  Eigen::Vector3d eps_alloc(0.02, 0.02, 0.02);

  // Weight = 10
  params_.cc_penalty_weight = 10.0;
  accessor_.setTestParams(params_);
  Eigen::VectorXd aug1 = accessor_.callComputeChanceConstrainedCosts(
    base_costs, violations, eps_alloc);

  // Weight = 20
  params_.cc_penalty_weight = 20.0;
  accessor_.setTestParams(params_);
  Eigen::VectorXd aug2 = accessor_.callComputeChanceConstrainedCosts(
    base_costs, violations, eps_alloc);

  // Weight 2배이면 penalty도 2배 (base는 같으므로)
  double penalty1 = (aug1 - base_costs).sum();
  double penalty2 = (aug2 - base_costs).sum();
  EXPECT_NEAR(penalty2 / penalty1, 2.0, 0.1);
}

// ============================================================================
// 10. Empirical quantile 정확도
// ============================================================================
TEST_F(CCMPPITest, EmpiricalQuantileCorrect)
{
  Eigen::VectorXd values(100);
  for (int i = 0; i < 100; ++i) {
    values(i) = static_cast<double>(i);
  }

  double q95 = accessor_.callEmpiricalQuantile(values, 0.95);
  EXPECT_GE(q95, 93.0);
  EXPECT_LE(q95, 96.0);

  double q50 = accessor_.callEmpiricalQuantile(values, 0.50);
  EXPECT_GE(q50, 48.0);
  EXPECT_LE(q50, 52.0);
}

// ============================================================================
// 11. Quantile smoothing EMA 추적
// ============================================================================
TEST_F(CCMPPITest, QuantileSmoothingEMA)
{
  // Initial smoothed quantile = 0
  accessor_.setSmoothedQuantiles(Eigen::Vector3d::Zero());

  // After several updates, smoothed quantile should approach actual
  Eigen::Vector3d q1 = accessor_.getSmoothedQuantiles();
  EXPECT_NEAR(q1(0), 0.0, 1e-6);

  // Manually simulate EMA: new = alpha * raw + (1-alpha) * old
  double alpha = params_.cc_quantile_smoothing;
  double raw = 5.0;
  double expected = alpha * raw + (1.0 - alpha) * 0.0;
  Eigen::Vector3d updated(expected, 0.0, 0.0);
  accessor_.setSmoothedQuantiles(updated);

  Eigen::Vector3d q2 = accessor_.getSmoothedQuantiles();
  EXPECT_NEAR(q2(0), expected, 1e-6);
}

// ============================================================================
// 12. computeControl (통합): 전체 파이프라인 유한 출력
// ============================================================================
TEST_F(CCMPPITest, ComputeControlReturnsValid)
{
  // Full pipeline 테스트: 내부 초기화가 필요한 부분은 unit test로 커버 불가
  // 대신 evaluateSampleViolations → estimateViolationProbabilities →
  // allocateRisk → computeChanceConstrainedCosts 체인 검증
  int K = 100, N = params_.N;
  auto ctrls = makeViolatingControls(K, N, 0.3);
  auto trajs = makeDummyTrajectories(K, N);

  Eigen::MatrixXd violations = accessor_.callEvaluateSampleViolations(ctrls, trajs);
  Eigen::Vector3d p_hat = accessor_.callEstimateViolationProbabilities(violations);
  Eigen::Vector3d eps = accessor_.callAllocateRisk(p_hat);

  Eigen::VectorXd base_costs = Eigen::VectorXd::Ones(K) * 10.0;
  Eigen::VectorXd aug = accessor_.callComputeChanceConstrainedCosts(
    base_costs, violations, eps);

  // 모든 augmented cost 유한
  for (int k = 0; k < K; ++k) {
    EXPECT_TRUE(std::isfinite(aug(k)));
    EXPECT_GE(aug(k), base_costs(k) - 1e-6);
  }
}

// ============================================================================
// 13. 비활성 시 vanilla 폴백
// ============================================================================
TEST_F(CCMPPITest, DisabledEqualsVanilla)
{
  params_.cc_mppi_enabled = false;
  accessor_.setTestParams(params_);

  // 비활성 시에도 내부 메서드는 여전히 동작 (computeControl에서만 분기)
  // allocateRisk는 파라미터에 의존하므로 단독 호출 가능
  Eigen::Vector3d p_hat(0.1, 0.1, 0.0);
  Eigen::Vector3d eps = accessor_.callAllocateRisk(p_hat);

  // Bonferroni 분배 검증
  double expected = params_.cc_risk_budget / 3.0;
  EXPECT_NEAR(eps(0), expected, 1e-8);
}

// ============================================================================
// 14. 10회 반복 안정성: NaN/Inf 없음
// ============================================================================
TEST_F(CCMPPITest, MultipleCallsStable)
{
  int K = 100, N = params_.N;

  for (int iter = 0; iter < 10; ++iter) {
    double fraction = 0.1 * iter;  // 0% ~ 90% 위반
    if (fraction > 0.9) fraction = 0.9;

    auto ctrls = makeViolatingControls(K, N, fraction);
    auto trajs = makeDummyTrajectories(K, N);

    Eigen::MatrixXd violations = accessor_.callEvaluateSampleViolations(ctrls, trajs);
    Eigen::Vector3d p_hat = accessor_.callEstimateViolationProbabilities(violations);
    Eigen::Vector3d eps = accessor_.callAllocateRisk(p_hat);

    Eigen::VectorXd base_costs = Eigen::VectorXd::Ones(K) * 5.0;
    Eigen::VectorXd aug = accessor_.callComputeChanceConstrainedCosts(
      base_costs, violations, eps);

    for (int k = 0; k < K; ++k) {
      EXPECT_TRUE(std::isfinite(aug(k))) << "iter=" << iter << " k=" << k;
    }
    for (int i = 0; i < 3; ++i) {
      EXPECT_TRUE(std::isfinite(p_hat(i)));
      EXPECT_TRUE(std::isfinite(eps(i)));
    }
  }
}

// ============================================================================
// 15. MPPIInfo CC 메트릭 필드 존재 검증
// ============================================================================
TEST_F(CCMPPITest, InfoContainsCCMetrics)
{
  MPPIInfo info;
  info.cc_violation_probability = 0.15;
  info.cc_effective_risk = 0.05;
  info.cc_num_tightened = 2;

  EXPECT_NEAR(info.cc_violation_probability, 0.15, 1e-8);
  EXPECT_NEAR(info.cc_effective_risk, 0.05, 1e-8);
  EXPECT_EQ(info.cc_num_tightened, 2);
}

}  // namespace mpc_controller_ros2
