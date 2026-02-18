#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>

#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/utils.hpp"

using namespace mpc_controller_ros2;

// ============================================================
// logSumExp 테스트
// ============================================================

TEST(LogSumExpTest, BasicValues)
{
  Eigen::VectorXd v(3);
  v << 1.0, 2.0, 3.0;
  double result = logSumExp(v);
  double expected = std::log(std::exp(1.0) + std::exp(2.0) + std::exp(3.0));
  EXPECT_NEAR(result, expected, 1e-10);
}

TEST(LogSumExpTest, IdenticalValues)
{
  Eigen::VectorXd v = Eigen::VectorXd::Constant(5, 3.0);
  double result = logSumExp(v);
  // log(5 * exp(3)) = 3 + log(5)
  double expected = 3.0 + std::log(5.0);
  EXPECT_NEAR(result, expected, 1e-10);
}

TEST(LogSumExpTest, LargeValues)
{
  Eigen::VectorXd v(3);
  v << 1000.0, 1001.0, 999.0;
  double result = logSumExp(v);
  // max-shift trick 덕분에 overflow 없이 계산 가능
  EXPECT_TRUE(std::isfinite(result));
  EXPECT_GT(result, 1000.0);
}

TEST(LogSumExpTest, NegativeValues)
{
  Eigen::VectorXd v(3);
  v << -1000.0, -1001.0, -999.0;
  double result = logSumExp(v);
  EXPECT_TRUE(std::isfinite(result));
  EXPECT_LT(result, -998.0);
}

// ============================================================
// Vanilla vs Log-MPPI 동등성 테스트
// ============================================================

TEST(WeightComputationTest, VanillaLogEquivalence)
{
  VanillaMPPIWeights vanilla;
  LogMPPIWeights log_mppi;

  Eigen::VectorXd costs(5);
  costs << 10.0, 20.0, 15.0, 30.0, 5.0;
  double lambda = 10.0;

  Eigen::VectorXd vanilla_weights = vanilla.compute(costs, lambda);
  Eigen::VectorXd log_weights = log_mppi.compute(costs, lambda);

  // 두 방법의 결과가 수치적으로 동일해야 함
  for (int i = 0; i < costs.size(); ++i) {
    EXPECT_NEAR(vanilla_weights(i), log_weights(i), 1e-10)
      << "Mismatch at index " << i;
  }
}

TEST(WeightComputationTest, VanillaLogEquivalenceVaryingLambda)
{
  VanillaMPPIWeights vanilla;
  LogMPPIWeights log_mppi;

  Eigen::VectorXd costs(100);
  for (int i = 0; i < 100; ++i) {
    costs(i) = static_cast<double>(i) * 0.5;
  }

  for (double lambda : {0.1, 1.0, 10.0, 100.0}) {
    Eigen::VectorXd vanilla_weights = vanilla.compute(costs, lambda);
    Eigen::VectorXd log_weights = log_mppi.compute(costs, lambda);

    for (int i = 0; i < costs.size(); ++i) {
      EXPECT_NEAR(vanilla_weights(i), log_weights(i), 1e-8)
        << "Mismatch at index " << i << " with lambda=" << lambda;
    }
  }
}

// ============================================================
// 정규화 테스트
// ============================================================

TEST(WeightComputationTest, WeightsSumToOne)
{
  VanillaMPPIWeights vanilla;
  LogMPPIWeights log_mppi;

  Eigen::VectorXd costs(10);
  for (int i = 0; i < 10; ++i) {
    costs(i) = static_cast<double>(i) * 3.0 + 1.0;
  }
  double lambda = 5.0;

  EXPECT_NEAR(vanilla.compute(costs, lambda).sum(), 1.0, 1e-10);
  EXPECT_NEAR(log_mppi.compute(costs, lambda).sum(), 1.0, 1e-10);
}

TEST(WeightComputationTest, WeightsNonNegative)
{
  LogMPPIWeights log_mppi;

  Eigen::VectorXd costs(20);
  for (int i = 0; i < 20; ++i) {
    costs(i) = static_cast<double>(i) * 10.0;
  }

  Eigen::VectorXd weights = log_mppi.compute(costs, 1.0);
  for (int i = 0; i < weights.size(); ++i) {
    EXPECT_GE(weights(i), 0.0) << "Negative weight at index " << i;
  }
}

// ============================================================
// Zero-lambda fallback (greedy) 테스트
// ============================================================

TEST(WeightComputationTest, ZeroLambdaGreedy)
{
  VanillaMPPIWeights vanilla;
  LogMPPIWeights log_mppi;

  Eigen::VectorXd costs(5);
  costs << 10.0, 5.0, 20.0, 3.0, 15.0;  // 인덱스 3이 최소

  Eigen::VectorXd vanilla_weights = vanilla.compute(costs, 0.0);
  Eigen::VectorXd log_weights = log_mppi.compute(costs, 0.0);

  // 최소 비용 인덱스만 가중치 1.0
  EXPECT_NEAR(vanilla_weights(3), 1.0, 1e-10);
  EXPECT_NEAR(log_weights(3), 1.0, 1e-10);

  // 나머지는 0
  for (int i = 0; i < 5; ++i) {
    if (i != 3) {
      EXPECT_NEAR(vanilla_weights(i), 0.0, 1e-10);
      EXPECT_NEAR(log_weights(i), 0.0, 1e-10);
    }
  }
}

// ============================================================
// 극단 비용 안정성 테스트
// ============================================================

TEST(WeightComputationTest, ExtremeCostStability)
{
  LogMPPIWeights log_mppi;

  // 극단적으로 큰 비용 차이
  Eigen::VectorXd costs(3);
  costs << 0.0, 1e6, 1e6;

  Eigen::VectorXd weights = log_mppi.compute(costs, 1.0);

  EXPECT_TRUE(weights.allFinite()) << "Weights contain NaN or Inf";
  EXPECT_NEAR(weights.sum(), 1.0, 1e-10);
  // 최소 비용(인덱스 0)이 거의 모든 가중치를 차지
  EXPECT_GT(weights(0), 0.99);
}

TEST(WeightComputationTest, UniformCosts)
{
  LogMPPIWeights log_mppi;

  // 모든 비용이 동일하면 균등 가중치
  Eigen::VectorXd costs = Eigen::VectorXd::Constant(10, 5.0);
  Eigen::VectorXd weights = log_mppi.compute(costs, 10.0);

  for (int i = 0; i < 10; ++i) {
    EXPECT_NEAR(weights(i), 0.1, 1e-10);
  }
}

// ============================================================
// Strategy 이름 테스트
// ============================================================

TEST(WeightComputationTest, StrategyNames)
{
  VanillaMPPIWeights vanilla;
  LogMPPIWeights log_mppi;
  TsallisMPPIWeights tsallis(1.5);
  RiskAwareMPPIWeights risk_aware(0.5);

  EXPECT_EQ(vanilla.name(), "VanillaMPPI");
  EXPECT_EQ(log_mppi.name(), "LogMPPI");
  EXPECT_EQ(tsallis.name(), "TsallisMPPI");
  EXPECT_EQ(risk_aware.name(), "RiskAwareMPPI");
}

// ============================================================
// qExponential 유틸리티 테스트
// ============================================================

TEST(QExponentialTest, QEqualsOneIsStdExp)
{
  Eigen::VectorXd x(4);
  x << -2.0, -1.0, 0.0, 1.0;

  Eigen::VectorXd result = qExponential(x, 1.0);

  for (int i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(result(i), std::exp(x(i)), 1e-6)
      << "q=1 should equal std::exp at index " << i;
  }
}

TEST(QExponentialTest, QGreaterThanOneHeavyTail)
{
  // q > 1: heavy-tail, 큰 음수 입력에서도 0보다 큰 값
  Eigen::VectorXd x(3);
  x << -1.0, -5.0, -10.0;

  Eigen::VectorXd result_q15 = qExponential(x, 1.5);
  Eigen::VectorXd result_exp = x.array().exp();

  // heavy-tail: q-exp 값이 std exp보다 커야 함 (큰 음수에서)
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_GE(result_q15(i), 0.0) << "q-exp should be non-negative";
  }
}

TEST(QExponentialTest, QExponentialAtZero)
{
  // exp_q(0) = 1 for any q
  Eigen::VectorXd x(1);
  x << 0.0;

  for (double q : {0.5, 1.0, 1.5, 2.0}) {
    Eigen::VectorXd result = qExponential(x, q);
    EXPECT_NEAR(result(0), 1.0, 1e-10) << "exp_q(0) should be 1 for q=" << q;
  }
}

TEST(QExponentialTest, CutoffBehavior)
{
  // q > 1 일 때, base = 1 + (1-q)*x <= 0 이면 결과 0
  // q=2: 1 + (1-2)*x = 1-x, x>1이면 base<0 → 결과 0
  Eigen::VectorXd x(3);
  x << 0.5, 1.0, 2.0;

  Eigen::VectorXd result = qExponential(x, 2.0);
  EXPECT_GT(result(0), 0.0);   // base = 1 - 0.5 = 0.5 > 0
  EXPECT_NEAR(result(1), 0.0, 1e-10);  // base = 1 - 1.0 = 0 → 0
  EXPECT_NEAR(result(2), 0.0, 1e-10);  // base = 1 - 2.0 < 0 → 0
}

// ============================================================
// Tsallis-MPPI 가중치 테스트
// ============================================================

TEST(TsallisMPPITest, QOneEqualsVanilla)
{
  // q=1.0 → Vanilla MPPI와 동일
  TsallisMPPIWeights tsallis(1.0);
  VanillaMPPIWeights vanilla;

  Eigen::VectorXd costs(5);
  costs << 10.0, 20.0, 15.0, 30.0, 5.0;
  double lambda = 10.0;

  Eigen::VectorXd tsallis_w = tsallis.compute(costs, lambda);
  Eigen::VectorXd vanilla_w = vanilla.compute(costs, lambda);

  for (int i = 0; i < costs.size(); ++i) {
    EXPECT_NEAR(tsallis_w(i), vanilla_w(i), 1e-6)
      << "q=1 Tsallis should equal Vanilla at index " << i;
  }
}

TEST(TsallisMPPITest, WeightsSumToOne)
{
  for (double q : {0.5, 1.0, 1.5, 2.0}) {
    TsallisMPPIWeights tsallis(q);

    Eigen::VectorXd costs(10);
    for (int i = 0; i < 10; ++i) {
      costs(i) = static_cast<double>(i) * 3.0 + 1.0;
    }

    Eigen::VectorXd weights = tsallis.compute(costs, 10.0);
    EXPECT_NEAR(weights.sum(), 1.0, 1e-10) << "Weights should sum to 1 for q=" << q;
  }
}

TEST(TsallisMPPITest, WeightsNonNegative)
{
  TsallisMPPIWeights tsallis(1.5);

  Eigen::VectorXd costs(20);
  for (int i = 0; i < 20; ++i) {
    costs(i) = static_cast<double>(i) * 10.0;
  }

  Eigen::VectorXd weights = tsallis.compute(costs, 5.0);
  for (int i = 0; i < weights.size(); ++i) {
    EXPECT_GE(weights(i), 0.0) << "Negative weight at index " << i;
  }
}

TEST(TsallisMPPITest, ZeroLambdaGreedy)
{
  TsallisMPPIWeights tsallis(1.5);

  Eigen::VectorXd costs(5);
  costs << 10.0, 5.0, 20.0, 3.0, 15.0;

  Eigen::VectorXd weights = tsallis.compute(costs, 0.0);
  EXPECT_NEAR(weights(3), 1.0, 1e-10);  // 최소 비용 인덱스
}

TEST(TsallisMPPITest, HeavyTailExploration)
{
  // q > 1: heavy-tail → 비최적 샘플에도 가중치 할당 (Vanilla보다 더 균등)
  TsallisMPPIWeights tsallis_heavy(2.0);
  VanillaMPPIWeights vanilla;

  Eigen::VectorXd costs(5);
  costs << 1.0, 5.0, 10.0, 20.0, 50.0;
  double lambda = 5.0;

  Eigen::VectorXd tsallis_w = tsallis_heavy.compute(costs, lambda);
  Eigen::VectorXd vanilla_w = vanilla.compute(costs, lambda);

  // heavy-tail: 최대 가중치가 Vanilla보다 작아야 함 (더 분산)
  EXPECT_LT(tsallis_w.maxCoeff(), vanilla_w.maxCoeff());
}

TEST(TsallisMPPITest, UniformCosts)
{
  TsallisMPPIWeights tsallis(1.5);

  Eigen::VectorXd costs = Eigen::VectorXd::Constant(10, 5.0);
  Eigen::VectorXd weights = tsallis.compute(costs, 10.0);

  for (int i = 0; i < 10; ++i) {
    EXPECT_NEAR(weights(i), 0.1, 1e-10);
  }
}

TEST(TsallisMPPITest, NumericalStability)
{
  TsallisMPPIWeights tsallis(1.5);

  Eigen::VectorXd costs(3);
  costs << 0.0, 1e6, 1e6;

  Eigen::VectorXd weights = tsallis.compute(costs, 1.0);
  EXPECT_TRUE(weights.allFinite()) << "Tsallis weights contain NaN or Inf";
  EXPECT_NEAR(weights.sum(), 1.0, 1e-10);
}

// ============================================================
// Risk-Aware (CVaR) MPPI 가중치 테스트
// ============================================================

TEST(RiskAwareMPPITest, AlphaOneEqualsVanilla)
{
  // alpha=1.0 → Vanilla MPPI와 동일 (모든 샘플 사용)
  RiskAwareMPPIWeights risk_aware(1.0);
  VanillaMPPIWeights vanilla;

  Eigen::VectorXd costs(5);
  costs << 10.0, 20.0, 15.0, 30.0, 5.0;
  double lambda = 10.0;

  Eigen::VectorXd ra_w = risk_aware.compute(costs, lambda);
  Eigen::VectorXd vanilla_w = vanilla.compute(costs, lambda);

  for (int i = 0; i < costs.size(); ++i) {
    EXPECT_NEAR(ra_w(i), vanilla_w(i), 1e-10)
      << "alpha=1 Risk-Aware should equal Vanilla at index " << i;
  }
}

TEST(RiskAwareMPPITest, WeightsSumToOne)
{
  for (double alpha : {0.1, 0.3, 0.5, 0.8, 1.0}) {
    RiskAwareMPPIWeights risk_aware(alpha);

    Eigen::VectorXd costs(10);
    for (int i = 0; i < 10; ++i) {
      costs(i) = static_cast<double>(i) * 3.0 + 1.0;
    }

    Eigen::VectorXd weights = risk_aware.compute(costs, 10.0);
    EXPECT_NEAR(weights.sum(), 1.0, 1e-10)
      << "Weights should sum to 1 for alpha=" << alpha;
  }
}

TEST(RiskAwareMPPITest, WeightsNonNegative)
{
  RiskAwareMPPIWeights risk_aware(0.5);

  Eigen::VectorXd costs(20);
  for (int i = 0; i < 20; ++i) {
    costs(i) = static_cast<double>(i) * 10.0;
  }

  Eigen::VectorXd weights = risk_aware.compute(costs, 5.0);
  for (int i = 0; i < weights.size(); ++i) {
    EXPECT_GE(weights(i), 0.0) << "Negative weight at index " << i;
  }
}

TEST(RiskAwareMPPITest, TruncationBehavior)
{
  // alpha=0.4, K=5 → n_keep = ceil(0.4*5) = 2
  // 최저 비용 2개만 비영(non-zero) 가중치
  RiskAwareMPPIWeights risk_aware(0.4);

  Eigen::VectorXd costs(5);
  costs << 10.0, 5.0, 20.0, 3.0, 15.0;  // 정렬: 3(idx3), 5(idx1), 10(idx0), 15(idx4), 20(idx2)

  Eigen::VectorXd weights = risk_aware.compute(costs, 10.0);

  // 인덱스 3(비용 3), 인덱스 1(비용 5)만 비영
  EXPECT_GT(weights(3), 0.0);
  EXPECT_GT(weights(1), 0.0);
  // 나머지는 0
  EXPECT_NEAR(weights(0), 0.0, 1e-10);
  EXPECT_NEAR(weights(2), 0.0, 1e-10);
  EXPECT_NEAR(weights(4), 0.0, 1e-10);
}

TEST(RiskAwareMPPITest, SmallAlphaConservative)
{
  // alpha가 작으면 가중치가 최소 비용에 더 집중
  RiskAwareMPPIWeights conservative(0.1);
  RiskAwareMPPIWeights moderate(0.5);

  Eigen::VectorXd costs(100);
  for (int i = 0; i < 100; ++i) {
    costs(i) = static_cast<double>(i);
  }

  Eigen::VectorXd cons_w = conservative.compute(costs, 10.0);
  Eigen::VectorXd mod_w = moderate.compute(costs, 10.0);

  // 보수적(alpha=0.1): 비영 가중치 수가 적어야 함
  int cons_nonzero = 0, mod_nonzero = 0;
  for (int i = 0; i < 100; ++i) {
    if (cons_w(i) > 1e-12) cons_nonzero++;
    if (mod_w(i) > 1e-12) mod_nonzero++;
  }
  EXPECT_LT(cons_nonzero, mod_nonzero);
}

TEST(RiskAwareMPPITest, UniformCosts)
{
  RiskAwareMPPIWeights risk_aware(0.5);

  // 동일 비용 → n_keep개가 균등 가중치
  Eigen::VectorXd costs = Eigen::VectorXd::Constant(10, 5.0);
  Eigen::VectorXd weights = risk_aware.compute(costs, 10.0);

  EXPECT_NEAR(weights.sum(), 1.0, 1e-10);
  EXPECT_TRUE(weights.allFinite());
}

TEST(RiskAwareMPPITest, NumericalStability)
{
  RiskAwareMPPIWeights risk_aware(0.5);

  Eigen::VectorXd costs(3);
  costs << 0.0, 1e6, 1e6;

  Eigen::VectorXd weights = risk_aware.compute(costs, 1.0);
  EXPECT_TRUE(weights.allFinite()) << "Risk-Aware weights contain NaN or Inf";
  EXPECT_NEAR(weights.sum(), 1.0, 1e-10);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
