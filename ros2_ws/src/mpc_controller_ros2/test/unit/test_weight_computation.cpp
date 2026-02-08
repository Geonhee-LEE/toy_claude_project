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

  EXPECT_EQ(vanilla.name(), "VanillaMPPI");
  EXPECT_EQ(log_mppi.name(), "LogMPPI");
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
