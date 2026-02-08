#include <gtest/gtest.h>
#include "mpc_controller_ros2/adaptive_temperature.hpp"
#include <cmath>

using namespace mpc_controller_ros2;

class AdaptiveTemperatureTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // 기본 파라미터로 초기화
    adaptive_temp_ = std::make_unique<AdaptiveTemperature>(
      10.0,   // initial_lambda
      0.5,    // target_ess_ratio
      0.1,    // adaptation_rate
      0.1,    // lambda_min
      100.0   // lambda_max
    );
  }

  std::unique_ptr<AdaptiveTemperature> adaptive_temp_;
};

TEST_F(AdaptiveTemperatureTest, InitialLambda)
{
  EXPECT_DOUBLE_EQ(adaptive_temp_->getLambda(), 10.0);
}

TEST_F(AdaptiveTemperatureTest, LowESSIncreasesLambda)
{
  // ESS가 낮으면 (weight collapse) λ 증가
  double initial_lambda = adaptive_temp_->getLambda();

  // ESS = 100, K = 1000 → ess_ratio = 0.1 < 0.5 (target)
  double new_lambda = adaptive_temp_->update(100.0, 1000);

  EXPECT_GT(new_lambda, initial_lambda);
  EXPECT_DOUBLE_EQ(new_lambda, adaptive_temp_->getLambda());
}

TEST_F(AdaptiveTemperatureTest, HighESSDecreasesLambda)
{
  // ESS가 높으면 (uniform weights) λ 감소
  double initial_lambda = adaptive_temp_->getLambda();

  // ESS = 800, K = 1000 → ess_ratio = 0.8 > 0.5 (target)
  double new_lambda = adaptive_temp_->update(800.0, 1000);

  EXPECT_LT(new_lambda, initial_lambda);
}

TEST_F(AdaptiveTemperatureTest, TargetESSMaintainsLambda)
{
  // ESS가 target과 같으면 λ 거의 유지
  double initial_lambda = adaptive_temp_->getLambda();

  // ESS = 500, K = 1000 → ess_ratio = 0.5 = target
  double new_lambda = adaptive_temp_->update(500.0, 1000);

  EXPECT_NEAR(new_lambda, initial_lambda, 0.01);
}

TEST_F(AdaptiveTemperatureTest, LambdaBoundsRespected)
{
  // λ_min 경계 테스트
  for (int i = 0; i < 100; ++i) {
    adaptive_temp_->update(990.0, 1000);  // 매우 높은 ESS
  }
  EXPECT_GE(adaptive_temp_->getLambda(), 0.1 - 1e-6);  // lambda_min

  // λ_max 경계 테스트
  adaptive_temp_->reset(10.0);
  for (int i = 0; i < 100; ++i) {
    adaptive_temp_->update(10.0, 1000);  // 매우 낮은 ESS
  }
  EXPECT_LE(adaptive_temp_->getLambda(), 100.0 + 1e-6);  // lambda_max (floating point tolerance)
}

TEST_F(AdaptiveTemperatureTest, ESSConvergence)
{
  // 여러 iteration 후 ESS가 target 근처로 수렴하는지 확인
  // (시뮬레이션: ESS가 λ에 반비례한다고 가정)
  int K = 1000;
  double target_ess = 0.5 * K;

  for (int i = 0; i < 50; ++i) {
    double lambda = adaptive_temp_->getLambda();
    // 간단한 모델: ESS ≈ K / (1 + λ/10)
    double simulated_ess = K / (1.0 + lambda / 10.0);
    adaptive_temp_->update(simulated_ess, K);
  }

  // 최종 ESS가 target 근처인지 확인
  double final_lambda = adaptive_temp_->getLambda();
  double final_ess = K / (1.0 + final_lambda / 10.0);
  double final_ratio = final_ess / K;

  EXPECT_NEAR(final_ratio, 0.5, 0.15);  // 15% 오차 허용
}

TEST_F(AdaptiveTemperatureTest, SetLambdaDirectly)
{
  adaptive_temp_->setLambda(25.0);
  EXPECT_DOUBLE_EQ(adaptive_temp_->getLambda(), 25.0);

  // 경계 밖 값은 클램프됨
  adaptive_temp_->setLambda(0.01);  // < lambda_min
  EXPECT_GE(adaptive_temp_->getLambda(), 0.1);

  adaptive_temp_->setLambda(500.0);  // > lambda_max
  EXPECT_LE(adaptive_temp_->getLambda(), 100.0);
}

TEST_F(AdaptiveTemperatureTest, Reset)
{
  adaptive_temp_->update(100.0, 1000);  // λ 변경
  double changed_lambda = adaptive_temp_->getLambda();
  EXPECT_NE(changed_lambda, 10.0);

  adaptive_temp_->reset(10.0);
  EXPECT_DOUBLE_EQ(adaptive_temp_->getLambda(), 10.0);
}

TEST_F(AdaptiveTemperatureTest, InfoStruct)
{
  adaptive_temp_->update(300.0, 1000);
  auto info = adaptive_temp_->getInfo();

  EXPECT_DOUBLE_EQ(info.ess_ratio, 0.3);
  EXPECT_DOUBLE_EQ(info.target_ratio, 0.5);
  EXPECT_NE(info.delta, 0.0);
  EXPECT_EQ(info.lambda, adaptive_temp_->getLambda());
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
