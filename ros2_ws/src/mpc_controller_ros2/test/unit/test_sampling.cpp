#include <gtest/gtest.h>
#include <cmath>
#include "mpc_controller_ros2/sampling.hpp"

namespace mpc_controller_ros2
{

class SamplingTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    sigma_ = Eigen::Vector2d(0.5, 0.3);
  }

  Eigen::VectorXd sigma_;
};

TEST_F(SamplingTest, GaussianSamplerDimensions)
{
  GaussianSampler sampler(sigma_, 42);

  int K = 100;
  int N = 20;
  int nu = 2;

  auto samples = sampler.sample(K, N, nu);

  EXPECT_EQ(samples.size(), K);
  for (const auto& sample : samples) {
    EXPECT_EQ(sample.rows(), N);
    EXPECT_EQ(sample.cols(), nu);
  }
}

TEST_F(SamplingTest, GaussianSamplerStatistics)
{
  // Test that samples have approximately correct mean and variance
  GaussianSampler sampler(sigma_, 42);

  int K = 10000;
  int N = 1;
  int nu = 2;

  auto samples = sampler.sample(K, N, nu);

  // Compute mean and variance
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(nu);
  for (const auto& sample : samples) {
    mean += sample.row(0).transpose();
  }
  mean /= K;

  Eigen::VectorXd variance = Eigen::VectorXd::Zero(nu);
  for (const auto& sample : samples) {
    Eigen::VectorXd diff = sample.row(0).transpose() - mean;
    variance += diff.array().square().matrix();
  }
  variance /= K;

  // Mean should be close to 0
  EXPECT_NEAR(mean(0), 0.0, 0.05);
  EXPECT_NEAR(mean(1), 0.0, 0.05);

  // Variance should be close to sigma^2
  EXPECT_NEAR(std::sqrt(variance(0)), sigma_(0), 0.05);
  EXPECT_NEAR(std::sqrt(variance(1)), sigma_(1), 0.05);
}

TEST_F(SamplingTest, GaussianSamplerReproducibility)
{
  // Same seed should produce same samples
  GaussianSampler sampler1(sigma_, 42);
  GaussianSampler sampler2(sigma_, 42);

  int K = 10;
  int N = 5;
  int nu = 2;

  auto samples1 = sampler1.sample(K, N, nu);
  auto samples2 = sampler2.sample(K, N, nu);

  for (int k = 0; k < K; ++k) {
    for (int t = 0; t < N; ++t) {
      for (int i = 0; i < nu; ++i) {
        EXPECT_NEAR(samples1[k](t, i), samples2[k](t, i), 1e-12);
      }
    }
  }
}

TEST_F(SamplingTest, GaussianSamplerResetSeed)
{
  GaussianSampler sampler(sigma_, 42);

  int K = 5;
  int N = 3;
  int nu = 2;

  auto samples1 = sampler.sample(K, N, nu);

  // Reset to same seed
  sampler.resetSeed(42);
  auto samples2 = sampler.sample(K, N, nu);

  // Should produce identical samples
  for (int k = 0; k < K; ++k) {
    for (int t = 0; t < N; ++t) {
      for (int i = 0; i < nu; ++i) {
        EXPECT_NEAR(samples1[k](t, i), samples2[k](t, i), 1e-12);
      }
    }
  }
}

TEST_F(SamplingTest, ColoredNoiseSamplerDimensions)
{
  ColoredNoiseSampler sampler(sigma_, 2.0, 42);

  int K = 100;
  int N = 20;
  int nu = 2;

  auto samples = sampler.sample(K, N, nu);

  EXPECT_EQ(samples.size(), K);
  for (const auto& sample : samples) {
    EXPECT_EQ(sample.rows(), N);
    EXPECT_EQ(sample.cols(), nu);
  }
}

TEST_F(SamplingTest, ColoredNoiseSamplerAutocorrelation)
{
  // Colored noise should have temporal correlation
  ColoredNoiseSampler sampler(sigma_, 1.0, 42);

  int K = 1;
  int N = 100;
  int nu = 1;

  auto samples = sampler.sample(K, N, nu);

  // Compute lag-1 autocorrelation
  double sum_product = 0.0;
  double sum_sq = 0.0;
  for (int t = 0; t < N - 1; ++t) {
    sum_product += samples[0](t, 0) * samples[0](t + 1, 0);
    sum_sq += samples[0](t, 0) * samples[0](t, 0);
  }

  double autocorr = sum_product / sum_sq;

  // Should have positive autocorrelation (> 0.3 for beta=1)
  // Note: exact value depends on decay = exp(-beta*dt)
  EXPECT_GT(autocorr, 0.3);
}

TEST_F(SamplingTest, ColoredNoiseSamplerBetaEffect)
{
  // Higher beta should reduce autocorrelation
  int K = 1;
  int N = 100;
  int nu = 1;

  ColoredNoiseSampler sampler_low_beta(sigma_, 0.5, 42);
  auto samples_low = sampler_low_beta.sample(K, N, nu);

  ColoredNoiseSampler sampler_high_beta(sigma_, 5.0, 43);
  auto samples_high = sampler_high_beta.sample(K, N, nu);

  // Compute lag-1 autocorrelation for both
  auto compute_autocorr = [](const Eigen::MatrixXd& sample) {
    double sum_product = 0.0;
    double sum_sq = 0.0;
    int N = sample.rows();
    for (int t = 0; t < N - 1; ++t) {
      sum_product += sample(t, 0) * sample(t + 1, 0);
      sum_sq += sample(t, 0) * sample(t, 0);
    }
    return sum_product / sum_sq;
  };

  double autocorr_low = compute_autocorr(samples_low[0]);
  double autocorr_high = compute_autocorr(samples_high[0]);

  // Low beta should have higher autocorrelation
  EXPECT_GT(autocorr_low, autocorr_high);
}

TEST_F(SamplingTest, ColoredNoiseSamplerStationaryVariance)
{
  // Variance should converge to sigma^2
  ColoredNoiseSampler sampler(sigma_, 2.0, 42);

  int K = 5000;
  int N = 50;
  int nu = 2;

  auto samples = sampler.sample(K, N, nu);

  // Compute variance at different time points
  for (int t : {0, 10, 30, 49}) {
    Eigen::VectorXd variance = Eigen::VectorXd::Zero(nu);
    for (const auto& sample : samples) {
      variance += sample.row(t).transpose().array().square().matrix();
    }
    variance /= K;

    // Should be close to sigma^2
    EXPECT_NEAR(std::sqrt(variance(0)), sigma_(0), 0.1);
    EXPECT_NEAR(std::sqrt(variance(1)), sigma_(1), 0.1);
  }
}

}  // namespace mpc_controller_ros2

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
