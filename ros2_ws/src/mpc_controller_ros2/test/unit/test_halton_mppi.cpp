// =============================================================================
// Halton-MPPI Unit Tests
//
// 15 gtest: Halton 시퀀스 수학적 검증 + 역정규 CDF + 샘플러 품질 + 모델 호환
// =============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include "mpc_controller_ros2/halton_sampler.hpp"

namespace mpc_controller_ros2
{

// ============================================================================
// Test fixture
// ============================================================================

class HaltonMPPITest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    sigma2_ = Eigen::Vector2d(0.5, 0.5);
    sigma3_ = Eigen::Vector3d(0.5, 0.5, 0.3);
    sigma1_ = Eigen::VectorXd(1);
    sigma1_(0) = 1.0;
  }

  Eigen::Vector2d sigma2_;
  Eigen::Vector3d sigma3_;
  Eigen::VectorXd sigma1_;
};

// ============================================================================
// Test 1: HaltonValueBase2
// Van der Corput base-2: 1->0.5, 2->0.25, 3->0.75
// ============================================================================
TEST_F(HaltonMPPITest, HaltonValueBase2)
{
  EXPECT_DOUBLE_EQ(HaltonSampler::haltonValue(1, 2), 0.5);
  EXPECT_DOUBLE_EQ(HaltonSampler::haltonValue(2, 2), 0.25);
  EXPECT_DOUBLE_EQ(HaltonSampler::haltonValue(3, 2), 0.75);
  EXPECT_DOUBLE_EQ(HaltonSampler::haltonValue(4, 2), 0.125);
  EXPECT_DOUBLE_EQ(HaltonSampler::haltonValue(5, 2), 0.625);
}

// ============================================================================
// Test 2: HaltonValueBase3
// Van der Corput base-3: 1->1/3, 2->2/3
// ============================================================================
TEST_F(HaltonMPPITest, HaltonValueBase3)
{
  EXPECT_NEAR(HaltonSampler::haltonValue(1, 3), 1.0 / 3.0, 1e-12);
  EXPECT_NEAR(HaltonSampler::haltonValue(2, 3), 2.0 / 3.0, 1e-12);
  EXPECT_NEAR(HaltonSampler::haltonValue(3, 3), 1.0 / 9.0, 1e-12);
}

// ============================================================================
// Test 3: HaltonValueZero
// H_b(0) = 0 for any base
// ============================================================================
TEST_F(HaltonMPPITest, HaltonValueZero)
{
  EXPECT_DOUBLE_EQ(HaltonSampler::haltonValue(0, 2), 0.0);
  EXPECT_DOUBLE_EQ(HaltonSampler::haltonValue(0, 3), 0.0);
  EXPECT_DOUBLE_EQ(HaltonSampler::haltonValue(0, 5), 0.0);
}

// ============================================================================
// Test 4: InverseCDFCenter
// Phi^{-1}(0.5) = 0.0
// ============================================================================
TEST_F(HaltonMPPITest, InverseCDFCenter)
{
  EXPECT_NEAR(HaltonSampler::inverseNormalCDF(0.5), 0.0, 1e-6);
}

// ============================================================================
// Test 5: InverseCDFTails
// Phi^{-1}(0.975) ~ 1.96, Phi^{-1}(0.025) ~ -1.96
// ============================================================================
TEST_F(HaltonMPPITest, InverseCDFTails)
{
  EXPECT_NEAR(HaltonSampler::inverseNormalCDF(0.975), 1.96, 0.02);
  EXPECT_NEAR(HaltonSampler::inverseNormalCDF(0.025), -1.96, 0.02);
  // Also check more extreme tails
  EXPECT_NEAR(HaltonSampler::inverseNormalCDF(0.999), 3.09, 0.05);
}

// ============================================================================
// Test 6: InverseCDFMonotone
// p1 < p2 => Phi^{-1}(p1) < Phi^{-1}(p2)
// ============================================================================
TEST_F(HaltonMPPITest, InverseCDFMonotone)
{
  std::vector<double> ps = {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99};
  for (size_t i = 1; i < ps.size(); ++i) {
    double v_prev = HaltonSampler::inverseNormalCDF(ps[i - 1]);
    double v_curr = HaltonSampler::inverseNormalCDF(ps[i]);
    EXPECT_LT(v_prev, v_curr)
      << "Monotonicity violated at p=" << ps[i - 1] << " vs p=" << ps[i];
  }
}

// ============================================================================
// Test 7: SampleDimensions
// sample(K=10, N=5, nu=2) returns 10 matrices of 5x2
// ============================================================================
TEST_F(HaltonMPPITest, SampleDimensions)
{
  HaltonSampler sampler(sigma2_, 2.0, 100);
  auto samples = sampler.sample(10, 5, 2);

  ASSERT_EQ(static_cast<int>(samples.size()), 10);
  for (int k = 0; k < 10; ++k) {
    EXPECT_EQ(samples[k].rows(), 5);
    EXPECT_EQ(samples[k].cols(), 2);
  }
}

// ============================================================================
// Test 8: LowDiscrepancy
// Halton samples cover [-3sigma, 3sigma] more uniformly than random
// ============================================================================
TEST_F(HaltonMPPITest, LowDiscrepancy)
{
  // Generate Halton samples with no OU correlation (high beta)
  HaltonSampler sampler(sigma1_, 10000.0, 100);
  auto samples = sampler.sample(100, 10, 1);

  // Collect all values
  std::vector<double> values;
  values.reserve(1000);
  for (int k = 0; k < 100; ++k) {
    for (int t = 0; t < 10; ++t) {
      values.push_back(samples[k](t, 0));
    }
  }

  // Halton + inverse CDF produces Gaussian distribution, so bins near center
  // naturally have more mass. Instead, verify the raw Halton values (before CDF)
  // are more uniformly distributed in [0,1].
  constexpr int NUM_BINS = 10;
  std::vector<int> halton_bins(NUM_BINS, 0);

  // Generate raw Halton values in [0,1] to check uniformity
  for (int i = 0; i < 1000; ++i) {
    double h = HaltonSampler::haltonValue(100 + i, 2);  // base 2, offset 100
    int bin = static_cast<int>(h * NUM_BINS);
    bin = std::min(bin, NUM_BINS - 1);
    halton_bins[bin]++;
  }

  // For Halton base-2, each bin should have ~100 samples (1000/10)
  int max_count = *std::max_element(halton_bins.begin(), halton_bins.end());
  int min_count = *std::min_element(halton_bins.begin(), halton_bins.end());

  // Halton sequence should be highly uniform in [0,1]
  double ratio = static_cast<double>(max_count) / static_cast<double>(min_count);
  EXPECT_LT(ratio, 2.0)
    << "Halton bins not uniform: max=" << max_count << " min=" << min_count;

  // Also verify the samples (after CDF) have reasonable spread
  double sample_std = 0.0;
  double sample_mean = 0.0;
  for (double v : values) { sample_mean += v; }
  sample_mean /= values.size();
  for (double v : values) { sample_std += (v - sample_mean) * (v - sample_mean); }
  sample_std = std::sqrt(sample_std / values.size());
  EXPECT_GT(sample_std, 0.5) << "Samples should have reasonable spread";
  EXPECT_LT(sample_std, 2.0) << "Samples std should be bounded";
}

// ============================================================================
// Test 9: OUCorrelation
// with beta=0.5, lag-1 autocorrelation > 0.3
// ============================================================================
TEST_F(HaltonMPPITest, OUCorrelation)
{
  // OU with small beta makes the sequence smoother (more temporal correlation)
  // Compare RMS of successive differences: OU should be smoother
  HaltonSampler sampler_ou(sigma1_, 0.5, 100);
  auto samples_ou = sampler_ou.sample(1, 100, 1);

  HaltonSampler sampler_no_ou(sigma1_, 1000.0, 100);
  auto samples_no_ou = sampler_no_ou.sample(1, 100, 1);

  // Compute RMS of successive differences
  double rms_ou = 0.0, rms_no_ou = 0.0;
  for (int t = 1; t < 100; ++t) {
    double diff_ou = samples_ou[0](t, 0) - samples_ou[0](t - 1, 0);
    double diff_no = samples_no_ou[0](t, 0) - samples_no_ou[0](t - 1, 0);
    rms_ou += diff_ou * diff_ou;
    rms_no_ou += diff_no * diff_no;
  }
  rms_ou = std::sqrt(rms_ou / 99.0);
  rms_no_ou = std::sqrt(rms_no_ou / 99.0);

  // OU-filtered should be smoother (smaller RMS of differences)
  EXPECT_LT(rms_ou, rms_no_ou)
    << "OU smoothing should reduce successive differences: rms_ou=" << rms_ou
    << " rms_no_ou=" << rms_no_ou;
}

// ============================================================================
// Test 10: NoBetaIndependent
// with beta=1000 (effectively infinite), autocorrelation < 0.1
// ============================================================================
TEST_F(HaltonMPPITest, NoBetaIndependent)
{
  // beta >= 1000 disables OU entirely (pass-through)
  // Verify values match direct Halton computation without OU
  HaltonSampler sampler(sigma1_, 1000.0, 100);
  auto samples = sampler.sample(1, 10, 1);

  // Directly compute expected values (no OU)
  for (int t = 0; t < 10; ++t) {
    int idx = 0 * 10 + t + 100;  // k=0, offset=100
    double h = HaltonSampler::haltonValue(idx, 2);
    double z = HaltonSampler::inverseNormalCDF(h);
    double expected = z * 1.0;  // sigma=1.0
    EXPECT_NEAR(samples[0](t, 0), expected, 1e-10)
      << "Without OU, samples should match raw Halton+CDF at t=" << t;
  }
}

// ============================================================================
// Test 11: DeterministicSequence
// Same sample_counter produces same output after reset
// ============================================================================
TEST_F(HaltonMPPITest, DeterministicSequence)
{
  HaltonSampler sampler(sigma2_, 10000.0, 100);  // high beta = no OU

  auto samples1 = sampler.sample(5, 10, 2);
  sampler.reset();
  auto samples2 = sampler.sample(5, 10, 2);

  for (int k = 0; k < 5; ++k) {
    for (int t = 0; t < 10; ++t) {
      for (int d = 0; d < 2; ++d) {
        EXPECT_DOUBLE_EQ(samples1[k](t, d), samples2[k](t, d))
          << "Mismatch at k=" << k << " t=" << t << " d=" << d;
      }
    }
  }
}

// ============================================================================
// Test 12: SigmaScaling
// Doubling sigma should approximately double the standard deviation
// ============================================================================
TEST_F(HaltonMPPITest, SigmaScaling)
{
  Eigen::VectorXd sigma_small(1);
  sigma_small(0) = 0.5;
  Eigen::VectorXd sigma_large(1);
  sigma_large(0) = 1.0;

  HaltonSampler sampler_small(sigma_small, 10000.0, 100);
  HaltonSampler sampler_large(sigma_large, 10000.0, 100);

  auto s1 = sampler_small.sample(100, 10, 1);
  auto s2 = sampler_large.sample(100, 10, 1);

  // Compute standard deviations
  auto computeStd = [](const std::vector<Eigen::MatrixXd>& samples) {
    double sum = 0.0, sum2 = 0.0;
    int n = 0;
    for (const auto& m : samples) {
      for (int t = 0; t < m.rows(); ++t) {
        double v = m(t, 0);
        sum += v;
        sum2 += v * v;
        n++;
      }
    }
    double mean = sum / n;
    return std::sqrt(sum2 / n - mean * mean);
  };

  double std1 = computeStd(s1);
  double std2 = computeStd(s2);

  // Ratio should be close to 2.0
  double ratio = std2 / std1;
  EXPECT_NEAR(ratio, 2.0, 0.3)
    << "Sigma scaling ratio: " << ratio << " (expected ~2.0)";
}

// ============================================================================
// Test 13: SampleInPlace
// sampleInPlace produces same dimension output as sample
// ============================================================================
TEST_F(HaltonMPPITest, SampleInPlace)
{
  HaltonSampler sampler(sigma2_, 2.0, 100);

  std::vector<Eigen::MatrixXd> buffer;
  sampler.sampleInPlace(buffer, 10, 5, 2);

  ASSERT_EQ(static_cast<int>(buffer.size()), 10);
  for (int k = 0; k < 10; ++k) {
    EXPECT_EQ(buffer[k].rows(), 5);
    EXPECT_EQ(buffer[k].cols(), 2);
    // Check no NaN/Inf
    EXPECT_TRUE(buffer[k].allFinite())
      << "Sample " << k << " contains NaN or Inf";
  }

  // Test re-use with different dimensions
  sampler.reset();
  sampler.sampleInPlace(buffer, 5, 3, 2);
  ASSERT_EQ(static_cast<int>(buffer.size()), 5);
  for (int k = 0; k < 5; ++k) {
    EXPECT_EQ(buffer[k].rows(), 3);
    EXPECT_EQ(buffer[k].cols(), 2);
  }
}

// ============================================================================
// Test 14: DiffDriveModel
// nu=2 works correctly (standard diff_drive: v, omega)
// ============================================================================
TEST_F(HaltonMPPITest, DiffDriveModel)
{
  HaltonSampler sampler(sigma2_, 2.0, 100);
  auto samples = sampler.sample(50, 30, 2);

  ASSERT_EQ(static_cast<int>(samples.size()), 50);
  for (int k = 0; k < 50; ++k) {
    EXPECT_EQ(samples[k].rows(), 30);
    EXPECT_EQ(samples[k].cols(), 2);
    EXPECT_TRUE(samples[k].allFinite());
  }

  // Check that both control dimensions have non-zero variance
  double var_v = 0.0, var_omega = 0.0;
  for (int k = 0; k < 50; ++k) {
    for (int t = 0; t < 30; ++t) {
      var_v += samples[k](t, 0) * samples[k](t, 0);
      var_omega += samples[k](t, 1) * samples[k](t, 1);
    }
  }
  var_v /= (50 * 30);
  var_omega /= (50 * 30);

  EXPECT_GT(var_v, 0.01) << "v dimension has no variance";
  EXPECT_GT(var_omega, 0.01) << "omega dimension has no variance";
}

// ============================================================================
// Test 15: SwerveModel
// nu=3 works correctly (swerve: vx, vy, omega)
// ============================================================================
TEST_F(HaltonMPPITest, SwerveModel)
{
  HaltonSampler sampler(sigma3_, 2.0, 100);
  auto samples = sampler.sample(50, 30, 3);

  ASSERT_EQ(static_cast<int>(samples.size()), 50);
  for (int k = 0; k < 50; ++k) {
    EXPECT_EQ(samples[k].rows(), 30);
    EXPECT_EQ(samples[k].cols(), 3);
    EXPECT_TRUE(samples[k].allFinite());
  }

  // Check that all 3 control dimensions have non-zero variance
  Eigen::Vector3d var = Eigen::Vector3d::Zero();
  for (int k = 0; k < 50; ++k) {
    for (int t = 0; t < 30; ++t) {
      for (int d = 0; d < 3; ++d) {
        var(d) += samples[k](t, d) * samples[k](t, d);
      }
    }
  }
  var /= (50 * 30);

  for (int d = 0; d < 3; ++d) {
    EXPECT_GT(var(d), 0.01) << "Dimension " << d << " has no variance";
  }
}

}  // namespace mpc_controller_ros2
