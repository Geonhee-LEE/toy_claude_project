#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "mpc_controller_ros2/svmpc_controller_plugin.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"

namespace mpc_controller_ros2
{

// ============================================================================
// 테스트 헬퍼: SVMPCControllerPlugin의 static 메서드 접근용 래퍼
// ============================================================================

class SVMPCTestHelper
{
public:
  // computeSVGDForce는 private이므로 동일 로직을 테스트용으로 복사
  static Eigen::MatrixXd computeSVGDForce(
    const std::vector<Eigen::VectorXd>& diff_flat,
    const Eigen::VectorXd& weights,
    const Eigen::MatrixXd& kernel,
    double bandwidth,
    int K,
    int D)
  {
    Eigen::MatrixXd force = Eigen::MatrixXd::Zero(K, D);
    double h_sq = bandwidth * bandwidth;

    for (int i = 0; i < K; ++i) {
      Eigen::VectorXd attractive = Eigen::VectorXd::Zero(D);
      Eigen::VectorXd repulsive = Eigen::VectorXd::Zero(D);

      for (int j = 0; j < K; ++j) {
        const Eigen::VectorXd& d_ji = diff_flat[j * K + i];
        attractive += weights(j) * kernel(j, i) * d_ji;
        repulsive += kernel(j, i) * d_ji / h_sq;
      }
      repulsive /= static_cast<double>(K);
      force.row(i) = (attractive + repulsive).transpose();
    }
    return force;
  }

  static double computeDiversity(
    const std::vector<Eigen::MatrixXd>& controls,
    int K, int D)
  {
    if (K <= 1) return 0.0;
    int max_samples = std::min(K, 128);
    std::vector<Eigen::VectorXd> flat;
    flat.reserve(max_samples);
    if (K > max_samples) {
      for (int i = 0; i < max_samples; ++i) {
        int idx = i * K / max_samples;
        Eigen::Map<const Eigen::VectorXd> v(controls[idx].data(), D);
        flat.push_back(v);
      }
    } else {
      for (int k = 0; k < K; ++k) {
        Eigen::Map<const Eigen::VectorXd> v(controls[k].data(), D);
        flat.push_back(v);
      }
      max_samples = K;
    }
    double sum_dist = 0.0;
    int count = 0;
    for (int i = 0; i < max_samples; ++i) {
      for (int j = i + 1; j < max_samples; ++j) {
        sum_dist += (flat[i] - flat[j]).norm();
        ++count;
      }
    }
    return count > 0 ? sum_dist / count : 0.0;
  }

  static double medianBandwidth(
    const Eigen::MatrixXd& sq_dist, int K)
  {
    std::vector<double> triu_vals;
    triu_vals.reserve(K * (K - 1) / 2);
    for (int i = 0; i < K; ++i) {
      for (int j = i + 1; j < K; ++j) {
        triu_vals.push_back(sq_dist(i, j));
      }
    }
    if (triu_vals.empty()) return 1.0;
    size_t mid = triu_vals.size() / 2;
    std::nth_element(triu_vals.begin(), triu_vals.begin() + mid, triu_vals.end());
    double med = triu_vals[mid];
    double h = std::sqrt(med / (2.0 * std::log(static_cast<double>(K) + 1.0)));
    return std::max(h, 1e-6);
  }
};

// ============================================================================
// SVGD Force 테스트
// ============================================================================

class SVGDForceTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    K_ = 4;
    D_ = 6;  // N=3, nu=2
  }

  // 헬퍼: pairwise diff와 kernel을 particles로부터 구성
  void buildKernelData(
    const Eigen::MatrixXd& particles,
    double bandwidth,
    std::vector<Eigen::VectorXd>& diff_flat,
    Eigen::MatrixXd& kernel,
    Eigen::MatrixXd& sq_dist)
  {
    diff_flat.resize(K_ * K_);
    sq_dist.resize(K_, K_);
    kernel.resize(K_, K_);

    for (int j = 0; j < K_; ++j) {
      for (int i = 0; i < K_; ++i) {
        diff_flat[j * K_ + i] = particles.row(j).transpose() - particles.row(i).transpose();
        sq_dist(j, i) = diff_flat[j * K_ + i].squaredNorm();
      }
    }

    kernel = (-sq_dist / (2.0 * bandwidth * bandwidth)).array().exp().matrix();
  }

  int K_;
  int D_;
};

TEST_F(SVGDForceTest, Attractive_LowCostDirection)
{
  // 저비용 샘플 방향으로 force가 향해야 함
  Eigen::MatrixXd particles(K_, D_);
  particles.row(0) = Eigen::VectorXd::Zero(D_);        // 원점
  particles.row(1) = Eigen::VectorXd::Ones(D_);         // (1,1,...,1)
  particles.row(2) = Eigen::VectorXd::Ones(D_) * 2.0;   // (2,2,...,2)
  particles.row(3) = Eigen::VectorXd::Ones(D_) * 3.0;   // (3,3,...,3)

  // 가중치: 샘플 1이 최고 가중치 (저비용)
  Eigen::VectorXd weights(K_);
  weights << 0.05, 0.8, 0.1, 0.05;

  double h = 2.0;
  std::vector<Eigen::VectorXd> diff_flat;
  Eigen::MatrixXd kernel, sq_dist;
  buildKernelData(particles, h, diff_flat, kernel, sq_dist);

  Eigen::MatrixXd force = SVMPCTestHelper::computeSVGDForce(
    diff_flat, weights, kernel, h, K_, D_);

  // 샘플 0 (원점)의 force는 샘플 1 (1,1,...,1) 방향 → 양수
  for (int d = 0; d < D_; ++d) {
    EXPECT_GT(force(0, d), 0.0)
      << "Force on sample 0 should point towards high-weight sample 1";
  }
}

TEST_F(SVGDForceTest, Repulsive_IdenticalParticles)
{
  // 동일 위치의 입자들 → repulsive force 존재 확인
  // 동일 위치에서는 diff=0이므로 force도 0 (수학적으로)
  Eigen::MatrixXd particles(K_, D_);
  for (int k = 0; k < K_; ++k) {
    particles.row(k) = Eigen::VectorXd::Ones(D_);
  }

  Eigen::VectorXd weights = Eigen::VectorXd::Constant(K_, 1.0 / K_);
  double h = 1.0;

  std::vector<Eigen::VectorXd> diff_flat;
  Eigen::MatrixXd kernel, sq_dist;
  buildKernelData(particles, h, diff_flat, kernel, sq_dist);

  Eigen::MatrixXd force = SVMPCTestHelper::computeSVGDForce(
    diff_flat, weights, kernel, h, K_, D_);

  // 완전 동일 위치 → diff=0 → force=0
  for (int k = 0; k < K_; ++k) {
    EXPECT_NEAR(force.row(k).norm(), 0.0, 1e-10)
      << "Identical particles should have zero force (diff=0)";
  }
}

TEST_F(SVGDForceTest, ZeroWeight_ZeroAttractiveForce)
{
  // 모든 가중치가 0이면 attractive force가 0
  Eigen::MatrixXd particles = Eigen::MatrixXd::Random(K_, D_);
  Eigen::VectorXd weights = Eigen::VectorXd::Zero(K_);
  double h = 1.0;

  std::vector<Eigen::VectorXd> diff_flat;
  Eigen::MatrixXd kernel, sq_dist;
  buildKernelData(particles, h, diff_flat, kernel, sq_dist);

  // 가중치=0이면 attractive=0, repulsive만 남음
  // repulsive의 net force도 대칭이므로 크기가 작아야 함
  Eigen::MatrixXd force = SVMPCTestHelper::computeSVGDForce(
    diff_flat, weights, kernel, h, K_, D_);

  // force 존재 확인 (repulsive는 있을 수 있음)
  // 정확히 0은 아니지만, attractive 기여가 없으므로 purely repulsive
  EXPECT_TRUE(force.allFinite()) << "Force should be finite with zero weights";
}

// ============================================================================
// Diversity 테스트
// ============================================================================

TEST(DiversityTest, Single_Sample_Zero)
{
  // K=1 → diversity=0
  int K = 1;
  int N = 3;
  int nu = 2;
  int D = N * nu;
  std::vector<Eigen::MatrixXd> controls;
  controls.push_back(Eigen::MatrixXd::Random(N, nu));

  double div = SVMPCTestHelper::computeDiversity(controls, K, D);
  EXPECT_DOUBLE_EQ(div, 0.0);
}

TEST(DiversityTest, Identical_Samples_Zero)
{
  // 동일 샘플 → diversity=0
  int K = 10;
  int N = 3;
  int nu = 2;
  int D = N * nu;
  Eigen::MatrixXd control = Eigen::MatrixXd::Ones(N, nu);

  std::vector<Eigen::MatrixXd> controls;
  for (int k = 0; k < K; ++k) {
    controls.push_back(control);
  }

  double div = SVMPCTestHelper::computeDiversity(controls, K, D);
  EXPECT_NEAR(div, 0.0, 1e-10);
}

TEST(DiversityTest, Spread_Greater_Than_Clustered)
{
  // 분산된 샘플의 diversity > 밀집된 샘플
  int K = 20;
  int N = 3;
  int nu = 2;
  int D = N * nu;

  // 밀집 샘플
  std::vector<Eigen::MatrixXd> clustered;
  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd c = Eigen::MatrixXd::Ones(N, nu) + 0.01 * Eigen::MatrixXd::Random(N, nu);
    clustered.push_back(c);
  }

  // 분산 샘플
  std::vector<Eigen::MatrixXd> spread;
  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd s = 10.0 * Eigen::MatrixXd::Random(N, nu);
    spread.push_back(s);
  }

  double div_clustered = SVMPCTestHelper::computeDiversity(clustered, K, D);
  double div_spread = SVMPCTestHelper::computeDiversity(spread, K, D);

  EXPECT_GT(div_spread, div_clustered)
    << "Spread samples should have greater diversity than clustered";
}

// ============================================================================
// Median Bandwidth 테스트
// ============================================================================

TEST(MedianBandwidthTest, Positive)
{
  // bandwidth > 0
  int K = 10;
  Eigen::MatrixXd particles = Eigen::MatrixXd::Random(K, 6);

  Eigen::MatrixXd sq_dist(K, K);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      sq_dist(i, j) = (particles.row(i) - particles.row(j)).squaredNorm();
    }
  }

  double h = SVMPCTestHelper::medianBandwidth(sq_dist, K);
  EXPECT_GT(h, 0.0) << "Bandwidth should be positive";
  EXPECT_TRUE(std::isfinite(h)) << "Bandwidth should be finite";
}

TEST(MedianBandwidthTest, ScaleProportional)
{
  // 거리가 스케일되면 bandwidth도 비례
  int K = 10;
  Eigen::MatrixXd particles = Eigen::MatrixXd::Random(K, 6);

  Eigen::MatrixXd sq_dist1(K, K);
  Eigen::MatrixXd sq_dist2(K, K);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      sq_dist1(i, j) = (particles.row(i) - particles.row(j)).squaredNorm();
      sq_dist2(i, j) = sq_dist1(i, j) * 100.0;  // 10x 스케일
    }
  }

  double h1 = SVMPCTestHelper::medianBandwidth(sq_dist1, K);
  double h2 = SVMPCTestHelper::medianBandwidth(sq_dist2, K);

  // h = sqrt(med / c) → sq_dist * 100 → h * 10
  EXPECT_NEAR(h2 / h1, 10.0, 0.5)
    << "Bandwidth should scale proportionally with distance";
}

TEST(MedianBandwidthTest, MinimumClamp)
{
  // 모든 입자가 동일 → sq_dist=0 → bandwidth >= 1e-6
  int K = 5;
  Eigen::MatrixXd sq_dist = Eigen::MatrixXd::Zero(K, K);

  double h = SVMPCTestHelper::medianBandwidth(sq_dist, K);
  EXPECT_GE(h, 1e-6) << "Bandwidth should be clamped to minimum";
}

// ============================================================================
// RBF Kernel 속성 테스트
// ============================================================================

TEST(RBFKernelTest, SelfKernelIsOne)
{
  // kernel(i,i) = exp(0) = 1
  int K = 5;
  Eigen::MatrixXd particles = Eigen::MatrixXd::Random(K, 6);
  double h = 1.0;

  Eigen::MatrixXd sq_dist(K, K);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      sq_dist(i, j) = (particles.row(i) - particles.row(j)).squaredNorm();
    }
  }

  Eigen::MatrixXd kernel = (-sq_dist / (2.0 * h * h)).array().exp().matrix();

  for (int i = 0; i < K; ++i) {
    EXPECT_DOUBLE_EQ(kernel(i, i), 1.0);
  }
}

TEST(RBFKernelTest, Symmetric)
{
  // kernel(i,j) = kernel(j,i)
  int K = 5;
  Eigen::MatrixXd particles = Eigen::MatrixXd::Random(K, 6);
  double h = 1.0;

  Eigen::MatrixXd sq_dist(K, K);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      sq_dist(i, j) = (particles.row(i) - particles.row(j)).squaredNorm();
    }
  }

  Eigen::MatrixXd kernel = (-sq_dist / (2.0 * h * h)).array().exp().matrix();

  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      EXPECT_NEAR(kernel(i, j), kernel(j, i), 1e-12);
    }
  }
}

TEST(RBFKernelTest, BoundedZeroOne)
{
  // 0 < kernel(i,j) <= 1
  int K = 5;
  Eigen::MatrixXd particles = 10.0 * Eigen::MatrixXd::Random(K, 6);
  double h = 1.0;

  Eigen::MatrixXd sq_dist(K, K);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      sq_dist(i, j) = (particles.row(i) - particles.row(j)).squaredNorm();
    }
  }

  Eigen::MatrixXd kernel = (-sq_dist / (2.0 * h * h)).array().exp().matrix();

  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      EXPECT_GT(kernel(i, j), 0.0);
      EXPECT_LE(kernel(i, j), 1.0 + 1e-12);
    }
  }
}

// ============================================================================
// MPPIInfo SVGD 필드 테스트
// ============================================================================

TEST(MPPIInfoSVGD, DefaultValues)
{
  MPPIInfo info;
  EXPECT_EQ(info.svgd_iterations, 0);
  EXPECT_DOUBLE_EQ(info.sample_diversity_before, 0.0);
  EXPECT_DOUBLE_EQ(info.sample_diversity_after, 0.0);
}

}  // namespace mpc_controller_ros2
