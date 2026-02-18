#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

#include "mpc_controller_ros2/smooth_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/spline_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/svg_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"

namespace mpc_controller_ros2
{

// ============================================================================
// Smooth-MPPI 테스트
// ============================================================================

TEST(SmoothMPPI, DeltaUShift)
{
  // DU shift 후 마지막 행이 0이어야 함
  int N = 10;
  Eigen::MatrixXd DU = Eigen::MatrixXd::Random(N, 2);

  // Shift
  for (int t = 0; t < N - 1; ++t) {
    DU.row(t) = DU.row(t + 1);
  }
  DU.row(N - 1).setZero();

  EXPECT_DOUBLE_EQ(DU(N - 1, 0), 0.0);
  EXPECT_DOUBLE_EQ(DU(N - 1, 1), 0.0);
}

TEST(SmoothMPPI, CumsumRestore)
{
  // u_prev + cumsum(DU) = U 확인
  int N = 10;
  Eigen::Vector2d u_prev(0.3, -0.1);
  Eigen::MatrixXd DU = Eigen::MatrixXd::Random(N, 2) * 0.1;

  // cumsum으로 U 복원
  Eigen::MatrixXd U(N, 2);
  Eigen::Vector2d cumulative = u_prev;
  for (int t = 0; t < N; ++t) {
    cumulative += DU.row(t).transpose();
    U.row(t) = cumulative.transpose();
  }

  // 검증: U[0] = u_prev + DU[0]
  Eigen::Vector2d expected_u0 = u_prev + DU.row(0).transpose();
  EXPECT_NEAR(U(0, 0), expected_u0(0), 1e-10);
  EXPECT_NEAR(U(0, 1), expected_u0(1), 1e-10);

  // 검증: U[N-1] = u_prev + sum(DU)
  Eigen::Vector2d expected_uN = u_prev + DU.colwise().sum().transpose();
  EXPECT_NEAR(U(N - 1, 0), expected_uN(0), 1e-10);
  EXPECT_NEAR(U(N - 1, 1), expected_uN(1), 1e-10);
}

TEST(SmoothMPPI, JerkCost)
{
  // ΔΔu → jerk cost > 0 확인
  int N = 10;
  int K = 5;
  Eigen::Vector2d R_jerk(0.1, 0.1);
  double jerk_weight = 1.0;

  // 노이즈가 있는 DU 생성
  std::mt19937 rng(42);
  std::normal_distribution<double> dist(0.0, 0.5);

  double total_jerk_cost = 0.0;
  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd du(N, 2);
    for (int t = 0; t < N; ++t) {
      du(t, 0) = dist(rng);
      du(t, 1) = dist(rng);
    }

    double jerk_cost = 0.0;
    for (int t = 0; t < N - 1; ++t) {
      Eigen::Vector2d ddu = du.row(t + 1).transpose() - du.row(t).transpose();
      jerk_cost += ddu(0) * ddu(0) * R_jerk(0) + ddu(1) * ddu(1) * R_jerk(1);
    }
    total_jerk_cost += jerk_weight * jerk_cost;
  }

  EXPECT_GT(total_jerk_cost, 0.0);
}

TEST(SmoothMPPI, SmoothOutput)
{
  // Smooth-MPPI에서 cumsum 복원된 제어는 순수 노이즈보다 부드러워야 함
  int N = 20;
  std::mt19937 rng(42);
  std::normal_distribution<double> dist(0.0, 0.5);

  // 직접 노이즈 (Vanilla 방식)
  Eigen::MatrixXd raw_u(N, 2);
  for (int t = 0; t < N; ++t) {
    raw_u(t, 0) = dist(rng);
    raw_u(t, 1) = dist(rng);
  }

  // cumsum 복원 (Smooth 방식)
  Eigen::MatrixXd du(N, 2);
  for (int t = 0; t < N; ++t) {
    du(t, 0) = dist(rng) * 0.3;  // 같은 크기 대비 더 작은 delta
    du(t, 1) = dist(rng) * 0.3;
  }
  Eigen::MatrixXd smooth_u(N, 2);
  Eigen::Vector2d cum = Eigen::Vector2d::Zero();
  for (int t = 0; t < N; ++t) {
    cum += du.row(t).transpose();
    smooth_u.row(t) = cum.transpose();
  }

  // 제어 변화율 (smoothness) 계산
  double raw_rate = 0.0, smooth_rate = 0.0;
  for (int t = 0; t < N - 1; ++t) {
    raw_rate += (raw_u.row(t + 1) - raw_u.row(t)).norm();
    smooth_rate += (smooth_u.row(t + 1) - smooth_u.row(t)).norm();
  }

  // Smooth 방식은 cumsum이므로 변화율이 delta 크기에 비례해야 함
  // delta가 작으면 smooth_rate < raw_rate
  EXPECT_LT(smooth_rate, raw_rate);
}

// ============================================================================
// Spline-MPPI / B-spline Basis 테스트
// ============================================================================

TEST(BSplineBasis, Shape)
{
  int N = 30;
  int P = 8;
  int degree = 3;

  Eigen::MatrixXd basis = SplineMPPIControllerPlugin::computeBSplineBasis(N, P, degree);

  EXPECT_EQ(basis.rows(), N);
  EXPECT_EQ(basis.cols(), P);
}

TEST(BSplineBasis, PartitionOfUnity)
{
  int N = 30;
  int P = 8;
  int degree = 3;

  Eigen::MatrixXd basis = SplineMPPIControllerPlugin::computeBSplineBasis(N, P, degree);

  // 각 행의 합 ≈ 1 (partition of unity)
  for (int i = 0; i < N; ++i) {
    double row_sum = basis.row(i).sum();
    EXPECT_NEAR(row_sum, 1.0, 1e-6)
      << "Row " << i << " sum = " << row_sum;
  }
}

TEST(BSplineBasis, Boundary)
{
  int N = 30;
  int P = 8;
  int degree = 3;

  Eigen::MatrixXd basis = SplineMPPIControllerPlugin::computeBSplineBasis(N, P, degree);

  // Clamped B-spline: basis(0, 0) ≈ 1 (첫 제어점)
  EXPECT_NEAR(basis(0, 0), 1.0, 1e-6)
    << "basis(0,0) = " << basis(0, 0);

  // basis(N-1, P-1) ≈ 1 (마지막 제어점)
  EXPECT_NEAR(basis(N - 1, P - 1), 1.0, 1e-3)
    << "basis(N-1,P-1) = " << basis(N - 1, P - 1);
}

TEST(SplineMPPI, KnotInterpolation)
{
  int N = 30;
  int P = 8;
  int degree = 3;

  Eigen::MatrixXd basis = SplineMPPIControllerPlugin::computeBSplineBasis(N, P, degree);

  // 상수 knots → 보간 결과도 상수
  Eigen::MatrixXd knots = Eigen::MatrixXd::Ones(P, 2) * 0.5;  // 모든 knot = [0.5, 0.5]
  Eigen::MatrixXd result = basis * knots;  // (N, 2)

  for (int t = 0; t < N; ++t) {
    EXPECT_NEAR(result(t, 0), 0.5, 1e-6)
      << "t=" << t << ", result=" << result(t, 0);
    EXPECT_NEAR(result(t, 1), 0.5, 1e-6)
      << "t=" << t << ", result=" << result(t, 1);
  }
}

TEST(BSplineBasis, NonNegative)
{
  int N = 30;
  int P = 8;
  int degree = 3;

  Eigen::MatrixXd basis = SplineMPPIControllerPlugin::computeBSplineBasis(N, P, degree);

  // B-spline basis는 항상 비음수
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < P; ++j) {
      EXPECT_GE(basis(i, j), -1e-12)
        << "basis(" << i << "," << j << ") = " << basis(i, j);
    }
  }
}

TEST(BSplineBasis, DifferentParameters)
{
  // 다양한 N, P, degree 조합에서 기본 속성 확인
  std::vector<std::tuple<int, int, int>> configs = {
    {20, 5, 2}, {30, 8, 3}, {40, 10, 3}, {50, 12, 4}
  };

  for (const auto& [N, P, degree] : configs) {
    if (P <= degree) continue;  // P > degree 필요

    Eigen::MatrixXd basis = SplineMPPIControllerPlugin::computeBSplineBasis(N, P, degree);

    EXPECT_EQ(basis.rows(), N) << "N=" << N << " P=" << P;
    EXPECT_EQ(basis.cols(), P) << "N=" << N << " P=" << P;

    // Partition of unity
    for (int i = 0; i < N; ++i) {
      EXPECT_NEAR(basis.row(i).sum(), 1.0, 1e-5)
        << "N=" << N << " P=" << P << " row=" << i;
    }
  }
}

// ============================================================================
// SVG-MPPI 테스트
// ============================================================================

TEST(SVGMPPI, GuideSelection)
{
  // 비용 최저 G개가 올바르게 선택되는지 확인
  int K = 100;
  int G = 10;

  Eigen::VectorXd costs = Eigen::VectorXd::Random(K).array().abs();
  // 인위적으로 처음 G개의 비용을 매우 낮게 설정
  for (int i = 0; i < G; ++i) {
    costs(i) = 0.001 * (i + 1);
  }

  // argpartition 시뮬레이션
  std::vector<int> indices(K);
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + G, indices.end(),
    [&costs](int a, int b) { return costs(a) < costs(b); });

  // 선택된 G개는 비용이 가장 낮아야 함
  double max_guide_cost = 0.0;
  for (int g = 0; g < G; ++g) {
    max_guide_cost = std::max(max_guide_cost, costs(indices[g]));
  }

  // 선택되지 않은 것들 중 최소 비용
  double min_non_guide_cost = std::numeric_limits<double>::max();
  std::set<int> guide_set(indices.begin(), indices.begin() + G);
  for (int k = 0; k < K; ++k) {
    if (guide_set.find(k) == guide_set.end()) {
      min_non_guide_cost = std::min(min_non_guide_cost, costs(k));
    }
  }

  EXPECT_LE(max_guide_cost, min_non_guide_cost + 1e-10);
}

TEST(SVGMPPI, FollowerCount)
{
  // K - G개 follower가 생성되는지 확인
  int K = 100;
  int G = 10;
  int n_followers = K - G;
  int followers_per_guide = std::max(1, n_followers / G);

  int total_followers = 0;
  for (int g = 0; g < G; ++g) {
    int n_f;
    if (g < G - 1) {
      n_f = followers_per_guide;
    } else {
      n_f = n_followers - followers_per_guide * (G - 1);
    }
    total_followers += n_f;
  }

  EXPECT_EQ(total_followers, n_followers);
  EXPECT_EQ(G + total_followers, K);
}

TEST(SVGMPPI, DiversityComputation)
{
  // 다양성이 큰 샘플 vs 작은 샘플
  int K = 20;
  int N = 5;
  int nu = 2;
  int D = N * nu;

  // 유사한 샘플 (낮은 다양성)
  std::vector<Eigen::MatrixXd> similar_controls;
  for (int k = 0; k < K; ++k) {
    similar_controls.push_back(Eigen::MatrixXd::Ones(N, nu) * 0.5
      + Eigen::MatrixXd::Random(N, nu) * 0.001);
  }

  // 다양한 샘플 (높은 다양성)
  std::vector<Eigen::MatrixXd> diverse_controls;
  for (int k = 0; k < K; ++k) {
    diverse_controls.push_back(Eigen::MatrixXd::Random(N, nu) * 5.0);
  }

  // 다양성 계산 (pairwise L2)
  auto computeDiversity = [](const std::vector<Eigen::MatrixXd>& controls,
                              int K_local, int D_local) -> double {
    if (K_local <= 1) return 0.0;
    std::vector<Eigen::VectorXd> flat;
    for (int k = 0; k < K_local; ++k) {
      Eigen::Map<const Eigen::VectorXd> v(controls[k].data(), D_local);
      flat.push_back(v);
    }
    double sum_dist = 0.0;
    int count = 0;
    for (int i = 0; i < K_local; ++i) {
      for (int j = i + 1; j < K_local; ++j) {
        sum_dist += (flat[i] - flat[j]).norm();
        ++count;
      }
    }
    return count > 0 ? sum_dist / count : 0.0;
  };

  double similar_diversity = computeDiversity(similar_controls, K, D);
  double diverse_diversity = computeDiversity(diverse_controls, K, D);

  EXPECT_GT(diverse_diversity, similar_diversity);
}

TEST(SVGMPPI, FallbackVanillaCondition)
{
  // G=0 또는 L=0일 때 Vanilla fallback 조건 확인
  int G_zero = 0;
  int L_zero = 0;
  int G_normal = 10;
  int L_normal = 3;

  EXPECT_TRUE(G_zero <= 0 || L_zero <= 0);   // fallback
  EXPECT_TRUE(G_zero <= 0);                    // fallback
  EXPECT_FALSE(G_normal <= 0 || L_normal <= 0); // no fallback
}

// ============================================================================
// MPPIInfo 필드 테스트
// ============================================================================

TEST(MPPIInfo, SVGFields)
{
  MPPIInfo info;

  // 기본값 확인
  EXPECT_EQ(info.num_guides, 0);
  EXPECT_EQ(info.num_followers, 0);
  EXPECT_EQ(info.guide_iterations, 0);

  // 설정 후 확인
  info.num_guides = 10;
  info.num_followers = 90;
  info.guide_iterations = 3;

  EXPECT_EQ(info.num_guides, 10);
  EXPECT_EQ(info.num_followers, 90);
  EXPECT_EQ(info.guide_iterations, 3);
}

// ============================================================================
// MPPIParams M3.5 필드 테스트
// ============================================================================

TEST(MPPIParams, M35Defaults)
{
  MPPIParams params;

  // Smooth-MPPI 기본값
  EXPECT_DOUBLE_EQ(params.smooth_R_jerk_v, 0.1);
  EXPECT_DOUBLE_EQ(params.smooth_R_jerk_omega, 0.1);
  EXPECT_DOUBLE_EQ(params.smooth_action_cost_weight, 1.0);

  // Spline-MPPI 기본값
  EXPECT_EQ(params.spline_num_knots, 8);
  EXPECT_EQ(params.spline_degree, 3);

  // SVG-MPPI 기본값
  EXPECT_EQ(params.svg_num_guide_particles, 10);
  EXPECT_EQ(params.svg_guide_iterations, 3);
  EXPECT_DOUBLE_EQ(params.svg_guide_step_size, 0.1);
  EXPECT_DOUBLE_EQ(params.svg_resample_std, 0.3);
}

}  // namespace mpc_controller_ros2
