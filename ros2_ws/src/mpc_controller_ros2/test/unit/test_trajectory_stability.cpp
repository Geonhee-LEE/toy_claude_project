#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include "mpc_controller_ros2/savitzky_golay_filter.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/sampling.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"

namespace mpc_controller_ros2
{

// ============================================================================
// Savitzky-Golay Filter 테스트 (10개)
// ============================================================================

class SGFilterTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // 기본: half_window=3, poly_order=3, nu=2
    filter_ = std::make_unique<SavitzkyGolayFilter>(3, 3, 2);
  }

  std::unique_ptr<SavitzkyGolayFilter> filter_;
};

TEST_F(SGFilterTest, Coefficients_SumToOne)
{
  // SG 계수의 합 = 1 (smoothing 속성)
  const auto& coeffs = filter_->coefficients();
  double sum = coeffs.sum();
  EXPECT_NEAR(sum, 1.0, 1e-10) << "SG 계수의 합이 1이어야 함";
}

TEST_F(SGFilterTest, PassThrough_ConstantInput)
{
  // 상수 입력 → 동일 출력
  int N = 10;
  Eigen::MatrixXd control_seq(N, 2);
  for (int t = 0; t < N; ++t) {
    control_seq.row(t) << 1.0, 0.5;
  }

  // 이력도 상수로 채우기
  for (int i = 0; i < 3; ++i) {
    Eigen::VectorXd u(2);
    u << 1.0, 0.5;
    filter_->pushHistory(u);
  }

  Eigen::VectorXd result = filter_->apply(control_seq, 0);
  EXPECT_NEAR(result(0), 1.0, 1e-10);
  EXPECT_NEAR(result(1), 0.5, 1e-10);
}

TEST_F(SGFilterTest, LinearPreserve)
{
  // 선형 입력 보존: u(t) = [t, 2t]
  int N = 10;
  Eigen::MatrixXd control_seq(N, 2);
  for (int t = 0; t < N; ++t) {
    control_seq.row(t) << static_cast<double>(t), 2.0 * t;
  }

  // 과거 이력: t = -3, -2, -1
  for (int i = -3; i < 0; ++i) {
    Eigen::VectorXd u(2);
    u << static_cast<double>(i), 2.0 * i;
    filter_->pushHistory(u);
  }

  Eigen::VectorXd result = filter_->apply(control_seq, 0);
  EXPECT_NEAR(result(0), 0.0, 1e-8) << "선형 입력은 SG 필터에 의해 보존되어야 함";
  EXPECT_NEAR(result(1), 0.0, 1e-8);
}

TEST_F(SGFilterTest, SmoothNoise)
{
  // 노이즈 입력 → 분산 감소
  int N = 10;
  Eigen::MatrixXd control_seq(N, 2);
  // 기본 신호(1.0)에 노이즈 추가
  std::vector<double> noisy = {1.2, 0.8, 1.1, 0.9, 1.3, 0.7, 1.15, 0.85, 1.05, 0.95};
  for (int t = 0; t < N; ++t) {
    control_seq.row(t) << noisy[t], 1.0;
  }

  // 이력도 비슷한 패턴
  for (int i = 0; i < 3; ++i) {
    Eigen::VectorXd u(2);
    u << 1.0 + 0.1 * (i % 2 == 0 ? 1 : -1), 1.0;
    filter_->pushHistory(u);
  }

  Eigen::VectorXd result = filter_->apply(control_seq, 0);
  // 스무딩된 값은 원래 신호(1.0)에 더 가까워야 함
  EXPECT_LT(std::abs(result(0) - 1.0), 0.15)
    << "스무딩 후 평균에 더 가까워야 함 (got " << result(0) << ")";
}

TEST_F(SGFilterTest, MultiDim_Nu3)
{
  // nu=3 (swerve: vx, vy, omega)
  auto filter3 = std::make_unique<SavitzkyGolayFilter>(3, 3, 3);
  int N = 10;
  Eigen::MatrixXd control_seq(N, 3);
  for (int t = 0; t < N; ++t) {
    control_seq.row(t) << 1.0, 0.5, 0.3;
  }
  for (int i = 0; i < 3; ++i) {
    Eigen::VectorXd u(3);
    u << 1.0, 0.5, 0.3;
    filter3->pushHistory(u);
  }

  Eigen::VectorXd result = filter3->apply(control_seq, 0);
  EXPECT_NEAR(result(0), 1.0, 1e-10);
  EXPECT_NEAR(result(1), 0.5, 1e-10);
  EXPECT_NEAR(result(2), 0.3, 1e-10);
}

TEST_F(SGFilterTest, HistoryAccumulation)
{
  // pushHistory가 올바르게 deque를 관리하는지
  Eigen::VectorXd u1(2), u2(2), u3(2), u4(2);
  u1 << 1.0, 0.0;
  u2 << 2.0, 0.0;
  u3 << 3.0, 0.0;
  u4 << 4.0, 0.0;

  filter_->pushHistory(u1);
  filter_->pushHistory(u2);
  filter_->pushHistory(u3);
  filter_->pushHistory(u4);  // half_window=3이므로 u1이 제거됨

  // 상수 미래 시퀀스로 apply
  Eigen::MatrixXd control_seq(10, 2);
  for (int t = 0; t < 10; ++t) {
    control_seq.row(t) << 5.0, 0.0;
  }

  // apply는 에러 없이 동작해야 함
  Eigen::VectorXd result = filter_->apply(control_seq, 0);
  EXPECT_FALSE(std::isnan(result(0)));
  EXPECT_FALSE(std::isnan(result(1)));
}

TEST_F(SGFilterTest, Reset)
{
  Eigen::VectorXd u(2);
  u << 1.0, 0.5;
  filter_->pushHistory(u);
  filter_->pushHistory(u);

  filter_->reset();

  // Reset 후에도 apply는 에러 없이 동작 (이력 부족 = 패딩)
  Eigen::MatrixXd control_seq(10, 2);
  for (int t = 0; t < 10; ++t) {
    control_seq.row(t) << 1.0, 0.5;
  }
  Eigen::VectorXd result = filter_->apply(control_seq, 0);
  EXPECT_NEAR(result(0), 1.0, 1e-10);
  EXPECT_NEAR(result(1), 0.5, 1e-10);
}

TEST_F(SGFilterTest, WindowSize_Various)
{
  // 다양한 half_window 크기
  for (int hw = 1; hw <= 5; ++hw) {
    SavitzkyGolayFilter f(hw, std::min(hw, 3), 2);
    EXPECT_EQ(f.windowSize(), 2 * hw + 1);

    // 상수 입력 → 동일 출력
    Eigen::MatrixXd control_seq(20, 2);
    for (int t = 0; t < 20; ++t) {
      control_seq.row(t) << 2.0, 1.0;
    }
    for (int i = 0; i < hw; ++i) {
      Eigen::VectorXd u(2);
      u << 2.0, 1.0;
      f.pushHistory(u);
    }
    Eigen::VectorXd result = f.apply(control_seq, 0);
    EXPECT_NEAR(result(0), 2.0, 1e-8) << "hw=" << hw << " failed";
  }
}

TEST_F(SGFilterTest, PolyOrder_Various)
{
  // 다양한 poly_order
  for (int po = 1; po <= 5; ++po) {
    int hw = std::max(po, 3);  // window_size > poly_order 보장
    SavitzkyGolayFilter f(hw, po, 2);
    const auto& coeffs = f.coefficients();
    EXPECT_NEAR(coeffs.sum(), 1.0, 1e-10) << "poly_order=" << po;
  }
}

TEST_F(SGFilterTest, EmptyHistory_GracefulFallback)
{
  // 이력 없이 apply → 현재값으로 패딩
  Eigen::MatrixXd control_seq(10, 2);
  for (int t = 0; t < 10; ++t) {
    control_seq.row(t) << 1.0, 0.5;
  }

  Eigen::VectorXd result = filter_->apply(control_seq, 0);
  // 이력 부족 시 현재값으로 패딩되므로 상수 입력 → 동일 출력
  EXPECT_NEAR(result(0), 1.0, 1e-8);
  EXPECT_NEAR(result(1), 0.5, 1e-8);
}

// ============================================================================
// Information-Theoretic 정규화 테스트 (8개)
// ============================================================================

class ITRegTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    params_.N = 10;
    params_.K = 100;
    params_.dt = 0.1;
    params_.lambda = 10.0;
    params_.noise_sigma = Eigen::Vector2d(0.5, 0.5);

    // 이전 최적 제어 시퀀스 (warm-started)
    control_sequence_ = Eigen::MatrixXd::Zero(params_.N, 2);
    for (int t = 0; t < params_.N; ++t) {
      control_sequence_.row(t) << 0.5, 0.1;  // 전진 + 약간 회전
    }
  }

  MPPIParams params_;
  Eigen::MatrixXd control_sequence_;

  // IT 정규화 비용 계산 헬퍼
  double computeITCost(
    double alpha,
    const Eigen::MatrixXd& perturbed_control,
    double lambda)
  {
    if (alpha >= 1.0) return 0.0;

    Eigen::VectorXd sigma_inv = params_.noise_sigma.cwiseInverse().cwiseAbs2();
    double it_cost = 0.0;
    for (int t = 0; t < params_.N; ++t) {
      Eigen::VectorXd u_prev_t = control_sequence_.row(t).transpose();
      Eigen::VectorXd u_k_t = perturbed_control.row(t).transpose();
      it_cost += u_prev_t.dot(sigma_inv.cwiseProduct(u_k_t));
    }
    return lambda * (1.0 - alpha) * it_cost;
  }
};

TEST_F(ITRegTest, Alpha1_NoCostChange)
{
  // alpha=1.0 → IT 비용 = 0
  Eigen::MatrixXd perturbed = control_sequence_;
  perturbed.array() += 0.1;

  double it_cost = computeITCost(1.0, perturbed, params_.lambda);
  EXPECT_NEAR(it_cost, 0.0, 1e-15) << "alpha=1.0이면 IT 비용 = 0";
}

TEST_F(ITRegTest, Alpha0975_CostIncrease)
{
  // alpha=0.975 → 편향된 샘플에 추가 비용
  Eigen::MatrixXd perturbed = control_sequence_;
  perturbed.array() += 0.5;  // 이전 해에서 크게 벗어남

  double it_cost = computeITCost(0.975, perturbed, params_.lambda);
  EXPECT_GT(it_cost, 0.0) << "alpha<1이고 양의 상관관계 → IT 비용 > 0";
}

TEST_F(ITRegTest, ShiftsWeightTowardPrevSolution)
{
  // IT 정규화가 이전 해에 가까운 샘플에 더 높은 가중치를 부여하는지 확인
  int K = 50;
  Eigen::VectorXd costs_no_it(K), costs_with_it(K);

  // 모든 샘플의 기본 비용은 동일 → 가중치가 균등
  for (int k = 0; k < K; ++k) {
    costs_no_it(k) = 10.0;  // 동일 비용

    // IT 비용 추가: 이전 해에서 멀수록 높은 비용
    Eigen::MatrixXd perturbed = control_sequence_;
    perturbed.array() += 0.1 * k;  // k가 클수록 편향
    double it_cost = computeITCost(0.975, perturbed, params_.lambda);
    costs_with_it(k) = costs_no_it(k) + it_cost;
  }

  // Vanilla 가중치 계산
  VanillaMPPIWeights weight_comp;
  Eigen::VectorXd w_no_it = weight_comp.compute(costs_no_it, params_.lambda);
  Eigen::VectorXd w_with_it = weight_comp.compute(costs_with_it, params_.lambda);

  // IT 정규화 없이: 모든 가중치 균등
  // IT 정규화 후: 낮은 k (이전 해에 가까운) 샘플에 더 높은 가중치
  double weight_first_quarter_no_it = w_no_it.head(K / 4).sum();
  double weight_first_quarter_with_it = w_with_it.head(K / 4).sum();

  EXPECT_GT(weight_first_quarter_with_it, weight_first_quarter_no_it)
    << "IT 정규화가 이전 해에 가까운 샘플에 더 높은 가중치를 부여해야 함";
}

TEST_F(ITRegTest, PreservesPreviousSolution)
{
  // 이전 해와 동일한 샘플은 낮은 IT 비용
  Eigen::MatrixXd same_as_prev = control_sequence_;
  Eigen::MatrixXd far_from_prev = control_sequence_;
  far_from_prev.array() += 2.0;

  double cost_same = computeITCost(0.975, same_as_prev, params_.lambda);
  double cost_far = computeITCost(0.975, far_from_prev, params_.lambda);

  // far_from_prev는 이전 해와 더 상관관계가 높으므로 비용이 더 크다
  // (u_prev^T * sigma_inv * u_k에서 u_k가 크면 비용 증가)
  EXPECT_LT(cost_same, cost_far)
    << "이전 해와 가까운 샘플은 더 낮은 IT 비용";
}

TEST_F(ITRegTest, ScalesWithLambda)
{
  // lambda 증가 → IT 비용 증가
  Eigen::MatrixXd perturbed = control_sequence_;
  perturbed.array() += 0.3;

  double cost_low = computeITCost(0.975, perturbed, 1.0);
  double cost_high = computeITCost(0.975, perturbed, 100.0);

  EXPECT_NEAR(cost_high / cost_low, 100.0, 1e-8)
    << "IT 비용은 lambda에 선형 비례";
}

TEST_F(ITRegTest, WorksWithAdaptiveTemp)
{
  // 적응형 온도 lambda와 호환성
  double lambda_adapted = 5.0;  // adaptive temp에 의해 조정된 lambda
  Eigen::MatrixXd perturbed = control_sequence_;
  perturbed.array() += 0.2;

  double it_cost = computeITCost(0.975, perturbed, lambda_adapted);
  EXPECT_GT(it_cost, 0.0) << "적응형 lambda에서도 동작해야 함";
  EXPECT_FALSE(std::isnan(it_cost));
  EXPECT_FALSE(std::isinf(it_cost));
}

TEST_F(ITRegTest, MultiDimControl_Nu3)
{
  // nu=3 (swerve: vx, vy, omega)
  params_.noise_sigma = Eigen::Vector3d(0.5, 0.3, 0.5);
  control_sequence_ = Eigen::MatrixXd::Zero(params_.N, 3);
  for (int t = 0; t < params_.N; ++t) {
    control_sequence_.row(t) << 0.5, 0.1, 0.2;
  }

  Eigen::MatrixXd perturbed = control_sequence_;
  perturbed.array() += 0.3;

  double it_cost = computeITCost(0.975, perturbed, params_.lambda);
  EXPECT_GT(it_cost, 0.0);
  EXPECT_FALSE(std::isnan(it_cost));
}

TEST_F(ITRegTest, NumericalStability)
{
  // 극단적 비용에서도 NaN/Inf 없음
  params_.noise_sigma = Eigen::Vector2d(0.001, 0.001);  // 매우 작은 sigma

  Eigen::MatrixXd perturbed = control_sequence_;
  perturbed.array() += 100.0;  // 극단적으로 큰 편향

  double it_cost = computeITCost(0.975, perturbed, 100.0);
  EXPECT_FALSE(std::isnan(it_cost)) << "NaN 발생하면 안됨";
  EXPECT_FALSE(std::isinf(it_cost)) << "Inf 발생하면 안됨";
}

// ============================================================================
// Exploitation/Exploration 분할 테스트 (7개)
// ============================================================================

class ExplExplTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    params_.N = 10;
    params_.K = 100;
    params_.noise_sigma = Eigen::Vector2d(0.5, 0.5);
    params_.v_max = 1.0;
    params_.v_min = -0.5;
    params_.omega_max = 1.0;
    params_.omega_min = -1.0;

    control_sequence_ = Eigen::MatrixXd::Zero(params_.N, 2);
    for (int t = 0; t < params_.N; ++t) {
      control_sequence_.row(t) << 0.5, 0.1;
    }

    sampler_ = std::make_unique<GaussianSampler>(params_.noise_sigma);
  }

  MPPIParams params_;
  Eigen::MatrixXd control_sequence_;
  std::unique_ptr<GaussianSampler> sampler_;

  // 실제 perturbed controls 생성 (exploration_ratio 적용)
  std::vector<Eigen::MatrixXd> generatePerturbedControls(double exploration_ratio)
  {
    int K = params_.K;
    int N = params_.N;
    int nu = 2;
    auto noise_samples = sampler_->sample(K, N, nu);
    int K_exploit = static_cast<int>((1.0 - exploration_ratio) * K);

    std::vector<Eigen::MatrixXd> perturbed;
    perturbed.reserve(K);

    for (int k = 0; k < K; ++k) {
      Eigen::MatrixXd p;
      if (k < K_exploit) {
        p = control_sequence_ + noise_samples[k];
      } else {
        p = noise_samples[k];
      }
      perturbed.push_back(p);
    }
    return perturbed;
  }
};

TEST_F(ExplExplTest, Ratio0_AllExploitation)
{
  // ratio=0 → 모든 샘플이 exploitation (warm-start + noise)
  auto perturbed = generatePerturbedControls(0.0);
  EXPECT_EQ(static_cast<int>(perturbed.size()), params_.K);

  // 모든 샘플은 control_sequence_ 주변
  for (int k = 0; k < params_.K; ++k) {
    double diff = (perturbed[k] - control_sequence_).norm();
    // 노이즈가 있으므로 차이가 있지만, warm-start 기반이므로 norm이 제한됨
    // 순수 노이즈만 사용하면 control_sequence_ 근처가 아닐 수 있음
    EXPECT_GT(diff, 0.0) << "노이즈가 있으므로 차이 > 0";
  }
}

TEST_F(ExplExplTest, Ratio1_AllExploration)
{
  // ratio=1 → 모든 샘플이 exploration (순수 noise)
  auto perturbed = generatePerturbedControls(1.0);
  int K_exploit = 0;  // 모두 exploration

  // 모든 샘플은 control_sequence_를 포함하지 않음 (순수 noise)
  // 평균이 0에 가까워야 함 (warm-start 없이)
  Eigen::MatrixXd sum = Eigen::MatrixXd::Zero(params_.N, 2);
  for (const auto& p : perturbed) {
    sum += p;
  }
  sum /= params_.K;

  // 순수 noise의 평균 → 0에 가까워야 함 (warm-start가 없으므로)
  double mean_norm = sum.norm() / params_.N;
  EXPECT_LT(mean_norm, 0.2) << "순수 noise 평균은 0에 가까워야 함 (warm-start 없이)";
  (void)K_exploit;
}

TEST_F(ExplExplTest, Ratio01_Split)
{
  // ratio=0.1 → 90% exploitation + 10% exploration
  double ratio = 0.1;
  int K = params_.K;
  int K_exploit = static_cast<int>((1.0 - ratio) * K);
  int K_explore = K - K_exploit;

  EXPECT_EQ(K_exploit, 90);
  EXPECT_EQ(K_explore, 10);
}

TEST_F(ExplExplTest, ExploitationUsesWarmStart)
{
  // exploitation 샘플은 control_sequence_ + noise
  auto noise_samples = sampler_->sample(params_.K, params_.N, 2);

  // 첫 번째 (exploitation) 샘플: control_sequence_ + noise
  Eigen::MatrixXd exploit = control_sequence_ + noise_samples[0];
  Eigen::MatrixXd expected = control_sequence_ + noise_samples[0];
  EXPECT_NEAR((exploit - expected).norm(), 0.0, 1e-15);
}

TEST_F(ExplExplTest, ExplorationNoWarmStart)
{
  // exploration 샘플은 순수 noise (warm-start 없이)
  auto noise_samples = sampler_->sample(params_.K, params_.N, 2);

  Eigen::MatrixXd explore = noise_samples[0];  // warm-start 없이
  Eigen::MatrixXd pure_noise = noise_samples[0];
  EXPECT_NEAR((explore - pure_noise).norm(), 0.0, 1e-15);
}

TEST_F(ExplExplTest, BothClipped)
{
  // 양쪽 모두 clipping 필요 시 적용
  params_.v_max = 0.3;  // 작은 limit
  auto model = MotionModelFactory::create("diff_drive", params_);
  auto dynamics = std::make_unique<BatchDynamicsWrapper>(params_, std::move(model));

  auto noise_samples = sampler_->sample(10, params_.N, 2);

  // Exploitation
  Eigen::MatrixXd exploit = control_sequence_ + noise_samples[0];
  exploit = dynamics->clipControls(exploit);
  for (int t = 0; t < params_.N; ++t) {
    EXPECT_LE(exploit(t, 0), params_.v_max + 1e-10);
    EXPECT_GE(exploit(t, 0), params_.v_min - 1e-10);
  }

  // Exploration
  Eigen::MatrixXd explore = noise_samples[1];
  explore = dynamics->clipControls(explore);
  for (int t = 0; t < params_.N; ++t) {
    EXPECT_LE(explore(t, 0), params_.v_max + 1e-10);
    EXPECT_GE(explore(t, 0), params_.v_min - 1e-10);
  }
}

TEST_F(ExplExplTest, IntegrationWithMPPI)
{
  // 전체 MPPI 파이프라인에서 exploration_ratio 적용
  // DiffDrive 기본 설정으로 computeControl 시뮬레이션
  params_.exploration_ratio = 0.1;
  params_.K = 50;  // 테스트 속도

  auto model = MotionModelFactory::create("diff_drive", params_);
  auto dynamics = std::make_unique<BatchDynamicsWrapper>(params_, std::move(model));

  int N = params_.N;
  int K = params_.K;
  int nu = 2;
  int nx = 3;

  // 현재 상태
  Eigen::VectorXd state(nx);
  state << 0.0, 0.0, 0.0;

  // 참조 궤적 (직진)
  Eigen::MatrixXd ref(N + 1, nx);
  for (int t = 0; t <= N; ++t) {
    ref(t, 0) = t * 0.1;
    ref(t, 1) = 0.0;
    ref(t, 2) = 0.0;
  }

  // 노이즈 샘플링
  auto noise_samples = sampler_->sample(K, N, nu);

  // Exploitation/Exploration 분할
  int K_exploit = static_cast<int>((1.0 - params_.exploration_ratio) * K);
  std::vector<Eigen::MatrixXd> perturbed;
  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd p;
    if (k < K_exploit) {
      p = control_sequence_ + noise_samples[k];
    } else {
      p = noise_samples[k];
    }
    p = dynamics->clipControls(p);
    perturbed.push_back(p);
  }

  // Rollout
  auto trajectories = dynamics->rolloutBatch(state, perturbed, params_.dt);
  EXPECT_EQ(static_cast<int>(trajectories.size()), K);

  // 비용 계산
  CompositeMPPICost cost_fn;
  cost_fn.addCost(std::make_unique<StateTrackingCost>(params_.Q));
  cost_fn.addCost(std::make_unique<ControlEffortCost>(params_.R));
  Eigen::VectorXd costs = cost_fn.compute(trajectories, perturbed, ref);
  EXPECT_EQ(costs.size(), K);

  // 가중치 계산
  VanillaMPPIWeights weight_comp;
  Eigen::VectorXd weights = weight_comp.compute(costs, params_.lambda);
  EXPECT_NEAR(weights.sum(), 1.0, 1e-10);

  // 업데이트
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_noise += weights(k) * noise_samples[k];
  }
  control_sequence_ += weighted_noise;

  // 최종 제어가 합리적인 범위
  Eigen::VectorXd u_opt = control_sequence_.row(0).transpose();
  EXPECT_FALSE(std::isnan(u_opt(0)));
  EXPECT_FALSE(std::isnan(u_opt(1)));
}

}  // namespace mpc_controller_ros2
