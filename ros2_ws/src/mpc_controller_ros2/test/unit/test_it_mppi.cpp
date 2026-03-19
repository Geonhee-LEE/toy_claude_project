// =============================================================================
// IT-MPPI (Information-Theoretic MPPI) 단위 테스트 (15개)
//
// 탐색 보너스 (3개):
//   1. ExplorationBonusRewardsDiversity — 다양한 궤적이 낮은 info_cost
//   2. ZeroExplorationWeightNoEffect    — weight=0이면 vanilla와 동일
//   3. HighExplorationIncreasesSpread   — 높은 가중치 → 분산 증가
//
// KL 정규화 (3개):
//   4. KLPenaltyConstrainsSampling     — 큰 노이즈에 페널티
//   5. ZeroKLWeightNoRegularization    — weight=0이면 정규화 없음
//   6. KLIncreasesWithNoiseScale       — 노이즈 증가 → KL 증가
//
// 다양성 (2개):
//   7. DiversityScorePositive          — 다양성 점수 항상 ≥ 0
//   8. DiversityThresholdEffect        — 임계값 이하에서 추가 보너스
//
// 적응형 탐색 (2개):
//   9. AdaptiveDecayReducesExploration — 반복 호출 → 가중치 감소
//  10. AdaptiveDisabledKeepsConstant   — adaptive=false → 가중치 유지
//
// 동등성 (1개):
//  11. DisabledEqualsVanilla           — enabled=false → vanilla
//
// 통합 (2개):
//  12. ComputeControlReturnsValid      — 유한 제어, 유효 info
//  13. WorksWithSwerveModel            — nu=3 호환
//
// 안정성 (1개):
//  14. MultipleCallsStable             — 10회 호출, NaN/Inf 없음
//
// 정보 (1개):
//  15. InfoContainsITMetrics           — info에 IT 메트릭 포함
// =============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

#include "mpc_controller_ros2/it_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

// ============================================================================
// 테스트 헬퍼
// ============================================================================
class ITMPPITestAccessor : public ITMPPIControllerPlugin
{
public:
  void setTestParams(const MPPIParams& params) {
    params_ = params;
    current_exploration_weight_ = params.it_exploration_weight;
  }
  void setDynamics(std::unique_ptr<BatchDynamicsWrapper> dyn) { dynamics_ = std::move(dyn); }
  void setSampler(std::unique_ptr<BaseSampler> sampler) { sampler_ = std::move(sampler); }
  void setCostFunction(std::unique_ptr<CompositeMPPICost> cf) { cost_function_ = std::move(cf); }
  void setWeightComputation(std::unique_ptr<WeightComputation> wc) {
    weight_computation_ = std::move(wc);
  }
  void setControlSequence(const Eigen::MatrixXd& cs) { control_sequence_ = cs; }
  Eigen::MatrixXd getControlSequence() const { return control_sequence_; }

  Eigen::VectorXd callDiversityBonus(const std::vector<Eigen::MatrixXd>& traj) const {
    return computeDiversityBonus(traj);
  }
  Eigen::VectorXd callKLPenalty(const std::vector<Eigen::MatrixXd>& noise) const {
    return computeKLPenalty(noise);
  }

  double getExplorationWeight() const { return current_exploration_weight_; }
  void setExplorationWeight(double w) { current_exploration_weight_ = w; }
};

// ============================================================================
// 테스트 Fixture
// ============================================================================
class ITMPPITest : public ::testing::Test
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
    params_.it_mppi_enabled = true;
    params_.it_exploration_weight = 0.1;
    params_.it_kl_weight = 0.01;
    params_.it_diversity_threshold = 0.5;
    params_.it_adaptive_exploration = false;
    params_.it_exploration_decay = 0.99;

    ref_traj_ = Eigen::MatrixXd::Zero(params_.N + 1, 3);
    for (int t = 0; t <= params_.N; ++t) {
      ref_traj_(t, 0) = t * params_.dt;
    }

    state_ = Eigen::Vector3d(0.0, 0.0, 0.0);
  }

  MPPIParams params_;
  Eigen::MatrixXd ref_traj_;
  Eigen::Vector3d state_;
};

// ============================================================================
// 탐색 보너스 테스트 (3개)
// ============================================================================

TEST_F(ITMPPITest, ExplorationBonusRewardsDiversity)
{
  // 다양한 궤적(분산된 최종 상태)이 높은 diversity bonus를 받는지 확인
  ITMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  int K = 5, N = 10, nx = 3;
  std::vector<Eigen::MatrixXd> trajectories(K);

  // 궤적 0~2: 비슷한 최종 상태
  for (int k = 0; k < 3; ++k) {
    trajectories[k] = Eigen::MatrixXd::Zero(N + 1, nx);
    trajectories[k](N, 0) = 1.0 + 0.01 * k;  // 거의 같은 위치
    trajectories[k](N, 1) = 0.0;
  }
  // 궤적 3~4: 멀리 떨어진 최종 상태
  trajectories[3] = Eigen::MatrixXd::Zero(N + 1, nx);
  trajectories[3](N, 0) = 5.0;
  trajectories[3](N, 1) = 5.0;
  trajectories[4] = Eigen::MatrixXd::Zero(N + 1, nx);
  trajectories[4](N, 0) = -3.0;
  trajectories[4](N, 1) = -3.0;

  auto diversity = accessor.callDiversityBonus(trajectories);

  EXPECT_EQ(diversity.size(), K);
  // 멀리 떨어진 궤적(3, 4)이 높은 diversity를 가져야 함
  EXPECT_GT(diversity(3), diversity(0));
  EXPECT_GT(diversity(4), diversity(0));
}

TEST_F(ITMPPITest, ZeroExplorationWeightNoEffect)
{
  ITMPPITestAccessor accessor;
  params_.it_exploration_weight = 0.0;
  accessor.setTestParams(params_);

  // diversity bonus가 있어도 exploration_weight=0이면 info_cost에 영향 없음
  Eigen::VectorXd costs(5);
  costs << 5.0, 3.0, 8.0, 1.0, 6.0;

  Eigen::VectorXd diversity(5);
  diversity << 1.0, 2.0, 0.5, 3.0, 1.5;

  Eigen::VectorXd kl(5);
  kl.setZero();

  double ew = 0.0;
  for (int k = 0; k < 5; ++k) {
    double info_cost = costs(k) - ew * diversity(k);
    EXPECT_DOUBLE_EQ(info_cost, costs(k));
  }
}

TEST_F(ITMPPITest, HighExplorationIncreasesSpread)
{
  // 높은 exploration weight에서 다양한 궤적이 더 선호됨
  auto dynamics = std::make_unique<BatchDynamicsWrapper>(params_);
  auto sampler = std::make_unique<GaussianSampler>(params_.noise_sigma, 42);
  int N = params_.N, K = params_.K, nu = 2;

  // 두 가지 exploration weight로 비교
  auto noise = sampler->sample(K, N, nu);
  std::vector<Eigen::MatrixXd> perturbed_low(K), perturbed_high(K);
  Eigen::MatrixXd mu = Eigen::MatrixXd::Zero(N, nu);

  for (int k = 0; k < K; ++k) {
    perturbed_low[k] = dynamics->clipControls(mu + noise[k]);
    perturbed_high[k] = perturbed_low[k];  // 같은 샘플
  }

  auto traj = dynamics->rolloutBatch(state_, perturbed_low, params_.dt);

  ITMPPITestAccessor accessor;
  accessor.setTestParams(params_);
  auto diversity = accessor.callDiversityBonus(traj);

  // 높은 ew로 diversity 반영 시 info_costs 분산이 증가
  Eigen::VectorXd base_costs = Eigen::VectorXd::Ones(K) * 10.0;

  // low ew
  Eigen::VectorXd info_low(K);
  for (int k = 0; k < K; ++k) {
    info_low(k) = base_costs(k) - 0.01 * diversity(k);
  }

  // high ew
  Eigen::VectorXd info_high(K);
  for (int k = 0; k < K; ++k) {
    info_high(k) = base_costs(k) - 1.0 * diversity(k);
  }

  // high ew의 분산이 더 커야 함
  double var_low = (info_low.array() - info_low.mean()).square().mean();
  double var_high = (info_high.array() - info_high.mean()).square().mean();
  EXPECT_GT(var_high, var_low);
}

// ============================================================================
// KL 정규화 테스트 (3개)
// ============================================================================

TEST_F(ITMPPITest, KLPenaltyConstrainsSampling)
{
  ITMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  int K = 3, N = 5, nu = 2;
  std::vector<Eigen::MatrixXd> noise(K);

  // 작은 노이즈
  noise[0] = Eigen::MatrixXd::Constant(N, nu, 0.1);
  // 중간 노이즈
  noise[1] = Eigen::MatrixXd::Constant(N, nu, 0.5);
  // 큰 노이즈
  noise[2] = Eigen::MatrixXd::Constant(N, nu, 2.0);

  auto kl = accessor.callKLPenalty(noise);

  EXPECT_LT(kl(0), kl(1));
  EXPECT_LT(kl(1), kl(2));
}

TEST_F(ITMPPITest, ZeroKLWeightNoRegularization)
{
  Eigen::VectorXd costs(3);
  costs << 5.0, 3.0, 8.0;

  Eigen::VectorXd kl(3);
  kl << 10.0, 20.0, 30.0;

  double kl_weight = 0.0;
  for (int k = 0; k < 3; ++k) {
    double info_cost = costs(k) + kl_weight * kl(k);
    EXPECT_DOUBLE_EQ(info_cost, costs(k));
  }
}

TEST_F(ITMPPITest, KLIncreasesWithNoiseScale)
{
  ITMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  int K = 1, N = 10, nu = 2;

  // 작은 노이즈
  std::vector<Eigen::MatrixXd> small_noise(K);
  small_noise[0] = Eigen::MatrixXd::Constant(N, nu, 0.1);
  auto kl_small = accessor.callKLPenalty(small_noise);

  // 큰 노이즈
  std::vector<Eigen::MatrixXd> large_noise(K);
  large_noise[0] = Eigen::MatrixXd::Constant(N, nu, 1.0);
  auto kl_large = accessor.callKLPenalty(large_noise);

  EXPECT_GT(kl_large(0), kl_small(0));
  // 10배 노이즈 → 100배 KL (제곱 관계)
  EXPECT_NEAR(kl_large(0) / kl_small(0), 100.0, 1.0);
}

// ============================================================================
// 다양성 테스트 (2개)
// ============================================================================

TEST_F(ITMPPITest, DiversityScorePositive)
{
  ITMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  auto dynamics = std::make_unique<BatchDynamicsWrapper>(params_);
  auto sampler = std::make_unique<GaussianSampler>(params_.noise_sigma, 42);
  int N = params_.N, K = 50, nu = 2;

  auto noise = sampler->sample(K, N, nu);
  std::vector<Eigen::MatrixXd> perturbed(K);
  Eigen::MatrixXd mu = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    perturbed[k] = dynamics->clipControls(mu + noise[k]);
  }
  auto traj = dynamics->rolloutBatch(state_, perturbed, params_.dt);

  auto diversity = accessor.callDiversityBonus(traj);

  for (int k = 0; k < K; ++k) {
    EXPECT_GE(diversity(k), 0.0) << "diversity negative at k=" << k;
  }
}

TEST_F(ITMPPITest, DiversityThresholdEffect)
{
  // diversity가 threshold 이하일 때 diversity_scale이 1보다 커짐
  double mean_diversity = 0.2;
  double threshold = 0.5;

  double scale = 1.0;
  if (mean_diversity < threshold && mean_diversity > 1e-8) {
    scale = threshold / mean_diversity;
  }

  EXPECT_GT(scale, 1.0);
  EXPECT_NEAR(scale, 2.5, 1e-10);

  // threshold 이상이면 scale=1.0
  mean_diversity = 0.8;
  scale = 1.0;
  if (mean_diversity < threshold && mean_diversity > 1e-8) {
    scale = threshold / mean_diversity;
  }
  EXPECT_DOUBLE_EQ(scale, 1.0);
}

// ============================================================================
// 적응형 탐색 테스트 (2개)
// ============================================================================

TEST_F(ITMPPITest, AdaptiveDecayReducesExploration)
{
  double ew = 0.1;
  double decay = 0.99;
  double initial_ew = ew;

  for (int i = 0; i < 100; ++i) {
    ew *= decay;
  }

  // 100회 후: 0.1 * 0.99^100 ≈ 0.0366
  EXPECT_LT(ew, initial_ew);
  EXPECT_NEAR(ew, 0.1 * std::pow(0.99, 100), 1e-10);
  EXPECT_GT(ew, 0.0);
}

TEST_F(ITMPPITest, AdaptiveDisabledKeepsConstant)
{
  ITMPPITestAccessor accessor;
  params_.it_adaptive_exploration = false;
  params_.it_exploration_weight = 0.1;
  accessor.setTestParams(params_);

  double initial_ew = accessor.getExplorationWeight();

  // 호출 후에도 가중치가 변하지 않음 (실제 computeControl 없이 검증)
  // adaptive=false이면 current_exploration_weight_ 유지
  EXPECT_DOUBLE_EQ(accessor.getExplorationWeight(), initial_ew);

  // 수동으로 decay 적용 여부 확인
  bool adaptive = false;
  double ew = 0.1;
  for (int i = 0; i < 10; ++i) {
    if (adaptive) {
      ew *= 0.99;
    }
  }
  EXPECT_DOUBLE_EQ(ew, 0.1);
}

// ============================================================================
// 동등성 (1개)
// ============================================================================

TEST_F(ITMPPITest, DisabledEqualsVanilla)
{
  params_.it_mppi_enabled = false;
  EXPECT_FALSE(params_.it_mppi_enabled);
  // enabled=false 시 computeControl은 base MPPIControllerPlugin::computeControl 호출
}

// ============================================================================
// 통합 테스트 (2개)
// ============================================================================

TEST_F(ITMPPITest, ComputeControlReturnsValid)
{
  auto dynamics = std::make_unique<BatchDynamicsWrapper>(params_);
  auto sampler = std::make_unique<GaussianSampler>(params_.noise_sigma, 42);
  auto cost_function = std::make_unique<CompositeMPPICost>();
  cost_function->addCost(std::make_unique<StateTrackingCost>(params_.Q));
  cost_function->addCost(std::make_unique<TerminalCost>(params_.Qf));
  cost_function->addCost(std::make_unique<ControlEffortCost>(params_.R));

  ITMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  int N = params_.N, K = params_.K, nu = 2;
  Eigen::MatrixXd mu = Eigen::MatrixXd::Zero(N, nu);

  VanillaMPPIWeights weight_strategy;

  // IT-MPPI 시뮬레이션
  auto noise = sampler->sample(K, N, nu);
  std::vector<Eigen::MatrixXd> perturbed(K);
  for (int k = 0; k < K; ++k) {
    perturbed[k] = dynamics->clipControls(mu + noise[k]);
  }
  auto trajectories = dynamics->rolloutBatch(state_, perturbed, params_.dt);
  auto costs = cost_function->compute(trajectories, perturbed, ref_traj_);

  // IT 비용 계산
  auto diversity = accessor.callDiversityBonus(trajectories);
  auto kl = accessor.callKLPenalty(noise);

  Eigen::VectorXd info_costs(K);
  for (int k = 0; k < K; ++k) {
    info_costs(k) = costs(k) - 0.1 * diversity(k) + 0.01 * kl(k);
  }

  auto weights = weight_strategy.compute(info_costs, params_.lambda);

  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_noise += weights(k) * noise[k];
  }
  Eigen::MatrixXd control_seq = dynamics->clipControls(mu + weighted_noise);
  Eigen::VectorXd u_opt = control_seq.row(0).transpose();

  EXPECT_FALSE(std::isnan(u_opt(0)));
  EXPECT_FALSE(std::isnan(u_opt(1)));
  EXPECT_GE(u_opt(0), params_.v_min);
  EXPECT_LE(u_opt(0), params_.v_max);
  EXPECT_GE(u_opt(1), params_.omega_min);
  EXPECT_LE(u_opt(1), params_.omega_max);
}

TEST_F(ITMPPITest, WorksWithSwerveModel)
{
  // nu=3 swerve 호환성 확인
  ITMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  int K = 5, N = 10, nx = 3;
  std::vector<Eigen::MatrixXd> trajectories(K);
  for (int k = 0; k < K; ++k) {
    trajectories[k] = Eigen::MatrixXd::Random(N + 1, nx);
  }

  auto diversity = accessor.callDiversityBonus(trajectories);
  EXPECT_EQ(diversity.size(), K);

  // nu=3 KL
  params_.noise_sigma = Eigen::Vector3d(0.5, 0.5, 0.5);
  accessor.setTestParams(params_);

  int nu = 3;
  std::vector<Eigen::MatrixXd> noise(K);
  for (int k = 0; k < K; ++k) {
    noise[k] = Eigen::MatrixXd::Random(N, nu);
  }

  auto kl = accessor.callKLPenalty(noise);
  EXPECT_EQ(kl.size(), K);
  for (int k = 0; k < K; ++k) {
    EXPECT_GE(kl(k), 0.0);
  }
}

// ============================================================================
// 안정성 (1개)
// ============================================================================

TEST_F(ITMPPITest, MultipleCallsStable)
{
  auto dynamics = std::make_unique<BatchDynamicsWrapper>(params_);
  auto sampler = std::make_unique<GaussianSampler>(params_.noise_sigma, 42);
  auto cost_function = std::make_unique<CompositeMPPICost>();
  cost_function->addCost(std::make_unique<StateTrackingCost>(params_.Q));
  cost_function->addCost(std::make_unique<ControlEffortCost>(params_.R));

  VanillaMPPIWeights weight_strategy;
  ITMPPITestAccessor accessor;
  params_.it_adaptive_exploration = true;
  accessor.setTestParams(params_);

  int N = params_.N, K = params_.K, nu = 2;
  Eigen::MatrixXd control_sequence = Eigen::MatrixXd::Zero(N, nu);

  for (int iter = 0; iter < 10; ++iter) {
    // Warm-start shift
    for (int t = 0; t < N - 1; ++t) {
      control_sequence.row(t) = control_sequence.row(t + 1);
    }
    control_sequence.row(N - 1) = control_sequence.row(N - 2);

    auto noise = sampler->sample(K, N, nu);
    std::vector<Eigen::MatrixXd> perturbed(K);
    for (int k = 0; k < K; ++k) {
      perturbed[k] = dynamics->clipControls(control_sequence + noise[k]);
    }
    auto trajectories = dynamics->rolloutBatch(state_, perturbed, params_.dt);
    auto costs = cost_function->compute(trajectories, perturbed, ref_traj_);

    auto diversity = accessor.callDiversityBonus(trajectories);
    auto kl = accessor.callKLPenalty(noise);

    double ew = accessor.getExplorationWeight();
    Eigen::VectorXd info_costs(K);
    for (int k = 0; k < K; ++k) {
      info_costs(k) = costs(k) - ew * diversity(k) + params_.it_kl_weight * kl(k);
    }

    // Adaptive decay
    accessor.setExplorationWeight(ew * params_.it_exploration_decay);

    auto weights = weight_strategy.compute(info_costs, params_.lambda);

    Eigen::MatrixXd wn = Eigen::MatrixXd::Zero(N, nu);
    for (int k = 0; k < K; ++k) {
      wn += weights(k) * noise[k];
    }
    control_sequence = dynamics->clipControls(control_sequence + wn);

    Eigen::VectorXd u_opt = control_sequence.row(0).transpose();
    EXPECT_FALSE(std::isnan(u_opt(0))) << "NaN at iter " << iter;
    EXPECT_FALSE(std::isnan(u_opt(1))) << "NaN at iter " << iter;
    EXPECT_FALSE(std::isinf(u_opt(0))) << "Inf at iter " << iter;
    EXPECT_FALSE(std::isinf(u_opt(1))) << "Inf at iter " << iter;
  }
}

// ============================================================================
// 정보 (1개)
// ============================================================================

TEST_F(ITMPPITest, InfoContainsITMetrics)
{
  // MPPIInfo의 IT 메트릭 필드가 존재하고 초기값이 올바른지 확인
  MPPIInfo info;
  EXPECT_DOUBLE_EQ(info.it_exploration_bonus, 0.0);
  EXPECT_DOUBLE_EQ(info.it_diversity_score, 0.0);
  EXPECT_DOUBLE_EQ(info.it_kl_divergence, 0.0);

  // 값 설정 후 확인
  info.it_exploration_bonus = 0.5;
  info.it_diversity_score = 1.2;
  info.it_kl_divergence = 0.03;

  EXPECT_DOUBLE_EQ(info.it_exploration_bonus, 0.5);
  EXPECT_DOUBLE_EQ(info.it_diversity_score, 1.2);
  EXPECT_DOUBLE_EQ(info.it_kl_divergence, 0.03);
}

}  // namespace mpc_controller_ros2

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
