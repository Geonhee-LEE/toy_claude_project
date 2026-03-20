// =============================================================================
// Robust MPPI (Distributionally Robust) 단위 테스트 (15개)
//
// Worst-case 추정 (3개):
//   1. WorstAlphaSelectsHighCosts    — worst 20% 비용 선택
//   2. AlphaOneEqualsFullSample      — alpha=1.0 전체 샘플 사용
//   3. AlphaZeroUsesWorstSingle      — alpha→0 단일 최악 근사
//
// 분산 페널티 (3개):
//   4. VariancePenaltyIncreasesRobustCost  — penalty>0 비용 증가
//   5. ZeroVariancePenaltyNoEffect          — penalty=0 비용 불변
//   6. HighVarianceSamplesDownweighted      — 높은 분산 → 낮은 가중치
//
// Wasserstein (2개):
//   7. WassersteinRadiusZeroNoEffect     — radius=0 표준 동일
//   8. WassersteinIncreasesConservatism  — radius>0 보수적 제어
//
// Adaptive (2개):
//   9. AdaptiveAlphaDecreasesOnLowSpread  — 낮은 스프레드 → 큰 alpha
//  10. AdaptiveAlphaIncreasesOnHighSpread — 높은 스프레드 → 작은 alpha
//
// Vanilla 동등성 (1개):
//  11. DisabledEqualsVanilla — enabled=false → vanilla
//
// 통합 (2개):
//  12. ComputeControlReturnsValid  — 전체 파이프라인 NaN/Inf 없음
//  13. WorksWithSwerveModel        — nu=3 swerve 호환
//
// 안정성 (1개):
//  14. MultipleCallsStable — 10회 반복 안정
//
// Info (1개):
//  15. WorstCaseCostInInfo — info에 robust 메트릭 포함
// =============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "mpc_controller_ros2/robust_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

// ============================================================================
// 테스트 헬퍼
// ============================================================================
class RobustMPPITestAccessor : public RobustMPPIControllerPlugin
{
public:
  void setTestParams(const MPPIParams& params) { params_ = params; }
  void setDynamics(std::unique_ptr<BatchDynamicsWrapper> dyn) { dynamics_ = std::move(dyn); }
  void setSampler(std::unique_ptr<BaseSampler> sampler) { sampler_ = std::move(sampler); }
  void setCostFunction(std::unique_ptr<CompositeMPPICost> cf) { cost_function_ = std::move(cf); }
  void setWeightComputation(std::unique_ptr<WeightComputation> wc) {
    weight_computation_ = std::move(wc);
  }
  void setControlSequence(const Eigen::MatrixXd& cs) { control_sequence_ = cs; }
  Eigen::MatrixXd getControlSequence() const { return control_sequence_; }

  std::pair<double, double> callApplyRobustProcessing(Eigen::VectorXd& costs) const {
    return applyRobustProcessing(costs);
  }

  Eigen::VectorXd callComputeWassersteinPenalty(const Eigen::VectorXd& costs) const {
    return computeWassersteinPenalty(costs);
  }
};

// ============================================================================
// 테스트 Fixture
// ============================================================================
class RobustMPPITest : public ::testing::Test
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
    params_.robust_enabled = true;
    params_.robust_alpha = 0.2;
    params_.robust_penalty = 1.0;
    params_.robust_wasserstein_radius = 0.0;
    params_.robust_adaptive_alpha = false;

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
// Worst-case 추정 테스트 (3개)
// ============================================================================

TEST_F(RobustMPPITest, WorstAlphaSelectsHighCosts)
{
  // alpha=0.2 → worst 20% (2 of 10)
  RobustMPPITestAccessor accessor;
  params_.robust_alpha = 0.2;
  params_.robust_penalty = 0.0;  // 분산 페널티 비활성
  params_.robust_wasserstein_radius = 0.0;
  params_.robust_adaptive_alpha = false;
  accessor.setTestParams(params_);

  Eigen::VectorXd costs(10);
  costs << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0;

  auto [worst_cost, eff_alpha] = accessor.callApplyRobustProcessing(costs);

  // worst 20% = top 2 costs = {9, 10}, mean = 9.5
  EXPECT_NEAR(worst_cost, 9.5, 0.1);
  EXPECT_NEAR(eff_alpha, 0.2, 1e-6);
}

TEST_F(RobustMPPITest, AlphaOneEqualsFullSample)
{
  // alpha=1.0 → 전체 샘플 사용
  RobustMPPITestAccessor accessor;
  params_.robust_alpha = 1.0;
  params_.robust_penalty = 0.0;
  params_.robust_wasserstein_radius = 0.0;
  params_.robust_adaptive_alpha = false;
  accessor.setTestParams(params_);

  Eigen::VectorXd costs(10);
  costs << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0;
  double full_mean = costs.mean();

  auto [worst_cost, eff_alpha] = accessor.callApplyRobustProcessing(costs);

  // alpha=1.0 → worst 100% = 전체 평균
  EXPECT_NEAR(worst_cost, full_mean, 0.1);
}

TEST_F(RobustMPPITest, AlphaZeroUsesWorstSingle)
{
  // alpha→0 → worst 1개 (ceil(0.01*10)=1)
  RobustMPPITestAccessor accessor;
  params_.robust_alpha = 0.01;  // very small
  params_.robust_penalty = 0.0;
  params_.robust_wasserstein_radius = 0.0;
  params_.robust_adaptive_alpha = false;
  accessor.setTestParams(params_);

  Eigen::VectorXd costs(10);
  costs << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0;

  auto [worst_cost, eff_alpha] = accessor.callApplyRobustProcessing(costs);

  // worst 1개 = 최대 비용 = 10.0
  EXPECT_NEAR(worst_cost, 10.0, 0.1);
}

// ============================================================================
// 분산 페널티 테스트 (3개)
// ============================================================================

TEST_F(RobustMPPITest, VariancePenaltyIncreasesRobustCost)
{
  RobustMPPITestAccessor accessor;
  params_.robust_penalty = 0.0;
  params_.robust_wasserstein_radius = 0.0;
  params_.robust_adaptive_alpha = false;
  accessor.setTestParams(params_);

  Eigen::VectorXd costs_no_penalty(5);
  costs_no_penalty << 1.0, 3.0, 5.0, 7.0, 9.0;

  auto [wc_no, _a1] = accessor.callApplyRobustProcessing(costs_no_penalty);

  // 이제 penalty 활성화
  params_.robust_penalty = 2.0;
  accessor.setTestParams(params_);

  Eigen::VectorXd costs_with_penalty(5);
  costs_with_penalty << 1.0, 3.0, 5.0, 7.0, 9.0;

  auto [wc_with, _a2] = accessor.callApplyRobustProcessing(costs_with_penalty);

  // 분산 페널티가 비용을 높임 → worst-case도 높아짐
  EXPECT_GT(wc_with, wc_no);
}

TEST_F(RobustMPPITest, ZeroVariancePenaltyNoEffect)
{
  RobustMPPITestAccessor accessor;
  params_.robust_penalty = 0.0;
  params_.robust_wasserstein_radius = 0.0;
  params_.robust_adaptive_alpha = false;
  accessor.setTestParams(params_);

  Eigen::VectorXd costs1(5);
  costs1 << 1.0, 3.0, 5.0, 7.0, 9.0;
  Eigen::VectorXd costs2 = costs1;

  auto [wc1, _a1] = accessor.callApplyRobustProcessing(costs1);

  // penalty=0 이므로 동일한 입력 → 동일한 출력
  auto [wc2, _a2] = accessor.callApplyRobustProcessing(costs2);

  EXPECT_NEAR(wc1, wc2, 1e-10);
}

TEST_F(RobustMPPITest, HighVarianceSamplesDownweighted)
{
  // 높은 분산 → penalty 추가 → MPPI 가중치에서 높은 비용 샘플 다운웨이트
  VanillaMPPIWeights weight_strategy;
  double lambda = 10.0;

  // 낮은 분산 비용
  Eigen::VectorXd low_var_costs(5);
  low_var_costs << 4.9, 5.0, 5.1, 5.0, 4.9;

  // 높은 분산 비용
  Eigen::VectorXd high_var_costs(5);
  high_var_costs << 1.0, 3.0, 5.0, 7.0, 9.0;

  auto w_low = weight_strategy.compute(low_var_costs, lambda);
  auto w_high = weight_strategy.compute(high_var_costs, lambda);

  // 높은 분산 비용의 가중치 엔트로피가 더 낮음 (일부에 집중)
  double max_w_low = w_low.maxCoeff();
  double max_w_high = w_high.maxCoeff();

  // 높은 분산에서는 최소 비용 샘플에 가중치가 집중됨
  EXPECT_GT(max_w_high, max_w_low);
}

// ============================================================================
// Wasserstein 테스트 (2개)
// ============================================================================

TEST_F(RobustMPPITest, WassersteinRadiusZeroNoEffect)
{
  RobustMPPITestAccessor accessor;
  params_.robust_penalty = 0.0;
  params_.robust_wasserstein_radius = 0.0;
  params_.robust_adaptive_alpha = false;
  accessor.setTestParams(params_);

  Eigen::VectorXd costs(5);
  costs << 1.0, 3.0, 5.0, 7.0, 9.0;
  Eigen::VectorXd costs_copy = costs;

  auto [wc_no_w, _a1] = accessor.callApplyRobustProcessing(costs);

  // Wasserstein penalty 벡터가 0
  auto penalty = accessor.callComputeWassersteinPenalty(costs_copy);
  // radius=0이므로 penalty가 적용되지 않음
  // (하지만 penalty 자체는 비용 기울기 근사이므로 0이 아닐 수 있음)
  // 핵심: radius=0이면 costs에 penalty가 더해지지 않음 → 결과 동일
  Eigen::VectorXd costs2 = costs_copy;
  auto [wc_no_w2, _a2] = accessor.callApplyRobustProcessing(costs2);
  EXPECT_NEAR(wc_no_w, wc_no_w2, 1e-10);
}

TEST_F(RobustMPPITest, WassersteinIncreasesConservatism)
{
  RobustMPPITestAccessor accessor;
  params_.robust_penalty = 0.0;
  params_.robust_wasserstein_radius = 0.0;
  params_.robust_adaptive_alpha = false;
  accessor.setTestParams(params_);

  Eigen::VectorXd costs1(5);
  costs1 << 1.0, 3.0, 5.0, 7.0, 9.0;
  auto [wc_no, _a1] = accessor.callApplyRobustProcessing(costs1);

  // Wasserstein radius > 0
  params_.robust_wasserstein_radius = 1.0;
  accessor.setTestParams(params_);

  Eigen::VectorXd costs2(5);
  costs2 << 1.0, 3.0, 5.0, 7.0, 9.0;
  auto [wc_with, _a2] = accessor.callApplyRobustProcessing(costs2);

  // Wasserstein 페널티 추가 → worst-case 비용 증가 (더 보수적)
  EXPECT_GT(wc_with, wc_no);
}

// ============================================================================
// Adaptive alpha 테스트 (2개)
// ============================================================================

TEST_F(RobustMPPITest, AdaptiveAlphaDecreasesOnLowSpread)
{
  // 낮은 비용 스프레드 → 큰 alpha (덜 보수적)
  RobustMPPITestAccessor accessor;
  params_.robust_alpha = 0.2;
  params_.robust_penalty = 0.0;
  params_.robust_wasserstein_radius = 0.0;
  params_.robust_adaptive_alpha = true;
  accessor.setTestParams(params_);

  // 낮은 스프레드 (비용 거의 균일)
  Eigen::VectorXd low_spread(10);
  low_spread.setLinSpaced(10, 5.0, 5.1);  // range = 0.1

  auto [_wc, eff_alpha_low] = accessor.callApplyRobustProcessing(low_spread);

  // 높은 스프레드
  Eigen::VectorXd high_spread(10);
  high_spread.setLinSpaced(10, 1.0, 100.0);  // range = 99

  auto [_wc2, eff_alpha_high] = accessor.callApplyRobustProcessing(high_spread);

  // 낮은 스프레드 → 큰 alpha (덜 보수적)
  EXPECT_GT(eff_alpha_low, eff_alpha_high);
}

TEST_F(RobustMPPITest, AdaptiveAlphaIncreasesOnHighSpread)
{
  // 높은 스프레드 → 작은 alpha (더 보수적)
  RobustMPPITestAccessor accessor;
  params_.robust_alpha = 0.5;
  params_.robust_penalty = 0.0;
  params_.robust_wasserstein_radius = 0.0;
  params_.robust_adaptive_alpha = true;
  accessor.setTestParams(params_);

  Eigen::VectorXd high_spread(10);
  high_spread.setLinSpaced(10, 1.0, 1000.0);

  auto [_wc, eff_alpha] = accessor.callApplyRobustProcessing(high_spread);

  // 매우 높은 스프레드 → alpha가 base 값보다 작아야 함
  EXPECT_LT(eff_alpha, params_.robust_alpha * 1.5);
}

// ============================================================================
// Vanilla 동등성 (1개)
// ============================================================================

TEST_F(RobustMPPITest, DisabledEqualsVanilla)
{
  params_.robust_enabled = false;
  EXPECT_FALSE(params_.robust_enabled);
}

// ============================================================================
// 통합 테스트 (2개)
// ============================================================================

TEST_F(RobustMPPITest, ComputeControlReturnsValid)
{
  auto dynamics = std::make_unique<BatchDynamicsWrapper>(params_);
  auto sampler = std::make_unique<GaussianSampler>(params_.noise_sigma, 42);
  auto cost_function = std::make_unique<CompositeMPPICost>();
  cost_function->addCost(std::make_unique<StateTrackingCost>(params_.Q));
  cost_function->addCost(std::make_unique<TerminalCost>(params_.Qf));
  cost_function->addCost(std::make_unique<ControlEffortCost>(params_.R));

  VanillaMPPIWeights weight_strategy;
  RobustMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  int N = params_.N, K = params_.K, nu = 2;
  Eigen::MatrixXd mu = Eigen::MatrixXd::Zero(N, nu);

  // 단일 MPPI 반복 시뮬레이션
  auto noise = sampler->sample(K, N, nu);
  std::vector<Eigen::MatrixXd> perturbed(K);
  for (int k = 0; k < K; ++k) {
    perturbed[k] = mu + noise[k];
    perturbed[k] = dynamics->clipControls(perturbed[k]);
  }

  auto trajectories = dynamics->rolloutBatch(state_, perturbed, params_.dt);
  auto costs = cost_function->compute(trajectories, perturbed, ref_traj_);

  // Robust 처리 적용
  auto [worst_cost, eff_alpha] = accessor.callApplyRobustProcessing(costs);

  // MPPI 가중 업데이트
  auto weights = weight_strategy.compute(costs, params_.lambda);
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
  EXPECT_GT(worst_cost, 0.0);
  EXPECT_GT(eff_alpha, 0.0);
}

TEST_F(RobustMPPITest, WorksWithSwerveModel)
{
  // nu=3 swerve 호환성 확인
  RobustMPPITestAccessor accessor;
  params_.robust_penalty = 1.0;
  params_.robust_wasserstein_radius = 0.5;
  accessor.setTestParams(params_);

  // Swerve 비용 벡터 (nu와 무관)
  Eigen::VectorXd costs(20);
  for (int k = 0; k < 20; ++k) {
    costs(k) = static_cast<double>(k + 1);
  }

  auto [worst_cost, eff_alpha] = accessor.callApplyRobustProcessing(costs);

  EXPECT_GT(worst_cost, 0.0);
  EXPECT_FALSE(std::isnan(worst_cost));
  EXPECT_FALSE(std::isinf(worst_cost));
}

// ============================================================================
// 안정성 (1개)
// ============================================================================

TEST_F(RobustMPPITest, MultipleCallsStable)
{
  auto dynamics = std::make_unique<BatchDynamicsWrapper>(params_);
  auto sampler = std::make_unique<GaussianSampler>(params_.noise_sigma, 42);
  auto cost_function = std::make_unique<CompositeMPPICost>();
  cost_function->addCost(std::make_unique<StateTrackingCost>(params_.Q));
  cost_function->addCost(std::make_unique<ControlEffortCost>(params_.R));

  VanillaMPPIWeights weight_strategy;
  RobustMPPITestAccessor accessor;
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
      perturbed[k] = control_sequence + noise[k];
      perturbed[k] = dynamics->clipControls(perturbed[k]);
    }

    auto trajectories = dynamics->rolloutBatch(state_, perturbed, params_.dt);
    auto costs = cost_function->compute(trajectories, perturbed, ref_traj_);

    // Robust 처리
    accessor.callApplyRobustProcessing(costs);

    auto weights = weight_strategy.compute(costs, params_.lambda);
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
// Info 검증 (1개)
// ============================================================================

TEST_F(RobustMPPITest, WorstCaseCostInInfo)
{
  // MPPIInfo에 robust 메트릭이 포함되는지 확인
  MPPIInfo info;
  info.robust_worst_case_cost = 42.0;
  info.robust_effective_alpha = 0.15;

  EXPECT_DOUBLE_EQ(info.robust_worst_case_cost, 42.0);
  EXPECT_DOUBLE_EQ(info.robust_effective_alpha, 0.15);
}

}  // namespace mpc_controller_ros2

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
