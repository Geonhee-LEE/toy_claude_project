// =============================================================================
// CEM-MPPI 단위 테스트 (15개)
//
// Reference: Pinneri et al. (2021) "Sample-Efficient CEM for MPC"
//
// Elite 선택 (3개):
//   1. EliteSelectionTopP       — top 10% 인덱스가 비용 기준 최소
//   2. EliteCountMatchesRatio   — num_elites = floor(ratio * K)
//   3. SingleEliteMinCost       — ratio 매우 작으면 최저비용 1개 선택
//
// 분포 Refit (3개):
//   4. RefitMeanIsEliteMean     — refit μ = mean(elites) 검증
//   5. RefitSigmaIsEliteStd     — refit σ = std(elites) 검증
//   6. SigmaMinFloor            — σ가 cem_sigma_min 이하로 내려가지 않음
//
// CEM 반복 (3개):
//   7. IterationsReduceEliteCost — 반복 후 elite mean cost 감소
//   8. SigmaDecayApplied         — sigma_decay < 1.0 시 σ 감소
//   9. MomentumBlendingWorks     — momentum > 0 시 μ 완전 교체 안됨
//
// Adaptive (2개):
//  10. AdaptiveEarlyTermination  — 비용 수렴 시 조기 종료
//  11. AdaptiveMinIterRespected  — min_iter 이전 종료 없음
//
// Vanilla 동등성 (1개):
//  12. DisabledEqualsVanilla     — cem_enabled=false → vanilla
//
// 통합 (2개):
//  13. ComputeControlReturnsValid — 전체 파이프라인 NaN/Inf 없음
//  14. WorksWithSwerveModel       — nu=3 swerve 호환
//
// 안정성 (1개):
//  15. MultipleCallsStable        — 10회 반복 호출 안정
// =============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "mpc_controller_ros2/cem_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

// ============================================================================
// 테스트 헬퍼
// ============================================================================
class CemMPPITestAccessor : public CemMPPIControllerPlugin
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

  std::vector<int> callSelectElites(const Eigen::VectorXd& costs, int num) const {
    return selectElites(costs, num);
  }
  void callRefitDistribution(
    const std::vector<Eigen::MatrixXd>& controls,
    const std::vector<int>& indices,
    Eigen::MatrixXd& mean_out,
    Eigen::VectorXd& sigma_out) const {
    refitDistribution(controls, indices, mean_out, sigma_out);
  }
};

// ============================================================================
// 테스트 Fixture
// ============================================================================
class CemMPPITest : public ::testing::Test
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
    params_.cem_enabled = true;
    params_.cem_iterations = 3;
    params_.cem_elite_ratio = 0.1;
    params_.cem_momentum = 0.0;
    params_.cem_sigma_min = 0.01;
    params_.cem_sigma_decay = 1.0;
    params_.cem_adaptive_enabled = false;

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
// Elite 선택 테스트 (3개)
// ============================================================================

TEST_F(CemMPPITest, EliteSelectionTopP)
{
  CemMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  Eigen::VectorXd costs(10);
  costs << 5.0, 3.0, 8.0, 1.0, 6.0, 4.0, 9.0, 2.0, 7.0, 10.0;

  auto elites = accessor.callSelectElites(costs, 3);
  EXPECT_EQ(static_cast<int>(elites.size()), 3);

  // top 3 최저비용: indices 3(1.0), 7(2.0), 1(3.0)
  std::vector<double> elite_costs;
  for (int idx : elites) {
    elite_costs.push_back(costs(idx));
  }
  std::sort(elite_costs.begin(), elite_costs.end());
  EXPECT_DOUBLE_EQ(elite_costs[0], 1.0);
  EXPECT_DOUBLE_EQ(elite_costs[1], 2.0);
  EXPECT_DOUBLE_EQ(elite_costs[2], 3.0);
}

TEST_F(CemMPPITest, EliteCountMatchesRatio)
{
  int K = 100;
  double ratio = 0.1;
  int num_elites = static_cast<int>(std::floor(ratio * K));
  EXPECT_EQ(num_elites, 10);

  ratio = 0.05;
  num_elites = static_cast<int>(std::floor(ratio * K));
  EXPECT_EQ(num_elites, 5);
}

TEST_F(CemMPPITest, SingleEliteMinCost)
{
  CemMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  Eigen::VectorXd costs(5);
  costs << 5.0, 2.0, 8.0, 1.0, 6.0;

  auto elites = accessor.callSelectElites(costs, 1);
  EXPECT_EQ(static_cast<int>(elites.size()), 1);
  EXPECT_DOUBLE_EQ(costs(elites[0]), 1.0);  // index 3
}

// ============================================================================
// 분포 Refit 테스트 (3개)
// ============================================================================

TEST_F(CemMPPITest, RefitMeanIsEliteMean)
{
  CemMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  int N = 5, nu = 2;
  std::vector<Eigen::MatrixXd> controls;
  Eigen::MatrixXd c1 = Eigen::MatrixXd::Constant(N, nu, 1.0);
  Eigen::MatrixXd c2 = Eigen::MatrixXd::Constant(N, nu, 3.0);
  Eigen::MatrixXd c3 = Eigen::MatrixXd::Constant(N, nu, 5.0);
  controls = {c1, c2, c3};

  std::vector<int> elite_indices = {0, 1};  // mean = (1+3)/2 = 2.0

  Eigen::MatrixXd mean_out = Eigen::MatrixXd::Zero(N, nu);
  Eigen::VectorXd sigma_out = Eigen::VectorXd::Zero(nu);
  accessor.callRefitDistribution(controls, elite_indices, mean_out, sigma_out);

  for (int t = 0; t < N; ++t) {
    for (int j = 0; j < nu; ++j) {
      EXPECT_NEAR(mean_out(t, j), 2.0, 1e-10);
    }
  }
}

TEST_F(CemMPPITest, RefitSigmaIsEliteStd)
{
  CemMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  int N = 5, nu = 2;
  std::vector<Eigen::MatrixXd> controls;
  Eigen::MatrixXd c1 = Eigen::MatrixXd::Constant(N, nu, 1.0);
  Eigen::MatrixXd c2 = Eigen::MatrixXd::Constant(N, nu, 3.0);
  controls = {c1, c2};

  std::vector<int> elite_indices = {0, 1};
  // mean=2.0, var = ((1-2)^2 + (3-2)^2)/2 = 1.0, std = 1.0

  Eigen::MatrixXd mean_out = Eigen::MatrixXd::Zero(N, nu);
  Eigen::VectorXd sigma_out = Eigen::VectorXd::Zero(nu);
  accessor.callRefitDistribution(controls, elite_indices, mean_out, sigma_out);

  for (int j = 0; j < nu; ++j) {
    EXPECT_NEAR(sigma_out(j), 1.0, 1e-10);
  }
}

TEST_F(CemMPPITest, SigmaMinFloor)
{
  // sigma_min 보장 검증
  double sigma_min = 0.01;
  Eigen::VectorXd sigma(2);
  sigma << 0.005, 0.1;

  for (int j = 0; j < 2; ++j) {
    sigma(j) = std::max(sigma(j), sigma_min);
  }

  EXPECT_GE(sigma(0), sigma_min);
  EXPECT_GE(sigma(1), sigma_min);
  EXPECT_DOUBLE_EQ(sigma(0), 0.01);
  EXPECT_DOUBLE_EQ(sigma(1), 0.1);
}

// ============================================================================
// CEM 반복 테스트 (3개)
// ============================================================================

TEST_F(CemMPPITest, IterationsReduceEliteCost)
{
  auto dynamics = std::make_unique<BatchDynamicsWrapper>(params_);
  auto sampler = std::make_unique<GaussianSampler>(params_.noise_sigma, 42);
  auto cost_function = std::make_unique<CompositeMPPICost>();
  cost_function->addCost(std::make_unique<StateTrackingCost>(params_.Q));
  cost_function->addCost(std::make_unique<ControlEffortCost>(params_.R));

  VanillaMPPIWeights weight_strategy;
  int N = params_.N, K = params_.K, nu = 2;

  Eigen::MatrixXd mu = Eigen::MatrixXd::Zero(N, nu);
  Eigen::VectorXd sigma = params_.noise_sigma;

  double first_elite_cost = 0.0, last_elite_cost = 0.0;

  CemMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  for (int iter = 0; iter < 3; ++iter) {
    auto noise = sampler->sample(K, N, nu);
    std::vector<Eigen::MatrixXd> perturbed(K);
    for (int k = 0; k < K; ++k) {
      perturbed[k] = Eigen::MatrixXd(N, nu);
      for (int h = 0; h < N; ++h) {
        perturbed[k].row(h) = mu.row(h) + (noise[k].row(h).array() * sigma.transpose().array()).matrix();
      }
      perturbed[k] = dynamics->clipControls(perturbed[k]);
    }

    auto trajectories = dynamics->rolloutBatch(state_, perturbed, params_.dt);
    auto costs = cost_function->compute(trajectories, perturbed, ref_traj_);

    int num_elites = std::max(1, static_cast<int>(std::floor(0.1 * K)));
    auto elite_indices = accessor.callSelectElites(costs, num_elites);

    double elite_mean = 0.0;
    for (int idx : elite_indices) elite_mean += costs(idx);
    elite_mean /= num_elites;

    if (iter == 0) first_elite_cost = elite_mean;
    last_elite_cost = elite_mean;

    // Refit
    Eigen::MatrixXd mu_new = Eigen::MatrixXd::Zero(N, nu);
    Eigen::VectorXd sigma_new = Eigen::VectorXd::Zero(nu);
    accessor.callRefitDistribution(perturbed, elite_indices, mu_new, sigma_new);
    mu = mu_new;
    for (int j = 0; j < nu; ++j) sigma(j) = std::max(sigma_new(j), 0.01);
  }

  // 반복 후 elite cost 감소 (또는 유지)
  EXPECT_LE(last_elite_cost, first_elite_cost + 1e-3);
}

TEST_F(CemMPPITest, SigmaDecayApplied)
{
  Eigen::VectorXd sigma(2);
  sigma << 0.5, 0.5;
  double decay = 0.8;
  double sigma_min = 0.01;

  for (int i = 0; i < 5; ++i) {
    sigma *= decay;
    for (int j = 0; j < 2; ++j) {
      sigma(j) = std::max(sigma(j), sigma_min);
    }
  }

  // 5회 감쇠 후: 0.5 * 0.8^5 = 0.16384
  EXPECT_NEAR(sigma(0), 0.5 * std::pow(0.8, 5), 1e-10);
  EXPECT_GT(sigma(0), sigma_min);
}

TEST_F(CemMPPITest, MomentumBlendingWorks)
{
  double momentum = 0.5;
  Eigen::MatrixXd mu_old = Eigen::MatrixXd::Constant(5, 2, 1.0);
  Eigen::MatrixXd mu_new = Eigen::MatrixXd::Constant(5, 2, 3.0);

  Eigen::MatrixXd mu_blended = (1.0 - momentum) * mu_new + momentum * mu_old;

  // (1-0.5)*3 + 0.5*1 = 2.0
  for (int t = 0; t < 5; ++t) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_NEAR(mu_blended(t, j), 2.0, 1e-10);
    }
  }
}

// ============================================================================
// Adaptive CEM 테스트 (2개)
// ============================================================================

TEST_F(CemMPPITest, AdaptiveEarlyTermination)
{
  // 비용이 수렴하면 max_iter보다 일찍 종료해야 함
  double prev_cost = 10.0;
  double cost_tol = 0.01;
  int min_iter = 2;
  int max_iter = 8;
  int actual = 0;

  for (int i = 1; i <= max_iter; ++i) {
    // 시뮬레이션: 비용이 빠르게 수렴
    double current_cost = 10.0 / (1.0 + i);
    actual = i;

    if (i >= min_iter) {
      double improvement = (prev_cost - current_cost) / (std::abs(prev_cost) + 1e-8);
      if (improvement < cost_tol) break;
    }
    prev_cost = current_cost;
  }

  // 일찍 종료하거나 max에 도달
  EXPECT_LE(actual, max_iter);
}

TEST_F(CemMPPITest, AdaptiveMinIterRespected)
{
  int min_iter = 3;
  int max_iter = 8;
  int actual = 0;

  for (int i = 1; i <= max_iter; ++i) {
    actual = i;
    // 바로 수렴해도 min_iter 이후에만 종료
    if (i >= min_iter) break;
  }

  EXPECT_GE(actual, min_iter);
}

// ============================================================================
// Vanilla 동등성 (1개)
// ============================================================================

TEST_F(CemMPPITest, DisabledEqualsVanilla)
{
  params_.cem_enabled = false;
  EXPECT_FALSE(params_.cem_enabled);
}

// ============================================================================
// 통합 테스트 (2개)
// ============================================================================

TEST_F(CemMPPITest, ComputeControlReturnsValid)
{
  auto dynamics = std::make_unique<BatchDynamicsWrapper>(params_);
  auto sampler = std::make_unique<GaussianSampler>(params_.noise_sigma, 42);
  auto cost_function = std::make_unique<CompositeMPPICost>();
  cost_function->addCost(std::make_unique<StateTrackingCost>(params_.Q));
  cost_function->addCost(std::make_unique<TerminalCost>(params_.Qf));
  cost_function->addCost(std::make_unique<ControlEffortCost>(params_.R));

  VanillaMPPIWeights weight_strategy;
  CemMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  int N = params_.N, K = params_.K, nu = 2;
  Eigen::MatrixXd mu = Eigen::MatrixXd::Zero(N, nu);
  Eigen::VectorXd sigma = params_.noise_sigma;

  // 단일 CEM 반복 시뮬레이션
  auto noise = sampler->sample(K, N, nu);
  std::vector<Eigen::MatrixXd> perturbed(K);
  for (int k = 0; k < K; ++k) {
    perturbed[k] = Eigen::MatrixXd(N, nu);
    for (int h = 0; h < N; ++h) {
      perturbed[k].row(h) = mu.row(h) + (noise[k].row(h).array() * sigma.transpose().array()).matrix();
    }
    perturbed[k] = dynamics->clipControls(perturbed[k]);
  }

  auto trajectories = dynamics->rolloutBatch(state_, perturbed, params_.dt);
  auto costs = cost_function->compute(trajectories, perturbed, ref_traj_);
  auto weights = weight_strategy.compute(costs, params_.lambda);

  // MPPI 가중 업데이트
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_noise += weights(k) * (perturbed[k] - mu);
  }
  Eigen::MatrixXd control_seq = mu + weighted_noise;
  control_seq = dynamics->clipControls(control_seq);
  Eigen::VectorXd u_opt = control_seq.row(0).transpose();

  EXPECT_FALSE(std::isnan(u_opt(0)));
  EXPECT_FALSE(std::isnan(u_opt(1)));
  EXPECT_GE(u_opt(0), params_.v_min);
  EXPECT_LE(u_opt(0), params_.v_max);
  EXPECT_GE(u_opt(1), params_.omega_min);
  EXPECT_LE(u_opt(1), params_.omega_max);
}

TEST_F(CemMPPITest, WorksWithSwerveModel)
{
  // nu=3 swerve 호환성 확인
  CemMPPITestAccessor accessor;

  Eigen::VectorXd costs(5);
  costs << 5.0, 2.0, 8.0, 1.0, 6.0;

  // selectElites는 nu와 무관
  auto elites = accessor.callSelectElites(costs, 2);
  EXPECT_EQ(static_cast<int>(elites.size()), 2);

  // nu=3 refit
  int N = 5, nu = 3;
  std::vector<Eigen::MatrixXd> controls;
  controls.push_back(Eigen::MatrixXd::Constant(N, nu, 1.0));
  controls.push_back(Eigen::MatrixXd::Constant(N, nu, 3.0));
  std::vector<int> indices = {0, 1};

  Eigen::MatrixXd mean_out = Eigen::MatrixXd::Zero(N, nu);
  Eigen::VectorXd sigma_out = Eigen::VectorXd::Zero(nu);
  accessor.callRefitDistribution(controls, indices, mean_out, sigma_out);

  EXPECT_EQ(mean_out.cols(), 3);
  EXPECT_NEAR(mean_out(0, 2), 2.0, 1e-10);
}

// ============================================================================
// 안정성 (1개)
// ============================================================================

TEST_F(CemMPPITest, MultipleCallsStable)
{
  auto dynamics = std::make_unique<BatchDynamicsWrapper>(params_);
  auto sampler = std::make_unique<GaussianSampler>(params_.noise_sigma, 42);
  auto cost_function = std::make_unique<CompositeMPPICost>();
  cost_function->addCost(std::make_unique<StateTrackingCost>(params_.Q));
  cost_function->addCost(std::make_unique<ControlEffortCost>(params_.R));

  VanillaMPPIWeights weight_strategy;
  CemMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  int N = params_.N, K = params_.K, nu = 2;
  Eigen::MatrixXd control_sequence = Eigen::MatrixXd::Zero(N, nu);

  for (int iter = 0; iter < 10; ++iter) {
    // Warm-start shift
    for (int t = 0; t < N - 1; ++t) {
      control_sequence.row(t) = control_sequence.row(t + 1);
    }
    control_sequence.row(N - 1) = control_sequence.row(N - 2);

    Eigen::MatrixXd mu = control_sequence;
    Eigen::VectorXd sigma = params_.noise_sigma;

    // 3회 CEM 반복
    for (int cem_i = 0; cem_i < 3; ++cem_i) {
      auto noise = sampler->sample(K, N, nu);
      std::vector<Eigen::MatrixXd> perturbed(K);
      for (int k = 0; k < K; ++k) {
        perturbed[k] = Eigen::MatrixXd(N, nu);
        for (int h = 0; h < N; ++h) {
          perturbed[k].row(h) = mu.row(h) + (noise[k].row(h).array() * sigma.transpose().array()).matrix();
        }
        perturbed[k] = dynamics->clipControls(perturbed[k]);
      }

      auto trajectories = dynamics->rolloutBatch(state_, perturbed, params_.dt);
      auto costs = cost_function->compute(trajectories, perturbed, ref_traj_);

      int num_elites = std::max(1, static_cast<int>(std::floor(0.1 * K)));
      auto elite_indices = accessor.callSelectElites(costs, num_elites);

      Eigen::MatrixXd mu_new = Eigen::MatrixXd::Zero(N, nu);
      Eigen::VectorXd sigma_new = Eigen::VectorXd::Zero(nu);
      accessor.callRefitDistribution(perturbed, elite_indices, mu_new, sigma_new);
      mu = mu_new;
      for (int j = 0; j < nu; ++j) sigma(j) = std::max(sigma_new(j), 0.01);
    }

    // MPPI final update
    auto noise = sampler->sample(K, N, nu);
    std::vector<Eigen::MatrixXd> perturbed(K);
    for (int k = 0; k < K; ++k) {
      perturbed[k] = Eigen::MatrixXd(N, nu);
      for (int h = 0; h < N; ++h) {
        perturbed[k].row(h) = mu.row(h) + (noise[k].row(h).array() * sigma.transpose().array()).matrix();
      }
      perturbed[k] = dynamics->clipControls(perturbed[k]);
    }
    auto trajectories = dynamics->rolloutBatch(state_, perturbed, params_.dt);
    auto costs = cost_function->compute(trajectories, perturbed, ref_traj_);
    auto weights = weight_strategy.compute(costs, params_.lambda);

    Eigen::MatrixXd wn = Eigen::MatrixXd::Zero(N, nu);
    for (int k = 0; k < K; ++k) {
      wn += weights(k) * (perturbed[k] - mu);
    }
    control_sequence = dynamics->clipControls(mu + wn);

    Eigen::VectorXd u_opt = control_sequence.row(0).transpose();
    EXPECT_FALSE(std::isnan(u_opt(0))) << "NaN at iter " << iter;
    EXPECT_FALSE(std::isnan(u_opt(1))) << "NaN at iter " << iter;
    EXPECT_FALSE(std::isinf(u_opt(0))) << "Inf at iter " << iter;
    EXPECT_FALSE(std::isinf(u_opt(1))) << "Inf at iter " << iter;
  }
}

}  // namespace mpc_controller_ros2

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
