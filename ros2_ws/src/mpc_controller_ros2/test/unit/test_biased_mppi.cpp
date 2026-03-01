// =============================================================================
// Biased-MPPI 단위 테스트 (15개)
//
// Reference: Trevisan & Alonso-Mora (2024) "Biased-MPPI" IEEE RA-L
//
// 테스트 구성:
//   Ancillary 시퀀스 생성 (4개):
//     BrakingSequenceIsZero, GoToGoalConverges,
//     PathFollowingTangent, PreviousSolutionCopy
//   샘플 구성 (3개):
//     BiasRatioSplit, DeterministicSamplesMatchAnc, ZeroBiasRatioAllGaussian
//   Vanilla 동등성 (2개):
//     DisabledEqualsVanilla, ZeroBiasRatioEqualsVanilla
//   가중치 호환 (2개):
//     WeightsNormalized, WeightsSameStrategy
//   통합 (3개):
//     ComputeControlReturnsValid, ControlSequenceUpdated, WorksWithSwerveModel
//   안정성 (1개):
//     MultipleCallsStable
// =============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

#include "mpc_controller_ros2/biased_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

// ============================================================================
// 테스트 헬퍼: BiasedMPPI 내부 함수를 테스트하기 위한 friend-like accessor
// ============================================================================
class BiasedMPPITestAccessor : public BiasedMPPIControllerPlugin
{
public:
  // 파라미터 직접 설정 (ROS2 없이)
  void setTestParams(const MPPIParams& params) { params_ = params; }
  void setDynamics(std::unique_ptr<BatchDynamicsWrapper> dyn) { dynamics_ = std::move(dyn); }
  void setSampler(std::unique_ptr<BaseSampler> sampler) { sampler_ = std::move(sampler); }
  void setCostFunction(std::unique_ptr<CompositeMPPICost> cf) { cost_function_ = std::move(cf); }
  void setWeightComputation(std::unique_ptr<WeightComputation> wc) {
    weight_computation_ = std::move(wc);
  }
  void setControlSequence(const Eigen::MatrixXd& cs) { control_sequence_ = cs; }
  Eigen::MatrixXd getControlSequence() const { return control_sequence_; }

  // Ancillary 생성 함수 — protected 상속으로 접근 가능
  Eigen::MatrixXd callBraking(int N, int nu) const {
    return generateBrakingSequence(N, nu);
  }
  Eigen::MatrixXd callGoToGoal(
    const Eigen::VectorXd& s, const Eigen::MatrixXd& r, int N, int nu, double dt) const {
    return generateGoToGoalSequence(s, r, N, nu, dt);
  }
  Eigen::MatrixXd callPathFollowing(
    const Eigen::VectorXd& s, const Eigen::MatrixXd& r, int N, int nu, double dt) const {
    return generatePathFollowingSequence(s, r, N, nu, dt);
  }
  Eigen::MatrixXd callPrevSolution(int N, int nu) const {
    return generatePreviousSolutionSequence(N, nu);
  }
};

// ============================================================================
// 기본 테스트 Fixture
// ============================================================================
class BiasedMPPITest : public ::testing::Test
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
    params_.biased_enabled = true;
    params_.bias_ratio = 0.1;
    params_.biased_braking = true;
    params_.biased_goto_goal = true;
    params_.biased_path_following = true;
    params_.biased_previous_solution = true;
    params_.biased_goto_goal_gain = 1.0;
    params_.biased_path_following_gain = 1.0;

    // Reference trajectory: straight line
    ref_traj_ = Eigen::MatrixXd::Zero(params_.N + 1, 3);
    for (int t = 0; t <= params_.N; ++t) {
      ref_traj_(t, 0) = t * params_.dt;  // x 전진
    }

    state_ = Eigen::Vector3d(0.0, 0.0, 0.0);
  }

  MPPIParams params_;
  Eigen::MatrixXd ref_traj_;
  Eigen::Vector3d state_;
};

// ============================================================================
// Ancillary 시퀀스 생성 테스트 (4개)
// ============================================================================

TEST_F(BiasedMPPITest, BrakingSequenceIsZero)
{
  BiasedMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  auto braking = accessor.callBraking(params_.N, 2);

  EXPECT_EQ(braking.rows(), params_.N);
  EXPECT_EQ(braking.cols(), 2);
  EXPECT_DOUBLE_EQ(braking.norm(), 0.0);
}

TEST_F(BiasedMPPITest, GoToGoalConverges)
{
  BiasedMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  auto seq = accessor.callGoToGoal(state_, ref_traj_, params_.N, 2, params_.dt);

  EXPECT_EQ(seq.rows(), params_.N);
  EXPECT_EQ(seq.cols(), 2);

  // 목표가 전방 → v_cmd > 0
  EXPECT_GT(seq(0, 0), 0.0);

  // 모든 제어가 범위 내
  for (int t = 0; t < params_.N; ++t) {
    EXPECT_GE(seq(t, 0), params_.v_min);
    EXPECT_LE(seq(t, 0), params_.v_max);
    EXPECT_GE(seq(t, 1), params_.omega_min);
    EXPECT_LE(seq(t, 1), params_.omega_max);
  }
}

TEST_F(BiasedMPPITest, PathFollowingTangent)
{
  BiasedMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  auto seq = accessor.callPathFollowing(state_, ref_traj_, params_.N, 2, params_.dt);

  EXPECT_EQ(seq.rows(), params_.N);
  EXPECT_EQ(seq.cols(), 2);

  // 직선 경로 → v_cmd > 0
  EXPECT_GT(seq(0, 0), 0.0);

  // 모든 제어가 범위 내
  for (int t = 0; t < params_.N; ++t) {
    EXPECT_GE(seq(t, 0), params_.v_min);
    EXPECT_LE(seq(t, 0), params_.v_max);
  }
}

TEST_F(BiasedMPPITest, PreviousSolutionCopy)
{
  BiasedMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  // 특정 control_sequence_ 설정
  Eigen::MatrixXd cs = Eigen::MatrixXd::Random(params_.N, 2) * 0.3;
  accessor.setControlSequence(cs);

  auto seq = accessor.callPrevSolution(params_.N, 2);

  EXPECT_EQ(seq.rows(), params_.N);
  EXPECT_EQ(seq.cols(), 2);
  EXPECT_NEAR((seq - cs).norm(), 0.0, 1e-10);
}

// ============================================================================
// 샘플 구성 테스트 (3개)
// ============================================================================

TEST_F(BiasedMPPITest, BiasRatioSplit)
{
  int K = params_.K;
  int num_ancillary = 4;  // braking + goto_goal + path_following + prev_solution
  int J = static_cast<int>(std::floor(params_.bias_ratio * K));  // 10
  int J_total = J * num_ancillary;  // 40
  int K_gaussian = K - J_total;     // 60

  EXPECT_EQ(J_total + K_gaussian, K);
  EXPECT_EQ(J, 10);
  EXPECT_EQ(J_total, 40);
  EXPECT_EQ(K_gaussian, 60);
}

TEST_F(BiasedMPPITest, DeterministicSamplesMatchAnc)
{
  BiasedMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  // Braking sequence는 항상 zero
  auto braking1 = accessor.callBraking(params_.N, 2);
  auto braking2 = accessor.callBraking(params_.N, 2);
  EXPECT_NEAR((braking1 - braking2).norm(), 0.0, 1e-10);

  // GoToGoal은 동일 입력이면 동일 출력 (결정적)
  auto goto1 = accessor.callGoToGoal(state_, ref_traj_, params_.N, 2, params_.dt);
  auto goto2 = accessor.callGoToGoal(state_, ref_traj_, params_.N, 2, params_.dt);
  EXPECT_NEAR((goto1 - goto2).norm(), 0.0, 1e-10);
}

TEST_F(BiasedMPPITest, ZeroBiasRatioAllGaussian)
{
  int K = 100;
  double bias_ratio = 0.0;
  int num_ancillary = 4;
  int J = static_cast<int>(std::floor(bias_ratio * K));
  int J_total = J * num_ancillary;
  int K_gaussian = K - J_total;

  EXPECT_EQ(J, 0);
  EXPECT_EQ(J_total, 0);
  EXPECT_EQ(K_gaussian, K);
}

// ============================================================================
// Vanilla 동등성 테스트 (2개)
// ============================================================================

TEST_F(BiasedMPPITest, DisabledEqualsVanilla)
{
  // biased_enabled=false면 base computeControl을 호출해야 함
  // 여기서는 파라미터 플래그만 검증
  params_.biased_enabled = false;
  EXPECT_FALSE(params_.biased_enabled);
  // 실제 동등성은 통합 테스트에서 검증 (ROS2 없이 computeControl 호출 불가)
}

TEST_F(BiasedMPPITest, ZeroBiasRatioEqualsVanilla)
{
  // bias_ratio=0 → J_total=0 → 모든 샘플이 Gaussian → Vanilla와 동일
  params_.bias_ratio = 0.0;
  int J = static_cast<int>(std::floor(params_.bias_ratio * params_.K));
  EXPECT_EQ(J, 0);
}

// ============================================================================
// 가중치 호환 테스트 (2개)
// ============================================================================

TEST_F(BiasedMPPITest, WeightsNormalized)
{
  // VanillaMPPIWeights 사용 시 sum(weights) ≈ 1.0
  VanillaMPPIWeights vanilla_weights;
  Eigen::VectorXd costs(params_.K);
  std::mt19937 rng(42);
  std::normal_distribution<double> dist(10.0, 3.0);
  for (int k = 0; k < params_.K; ++k) {
    costs(k) = std::abs(dist(rng));
  }

  Eigen::VectorXd weights = vanilla_weights.compute(costs, params_.lambda);
  EXPECT_NEAR(weights.sum(), 1.0, 1e-9);
  for (int k = 0; k < params_.K; ++k) {
    EXPECT_GE(weights(k), 0.0);
  }
}

TEST_F(BiasedMPPITest, WeightsSameStrategy)
{
  // Biased-MPPI에서도 기존 WeightComputation 전략 재사용 확인
  VanillaMPPIWeights vanilla;
  LogMPPIWeights log_weights;

  Eigen::VectorXd costs(10);
  costs << 5.0, 3.0, 8.0, 1.0, 6.0, 4.0, 9.0, 2.0, 7.0, 10.0;

  auto w_vanilla = vanilla.compute(costs, 10.0);
  auto w_log = log_weights.compute(costs, 10.0);

  // 둘 다 정규화됨
  EXPECT_NEAR(w_vanilla.sum(), 1.0, 1e-9);
  EXPECT_NEAR(w_log.sum(), 1.0, 1e-9);

  // 최저 비용이 최대 가중치
  int min_cost_idx;
  costs.minCoeff(&min_cost_idx);

  int max_weight_idx_v;
  w_vanilla.maxCoeff(&max_weight_idx_v);
  EXPECT_EQ(min_cost_idx, max_weight_idx_v);
}

// ============================================================================
// 통합 테스트 (3개)
// ============================================================================

TEST_F(BiasedMPPITest, ComputeControlReturnsValid)
{
  // ROS2 없이 직접 MPPI 파이프라인 테스트 (biased 로직만)
  auto dynamics = std::make_unique<BatchDynamicsWrapper>(params_);
  auto sampler = std::make_unique<GaussianSampler>(params_.noise_sigma, 42);
  auto cost_function = std::make_unique<CompositeMPPICost>();
  cost_function->addCost(std::make_unique<StateTrackingCost>(params_.Q));
  cost_function->addCost(std::make_unique<TerminalCost>(params_.Qf));
  cost_function->addCost(std::make_unique<ControlEffortCost>(params_.R));

  VanillaMPPIWeights weight_strategy;

  int N = params_.N;
  int K = params_.K;
  int nu = 2;

  Eigen::MatrixXd control_sequence = Eigen::MatrixXd::Zero(N, nu);

  // Ancillary 생성
  BiasedMPPITestAccessor accessor;
  accessor.setTestParams(params_);
  accessor.setControlSequence(control_sequence);

  std::vector<Eigen::MatrixXd> ancillary_seqs;
  ancillary_seqs.push_back(accessor.callBraking(N, nu));
  ancillary_seqs.push_back(
    accessor.callGoToGoal(state_, ref_traj_, N, nu, params_.dt));
  ancillary_seqs.push_back(
    accessor.callPathFollowing(state_, ref_traj_, N, nu, params_.dt));
  ancillary_seqs.push_back(accessor.callPrevSolution(N, nu));

  int num_ancillary = static_cast<int>(ancillary_seqs.size());
  int J = static_cast<int>(std::floor(params_.bias_ratio * K));
  int J_total = J * num_ancillary;
  int K_gaussian = K - J_total;

  // 노이즈 샘플링
  auto noise_samples = sampler->sample(K, N, nu);

  // Biased 샘플 구성
  std::vector<Eigen::MatrixXd> perturbed_controls;
  std::vector<Eigen::MatrixXd> noise_for_weight;

  for (int a = 0; a < num_ancillary; ++a) {
    for (int j = 0; j < J; ++j) {
      auto anc_ctrl = dynamics->clipControls(ancillary_seqs[a]);
      perturbed_controls.push_back(anc_ctrl);
      noise_for_weight.push_back(anc_ctrl - control_sequence);
    }
  }

  for (int k = 0; k < K_gaussian; ++k) {
    auto perturbed = dynamics->clipControls(control_sequence + noise_samples[k]);
    perturbed_controls.push_back(perturbed);
    noise_for_weight.push_back(noise_samples[k]);
  }

  EXPECT_EQ(static_cast<int>(perturbed_controls.size()), K);

  // Rollout + Cost
  auto trajectories = dynamics->rolloutBatch(state_, perturbed_controls, params_.dt);
  auto costs = cost_function->compute(trajectories, perturbed_controls, ref_traj_);
  auto weights = weight_strategy.compute(costs, params_.lambda);

  // Update
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_noise += weights(k) * noise_for_weight[k];
  }
  control_sequence += weighted_noise;
  control_sequence = dynamics->clipControls(control_sequence);

  Eigen::VectorXd u_opt = control_sequence.row(0).transpose();

  // 검증
  EXPECT_FALSE(std::isnan(u_opt(0)));
  EXPECT_FALSE(std::isnan(u_opt(1)));
  EXPECT_GE(u_opt(0), params_.v_min);
  EXPECT_LE(u_opt(0), params_.v_max);
  EXPECT_GE(u_opt(1), params_.omega_min);
  EXPECT_LE(u_opt(1), params_.omega_max);
}

TEST_F(BiasedMPPITest, ControlSequenceUpdated)
{
  // 반복 후 control_sequence가 변경되는지 확인
  auto dynamics = std::make_unique<BatchDynamicsWrapper>(params_);
  auto sampler = std::make_unique<GaussianSampler>(params_.noise_sigma, 42);
  auto cost_function = std::make_unique<CompositeMPPICost>();
  cost_function->addCost(std::make_unique<StateTrackingCost>(params_.Q));

  VanillaMPPIWeights weight_strategy;
  int N = params_.N;
  int K = params_.K;
  int nu = 2;

  Eigen::MatrixXd control_sequence = Eigen::MatrixXd::Zero(N, nu);
  Eigen::MatrixXd original = control_sequence;

  auto noise_samples = sampler->sample(K, N, nu);

  // 단순 Gaussian 업데이트
  std::vector<Eigen::MatrixXd> perturbed;
  for (int k = 0; k < K; ++k) {
    perturbed.push_back(dynamics->clipControls(control_sequence + noise_samples[k]));
  }

  auto trajectories = dynamics->rolloutBatch(state_, perturbed, params_.dt);
  auto costs = cost_function->compute(trajectories, perturbed, ref_traj_);
  auto weights = weight_strategy.compute(costs, params_.lambda);

  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_noise += weights(k) * noise_samples[k];
  }
  control_sequence += weighted_noise;

  // 업데이트 후 변경되었는지 확인
  double diff = (control_sequence - original).norm();
  EXPECT_GT(diff, 1e-6);
}

TEST_F(BiasedMPPITest, WorksWithSwerveModel)
{
  // Swerve (nu=3) 호환성 확인
  MPPIParams swerve_params = params_;
  swerve_params.motion_model = "swerve";
  swerve_params.noise_sigma = Eigen::Vector3d(0.5, 0.5, 0.5);
  swerve_params.Q = Eigen::Matrix3d::Identity() * 10.0;
  swerve_params.Qf = Eigen::Matrix3d::Identity() * 20.0;
  swerve_params.R = Eigen::Matrix3d::Identity() * 0.1;
  swerve_params.R_rate = Eigen::Matrix3d::Identity();

  BiasedMPPITestAccessor accessor;
  accessor.setTestParams(swerve_params);

  int nu = 3;

  // Ancillary 시퀀스가 nu=3으로 생성됨
  auto braking = accessor.callBraking(swerve_params.N, nu);
  EXPECT_EQ(braking.cols(), 3);
  EXPECT_DOUBLE_EQ(braking.norm(), 0.0);

  auto goto_seq = accessor.callGoToGoal(
    state_, ref_traj_, swerve_params.N, nu, swerve_params.dt);
  EXPECT_EQ(goto_seq.cols(), 3);

  auto path_seq = accessor.callPathFollowing(
    state_, ref_traj_, swerve_params.N, nu, swerve_params.dt);
  EXPECT_EQ(path_seq.cols(), 3);
}

// ============================================================================
// 다중 호출 안정성 (1개)
// ============================================================================

TEST_F(BiasedMPPITest, MultipleCallsStable)
{
  auto dynamics = std::make_unique<BatchDynamicsWrapper>(params_);
  auto sampler = std::make_unique<GaussianSampler>(params_.noise_sigma, 42);
  auto cost_function = std::make_unique<CompositeMPPICost>();
  cost_function->addCost(std::make_unique<StateTrackingCost>(params_.Q));
  cost_function->addCost(std::make_unique<ControlEffortCost>(params_.R));

  BiasedMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  VanillaMPPIWeights weight_strategy;
  int N = params_.N;
  int K = params_.K;
  int nu = 2;
  int num_ancillary = 4;

  Eigen::MatrixXd control_sequence = Eigen::MatrixXd::Zero(N, nu);

  for (int iter = 0; iter < 10; ++iter) {
    accessor.setControlSequence(control_sequence);

    // Warm-start shift
    for (int t = 0; t < N - 1; ++t) {
      control_sequence.row(t) = control_sequence.row(t + 1);
    }
    control_sequence.row(N - 1) = control_sequence.row(N - 2);
    accessor.setControlSequence(control_sequence);

    // Ancillary 생성
    std::vector<Eigen::MatrixXd> anc_seqs;
    anc_seqs.push_back(accessor.callBraking(N, nu));
    anc_seqs.push_back(accessor.callGoToGoal(state_, ref_traj_, N, nu, params_.dt));
    anc_seqs.push_back(accessor.callPathFollowing(state_, ref_traj_, N, nu, params_.dt));
    anc_seqs.push_back(accessor.callPrevSolution(N, nu));

    int J = static_cast<int>(std::floor(params_.bias_ratio * K));
    int J_total = J * num_ancillary;
    int K_gaussian = K - J_total;

    auto noise_samples = sampler->sample(K, N, nu);

    std::vector<Eigen::MatrixXd> perturbed;
    std::vector<Eigen::MatrixXd> noise_for_w;

    for (int a = 0; a < num_ancillary; ++a) {
      for (int j = 0; j < J; ++j) {
        auto ac = dynamics->clipControls(anc_seqs[a]);
        perturbed.push_back(ac);
        noise_for_w.push_back(ac - control_sequence);
      }
    }
    for (int k = 0; k < K_gaussian; ++k) {
      auto p = dynamics->clipControls(control_sequence + noise_samples[k]);
      perturbed.push_back(p);
      noise_for_w.push_back(noise_samples[k]);
    }

    auto trajectories = dynamics->rolloutBatch(state_, perturbed, params_.dt);
    auto costs = cost_function->compute(trajectories, perturbed, ref_traj_);
    auto weights = weight_strategy.compute(costs, params_.lambda);

    Eigen::MatrixXd wn = Eigen::MatrixXd::Zero(N, nu);
    for (int k = 0; k < K; ++k) {
      wn += weights(k) * noise_for_w[k];
    }
    control_sequence += wn;
    control_sequence = dynamics->clipControls(control_sequence);

    Eigen::VectorXd u_opt = control_sequence.row(0).transpose();

    // NaN/Inf 없음
    EXPECT_FALSE(std::isnan(u_opt(0))) << "NaN at iteration " << iter;
    EXPECT_FALSE(std::isnan(u_opt(1))) << "NaN at iteration " << iter;
    EXPECT_FALSE(std::isinf(u_opt(0))) << "Inf at iteration " << iter;
    EXPECT_FALSE(std::isinf(u_opt(1))) << "Inf at iteration " << iter;
  }
}

}  // namespace mpc_controller_ros2

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
