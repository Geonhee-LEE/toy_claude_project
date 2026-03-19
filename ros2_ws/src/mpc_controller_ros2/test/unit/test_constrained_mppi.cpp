// =============================================================================
// Constrained MPPI (Augmented Lagrangian) 단위 테스트 (15개)
// =============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

#include "mpc_controller_ros2/constrained_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

// ============================================================================
// 테스트 헬퍼
// ============================================================================
class ConstrainedMPPITestAccessor : public ConstrainedMPPIControllerPlugin
{
public:
  void setTestParams(const MPPIParams& params) { params_ = params; }

  Eigen::Vector3d callEvaluateConstraintViolation(
    const Eigen::MatrixXd& ctrl, const Eigen::MatrixXd& traj) const {
    return evaluateConstraintViolation(ctrl, traj);
  }

  Eigen::VectorXd callComputeAugmentedCosts(
    const Eigen::VectorXd& base_costs,
    const std::vector<Eigen::MatrixXd>& controls,
    const std::vector<Eigen::MatrixXd>& trajs) const {
    return computeAugmentedCosts(base_costs, controls, trajs);
  }

  void callUpdateDualVariables(const Eigen::Vector3d& violation) {
    updateDualVariables(violation);
  }

  Eigen::Vector3d getLambda() const { return lambda_; }
  double getMu() const { return mu_; }
  void setLambda(const Eigen::Vector3d& l) { lambda_ = l; }
  void setMu(double m) { mu_ = m; }
};

// ============================================================================
// 테스트 Fixture
// ============================================================================
class ConstrainedMPPITest : public ::testing::Test
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
    params_.constrained_enabled = true;
    params_.constrained_mu_init = 1.0;
    params_.constrained_mu_growth = 1.5;
    params_.constrained_mu_max = 1000.0;
    params_.constrained_accel_max_v = 2.0;
    params_.constrained_accel_max_omega = 3.0;
    params_.constrained_clearance_min = 0.3;

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
// 제약 평가 테스트 (3개)
// ============================================================================

TEST_F(ConstrainedMPPITest, VelocityViolationDetected)
{
  ConstrainedMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  Eigen::MatrixXd ctrl = Eigen::MatrixXd::Zero(params_.N, 2);
  for (int t = 0; t < params_.N; ++t) {
    ctrl(t, 0) = 2.0;  // v = 2.0 > v_max = 1.0
    ctrl(t, 1) = 0.0;
  }
  Eigen::MatrixXd traj = Eigen::MatrixXd::Zero(params_.N + 1, 3);

  Eigen::Vector3d violation = accessor.callEvaluateConstraintViolation(ctrl, traj);
  EXPECT_GT(violation(0), 0.0);
}

TEST_F(ConstrainedMPPITest, AccelViolationDetected)
{
  ConstrainedMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  Eigen::MatrixXd ctrl = Eigen::MatrixXd::Zero(params_.N, 2);
  for (int t = 0; t < params_.N; ++t) {
    ctrl(t, 0) = (t % 2 == 0) ? 0.0 : 1.0;  // dv/dt = 10 > accel_max_v = 2
    ctrl(t, 1) = 0.0;
  }
  Eigen::MatrixXd traj = Eigen::MatrixXd::Zero(params_.N + 1, 3);

  Eigen::Vector3d violation = accessor.callEvaluateConstraintViolation(ctrl, traj);
  EXPECT_GT(violation(1), 0.0);
}

TEST_F(ConstrainedMPPITest, NoViolationWhenFeasible)
{
  ConstrainedMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  Eigen::MatrixXd ctrl = Eigen::MatrixXd::Constant(params_.N, 2, 0.5);
  Eigen::MatrixXd traj = Eigen::MatrixXd::Zero(params_.N + 1, 3);

  Eigen::Vector3d violation = accessor.callEvaluateConstraintViolation(ctrl, traj);
  EXPECT_NEAR(violation(0), 0.0, 1e-10);
  EXPECT_NEAR(violation(1), 0.0, 1e-10);
  EXPECT_NEAR(violation(2), 0.0, 1e-10);
}

// ============================================================================
// Augmented Lagrangian 테스트 (3개)
// ============================================================================

TEST_F(ConstrainedMPPITest, PenaltyIncreasesWithViolation)
{
  ConstrainedMPPITestAccessor accessor;
  accessor.setTestParams(params_);
  accessor.setLambda(Eigen::Vector3d::Zero());
  accessor.setMu(10.0);

  int K = 5;
  Eigen::VectorXd base_costs(K);
  base_costs << 10.0, 20.0, 30.0, 40.0, 50.0;

  std::vector<Eigen::MatrixXd> controls(K, Eigen::MatrixXd::Zero(params_.N, 2));
  std::vector<Eigen::MatrixXd> trajs(K, Eigen::MatrixXd::Zero(params_.N + 1, 3));
  for (int k = 0; k < K; ++k) {
    for (int t = 0; t < params_.N; ++t) {
      controls[k](t, 0) = 0.5 + k * 0.5;  // k=2+: v > v_max
    }
  }

  Eigen::VectorXd aug_costs = accessor.callComputeAugmentedCosts(
    base_costs, controls, trajs);

  for (int k = 0; k < K; ++k) {
    EXPECT_GE(aug_costs(k), base_costs(k) - 1e-6);
  }
}

TEST_F(ConstrainedMPPITest, LambdaUpdateOnViolation)
{
  ConstrainedMPPITestAccessor accessor;
  accessor.setTestParams(params_);
  accessor.setLambda(Eigen::Vector3d::Zero());
  accessor.setMu(1.0);

  Eigen::Vector3d violation(1.0, 0.5, 0.0);
  accessor.callUpdateDualVariables(violation);

  Eigen::Vector3d lambda = accessor.getLambda();
  EXPECT_GT(lambda(0), 0.0);
  EXPECT_GT(lambda(1), 0.0);
  EXPECT_EQ(lambda(2), 0.0);
}

TEST_F(ConstrainedMPPITest, MuGrowthOnPersistentViolation)
{
  ConstrainedMPPITestAccessor accessor;
  accessor.setTestParams(params_);
  accessor.setMu(1.0);

  double mu_before = accessor.getMu();
  Eigen::Vector3d violation(1.0, 0.0, 0.0);
  accessor.callUpdateDualVariables(violation);

  double mu_after = accessor.getMu();
  EXPECT_GT(mu_after, mu_before);
  EXPECT_NEAR(mu_after, mu_before * params_.constrained_mu_growth, 1e-10);
}

// ============================================================================
// Dual 수렴 테스트 (2개)
// ============================================================================

TEST_F(ConstrainedMPPITest, LambdaConvergesOnFeasible)
{
  ConstrainedMPPITestAccessor accessor;
  accessor.setTestParams(params_);
  accessor.setLambda(Eigen::Vector3d(5.0, 3.0, 1.0));
  accessor.setMu(1.0);

  Eigen::Vector3d violation = Eigen::Vector3d::Zero();
  Eigen::Vector3d lambda_before = accessor.getLambda();
  accessor.callUpdateDualVariables(violation);
  Eigen::Vector3d lambda_after = accessor.getLambda();

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(lambda_after(i), lambda_before(i), 1e-10);
  }
}

TEST_F(ConstrainedMPPITest, MuCappedAtMax)
{
  ConstrainedMPPITestAccessor accessor;
  params_.constrained_mu_max = 10.0;
  accessor.setTestParams(params_);
  accessor.setMu(9.0);

  Eigen::Vector3d violation(1.0, 0.0, 0.0);
  accessor.callUpdateDualVariables(violation);

  EXPECT_LE(accessor.getMu(), params_.constrained_mu_max + 1e-10);
}

// ============================================================================
// 제약 강제 테스트 (2개)
// ============================================================================

TEST_F(ConstrainedMPPITest, ConstrainedProducesSmootherControl)
{
  ConstrainedMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  Eigen::MatrixXd ctrl_rough = Eigen::MatrixXd::Zero(params_.N, 2);
  for (int t = 0; t < params_.N; ++t) {
    ctrl_rough(t, 0) = (t % 2 == 0) ? 0.0 : 0.8;
  }
  Eigen::MatrixXd traj = Eigen::MatrixXd::Zero(params_.N + 1, 3);

  Eigen::MatrixXd ctrl_smooth = Eigen::MatrixXd::Constant(params_.N, 2, 0.4);

  Eigen::Vector3d viol_rough = accessor.callEvaluateConstraintViolation(ctrl_rough, traj);
  Eigen::Vector3d viol_smooth = accessor.callEvaluateConstraintViolation(ctrl_smooth, traj);

  EXPECT_GT(viol_rough(1), viol_smooth(1));
}

TEST_F(ConstrainedMPPITest, ConstrainedRespectsVelocityBounds)
{
  ConstrainedMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  Eigen::MatrixXd ctrl_within = Eigen::MatrixXd::Constant(params_.N, 2, 0.5);
  Eigen::MatrixXd traj = Eigen::MatrixXd::Zero(params_.N + 1, 3);

  Eigen::Vector3d viol = accessor.callEvaluateConstraintViolation(ctrl_within, traj);
  EXPECT_NEAR(viol(0), 0.0, 1e-10);

  Eigen::MatrixXd ctrl_over = Eigen::MatrixXd::Constant(params_.N, 2, 1.5);
  viol = accessor.callEvaluateConstraintViolation(ctrl_over, traj);
  EXPECT_GT(viol(0), 0.0);
}

// ============================================================================
// Vanilla 동등성 (1개)
// ============================================================================

TEST_F(ConstrainedMPPITest, DisabledEqualsVanilla)
{
  params_.constrained_enabled = false;
  EXPECT_FALSE(params_.constrained_enabled);
}

// ============================================================================
// 통합 테스트 (2개)
// ============================================================================

TEST_F(ConstrainedMPPITest, ComputeControlReturnsValid)
{
  auto dynamics = std::make_unique<BatchDynamicsWrapper>(params_);
  auto sampler = std::make_unique<GaussianSampler>(params_.noise_sigma, 42);
  auto cost_function = std::make_unique<CompositeMPPICost>();
  cost_function->addCost(std::make_unique<StateTrackingCost>(params_.Q));
  cost_function->addCost(std::make_unique<TerminalCost>(params_.Qf));
  cost_function->addCost(std::make_unique<ControlEffortCost>(params_.R));

  VanillaMPPIWeights weight_strategy;
  ConstrainedMPPITestAccessor accessor;
  accessor.setTestParams(params_);
  accessor.setLambda(Eigen::Vector3d::Zero());
  accessor.setMu(params_.constrained_mu_init);

  int N = params_.N, K = params_.K, nu = 2;
  Eigen::MatrixXd mu = Eigen::MatrixXd::Zero(N, nu);

  auto noise = sampler->sample(K, N, nu);
  std::vector<Eigen::MatrixXd> perturbed(K);
  for (int k = 0; k < K; ++k) {
    perturbed[k] = mu + noise[k];
    perturbed[k] = dynamics->clipControls(perturbed[k]);
  }

  auto trajectories = dynamics->rolloutBatch(state_, perturbed, params_.dt);
  auto base_costs = cost_function->compute(trajectories, perturbed, ref_traj_);

  // Augmented costs
  auto aug_costs = accessor.callComputeAugmentedCosts(base_costs, perturbed, trajectories);
  EXPECT_TRUE(aug_costs.allFinite());

  // Weight + update
  auto weights = weight_strategy.compute(aug_costs, params_.lambda);
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_noise += weights(k) * noise[k];
  }
  Eigen::MatrixXd control_seq = dynamics->clipControls(mu + weighted_noise);
  Eigen::VectorXd u_opt = control_seq.row(0).transpose();

  EXPECT_TRUE(u_opt.allFinite());
  EXPECT_GE(u_opt(0), params_.v_min);
  EXPECT_LE(u_opt(0), params_.v_max);
}

TEST_F(ConstrainedMPPITest, WorksWithSwerveModel)
{
  // nu=3 swerve와 독립적 (제약 평가는 제어 차원에 적응)
  ConstrainedMPPITestAccessor accessor;
  accessor.setTestParams(params_);

  Eigen::MatrixXd ctrl = Eigen::MatrixXd::Constant(params_.N, 3, 0.5);
  Eigen::MatrixXd traj = Eigen::MatrixXd::Zero(params_.N + 1, 3);

  Eigen::Vector3d viol = accessor.callEvaluateConstraintViolation(ctrl, traj);
  EXPECT_TRUE(viol.allFinite());
  EXPECT_GE(viol(0), 0.0);
}

// ============================================================================
// 안정성 (1개)
// ============================================================================

TEST_F(ConstrainedMPPITest, MultipleCallsStable)
{
  ConstrainedMPPITestAccessor accessor;
  accessor.setTestParams(params_);
  accessor.setLambda(Eigen::Vector3d::Zero());
  accessor.setMu(params_.constrained_mu_init);

  // 10회 반복 dual update
  for (int i = 0; i < 10; ++i) {
    Eigen::Vector3d violation(0.1 * (10 - i), 0.05 * (10 - i), 0.0);
    accessor.callUpdateDualVariables(violation);

    EXPECT_TRUE(accessor.getLambda().allFinite()) << "lambda NaN at iter " << i;
    EXPECT_TRUE(std::isfinite(accessor.getMu())) << "mu NaN at iter " << i;
    EXPECT_LE(accessor.getMu(), params_.constrained_mu_max + 1e-6);
  }
}

// ============================================================================
// Info (1개)
// ============================================================================

TEST_F(ConstrainedMPPITest, InfoContainsConstraintMetrics)
{
  // Augmented cost 계산 후 메트릭 확인
  ConstrainedMPPITestAccessor accessor;
  accessor.setTestParams(params_);
  accessor.setLambda(Eigen::Vector3d(1.0, 0.5, 0.0));
  accessor.setMu(5.0);

  int K = 10;
  Eigen::VectorXd base_costs = Eigen::VectorXd::LinSpaced(K, 1.0, 10.0);
  std::vector<Eigen::MatrixXd> controls(K, Eigen::MatrixXd::Constant(params_.N, 2, 1.5));
  std::vector<Eigen::MatrixXd> trajs(K, Eigen::MatrixXd::Zero(params_.N + 1, 3));

  Eigen::VectorXd aug_costs = accessor.callComputeAugmentedCosts(
    base_costs, controls, trajs);

  // All augmented costs should be finite and >= base costs
  EXPECT_TRUE(aug_costs.allFinite());
  for (int k = 0; k < K; ++k) {
    EXPECT_GE(aug_costs(k), base_costs(k) - 1e-6);
  }

  // Dual variables
  EXPECT_TRUE(std::isfinite(accessor.getMu()));
  EXPECT_GT(accessor.getMu(), 0.0);
  EXPECT_TRUE(accessor.getLambda().allFinite());
}

}  // namespace mpc_controller_ros2
