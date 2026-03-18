// =============================================================================
// Multi-Agent MPPI Unit Tests
//
// 15 gtest: AgentTrajectoryManager(8) + InterAgentCost(5) + Integration(2)
// =============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <memory>

#include "mpc_controller_ros2/agent_trajectory_manager.hpp"
#include "mpc_controller_ros2/inter_agent_cost.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/sampling.hpp"
#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"

using namespace mpc_controller_ros2;

// =============================================================================
// Helper: MPPI 컴포넌트 셋업 (Integration 테스트용)
// =============================================================================

struct MultiAgentTestSetup
{
  MPPIParams params;
  std::unique_ptr<BatchDynamicsWrapper> dynamics;
  std::unique_ptr<CompositeMPPICost> cost_function;
  std::unique_ptr<GaussianSampler> sampler;

  void init(int K = 64, int N = 10)
  {
    params = MPPIParams();
    params.N = N;
    params.dt = 0.1;
    params.K = K;
    params.lambda = 10.0;
    params.motion_model = "diff_drive";

    params.noise_sigma = Eigen::Vector2d(0.5, 0.5);
    params.Q = Eigen::MatrixXd::Identity(3, 3) * 10.0;
    params.Q(2, 2) = 1.0;
    params.Qf = params.Q * 2.0;
    params.R = Eigen::MatrixXd::Identity(2, 2) * 0.1;
    params.R_rate = Eigen::MatrixXd::Identity(2, 2) * 1.0;

    auto model = MotionModelFactory::create("diff_drive", params);
    dynamics = std::make_unique<BatchDynamicsWrapper>(
      params, std::shared_ptr<MotionModel>(std::move(model)));
    sampler = std::make_unique<GaussianSampler>(params.noise_sigma, 42);

    cost_function = std::make_unique<CompositeMPPICost>();
    cost_function->addCost(std::make_unique<StateTrackingCost>(params.Q));
    cost_function->addCost(std::make_unique<TerminalCost>(params.Qf));
    cost_function->addCost(std::make_unique<ControlEffortCost>(params.R));
  }
};

// =============================================================================
// AgentTrajectoryManager Tests
// =============================================================================

// 1. Construction — own_id, numAgents=0
TEST(MultiAgentMPPI, ManagerConstruction)
{
  AgentTrajectoryManager manager(0, 2.0);
  EXPECT_EQ(manager.ownId(), 0);
  EXPECT_EQ(manager.numAgents(), 0u);
}

// 2. UpdateOwnTrajectory — stores own, getOtherAgents excludes own
TEST(MultiAgentMPPI, UpdateOwnTrajectory)
{
  AgentTrajectoryManager manager(0, 2.0);

  Eigen::MatrixXd traj(5, 3);
  traj.setZero();
  traj(0, 0) = 1.0;  // x position
  Eigen::Vector2d vel(0.5, 0.0);

  manager.updateOwnTrajectory(traj, vel, 0.1);

  // Own trajectory is not counted as "other"
  EXPECT_EQ(manager.numAgents(), 0u);
  auto others = manager.getOtherAgents();
  EXPECT_TRUE(others.empty());
}

// 3. UpdateOtherAgent — store and retrieve
TEST(MultiAgentMPPI, UpdateOtherAgent)
{
  AgentTrajectoryManager manager(0, 2.0);

  AgentPrediction pred;
  pred.agent_id = 1;
  pred.trajectory = Eigen::MatrixXd::Zero(5, 3);
  pred.trajectory(0, 0) = 2.0;
  pred.trajectory(0, 1) = 3.0;
  pred.velocity = Eigen::Vector2d(0.3, 0.1);
  pred.radius = 0.25;
  pred.timestamp = 10.0;
  pred.dt = 0.1;

  manager.updateAgentTrajectory(1, pred);

  EXPECT_EQ(manager.numAgents(), 1u);
  auto others = manager.getOtherAgents();
  ASSERT_EQ(others.size(), 1u);
  EXPECT_EQ(others[0].agent_id, 1);
  EXPECT_NEAR(others[0].trajectory(0, 0), 2.0, 1e-10);
  EXPECT_NEAR(others[0].trajectory(0, 1), 3.0, 1e-10);
}

// 4. PruneStale — add agent, prune after timeout, verify removed
TEST(MultiAgentMPPI, PruneStale)
{
  AgentTrajectoryManager manager(0, 2.0);

  AgentPrediction pred;
  pred.agent_id = 1;
  pred.trajectory = Eigen::MatrixXd::Zero(5, 3);
  pred.timestamp = 10.0;
  pred.dt = 0.1;

  manager.updateAgentTrajectory(1, pred);
  EXPECT_EQ(manager.numAgents(), 1u);

  // Prune at time 11.0 — within timeout (2.0)
  manager.pruneStale(11.0);
  EXPECT_EQ(manager.numAgents(), 1u);

  // Prune at time 13.0 — exceeds timeout
  manager.pruneStale(13.0);
  EXPECT_EQ(manager.numAgents(), 0u);
}

// 5. ToObstaclesWithVelocity — correct position/velocity conversion
TEST(MultiAgentMPPI, ToObstaclesWithVelocity)
{
  AgentTrajectoryManager manager(0, 2.0);

  AgentPrediction pred;
  pred.agent_id = 1;
  pred.trajectory = Eigen::MatrixXd::Zero(5, 3);
  pred.trajectory(0, 0) = 3.0;
  pred.trajectory(0, 1) = 4.0;
  pred.velocity = Eigen::Vector2d(0.5, -0.2);
  pred.radius = 0.3;
  pred.timestamp = 10.0;

  manager.updateAgentTrajectory(1, pred);

  auto [obs, vels] = manager.toObstaclesWithVelocity(10.0);
  ASSERT_EQ(obs.size(), 1u);
  ASSERT_EQ(vels.size(), 1u);
  EXPECT_NEAR(obs[0](0), 3.0, 1e-10);  // x
  EXPECT_NEAR(obs[0](1), 4.0, 1e-10);  // y
  EXPECT_NEAR(obs[0](2), 0.3, 1e-10);  // radius
  EXPECT_NEAR(vels[0](0), 0.5, 1e-10);
  EXPECT_NEAR(vels[0](1), -0.2, 1e-10);
}

// 6. ExcludesOwnId — own not in obstacles
TEST(MultiAgentMPPI, ExcludesOwnId)
{
  AgentTrajectoryManager manager(0, 2.0);

  // Add own trajectory
  Eigen::MatrixXd own_traj(5, 3);
  own_traj.setZero();
  manager.updateOwnTrajectory(own_traj, Eigen::Vector2d(0.0, 0.0), 0.1);

  // Should not appear in obstacles
  auto [obs, vels] = manager.toObstaclesWithVelocity(0.0);
  EXPECT_TRUE(obs.empty());
  EXPECT_TRUE(vels.empty());
}

// 7. MultipleAgents — 3 agents, getOtherAgents returns 2
TEST(MultiAgentMPPI, MultipleAgents)
{
  AgentTrajectoryManager manager(0, 2.0);

  for (int i = 1; i <= 2; ++i) {
    AgentPrediction pred;
    pred.agent_id = i;
    pred.trajectory = Eigen::MatrixXd::Zero(5, 3);
    pred.trajectory(0, 0) = static_cast<double>(i);
    pred.velocity = Eigen::Vector2d(0.0, 0.0);
    pred.radius = 0.2;
    pred.timestamp = 10.0;
    manager.updateAgentTrajectory(i, pred);
  }

  EXPECT_EQ(manager.numAgents(), 2u);
  auto others = manager.getOtherAgents();
  EXPECT_EQ(others.size(), 2u);

  auto [obs, vels] = manager.toObstaclesWithVelocity(10.0);
  EXPECT_EQ(obs.size(), 2u);
}

// 8. Reset — clear all
TEST(MultiAgentMPPI, Reset)
{
  AgentTrajectoryManager manager(0, 2.0);

  AgentPrediction pred;
  pred.agent_id = 1;
  pred.trajectory = Eigen::MatrixXd::Zero(5, 3);
  pred.timestamp = 10.0;
  manager.updateAgentTrajectory(1, pred);

  // Also add own
  manager.updateOwnTrajectory(Eigen::MatrixXd::Zero(5, 3),
                               Eigen::Vector2d(0.0, 0.0), 0.1);

  EXPECT_EQ(manager.numAgents(), 1u);

  manager.reset();
  EXPECT_EQ(manager.numAgents(), 0u);
  EXPECT_TRUE(manager.getOtherAgents().empty());
}

// =============================================================================
// InterAgentCost Tests
// =============================================================================

// Helper: 직선 궤적 생성
static std::vector<Eigen::MatrixXd> makeTrajectories(
  int K, int N, double x0, double y0, double dx)
{
  std::vector<Eigen::MatrixXd> trajs(K);
  for (int k = 0; k < K; ++k) {
    trajs[k] = Eigen::MatrixXd::Zero(N + 1, 3);
    for (int t = 0; t <= N; ++t) {
      trajs[k](t, 0) = x0 + t * dx;
      trajs[k](t, 1) = y0;
    }
  }
  return trajs;
}

static std::vector<Eigen::MatrixXd> makeControls(int K, int N)
{
  std::vector<Eigen::MatrixXd> ctrls(K);
  for (int k = 0; k < K; ++k) {
    ctrls[k] = Eigen::MatrixXd::Zero(N, 2);
  }
  return ctrls;
}

// 9. ZeroCostNoAgents — no agents → costs all zero
TEST(MultiAgentMPPI, ZeroCostNoAgents)
{
  AgentTrajectoryManager manager(0, 2.0);
  InterAgentCost cost(&manager, 500.0, 0.3, 0.2);

  int K = 10, N = 5;
  auto trajs = makeTrajectories(K, N, 0.0, 0.0, 0.1);
  auto ctrls = makeControls(K, N);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, 3);

  Eigen::VectorXd costs = cost.compute(trajs, ctrls, ref);
  EXPECT_EQ(costs.size(), K);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-10);
}

// 10. HighCostNearby — agent at (0.1, 0) → high cost
TEST(MultiAgentMPPI, HighCostNearby)
{
  AgentTrajectoryManager manager(0, 2.0);

  // Agent 1 at (0.1, 0) — very close
  AgentPrediction pred;
  pred.agent_id = 1;
  pred.trajectory = Eigen::MatrixXd::Zero(6, 3);
  pred.trajectory.col(0).setConstant(0.1);  // x = 0.1 for all timesteps
  pred.velocity = Eigen::Vector2d(0.0, 0.0);
  pred.radius = 0.2;
  pred.timestamp = 10.0;
  manager.updateAgentTrajectory(1, pred);

  InterAgentCost cost(&manager, 500.0, 0.3, 0.2);

  int K = 4, N = 5;
  // Robot trajectories at origin
  auto trajs = makeTrajectories(K, N, 0.0, 0.0, 0.0);
  auto ctrls = makeControls(K, N);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, 3);

  Eigen::VectorXd costs = cost.compute(trajs, ctrls, ref);

  // dist = 0.1, d_safe = 0.2 + 0.2 + 0.3 = 0.7
  // violation = 0.7 - 0.1 = 0.6 at each timestep
  // cost per step = 500 * 0.6^2 = 180.0
  for (int k = 0; k < K; ++k) {
    EXPECT_GT(costs(k), 100.0) << "Should have high cost for nearby agent";
  }
}

// 11. ZeroCostFaraway — agent at (100, 100) → zero cost
TEST(MultiAgentMPPI, ZeroCostFaraway)
{
  AgentTrajectoryManager manager(0, 2.0);

  AgentPrediction pred;
  pred.agent_id = 1;
  pred.trajectory = Eigen::MatrixXd::Zero(6, 3);
  pred.trajectory.col(0).setConstant(100.0);  // x = 100
  pred.trajectory.col(1).setConstant(100.0);  // y = 100
  pred.velocity = Eigen::Vector2d(0.0, 0.0);
  pred.radius = 0.2;
  pred.timestamp = 10.0;
  manager.updateAgentTrajectory(1, pred);

  InterAgentCost cost(&manager, 500.0, 0.3, 0.2);

  int K = 4, N = 5;
  auto trajs = makeTrajectories(K, N, 0.0, 0.0, 0.0);
  auto ctrls = makeControls(K, N);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, 3);

  Eigen::VectorXd costs = cost.compute(trajs, ctrls, ref);
  EXPECT_NEAR(costs.sum(), 0.0, 1e-10);
}

// 12. MultipleAgentsCostAccumulate — 2 agents, costs accumulate
TEST(MultiAgentMPPI, MultipleAgentsCostAccumulate)
{
  AgentTrajectoryManager manager(0, 2.0);

  // Agent 1 nearby
  AgentPrediction pred1;
  pred1.agent_id = 1;
  pred1.trajectory = Eigen::MatrixXd::Zero(6, 3);
  pred1.trajectory.col(0).setConstant(0.1);
  pred1.velocity = Eigen::Vector2d(0.0, 0.0);
  pred1.radius = 0.2;
  pred1.timestamp = 10.0;
  manager.updateAgentTrajectory(1, pred1);

  InterAgentCost cost_single(&manager, 500.0, 0.3, 0.2);

  int K = 4, N = 5;
  auto trajs = makeTrajectories(K, N, 0.0, 0.0, 0.0);
  auto ctrls = makeControls(K, N);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, 3);

  Eigen::VectorXd costs_one = cost_single.compute(trajs, ctrls, ref);

  // Agent 2 also nearby (symmetric position)
  AgentPrediction pred2;
  pred2.agent_id = 2;
  pred2.trajectory = Eigen::MatrixXd::Zero(6, 3);
  pred2.trajectory.col(1).setConstant(0.1);  // y = 0.1
  pred2.velocity = Eigen::Vector2d(0.0, 0.0);
  pred2.radius = 0.2;
  pred2.timestamp = 10.0;
  manager.updateAgentTrajectory(2, pred2);

  InterAgentCost cost_both(&manager, 500.0, 0.3, 0.2);
  Eigen::VectorXd costs_two = cost_both.compute(trajs, ctrls, ref);

  // Two agents should produce higher cost than one
  for (int k = 0; k < K; ++k) {
    EXPECT_GT(costs_two(k), costs_one(k));
  }
}

// 13. TimeAligned — costs evaluated at matching timesteps
TEST(MultiAgentMPPI, TimeAligned)
{
  AgentTrajectoryManager manager(0, 2.0);

  // Agent starts far (x=10) at t=0, gets close (x=0.1) at t=3
  AgentPrediction pred;
  pred.agent_id = 1;
  pred.trajectory = Eigen::MatrixXd::Zero(6, 3);
  for (int t = 0; t < 6; ++t) {
    pred.trajectory(t, 0) = (t < 3) ? 10.0 : 0.1;
  }
  pred.velocity = Eigen::Vector2d(0.0, 0.0);
  pred.radius = 0.2;
  pred.timestamp = 10.0;
  manager.updateAgentTrajectory(1, pred);

  InterAgentCost cost(&manager, 500.0, 0.3, 0.2);

  int K = 1, N = 5;
  // Robot stays at origin
  auto trajs = makeTrajectories(K, N, 0.0, 0.0, 0.0);
  auto ctrls = makeControls(K, N);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, 3);

  Eigen::VectorXd costs = cost.compute(trajs, ctrls, ref);
  // Cost should be > 0 only for timesteps 3+ where agent is close
  EXPECT_GT(costs(0), 0.0);

  // Compare with agent always far
  AgentPrediction pred_far;
  pred_far.agent_id = 1;
  pred_far.trajectory = Eigen::MatrixXd::Zero(6, 3);
  pred_far.trajectory.col(0).setConstant(10.0);
  pred_far.velocity = Eigen::Vector2d(0.0, 0.0);
  pred_far.radius = 0.2;
  pred_far.timestamp = 10.0;
  manager.updateAgentTrajectory(1, pred_far);

  InterAgentCost cost_far(&manager, 500.0, 0.3, 0.2);
  Eigen::VectorXd costs_far = cost_far.compute(trajs, ctrls, ref);
  // Always-far should have lower cost
  EXPECT_LT(costs_far(0), costs(0));
}

// =============================================================================
// Integration Tests (without ROS2)
// =============================================================================

// 14. DisabledFallback — multi_agent_enabled=false → same as base MPPI
TEST(MultiAgentMPPI, DisabledFallback)
{
  MultiAgentTestSetup setup;
  setup.init(64, 10);

  int K = setup.params.K;
  int N = setup.params.N;
  int nu = setup.dynamics->model().controlDim();

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(3);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, 3);
  ref.col(0).setConstant(2.0);

  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(N, nu);

  // Run MPPI without any multi-agent cost
  auto noise = setup.sampler->sample(K, N, nu);
  std::vector<Eigen::MatrixXd> perturbed(K);
  for (int k = 0; k < K; ++k) {
    perturbed[k] = setup.dynamics->clipControls(control_seq + noise[k]);
  }

  std::vector<Eigen::MatrixXd> trajectories;
  setup.dynamics->rolloutBatchInPlace(x0, perturbed, setup.params.dt, trajectories);

  Eigen::VectorXd costs = setup.cost_function->compute(trajectories, perturbed, ref);

  VanillaMPPIWeights weight_comp;
  Eigen::VectorXd weights = weight_comp.compute(costs, setup.params.lambda);

  // Should produce valid weights
  EXPECT_NEAR(weights.sum(), 1.0, 1e-5);
  EXPECT_EQ(weights.size(), K);
}

// 15. ManagerIntegration — InterAgentCost integrated into CompositeCost
TEST(MultiAgentMPPI, ManagerIntegration)
{
  MultiAgentTestSetup setup;
  setup.init(64, 10);

  int K = setup.params.K;
  int N = setup.params.N;
  int nu = setup.dynamics->model().controlDim();

  // Create manager and add nearby agent
  AgentTrajectoryManager manager(0, 2.0);
  AgentPrediction pred;
  pred.agent_id = 1;
  pred.trajectory = Eigen::MatrixXd::Zero(N + 1, 3);
  pred.trajectory.col(0).setConstant(0.15);  // Very close x position
  pred.velocity = Eigen::Vector2d(0.0, 0.0);
  pred.radius = 0.2;
  pred.timestamp = 10.0;
  manager.updateAgentTrajectory(1, pred);

  // Cost without inter-agent
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(3);
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(N + 1, 3);
  ref.col(0).setConstant(2.0);

  Eigen::MatrixXd control_seq = Eigen::MatrixXd::Zero(N, nu);
  auto noise = setup.sampler->sample(K, N, nu);
  std::vector<Eigen::MatrixXd> perturbed(K);
  for (int k = 0; k < K; ++k) {
    perturbed[k] = setup.dynamics->clipControls(control_seq + noise[k]);
  }

  std::vector<Eigen::MatrixXd> trajectories;
  setup.dynamics->rolloutBatchInPlace(x0, perturbed, setup.params.dt, trajectories);

  Eigen::VectorXd costs_base = setup.cost_function->compute(trajectories, perturbed, ref);

  // Add inter-agent cost
  setup.cost_function->addCost(std::make_unique<InterAgentCost>(
    &manager, 500.0, 0.3, 0.2));

  Eigen::VectorXd costs_with_agent = setup.cost_function->compute(
    trajectories, perturbed, ref);

  // Costs should increase when nearby agent is present
  double total_base = costs_base.sum();
  double total_with = costs_with_agent.sum();
  EXPECT_GT(total_with, total_base)
    << "InterAgentCost should increase total costs when agent is nearby";
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
