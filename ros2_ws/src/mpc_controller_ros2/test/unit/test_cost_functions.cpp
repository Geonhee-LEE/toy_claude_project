#include <gtest/gtest.h>
#include <cmath>
#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"

namespace mpc_controller_ros2
{

class CostFunctionsTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Create simple test data
    K_ = 2;  // 2 samples
    N_ = 3;  // 3 timesteps (so N+1=4 states)

    // Reference trajectory: straight line
    reference_ = Eigen::MatrixXd::Zero(N_ + 1, 3);
    for (int t = 0; t <= N_; ++t) {
      reference_(t, 0) = t * 0.1;  // x increases
      reference_(t, 1) = 0.0;      // y = 0
      reference_(t, 2) = 0.0;      // theta = 0
    }

    // Sample 1: perfect tracking
    Eigen::MatrixXd traj1 = reference_;

    // Sample 2: offset trajectory
    Eigen::MatrixXd traj2 = Eigen::MatrixXd::Zero(N_ + 1, 3);
    for (int t = 0; t <= N_; ++t) {
      traj2(t, 0) = t * 0.1 + 0.1;  // x offset by 0.1
      traj2(t, 1) = 0.05;           // y offset by 0.05
      traj2(t, 2) = 0.1;            // theta offset by 0.1
    }

    trajectories_.push_back(traj1);
    trajectories_.push_back(traj2);

    // Controls: constant
    Eigen::MatrixXd ctrl1 = Eigen::MatrixXd::Zero(N_, 2);
    ctrl1.col(0).setConstant(1.0);  // v = 1.0
    ctrl1.col(1).setConstant(0.0);  // omega = 0.0

    Eigen::MatrixXd ctrl2 = Eigen::MatrixXd::Zero(N_, 2);
    ctrl2.col(0).setConstant(0.5);  // v = 0.5
    ctrl2.col(1).setConstant(0.2);  // omega = 0.2

    controls_.push_back(ctrl1);
    controls_.push_back(ctrl2);

    // Weight matrices
    Q_ = Eigen::Matrix3d::Identity();
    Q_(0, 0) = 10.0;
    Q_(1, 1) = 10.0;
    Q_(2, 2) = 1.0;

    Qf_ = 2.0 * Q_;

    R_ = Eigen::Matrix2d::Identity();
    R_(0, 0) = 0.1;
    R_(1, 1) = 0.1;

    R_rate_ = Eigen::Matrix2d::Identity();
  }

  int K_;
  int N_;
  Eigen::MatrixXd reference_;
  std::vector<Eigen::MatrixXd> trajectories_;
  std::vector<Eigen::MatrixXd> controls_;
  Eigen::Matrix3d Q_;
  Eigen::Matrix3d Qf_;
  Eigen::Matrix2d R_;
  Eigen::Matrix2d R_rate_;
};

TEST_F(CostFunctionsTest, StateTrackingCostPerfectTracking)
{
  StateTrackingCost cost(Q_);

  auto costs = cost.compute(trajectories_, controls_, reference_);

  EXPECT_EQ(costs.size(), K_);

  // Sample 1 (perfect tracking) should have zero cost
  EXPECT_NEAR(costs(0), 0.0, 1e-9);

  // Sample 2 (offset) should have non-zero cost
  EXPECT_GT(costs(1), 0.0);
}

TEST_F(CostFunctionsTest, StateTrackingCostCalculation)
{
  // Simple Q matrix for manual calculation
  Eigen::Matrix3d Q_simple = Eigen::Matrix3d::Zero();
  Q_simple(0, 0) = 1.0;  // Only x error

  StateTrackingCost cost(Q_simple);

  auto costs = cost.compute(trajectories_, controls_, reference_);

  // Sample 2: x offset by 0.1 at each of N timesteps (not N+1, terminal excluded)
  // cost = N * (0.1)^2 * 1.0 = 3 * 0.01 = 0.03
  EXPECT_NEAR(costs(1), 3 * 0.1 * 0.1, 1e-9);
}

TEST_F(CostFunctionsTest, StateTrackingCostAngleNormalization)
{
  // Test angle wrapping
  Eigen::Matrix3d Q_theta = Eigen::Matrix3d::Zero();
  Q_theta(2, 2) = 1.0;  // Only theta error

  StateTrackingCost cost(Q_theta);

  // Create trajectory with angle > π
  std::vector<Eigen::MatrixXd> test_trajs;
  Eigen::MatrixXd traj(2, 3);
  traj << 0.0, 0.0, 0.0,
          0.0, 0.0, 3.5;  // theta = 3.5 rad (> π)
  test_trajs.push_back(traj);

  Eigen::MatrixXd ref(2, 3);
  ref << 0.0, 0.0, 0.0,
         0.0, 0.0, -3.0;  // theta = -3.0 rad

  // Normalized error should be small (both wrap to similar values)
  auto costs = cost.compute(test_trajs, controls_, ref);

  // 3.5 - (-3.0) = 6.5, normalized to ~-0.78
  // Error should be small after normalization
  EXPECT_LT(costs(0), 1.0);
}

TEST_F(CostFunctionsTest, TerminalCostPerfectTracking)
{
  TerminalCost cost(Qf_);

  auto costs = cost.compute(trajectories_, controls_, reference_);

  EXPECT_EQ(costs.size(), K_);

  // Sample 1 should have zero terminal cost
  EXPECT_NEAR(costs(0), 0.0, 1e-9);

  // Sample 2 should have non-zero terminal cost
  EXPECT_GT(costs(1), 0.0);
}

TEST_F(CostFunctionsTest, TerminalCostCalculation)
{
  // Simple Qf for manual calculation
  Eigen::Matrix3d Qf_simple = Eigen::Matrix3d::Zero();
  Qf_simple(0, 0) = 1.0;

  TerminalCost cost(Qf_simple);

  auto costs = cost.compute(trajectories_, controls_, reference_);

  // Sample 2: terminal x error = 0.1
  // cost = (0.1)^2 * 1.0 = 0.01
  EXPECT_NEAR(costs(1), 0.1 * 0.1, 1e-9);
}

TEST_F(CostFunctionsTest, ControlEffortCostZeroControl)
{
  ControlEffortCost cost(R_);

  // Create zero controls
  std::vector<Eigen::MatrixXd> zero_controls;
  zero_controls.push_back(Eigen::MatrixXd::Zero(N_, 2));
  zero_controls.push_back(Eigen::MatrixXd::Zero(N_, 2));

  auto costs = cost.compute(trajectories_, zero_controls, reference_);

  EXPECT_NEAR(costs(0), 0.0, 1e-9);
  EXPECT_NEAR(costs(1), 0.0, 1e-9);
}

TEST_F(CostFunctionsTest, ControlEffortCostCalculation)
{
  // Simple R for manual calculation
  Eigen::Matrix2d R_simple = Eigen::Matrix2d::Zero();
  R_simple(0, 0) = 1.0;  // Only v effort

  ControlEffortCost cost(R_simple);

  auto costs = cost.compute(trajectories_, controls_, reference_);

  // Sample 1: v=1.0 at N timesteps, cost = N * (1.0)^2 * 1.0 = 3
  EXPECT_NEAR(costs(0), 3.0, 1e-9);

  // Sample 2: v=0.5 at N timesteps, cost = N * (0.5)^2 * 1.0 = 0.75
  EXPECT_NEAR(costs(1), 3 * 0.5 * 0.5, 1e-9);
}

TEST_F(CostFunctionsTest, ControlRateCostConstantControl)
{
  ControlRateCost cost(R_rate_);

  // Create constant controls (zero rate)
  std::vector<Eigen::MatrixXd> const_controls;
  Eigen::MatrixXd ctrl = Eigen::MatrixXd::Ones(N_, 2) * 0.5;
  const_controls.push_back(ctrl);

  auto costs = cost.compute(trajectories_, const_controls, reference_);

  // Constant control → zero rate → zero cost
  EXPECT_NEAR(costs(0), 0.0, 1e-9);
}

TEST_F(CostFunctionsTest, ControlRateCostVaryingControl)
{
  // Simple R_rate for manual calculation
  Eigen::Matrix2d R_rate_simple = Eigen::Matrix2d::Zero();
  R_rate_simple(0, 0) = 1.0;

  ControlRateCost cost(R_rate_simple);

  // Create varying control
  std::vector<Eigen::MatrixXd> varying_controls;
  Eigen::MatrixXd ctrl(3, 2);
  ctrl << 1.0, 0.0,
          1.5, 0.0,  // dv = 0.5
          2.0, 0.0;  // dv = 0.5
  varying_controls.push_back(ctrl);

  auto costs = cost.compute(trajectories_, varying_controls, reference_);

  // cost = 2 * (0.5)^2 * 1.0 = 0.5 (2 differences)
  EXPECT_NEAR(costs(0), 2 * 0.5 * 0.5, 1e-9);
}

TEST_F(CostFunctionsTest, ObstacleCostNoObstacles)
{
  ObstacleCost cost(100.0, 0.5);
  cost.setObstacles({});  // No obstacles

  auto costs = cost.compute(trajectories_, controls_, reference_);

  EXPECT_NEAR(costs(0), 0.0, 1e-9);
  EXPECT_NEAR(costs(1), 0.0, 1e-9);
}

TEST_F(CostFunctionsTest, ObstacleCostNoCollision)
{
  ObstacleCost cost(100.0, 0.5);

  // Obstacle far away from trajectory
  std::vector<Eigen::Vector3d> obstacles;
  obstacles.push_back(Eigen::Vector3d(10.0, 10.0, 0.2));
  cost.setObstacles(obstacles);

  auto costs = cost.compute(trajectories_, controls_, reference_);

  // No collision → zero cost
  EXPECT_NEAR(costs(0), 0.0, 1e-9);
  EXPECT_NEAR(costs(1), 0.0, 1e-9);
}

TEST_F(CostFunctionsTest, ObstacleCostWithCollision)
{
  double weight = 100.0;
  double safety = 0.5;
  ObstacleCost cost(weight, safety);

  // Obstacle at origin (trajectory passes through it)
  std::vector<Eigen::Vector3d> obstacles;
  obstacles.push_back(Eigen::Vector3d(0.0, 0.0, 0.1));  // radius = 0.1
  cost.setObstacles(obstacles);

  auto costs = cost.compute(trajectories_, controls_, reference_);

  // Should have collision cost
  EXPECT_GT(costs(0), 0.0);
}

TEST_F(CostFunctionsTest, CompositeCostEmpty)
{
  CompositeMPPICost composite;

  auto costs = composite.compute(trajectories_, controls_, reference_);

  EXPECT_EQ(costs.size(), K_);
  EXPECT_NEAR(costs(0), 0.0, 1e-9);
  EXPECT_NEAR(costs(1), 0.0, 1e-9);
}

TEST_F(CostFunctionsTest, CompositeCostSingleComponent)
{
  CompositeMPPICost composite;

  Eigen::Matrix3d Q_simple = Eigen::Matrix3d::Identity();
  composite.addCost(std::make_unique<StateTrackingCost>(Q_simple));

  auto composite_costs = composite.compute(trajectories_, controls_, reference_);

  // Should match individual cost
  StateTrackingCost individual(Q_simple);
  auto individual_costs = individual.compute(trajectories_, controls_, reference_);

  EXPECT_NEAR(composite_costs(0), individual_costs(0), 1e-9);
  EXPECT_NEAR(composite_costs(1), individual_costs(1), 1e-9);
}

TEST_F(CostFunctionsTest, CompositeCostMultipleComponents)
{
  CompositeMPPICost composite;

  composite.addCost(std::make_unique<StateTrackingCost>(Q_));
  composite.addCost(std::make_unique<TerminalCost>(Qf_));
  composite.addCost(std::make_unique<ControlEffortCost>(R_));

  auto composite_costs = composite.compute(trajectories_, controls_, reference_);

  // Compute individual costs
  StateTrackingCost state_cost(Q_);
  TerminalCost terminal_cost(Qf_);
  ControlEffortCost control_cost(R_);

  auto state_costs = state_cost.compute(trajectories_, controls_, reference_);
  auto terminal_costs = terminal_cost.compute(trajectories_, controls_, reference_);
  auto control_costs = control_cost.compute(trajectories_, controls_, reference_);

  // Composite should be sum
  EXPECT_NEAR(
    composite_costs(0),
    state_costs(0) + terminal_costs(0) + control_costs(0),
    1e-9
  );
  EXPECT_NEAR(
    composite_costs(1),
    state_costs(1) + terminal_costs(1) + control_costs(1),
    1e-9
  );
}

// ============================================================================
// PreferForwardCost 테스트
// ============================================================================

TEST_F(CostFunctionsTest, PreferForwardCostForwardNoCost)
{
  // v >= 0 시 비용 0
  PreferForwardCost cost(50.0, 0.5);

  // 양의 속도만 포함하는 controls 사용 (controls_ 에서 v>=0)
  auto costs = cost.compute(trajectories_, controls_, reference_);

  // controls_ 의 두 샘플 모두 v >= 0 이므로 비용 0
  EXPECT_NEAR(costs(0), 0.0, 1e-9);
  EXPECT_NEAR(costs(1), 0.0, 1e-9);
}

TEST_F(CostFunctionsTest, PreferForwardCostBackwardPenalty)
{
  // v < 0 시 비용 > 0
  double weight = 50.0;
  PreferForwardCost cost(weight, 0.5);

  // 후진 속도를 포함하는 controls 생성
  std::vector<Eigen::MatrixXd> backward_controls;
  Eigen::MatrixXd ctrl(3, 2);
  ctrl << -0.2, 0.0,
          -0.2, 0.0,
          -0.2, 0.0;
  backward_controls.push_back(ctrl);

  auto costs = cost.compute(trajectories_, backward_controls, reference_);

  // v=-0.2, ratio=0.5: cost_per_step = 50 * (0.5*0.2 + 0.5*0.04) = 50 * 0.12 = 6.0
  // total = 3 * 6.0 = 18.0
  EXPECT_GT(costs(0), 0.0);
  EXPECT_NEAR(costs(0), 18.0, 1e-9);
}

TEST_F(CostFunctionsTest, PreferForwardCostLinearMixed)
{
  // linear_ratio=0.5 비용 검증
  double weight = 100.0;
  double linear_ratio = 0.5;
  PreferForwardCost cost(weight, linear_ratio);

  std::vector<Eigen::MatrixXd> test_controls;
  Eigen::MatrixXd ctrl(1, 2);
  ctrl << -0.5, 0.0;  // v = -0.5
  test_controls.push_back(ctrl);

  auto costs = cost.compute(trajectories_, test_controls, reference_);

  // cost = weight * (ratio * |v| + (1-ratio) * v²)
  //      = 100 * (0.5 * 0.5 + 0.5 * 0.25)
  //      = 100 * (0.25 + 0.125) = 37.5
  EXPECT_NEAR(costs(0), 37.5, 1e-9);
}

TEST_F(CostFunctionsTest, PreferForwardCostQuadraticOnly)
{
  // linear_ratio=0.0 → 기존 이차 비용과 동일
  double weight = 15.0;
  PreferForwardCost cost_new(weight, 0.0);  // 혼합 비용 (이차만)

  std::vector<Eigen::MatrixXd> test_controls;
  Eigen::MatrixXd ctrl(3, 2);
  ctrl << -0.2, 0.0,
          -0.3, 0.0,
           0.5, 0.0;  // 전진은 무비용
  test_controls.push_back(ctrl);

  auto costs = cost_new.compute(trajectories_, test_controls, reference_);

  // cost = weight * (v1² + v2²) = 15 * (0.04 + 0.09) = 15 * 0.13 = 1.95
  EXPECT_NEAR(costs(0), 15.0 * (0.2 * 0.2 + 0.3 * 0.3), 1e-9);
}

}  // namespace mpc_controller_ros2

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
