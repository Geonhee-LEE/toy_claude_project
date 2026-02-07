#include <gtest/gtest.h>
#include <cmath>
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

class BatchDynamicsTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    params_ = MPPIParams();
    params_.v_max = 1.0;
    params_.v_min = -0.5;
    params_.omega_max = 1.0;
    params_.omega_min = -1.0;
    dynamics_ = std::make_unique<BatchDynamicsWrapper>(params_);
  }

  MPPIParams params_;
  std::unique_ptr<BatchDynamicsWrapper> dynamics_;
};

TEST_F(BatchDynamicsTest, DynamicsBatchSingleState)
{
  // Test case from Python: state = [0, 0, 0], control = [1.0, 0.5]
  // Expected: [1.0, 0.0, 0.5] (v*cos(0), v*sin(0), omega)

  Eigen::MatrixXd states(1, 3);
  states << 0.0, 0.0, 0.0;

  Eigen::MatrixXd controls(1, 2);
  controls << 1.0, 0.5;

  Eigen::MatrixXd state_dot = dynamics_->dynamicsBatch(states, controls);

  EXPECT_NEAR(state_dot(0, 0), 1.0, 1e-9);  // x_dot = v*cos(0) = 1
  EXPECT_NEAR(state_dot(0, 1), 0.0, 1e-9);  // y_dot = v*sin(0) = 0
  EXPECT_NEAR(state_dot(0, 2), 0.5, 1e-9);  // theta_dot = omega = 0.5
}

TEST_F(BatchDynamicsTest, DynamicsBatchMultipleStates)
{
  // Test batch processing with 3 states
  Eigen::MatrixXd states(3, 3);
  states << 0.0, 0.0, 0.0,
            1.0, 1.0, M_PI / 2,
            2.0, 2.0, M_PI;

  Eigen::MatrixXd controls(3, 2);
  controls << 1.0, 0.5,
              0.5, -0.3,
              0.8, 0.0;

  Eigen::MatrixXd state_dot = dynamics_->dynamicsBatch(states, controls);

  // State 0: theta=0, v=1.0, omega=0.5
  EXPECT_NEAR(state_dot(0, 0), 1.0, 1e-9);
  EXPECT_NEAR(state_dot(0, 1), 0.0, 1e-9);
  EXPECT_NEAR(state_dot(0, 2), 0.5, 1e-9);

  // State 1: theta=π/2, v=0.5, omega=-0.3
  EXPECT_NEAR(state_dot(1, 0), 0.0, 1e-9);  // cos(π/2) ≈ 0
  EXPECT_NEAR(state_dot(1, 1), 0.5, 1e-9);  // sin(π/2) = 1
  EXPECT_NEAR(state_dot(1, 2), -0.3, 1e-9);

  // State 2: theta=π, v=0.8, omega=0.0
  EXPECT_NEAR(state_dot(2, 0), -0.8, 1e-9);  // cos(π) = -1
  EXPECT_NEAR(state_dot(2, 1), 0.0, 1e-9);   // sin(π) ≈ 0
  EXPECT_NEAR(state_dot(2, 2), 0.0, 1e-9);
}

TEST_F(BatchDynamicsTest, PropagateBatchRK4Integration)
{
  // Test RK4 integration for one step
  // Compare with known analytical solution for simple case

  Eigen::MatrixXd states(1, 3);
  states << 0.0, 0.0, 0.0;

  Eigen::MatrixXd controls(1, 2);
  controls << 1.0, 0.0;  // Straight line motion

  double dt = 0.1;
  Eigen::MatrixXd next_state = dynamics_->propagateBatch(states, controls, dt);

  // Analytical solution for straight line: x = v*t, y = 0, theta = 0
  EXPECT_NEAR(next_state(0, 0), 0.1, 1e-6);  // x = 1.0 * 0.1
  EXPECT_NEAR(next_state(0, 1), 0.0, 1e-9);  // y = 0
  EXPECT_NEAR(next_state(0, 2), 0.0, 1e-9);  // theta = 0
}

TEST_F(BatchDynamicsTest, PropagateBatchCircularMotion)
{
  // Test circular motion: v=1, omega=1, dt=0.1
  Eigen::MatrixXd states(1, 3);
  states << 0.0, 0.0, 0.0;

  Eigen::MatrixXd controls(1, 2);
  controls << 1.0, 1.0;

  double dt = 0.1;
  Eigen::MatrixXd next_state = dynamics_->propagateBatch(states, controls, dt);

  // After dt, theta should increase
  EXPECT_GT(next_state(0, 2), 0.0);
  EXPECT_LT(next_state(0, 2), 0.2);  // Approximately 0.1 rad

  // Position should move
  EXPECT_GT(next_state(0, 0), 0.0);
  EXPECT_GE(next_state(0, 1), 0.0);
}

TEST_F(BatchDynamicsTest, RolloutBatchConsistency)
{
  // Test rollout with 2 samples, 5 timesteps
  Eigen::Vector3d x0(0.0, 0.0, 0.0);

  std::vector<Eigen::MatrixXd> control_sequences;

  // Sample 1: straight line
  Eigen::MatrixXd controls1(5, 2);
  controls1 << 1.0, 0.0,
               1.0, 0.0,
               1.0, 0.0,
               1.0, 0.0,
               1.0, 0.0;
  control_sequences.push_back(controls1);

  // Sample 2: turning
  Eigen::MatrixXd controls2(5, 2);
  controls2 << 0.5, 0.5,
               0.5, 0.5,
               0.5, 0.5,
               0.5, 0.5,
               0.5, 0.5;
  control_sequences.push_back(controls2);

  double dt = 0.1;
  auto trajectories = dynamics_->rolloutBatch(x0, control_sequences, dt);

  // Check dimensions
  EXPECT_EQ(trajectories.size(), 2);
  EXPECT_EQ(trajectories[0].rows(), 6);  // N+1 = 5+1
  EXPECT_EQ(trajectories[0].cols(), 3);

  // Check initial state
  EXPECT_NEAR(trajectories[0](0, 0), 0.0, 1e-9);
  EXPECT_NEAR(trajectories[0](0, 1), 0.0, 1e-9);
  EXPECT_NEAR(trajectories[0](0, 2), 0.0, 1e-9);

  // Trajectory 1 should move in +x direction
  EXPECT_GT(trajectories[0](5, 0), 0.4);  // After 5 steps
  EXPECT_NEAR(trajectories[0](5, 1), 0.0, 1e-3);
  EXPECT_NEAR(trajectories[0](5, 2), 0.0, 1e-3);

  // Trajectory 2 should turn
  EXPECT_GT(trajectories[1](5, 2), 0.0);  // Theta increases
}

TEST_F(BatchDynamicsTest, ClipControlsVelocity)
{
  Eigen::MatrixXd controls(3, 2);
  controls << 2.0, 0.5,      // v exceeds max
              -1.0, -0.5,    // v exceeds min
              0.5, 0.0;      // within bounds

  Eigen::MatrixXd clipped = dynamics_->clipControls(controls);

  EXPECT_NEAR(clipped(0, 0), 1.0, 1e-9);   // Clipped to v_max
  EXPECT_NEAR(clipped(0, 1), 0.5, 1e-9);   // omega unchanged

  EXPECT_NEAR(clipped(1, 0), -0.5, 1e-9);  // Clipped to v_min
  EXPECT_NEAR(clipped(1, 1), -0.5, 1e-9);  // omega unchanged

  EXPECT_NEAR(clipped(2, 0), 0.5, 1e-9);   // No clipping
  EXPECT_NEAR(clipped(2, 1), 0.0, 1e-9);
}

TEST_F(BatchDynamicsTest, ClipControlsOmega)
{
  Eigen::MatrixXd controls(3, 2);
  controls << 0.5, 2.0,      // omega exceeds max
              0.5, -2.0,     // omega exceeds min
              0.5, 0.5;      // within bounds

  Eigen::MatrixXd clipped = dynamics_->clipControls(controls);

  EXPECT_NEAR(clipped(0, 0), 0.5, 1e-9);
  EXPECT_NEAR(clipped(0, 1), 1.0, 1e-9);   // Clipped to omega_max

  EXPECT_NEAR(clipped(1, 0), 0.5, 1e-9);
  EXPECT_NEAR(clipped(1, 1), -1.0, 1e-9);  // Clipped to omega_min

  EXPECT_NEAR(clipped(2, 0), 0.5, 1e-9);
  EXPECT_NEAR(clipped(2, 1), 0.5, 1e-9);   // No clipping
}

TEST_F(BatchDynamicsTest, AngleNormalization)
{
  // Test that angles are normalized in propagate
  Eigen::MatrixXd states(1, 3);
  states << 0.0, 0.0, 3.0;  // theta = 3 rad (> π)

  Eigen::MatrixXd controls(1, 2);
  controls << 0.0, 1.0;  // Turn

  double dt = 1.0;
  Eigen::MatrixXd next_state = dynamics_->propagateBatch(states, controls, dt);

  // Result should be normalized to [-π, π]
  EXPECT_GE(next_state(0, 2), -M_PI);
  EXPECT_LE(next_state(0, 2), M_PI);
}

}  // namespace mpc_controller_ros2

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
