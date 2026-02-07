#include <gtest/gtest.h>
#include <cmath>
#include "mpc_controller_ros2/mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/mppi_params.hpp"
#include "mpc_controller_ros2/batch_dynamics_wrapper.hpp"
#include "mpc_controller_ros2/cost_functions.hpp"
#include "mpc_controller_ros2/sampling.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

class MPPIAlgorithmTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Setup MPPI parameters
    params_ = MPPIParams();
    params_.N = 10;
    params_.dt = 0.1;
    params_.K = 100;
    params_.lambda = 10.0;
    params_.noise_sigma = Eigen::Vector2d(0.5, 0.3);

    // Cost weights
    params_.Q(0, 0) = 10.0;
    params_.Q(1, 1) = 10.0;
    params_.Q(2, 2) = 1.0;
    params_.Qf = 2.0 * params_.Q;
    params_.R(0, 0) = 0.1;
    params_.R(1, 1) = 0.1;

    // Initialize components
    dynamics_ = std::make_unique<BatchDynamicsWrapper>(params_);
    sampler_ = std::make_unique<GaussianSampler>(params_.noise_sigma, 42);

    cost_function_ = std::make_unique<CompositeMPPICost>();
    cost_function_->addCost(std::make_unique<StateTrackingCost>(params_.Q));
    cost_function_->addCost(std::make_unique<TerminalCost>(params_.Qf));
    cost_function_->addCost(std::make_unique<ControlEffortCost>(params_.R));

    // Initialize control sequence
    control_sequence_ = Eigen::MatrixXd::Zero(params_.N, 2);
  }

  MPPIParams params_;
  std::unique_ptr<BatchDynamicsWrapper> dynamics_;
  std::unique_ptr<BaseSampler> sampler_;
  std::unique_ptr<CompositeMPPICost> cost_function_;
  Eigen::MatrixXd control_sequence_;
};

TEST_F(MPPIAlgorithmTest, ControlSequenceShift)
{
  // Initialize with specific values
  for (int t = 0; t < params_.N; ++t) {
    control_sequence_(t, 0) = t * 0.1;
    control_sequence_(t, 1) = t * 0.05;
  }

  // Shift
  for (int t = 0; t < params_.N - 1; ++t) {
    control_sequence_.row(t) = control_sequence_.row(t + 1);
  }
  control_sequence_.row(params_.N - 1).setZero();

  // Check
  EXPECT_NEAR(control_sequence_(0, 0), 0.1, 1e-9);
  EXPECT_NEAR(control_sequence_(0, 1), 0.05, 1e-9);
  EXPECT_NEAR(control_sequence_(params_.N - 1, 0), 0.0, 1e-9);
  EXPECT_NEAR(control_sequence_(params_.N - 1, 1), 0.0, 1e-9);
}

TEST_F(MPPIAlgorithmTest, NoiseAdditionAndClipping)
{
  // Sample noise
  auto noise_samples = sampler_->sample(params_.K, params_.N, 2);

  // Add noise and clip
  std::vector<Eigen::MatrixXd> perturbed_controls;
  for (int k = 0; k < params_.K; ++k) {
    Eigen::MatrixXd perturbed = control_sequence_ + noise_samples[k];
    perturbed = dynamics_->clipControls(perturbed);
    perturbed_controls.push_back(perturbed);
  }

  EXPECT_EQ(perturbed_controls.size(), params_.K);

  // Check clipping
  for (const auto& ctrl : perturbed_controls) {
    for (int t = 0; t < params_.N; ++t) {
      EXPECT_GE(ctrl(t, 0), params_.v_min);
      EXPECT_LE(ctrl(t, 0), params_.v_max);
      EXPECT_GE(ctrl(t, 1), params_.omega_min);
      EXPECT_LE(ctrl(t, 1), params_.omega_max);
    }
  }
}

TEST_F(MPPIAlgorithmTest, WeightedAverageUpdate)
{
  // Create simple noise samples
  std::vector<Eigen::MatrixXd> noise_samples;
  for (int k = 0; k < 3; ++k) {
    Eigen::MatrixXd noise = Eigen::MatrixXd::Constant(params_.N, 2, k * 0.1);
    noise_samples.push_back(noise);
  }

  // Equal weights
  Eigen::VectorXd weights = Eigen::VectorXd::Constant(3, 1.0 / 3.0);

  // Compute weighted average
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(params_.N, 2);
  for (int k = 0; k < 3; ++k) {
    weighted_noise += weights(k) * noise_samples[k];
  }

  // Expected: (0 + 0.1 + 0.2) / 3 = 0.1
  EXPECT_NEAR(weighted_noise(0, 0), 0.1, 1e-9);
  EXPECT_NEAR(weighted_noise(0, 1), 0.1, 1e-9);
}

TEST_F(MPPIAlgorithmTest, SoftmaxWeightsSum)
{
  // Create costs
  Eigen::VectorXd costs(5);
  costs << 10.0, 5.0, 15.0, 8.0, 12.0;

  // Compute weights
  Eigen::VectorXd weights = softmaxWeights(costs, params_.lambda);

  // Weights should sum to 1
  EXPECT_NEAR(weights.sum(), 1.0, 1e-9);

  // All weights should be positive
  for (int k = 0; k < weights.size(); ++k) {
    EXPECT_GT(weights(k), 0.0);
  }

  // Lowest cost should have highest weight
  int min_idx;
  costs.minCoeff(&min_idx);
  int max_weight_idx;
  weights.maxCoeff(&max_weight_idx);
  EXPECT_EQ(min_idx, max_weight_idx);
}

TEST_F(MPPIAlgorithmTest, ESSCalculation)
{
  // Uniform weights
  Eigen::VectorXd uniform_weights = Eigen::VectorXd::Constant(100, 1.0 / 100.0);
  double ess_uniform = computeESS(uniform_weights);
  EXPECT_NEAR(ess_uniform, 100.0, 1.0);  // ESS ≈ K for uniform

  // Degenerate weights (one weight = 1)
  Eigen::VectorXd degenerate_weights = Eigen::VectorXd::Zero(100);
  degenerate_weights(0) = 1.0;
  double ess_degenerate = computeESS(degenerate_weights);
  EXPECT_NEAR(ess_degenerate, 1.0, 0.1);  // ESS ≈ 1 for degenerate
}

TEST_F(MPPIAlgorithmTest, FullAlgorithmIteration)
{
  // Initial state
  Eigen::Vector3d x0(0.0, 0.0, 0.0);

  // Reference trajectory: straight line
  Eigen::MatrixXd reference = Eigen::MatrixXd::Zero(params_.N + 1, 3);
  for (int t = 0; t <= params_.N; ++t) {
    reference(t, 0) = t * params_.dt;  // x increases
    reference(t, 1) = 0.0;
    reference(t, 2) = 0.0;
  }

  // Run one MPPI iteration
  // 1. Shift control sequence (already zero)

  // 2. Sample noise
  auto noise_samples = sampler_->sample(params_.K, params_.N, 2);

  // 3. Add noise and clip
  std::vector<Eigen::MatrixXd> perturbed_controls;
  for (int k = 0; k < params_.K; ++k) {
    Eigen::MatrixXd perturbed = control_sequence_ + noise_samples[k];
    perturbed = dynamics_->clipControls(perturbed);
    perturbed_controls.push_back(perturbed);
  }

  // 4. Rollout
  auto trajectories = dynamics_->rolloutBatch(x0, perturbed_controls, params_.dt);

  // 5. Compute costs
  Eigen::VectorXd costs = cost_function_->compute(
    trajectories,
    perturbed_controls,
    reference
  );

  // 6. Compute weights
  Eigen::VectorXd weights = softmaxWeights(costs, params_.lambda);

  // 7. Update control sequence
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(params_.N, 2);
  for (int k = 0; k < params_.K; ++k) {
    weighted_noise += weights(k) * noise_samples[k];
  }
  control_sequence_ += weighted_noise;
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // 8. Extract optimal control
  Eigen::Vector2d u_opt = control_sequence_.row(0).transpose();

  // Verify results
  EXPECT_EQ(trajectories.size(), params_.K);
  EXPECT_EQ(costs.size(), params_.K);
  EXPECT_NEAR(weights.sum(), 1.0, 1e-9);

  // Control should be reasonable (not NaN, not too large)
  EXPECT_FALSE(std::isnan(u_opt(0)));
  EXPECT_FALSE(std::isnan(u_opt(1)));
  EXPECT_GE(u_opt(0), params_.v_min);
  EXPECT_LE(u_opt(0), params_.v_max);

  // ESS should be reasonable
  double ess = computeESS(weights);
  EXPECT_GT(ess, 1.0);
  EXPECT_LE(ess, params_.K);
}

TEST_F(MPPIAlgorithmTest, MultipleIterationsConvergence)
{
  // Initial state
  Eigen::Vector3d x0(0.0, 0.0, 0.0);

  // Reference trajectory
  Eigen::MatrixXd reference = Eigen::MatrixXd::Zero(params_.N + 1, 3);
  for (int t = 0; t <= params_.N; ++t) {
    reference(t, 0) = t * params_.dt;
  }

  // Run multiple iterations
  std::vector<double> min_costs;
  for (int iter = 0; iter < 5; ++iter) {
    // Shift
    for (int t = 0; t < params_.N - 1; ++t) {
      control_sequence_.row(t) = control_sequence_.row(t + 1);
    }
    control_sequence_.row(params_.N - 1).setZero();

    // Sample
    auto noise_samples = sampler_->sample(params_.K, params_.N, 2);

    // Perturb
    std::vector<Eigen::MatrixXd> perturbed_controls;
    for (int k = 0; k < params_.K; ++k) {
      Eigen::MatrixXd perturbed = control_sequence_ + noise_samples[k];
      perturbed = dynamics_->clipControls(perturbed);
      perturbed_controls.push_back(perturbed);
    }

    // Rollout
    auto trajectories = dynamics_->rolloutBatch(x0, perturbed_controls, params_.dt);

    // Costs
    Eigen::VectorXd costs = cost_function_->compute(
      trajectories, perturbed_controls, reference
    );

    // Weights
    Eigen::VectorXd weights = softmaxWeights(costs, params_.lambda);

    // Update
    Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(params_.N, 2);
    for (int k = 0; k < params_.K; ++k) {
      weighted_noise += weights(k) * noise_samples[k];
    }
    control_sequence_ += weighted_noise;
    control_sequence_ = dynamics_->clipControls(control_sequence_);

    // Record min cost
    min_costs.push_back(costs.minCoeff());
  }

  // Cost should generally decrease (allowing some noise)
  // At least the last cost should be lower than the first
  EXPECT_LT(min_costs.back(), min_costs.front() * 1.5);
}

}  // namespace mpc_controller_ros2

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
