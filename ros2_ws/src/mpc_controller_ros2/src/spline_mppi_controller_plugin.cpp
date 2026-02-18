#include "mpc_controller_ros2/spline_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>
#include <random>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::SplineMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

Eigen::MatrixXd SplineMPPIControllerPlugin::computeBSplineBasis(int N, int P, int degree)
{
  int k = degree;
  int n_knots = P + k + 1;

  // Clamped uniform knot vector
  Eigen::VectorXd knots = Eigen::VectorXd::Zero(n_knots);
  int n_internal = P - k - 1;
  if (n_internal > 0) {
    for (int i = 0; i < n_internal; ++i) {
      knots(k + 1 + i) = static_cast<double>(i + 1) / (n_internal + 1);
    }
  }
  for (int i = P; i < n_knots; ++i) {
    knots(i) = 1.0;
  }

  // Evaluation points
  Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(N, 0.0, 1.0);
  t(N - 1) = 1.0 - 1e-10;  // avoid boundary issue

  // de Boor recursion — degree 0
  int cols0 = n_knots - 1;
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(N, cols0);
  for (int i = 0; i < cols0; ++i) {
    for (int j = 0; j < N; ++j) {
      if (t(j) >= knots(i) && t(j) < knots(i + 1)) {
        B(j, i) = 1.0;
      }
    }
  }

  // Recursively raise degree
  for (int d = 1; d <= k; ++d) {
    int new_cols = n_knots - 1 - d;
    Eigen::MatrixXd B_new = Eigen::MatrixXd::Zero(N, new_cols);
    for (int i = 0; i < new_cols; ++i) {
      double denom1 = knots(i + d) - knots(i);
      double denom2 = knots(i + d + 1) - knots(i + 1);

      for (int j = 0; j < N; ++j) {
        double term1 = 0.0;
        double term2 = 0.0;
        if (denom1 > 1e-12) {
          term1 = (t(j) - knots(i)) / denom1 * B(j, i);
        }
        if (denom2 > 1e-12) {
          term2 = (knots(i + d + 1) - t(j)) / denom2 * B(j, i + 1);
        }
        B_new(j, i) = term1 + term2;
      }
    }
    B = B_new;
  }

  // Extract first P columns
  Eigen::MatrixXd basis = B.leftCols(P);

  // Row normalization (partition of unity)
  for (int j = 0; j < N; ++j) {
    double row_sum = basis.row(j).sum();
    if (row_sum > 1e-12) {
      basis.row(j) /= row_sum;
    }
  }

  return basis;
}

void SplineMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  // 부모 configure 호출
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  P_ = params_.spline_num_knots;
  degree_ = params_.spline_degree;

  // B-spline basis 사전 계산 (N, P)
  basis_ = computeBSplineBasis(params_.N, P_, degree_);

  // Knot warm-start
  u_knots_ = Eigen::MatrixXd::Zero(P_, 2);

  auto node = parent.lock();
  RCLCPP_INFO(
    node->get_logger(),
    "Spline-MPPI plugin configured: num_knots=%d, degree=%d, basis=(%dx%d)",
    P_, degree_, static_cast<int>(basis_.rows()), static_cast<int>(basis_.cols()));
}

std::pair<Eigen::Vector2d, MPPIInfo> SplineMPPIControllerPlugin::computeControl(
  const Eigen::Vector3d& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  int N = params_.N;
  int K = params_.K;
  int nu = 2;

  // 1. Shift knot sequence
  for (int p = 0; p < P_ - 1; ++p) {
    u_knots_.row(p) = u_knots_.row(p + 1);
  }
  u_knots_.row(P_ - 1).setZero();

  // 2. Sample knot noise (K, P, nu) using standard normal
  static std::mt19937 rng(42);
  std::normal_distribution<double> dist(0.0, 1.0);

  // knot_noise[k] = (P, nu)
  std::vector<Eigen::MatrixXd> knot_noise;
  knot_noise.reserve(K);
  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd noise(P_, nu);
    for (int p = 0; p < P_; ++p) {
      noise(p, 0) = dist(rng) * params_.noise_sigma(0);
      noise(p, 1) = dist(rng) * params_.noise_sigma(1);
    }
    knot_noise.push_back(noise);
  }

  // 3. Perturb knots and interpolate via B-spline basis
  std::vector<Eigen::MatrixXd> perturbed_controls;
  perturbed_controls.reserve(K);

  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd perturbed_knots = u_knots_ + knot_noise[k];  // (P, nu)
    Eigen::MatrixXd u_interp = basis_ * perturbed_knots;           // (N, nu)
    u_interp = dynamics_->clipControls(u_interp);
    perturbed_controls.push_back(u_interp);
  }

  // 4. Batch rollout
  auto trajectories = dynamics_->rolloutBatch(
    current_state, perturbed_controls, params_.dt);

  // 5. Compute costs
  Eigen::VectorXd costs = cost_function_->compute(
    trajectories, perturbed_controls, reference_trajectory);

  // 6. Compute weights
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);
  }
  Eigen::VectorXd weights = weight_computation_->compute(costs, current_lambda);

  // 7. Update knots in knot space
  Eigen::MatrixXd weighted_knot_noise = Eigen::MatrixXd::Zero(P_, nu);
  for (int k = 0; k < K; ++k) {
    weighted_knot_noise += weights(k) * knot_noise[k];
  }
  u_knots_ += weighted_knot_noise;

  // 8. Restore U via B-spline interpolation
  control_sequence_ = basis_ * u_knots_;
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // 9. Extract optimal control
  Eigen::Vector2d u_opt = control_sequence_.row(0).transpose();

  // Weighted average trajectory
  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, 3);
  for (int k = 0; k < K; ++k) {
    weighted_traj += weights(k) * trajectories[k];
  }

  // Best sample
  int best_idx;
  double min_cost = costs.minCoeff(&best_idx);

  // ESS
  double ess = computeESS(weights);

  // Build info
  MPPIInfo info;
  info.sample_trajectories = trajectories;
  info.sample_weights = weights;
  info.best_trajectory = trajectories[best_idx];
  info.weighted_avg_trajectory = weighted_traj;
  info.temperature = (params_.adaptive_temperature && adaptive_temp_) ?
    adaptive_temp_->getLambda() : params_.lambda;
  info.ess = ess;
  info.costs = costs;

  info.colored_noise_used = params_.colored_noise;
  info.adaptive_temp_used = params_.adaptive_temperature;
  info.tube_mppi_used = params_.tube_enabled;

  RCLCPP_DEBUG(
    node_->get_logger(),
    "Spline-MPPI: min_cost=%.4f, ESS=%.1f/%d, knots=%d",
    min_cost, ess, K, P_);

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
