#include "mpc_controller_ros2/smooth_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::SmoothMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void SmoothMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  // 부모 configure 호출 (파라미터 declare/load 포함)
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  // Δu 시퀀스 초기화
  delta_u_sequence_ = Eigen::MatrixXd::Zero(params_.N, 2);
  u_prev_ = Eigen::Vector2d::Zero();

  auto node = parent.lock();
  RCLCPP_INFO(
    node->get_logger(),
    "Smooth-MPPI plugin configured: R_jerk=[%.3f, %.3f], action_cost_weight=%.3f",
    params_.smooth_R_jerk_v,
    params_.smooth_R_jerk_omega,
    params_.smooth_action_cost_weight);
}

std::pair<Eigen::Vector2d, MPPIInfo> SmoothMPPIControllerPlugin::computeControl(
  const Eigen::Vector3d& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  int N = params_.N;
  int K = params_.K;
  int nu = 2;  // [v, omega]

  // 1. Shift previous DU sequence (warm start)
  for (int t = 0; t < N - 1; ++t) {
    delta_u_sequence_.row(t) = delta_u_sequence_.row(t + 1);
  }
  delta_u_sequence_.row(N - 1).setZero();

  // 2. Sample noise in Δu space
  auto delta_noise = sampler_->sample(K, N, nu);

  // 3. Perturb Δu sequence
  std::vector<Eigen::MatrixXd> perturbed_du;
  perturbed_du.reserve(K);
  for (int k = 0; k < K; ++k) {
    perturbed_du.push_back(delta_u_sequence_ + delta_noise[k]);
  }

  // 4. Cumsum: u[t] = u_prev + Σ_{i=0}^{t} Δu[i]
  std::vector<Eigen::MatrixXd> perturbed_controls;
  perturbed_controls.reserve(K);
  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd u_seq(N, nu);
    Eigen::Vector2d cumulative = u_prev_;
    for (int t = 0; t < N; ++t) {
      cumulative += perturbed_du[k].row(t).transpose();
      u_seq.row(t) = cumulative.transpose();
    }
    // Clip
    u_seq = dynamics_->clipControls(u_seq);
    perturbed_controls.push_back(u_seq);
  }

  // 5. Batch rollout
  auto trajectories = dynamics_->rolloutBatch(
    current_state, perturbed_controls, params_.dt);

  // 6. Compute costs
  Eigen::VectorXd costs = cost_function_->compute(
    trajectories, perturbed_controls, reference_trajectory);

  // 7. Jerk cost: ‖ΔΔu‖² = ‖Δu[t+1] - Δu[t]‖²
  if (params_.smooth_action_cost_weight > 0.0 && N > 1) {
    Eigen::Vector2d R_jerk(params_.smooth_R_jerk_v, params_.smooth_R_jerk_omega);

    for (int k = 0; k < K; ++k) {
      double jerk_cost = 0.0;
      for (int t = 0; t < N - 1; ++t) {
        Eigen::Vector2d ddu = perturbed_du[k].row(t + 1).transpose()
                            - perturbed_du[k].row(t).transpose();
        jerk_cost += ddu(0) * ddu(0) * R_jerk(0) + ddu(1) * ddu(1) * R_jerk(1);
      }
      costs(k) += params_.smooth_action_cost_weight * jerk_cost;
    }
  }

  // 8. Compute weights (Adaptive Temperature)
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);
  }
  Eigen::VectorXd weights = weight_computation_->compute(costs, current_lambda);

  // 9. Update DU in Δu space (weighted average of noise)
  Eigen::MatrixXd weighted_delta_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_delta_noise += weights(k) * delta_noise[k];
  }
  delta_u_sequence_ += weighted_delta_noise;

  // 10. Restore optimal U from DU
  control_sequence_ = Eigen::MatrixXd::Zero(N, nu);
  Eigen::Vector2d cumulative = u_prev_;
  for (int t = 0; t < N; ++t) {
    cumulative += delta_u_sequence_.row(t).transpose();
    control_sequence_.row(t) = cumulative.transpose();
  }
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // 11. Extract optimal control
  Eigen::Vector2d u_opt = control_sequence_.row(0).transpose();
  u_prev_ = u_opt;

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

  // Build info struct
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
    "Smooth-MPPI: min_cost=%.4f, ESS=%.1f/%d, du_norm=%.4f",
    min_cost, ess, K, delta_u_sequence_.norm());

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
