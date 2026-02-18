#include "mpc_controller_ros2/svg_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::SVGMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void SVGMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  // 부모 configure 호출 (SVMPC → MPPI base)
  SVMPCControllerPlugin::configure(parent, name, tf, costmap_ros);

  auto node = parent.lock();
  RCLCPP_INFO(
    node->get_logger(),
    "SVG-MPPI plugin configured: guides=%d, iterations=%d, step_size=%.3f, resample_std=%.3f",
    params_.svg_num_guide_particles,
    params_.svg_guide_iterations,
    params_.svg_guide_step_size,
    params_.svg_resample_std);
}

std::pair<Eigen::Vector2d, MPPIInfo> SVGMPPIControllerPlugin::computeControl(
  const Eigen::Vector3d& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  int G = params_.svg_num_guide_particles;
  int L = params_.svg_guide_iterations;
  int K = params_.K;
  int N = params_.N;
  int nu = 2;
  int D = N * nu;

  // G=0 또는 L=0이면 Vanilla fallback (SVMPC로)
  if (G <= 0 || L <= 0) {
    return SVMPCControllerPlugin::computeControl(current_state, reference_trajectory);
  }

  G = std::min(G, K);

  // ──── Phase 1: 초기 샘플링 & 비용 ────

  // Shift previous control sequence
  for (int t = 0; t < N - 1; ++t) {
    control_sequence_.row(t) = control_sequence_.row(t + 1);
  }
  control_sequence_.row(N - 1).setZero();

  // Sample noise (전체 K개)
  auto noise_samples = sampler_->sample(K, N, nu);

  std::vector<Eigen::MatrixXd> perturbed_controls;
  perturbed_controls.reserve(K);
  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd perturbed = control_sequence_ + noise_samples[k];
    perturbed = dynamics_->clipControls(perturbed);
    perturbed_controls.push_back(perturbed);
  }

  auto trajectories = dynamics_->rolloutBatch(
    current_state, perturbed_controls, params_.dt);

  Eigen::VectorXd costs = cost_function_->compute(
    trajectories, perturbed_controls, reference_trajectory);

  // ──── Phase 2: Guide particle 선택 (비용 최저 G개) ────

  // 인덱스를 비용 순으로 부분 정렬
  std::vector<int> indices(K);
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + G, indices.end(),
    [&costs](int a, int b) { return costs(a) < costs(b); });

  // Guide particles flatten (G, D)
  Eigen::MatrixXd guide_particles(G, D);
  Eigen::VectorXd guide_costs(G);
  for (int g = 0; g < G; ++g) {
    int idx = indices[g];
    Eigen::Map<Eigen::VectorXd> flat(
      const_cast<double*>(perturbed_controls[idx].data()), D);
    guide_particles.row(g) = flat.transpose();
    guide_costs(g) = costs(idx);
  }

  // Diversity before SVGD
  std::vector<Eigen::MatrixXd> guide_ctrl_vec;
  guide_ctrl_vec.reserve(G);
  for (int g = 0; g < G; ++g) {
    Eigen::MatrixXd ctrl(N, nu);
    Eigen::Map<Eigen::VectorXd> flat(ctrl.data(), D);
    flat = guide_particles.row(g).transpose();
    guide_ctrl_vec.push_back(ctrl);
  }
  double diversity_before = computeDiversity(guide_ctrl_vec, G, D);

  // ──── Phase 3: SVGD on guides only (G×G) ────

  double step_size = params_.svg_guide_step_size;

  for (int svgd_iter = 0; svgd_iter < L; ++svgd_iter) {
    // Softmax weights on guide costs
    double current_lambda = params_.lambda;
    if (params_.adaptive_temperature && adaptive_temp_) {
      Eigen::VectorXd temp_w = weight_computation_->compute(guide_costs, current_lambda);
      double ess_temp = computeESS(temp_w);
      current_lambda = adaptive_temp_->update(ess_temp, G);
    }
    Eigen::VectorXd guide_weights = weight_computation_->compute(guide_costs, current_lambda);

    // Pairwise diff (G×G)
    Eigen::MatrixXd sq_dist(G, G);
    std::vector<Eigen::VectorXd> diff_flat(G * G);

    for (int j = 0; j < G; ++j) {
      for (int i = 0; i < G; ++i) {
        diff_flat[j * G + i] = guide_particles.row(j).transpose() - guide_particles.row(i).transpose();
        sq_dist(j, i) = diff_flat[j * G + i].squaredNorm();
      }
    }

    // Bandwidth
    double h;
    if (params_.svgd_bandwidth > 0.0) {
      h = params_.svgd_bandwidth;
    } else {
      h = medianBandwidth(sq_dist, G);
    }

    // RBF kernel
    Eigen::MatrixXd kernel = (-sq_dist / (2.0 * h * h)).array().exp().matrix();

    // SVGD force (재사용)
    Eigen::MatrixXd force = computeSVGDForce(diff_flat, guide_weights, kernel, h, G, D);

    // Update guides
    guide_particles += step_size * force;

    // Clip & re-evaluate
    for (int g = 0; g < G; ++g) {
      Eigen::MatrixXd ctrl(N, nu);
      Eigen::Map<Eigen::VectorXd> flat(ctrl.data(), D);
      flat = guide_particles.row(g).transpose();
      ctrl = dynamics_->clipControls(ctrl);
      Eigen::Map<Eigen::VectorXd> flat2(ctrl.data(), D);
      guide_particles.row(g) = flat2.transpose();
      guide_ctrl_vec[g] = ctrl;
    }

    // Re-rollout & re-cost for guides
    auto guide_trajectories = dynamics_->rolloutBatch(
      current_state, guide_ctrl_vec, params_.dt);
    guide_costs = cost_function_->compute(
      guide_trajectories, guide_ctrl_vec, reference_trajectory);
  }

  // Diversity after SVGD
  double diversity_after = computeDiversity(guide_ctrl_vec, G, D);

  // ──── Phase 4: Follower resampling ────

  int n_followers = K - G;
  int followers_per_guide = std::max(1, n_followers / G);

  // Collect all controls: guides + followers
  std::vector<Eigen::MatrixXd> all_controls;
  all_controls.reserve(K);

  // Add guides
  for (int g = 0; g < G; ++g) {
    all_controls.push_back(guide_ctrl_vec[g]);
  }

  // Add followers around each guide
  double resample_std = params_.svg_resample_std;
  for (int g = 0; g < G; ++g) {
    int n_f;
    if (g < G - 1) {
      n_f = followers_per_guide;
    } else {
      n_f = n_followers - followers_per_guide * (G - 1);
    }
    if (n_f <= 0) {
      continue;
    }

    // Sample follower noise
    auto follower_noise = sampler_->sample(n_f, N, nu);
    for (int f = 0; f < n_f; ++f) {
      Eigen::MatrixXd follower_ctrl = guide_ctrl_vec[g] + resample_std * follower_noise[f];
      follower_ctrl = dynamics_->clipControls(follower_ctrl);
      all_controls.push_back(follower_ctrl);
    }
  }

  int K_total = static_cast<int>(all_controls.size());

  // ──── Phase 5: 전체 rollout & weight ────

  auto all_trajectories = dynamics_->rolloutBatch(
    current_state, all_controls, params_.dt);

  Eigen::VectorXd all_costs = cost_function_->compute(
    all_trajectories, all_controls, reference_trajectory);

  // Final weights
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_w = weight_computation_->compute(all_costs, current_lambda);
    double ess_temp = computeESS(temp_w);
    current_lambda = adaptive_temp_->update(ess_temp, K_total);
  }
  Eigen::VectorXd weights = weight_computation_->compute(all_costs, current_lambda);

  // Update U via effective noise
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K_total; ++k) {
    Eigen::MatrixXd effective_noise = all_controls[k] - control_sequence_;
    weighted_noise += weights(k) * effective_noise;
  }
  control_sequence_ += weighted_noise;
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // Extract optimal control
  Eigen::Vector2d u_opt = control_sequence_.row(0).transpose();

  // Weighted average trajectory
  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, 3);
  for (int k = 0; k < K_total; ++k) {
    weighted_traj += weights(k) * all_trajectories[k];
  }

  // Best sample
  int best_idx;
  double min_cost = all_costs.minCoeff(&best_idx);

  // ESS
  double ess = computeESS(weights);

  // Build info struct
  MPPIInfo info;
  info.sample_trajectories = all_trajectories;
  info.sample_weights = weights;
  info.best_trajectory = all_trajectories[best_idx];
  info.weighted_avg_trajectory = weighted_traj;
  info.temperature = (params_.adaptive_temperature && adaptive_temp_) ?
    adaptive_temp_->getLambda() : params_.lambda;
  info.ess = ess;
  info.costs = all_costs;

  info.colored_noise_used = params_.colored_noise;
  info.adaptive_temp_used = params_.adaptive_temperature;
  info.tube_mppi_used = params_.tube_enabled;

  // SVGD info
  info.svgd_iterations = L;
  info.sample_diversity_before = diversity_before;
  info.sample_diversity_after = diversity_after;

  // SVG-MPPI 전용 info
  info.num_guides = G;
  info.num_followers = n_followers;
  info.guide_iterations = L;

  RCLCPP_DEBUG(
    node_->get_logger(),
    "SVG-MPPI: min_cost=%.4f, guides=%d, followers=%d, "
    "diversity=%.4f->%.4f, ESS=%.1f/%d",
    min_cost, G, n_followers,
    diversity_before, diversity_after, ess, K_total);

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
