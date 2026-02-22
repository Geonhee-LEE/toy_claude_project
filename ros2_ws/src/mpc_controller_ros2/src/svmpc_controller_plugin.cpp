#include "mpc_controller_ros2/svmpc_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::SVMPCControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void SVMPCControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  // 부모 configure 호출 (파라미터 declare/load 포함)
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  auto node = parent.lock();
  RCLCPP_INFO(
    node->get_logger(),
    "SVMPC plugin configured: svgd_iterations=%d, step_size=%.3f, bandwidth=%.3f",
    params_.svgd_num_iterations,
    params_.svgd_step_size,
    params_.svgd_bandwidth);
}

std::pair<Eigen::VectorXd, MPPIInfo> SVMPCControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  int L = params_.svgd_num_iterations;

  // svgd_num_iterations=0 → Vanilla 동등
  if (L == 0) {
    return MPPIControllerPlugin::computeControl(current_state, reference_trajectory);
  }

  int N = params_.N;
  int K = params_.K;
  int nu = dynamics_->model().controlDim();
  int nx = dynamics_->model().stateDim();
  int D = N * nu;  // flatten 차원

  // ──── Phase 1: Vanilla 동일 (shift, sample, perturb, rollout, cost) ────

  // 1. Shift previous control sequence (warm start)
  for (int t = 0; t < N - 1; ++t) {
    control_sequence_.row(t) = control_sequence_.row(t + 1);
  }
  control_sequence_.row(N - 1).setZero();

  // 2. Sample noise
  auto noise_samples = sampler_->sample(K, N, nu);

  // 3. Add noise to control sequence and clip
  std::vector<Eigen::MatrixXd> perturbed_controls;
  perturbed_controls.reserve(K);

  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd perturbed = control_sequence_ + noise_samples[k];
    perturbed = dynamics_->clipControls(perturbed);
    perturbed_controls.push_back(perturbed);
  }

  // 4. Batch rollout
  auto trajectories = dynamics_->rolloutBatch(
    current_state, perturbed_controls, params_.dt);

  // 5. Compute costs
  Eigen::VectorXd costs = cost_function_->compute(
    trajectories, perturbed_controls, reference_trajectory);

  // 샘플 다양성 측정 (SVGD 전)
  double diversity_before = computeDiversity(perturbed_controls, K, D);

  // ──── Phase 2: SVGD Loop ────

  for (int svgd_iter = 0; svgd_iter < L; ++svgd_iter) {
    // flatten: (K, N, nu) → (K, D) — 각 샘플을 하나의 벡터로
    Eigen::MatrixXd particles(K, D);
    for (int k = 0; k < K; ++k) {
      Eigen::Map<Eigen::VectorXd> flat(
        const_cast<double*>(perturbed_controls[k].data()),
        D);
      particles.row(k) = flat.transpose();
    }

    // softmax 가중치 (현재 비용 기반)
    double current_lambda = params_.lambda;
    if (params_.adaptive_temperature && adaptive_temp_) {
      Eigen::VectorXd temp_weights = weight_computation_->compute(costs, current_lambda);
      double ess = computeESS(temp_weights);
      current_lambda = adaptive_temp_->update(ess, K);
    }
    Eigen::VectorXd weights = weight_computation_->compute(costs, current_lambda);

    // ── pairwise diff 1회 계산 후 재사용 ──
    // sq_dist(j, i) = ||x_j - x_i||^2
    Eigen::MatrixXd sq_dist(K, K);
    // diff를 flatten해서 저장 (K*K개 D차원 벡터)
    std::vector<Eigen::VectorXd> diff_flat(K * K);

    for (int j = 0; j < K; ++j) {
      for (int i = 0; i < K; ++i) {
        diff_flat[j * K + i] = particles.row(j).transpose() - particles.row(i).transpose();
        sq_dist(j, i) = diff_flat[j * K + i].squaredNorm();
      }
    }

    // bandwidth (median heuristic 또는 고정값)
    double h;
    if (params_.svgd_bandwidth > 0.0) {
      h = params_.svgd_bandwidth;
    } else {
      h = medianBandwidth(sq_dist, K);
    }

    // RBF 커널: kernel(j,i) = exp(-sq_dist(j,i) / (2h²))
    Eigen::MatrixXd kernel = (-sq_dist / (2.0 * h * h)).array().exp().matrix();

    // SVGD force 계산
    Eigen::MatrixXd force = computeSVGDForce(diff_flat, weights, kernel, h, K, D);

    // particles 업데이트
    particles += params_.svgd_step_size * force;

    // unflatten: (K, D) → 각 perturbed_controls[k]
    for (int k = 0; k < K; ++k) {
      Eigen::Map<Eigen::VectorXd> flat(perturbed_controls[k].data(), D);
      flat = particles.row(k).transpose();
    }

    // 클리핑
    for (int k = 0; k < K; ++k) {
      perturbed_controls[k] = dynamics_->clipControls(perturbed_controls[k]);
    }

    // re-rollout
    trajectories = dynamics_->rolloutBatch(
      current_state, perturbed_controls, params_.dt);

    // re-cost
    costs = cost_function_->compute(
      trajectories, perturbed_controls, reference_trajectory);
  }

  // 샘플 다양성 측정 (SVGD 후)
  double diversity_after = computeDiversity(perturbed_controls, K, D);

  // ──── Phase 3: final weight, update U, return ────

  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);
  }
  Eigen::VectorXd weights = weight_computation_->compute(costs, current_lambda);

  // effective noise 역산: perturbed_controls - U
  // 가중 평균으로 제어열 업데이트
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd effective_noise = perturbed_controls[k] - control_sequence_;
    weighted_noise += weights(k) * effective_noise;
  }
  control_sequence_ += weighted_noise;

  // Clip updated control sequence
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // 최적 제어 추출 (first timestep)
  Eigen::VectorXd u_opt = control_sequence_.row(0).transpose();

  // 가중 평균 궤적
  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int k = 0; k < K; ++k) {
    weighted_traj += weights(k) * trajectories[k];
  }

  // 최적 샘플 인덱스
  int best_idx;
  double min_cost = costs.minCoeff(&best_idx);

  // ESS 계산
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

  // M2 확장 정보
  info.colored_noise_used = params_.colored_noise;
  info.adaptive_temp_used = params_.adaptive_temperature;
  info.tube_mppi_used = params_.tube_enabled;

  // SVGD 전용 정보
  info.svgd_iterations = L;
  info.sample_diversity_before = diversity_before;
  info.sample_diversity_after = diversity_after;

  RCLCPP_DEBUG(
    node_->get_logger(),
    "SVMPC: min_cost=%.4f, svgd_iters=%d, diversity=%.4f→%.4f, ESS=%.1f/%d",
    min_cost, L, diversity_before, diversity_after, ess, K);

  return {u_opt, info};
}

Eigen::MatrixXd SVMPCControllerPlugin::computeSVGDForce(
  const std::vector<Eigen::VectorXd>& diff_flat,
  const Eigen::VectorXd& weights,
  const Eigen::MatrixXd& kernel,
  double bandwidth,
  int K,
  int D) const
{
  Eigen::MatrixXd force = Eigen::MatrixXd::Zero(K, D);
  double h_sq = bandwidth * bandwidth;

  for (int i = 0; i < K; ++i) {
    Eigen::VectorXd attractive = Eigen::VectorXd::Zero(D);
    Eigen::VectorXd repulsive = Eigen::VectorXd::Zero(D);

    for (int j = 0; j < K; ++j) {
      // diff[j,i] = x_j - x_i
      const Eigen::VectorXd& d_ji = diff_flat[j * K + i];

      // Attractive: w_j * k(x_j, x_i) * (x_j - x_i)
      attractive += weights(j) * kernel(j, i) * d_ji;

      // Repulsive: (1/K) * k(x_j, x_i) * (x_j - x_i) / h²
      repulsive += kernel(j, i) * d_ji / h_sq;
    }
    repulsive /= static_cast<double>(K);

    force.row(i) = (attractive + repulsive).transpose();
  }

  return force;
}

double SVMPCControllerPlugin::computeDiversity(
  const std::vector<Eigen::MatrixXd>& controls,
  int K, int D)
{
  if (K <= 1) {
    return 0.0;
  }

  // flatten
  int max_samples = std::min(K, 128);
  std::vector<Eigen::VectorXd> flat;
  flat.reserve(max_samples);

  if (K > max_samples) {
    // 서브샘플: 균등 간격
    for (int i = 0; i < max_samples; ++i) {
      int idx = i * K / max_samples;
      Eigen::Map<const Eigen::VectorXd> v(controls[idx].data(), D);
      flat.push_back(v);
    }
  } else {
    for (int k = 0; k < K; ++k) {
      Eigen::Map<const Eigen::VectorXd> v(controls[k].data(), D);
      flat.push_back(v);
    }
    max_samples = K;
  }

  // 상삼각 pairwise L2 거리 평균
  double sum_dist = 0.0;
  int count = 0;
  for (int i = 0; i < max_samples; ++i) {
    for (int j = i + 1; j < max_samples; ++j) {
      sum_dist += (flat[i] - flat[j]).norm();
      ++count;
    }
  }

  return count > 0 ? sum_dist / count : 0.0;
}

double SVMPCControllerPlugin::medianBandwidth(
  const Eigen::MatrixXd& sq_dist, int K)
{
  // 상삼각 원소 추출
  std::vector<double> triu_vals;
  triu_vals.reserve(K * (K - 1) / 2);

  for (int i = 0; i < K; ++i) {
    for (int j = i + 1; j < K; ++j) {
      triu_vals.push_back(sq_dist(i, j));
    }
  }

  if (triu_vals.empty()) {
    return 1.0;
  }

  // median 계산
  size_t mid = triu_vals.size() / 2;
  std::nth_element(triu_vals.begin(), triu_vals.begin() + mid, triu_vals.end());
  double med = triu_vals[mid];

  // h = sqrt(median / (2 * log(K+1)))
  double h = std::sqrt(med / (2.0 * std::log(static_cast<double>(K) + 1.0)));
  return std::max(h, 1e-6);
}

}  // namespace mpc_controller_ros2
