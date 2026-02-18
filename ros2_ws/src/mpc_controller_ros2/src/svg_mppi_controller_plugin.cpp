// =============================================================================
// SVG-MPPI (Stein Variational Guided MPPI) Controller Plugin
//
// Reference: Kondo et al. (2024) "SVG-MPPI: Steering Stein Variational
//            Guided MPPI for Efficient Navigation" — ICRA 2024
//
// 핵심 아이디어 (Guide Particle + Follower Resampling):
//   SVMPC는 K×K pairwise SVGD를 수행하여 O(K²D)로 비용이 큼.
//   SVG-MPPI는 G개 guide particle만 SVGD(O(G²D))로 최적화한 뒤,
//   나머지 K-G개를 guide 주변에서 리샘플링하여 계산량을 대폭 절감.
//   G << K이므로 총 복잡도: O(G²D) + O(KND) << O(K²D)
//
// 수식:
//   Guide 선택:    {x_g}_{g=1}^G = argmin_G S(x)      ... (1) Top-G selection
//   SVGD update:   φ*(x_i) ← (1/G) Σ_j [k(x_j,x_i)·∇log p(x_j)
//                            + ∇_{x_j} k(x_j,x_i)]    ... (2) Stein force
//   Gradient-free attractive force:
//                  attract_i = Σ_j w_j · k(x_j,x_i) · (x_j - x_i)  ... (2a)
//   Repulsive force (diversity):
//                  repel_i = (1/G) Σ_j k(x_j,x_i) · (x_j - x_i) / h²  ... (2b)
//   RBF kernel:    k(x_i,x_j) = exp(-‖x_i-x_j‖² / 2h²)  ... (3)
//   Median bandwidth:  h = √(med(‖x_i-x_j‖²) / (2·log(G+1)))  ... (4)
//   Follower:      x_f ~ N(x_guide, σ²_resample)        ... (5) Resampling
//
// Python 대응: mpc_controller/controllers/mppi/svg_mppi.py
// =============================================================================

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
  // SVMPCControllerPlugin::configure() → MPPIControllerPlugin::configure()
  // SVGD 메서드(computeSVGDForce, computeDiversity, medianBandwidth) 상속
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
  int G = params_.svg_num_guide_particles;  // guide particle 수
  int L = params_.svg_guide_iterations;     // SVGD 반복 횟수
  int K = params_.K;
  int N = params_.N;
  int nu = 2;
  int D = N * nu;  // flatten 차원 (제어 시퀀스 전체)

  // G=0 또는 L=0이면 SVMPC (전체 K×K SVGD) fallback
  // Python 차이점: Python은 MPPIController (Vanilla) fallback,
  //             C++은 SVMPCControllerPlugin (SVMPC) fallback
  if (G <= 0 || L <= 0) {
    return SVMPCControllerPlugin::computeControl(current_state, reference_trajectory);
  }

  G = std::min(G, K);

  // ════════════════════════════════════════════════════════════════════
  // Phase 1: 초기 샘플링 & 비용 — 전체 K개 sample → rollout → cost
  // ════════════════════════════════════════════════════════════════════

  // Shift (warm start)
  for (int t = 0; t < N - 1; ++t) {
    control_sequence_.row(t) = control_sequence_.row(t + 1);
  }
  control_sequence_.row(N - 1).setZero();

  // 전체 K개 노이즈 샘플링 (guide 후보 선택용)
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

  // ════════════════════════════════════════════════════════════════════
  // Phase 2: Guide particle 선택 — 수식 (1)
  //   비용 최저 G개를 guide로 선택
  //   Python: np.argpartition(costs, G)[:G]  — O(K) average
  //   C++ 차이점: std::partial_sort — O(K·logG)
  // ════════════════════════════════════════════════════════════════════

  std::vector<int> indices(K);
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + G, indices.end(),
    [&costs](int a, int b) { return costs(a) < costs(b); });

  // Guide particles를 flatten — (G, D) where D = N·nu
  // Python: guide_particles = perturbed_controls[guide_idx].reshape(G, D)
  // C++ 차이점: Eigen::Map으로 zero-copy flatten (Python은 np.reshape view)
  Eigen::MatrixXd guide_particles(G, D);
  Eigen::VectorXd guide_costs(G);
  for (int g = 0; g < G; ++g) {
    int idx = indices[g];
    Eigen::Map<Eigen::VectorXd> flat(
      const_cast<double*>(perturbed_controls[idx].data()), D);
    guide_particles.row(g) = flat.transpose();
    guide_costs(g) = costs(idx);
  }

  // Diversity before SVGD (평균 pairwise L2 거리)
  std::vector<Eigen::MatrixXd> guide_ctrl_vec;
  guide_ctrl_vec.reserve(G);
  for (int g = 0; g < G; ++g) {
    Eigen::MatrixXd ctrl(N, nu);
    Eigen::Map<Eigen::VectorXd> flat(ctrl.data(), D);
    flat = guide_particles.row(g).transpose();
    guide_ctrl_vec.push_back(ctrl);
  }
  double diversity_before = computeDiversity(guide_ctrl_vec, G, D);

  // ════════════════════════════════════════════════════════════════════
  // Phase 3: SVGD on guides only — G×G 커널 (수식 2, 3, 4)
  //   SVMPC와 달리 G << K이므로 O(G²D) << O(K²D)
  //
  //   수식 (2a): attract_i = Σ_j w_j · k(x_j,x_i) · (x_j - x_i)
  //   수식 (2b): repel_i = (1/G) Σ_j k(x_j,x_i) · (x_j-x_i) / h²
  //   수식 (3):  k(x_i,x_j) = exp(-‖x_i-x_j‖² / 2h²)
  //   수식 (4):  h = √(median(‖x_i-x_j‖²) / 2·log(G+1))
  //
  //   Python: self._svgd_update(diff, weights, kernel, h, K)
  //   C++:    부모 SVMPCControllerPlugin의 protected 메서드 재사용
  //           computeSVGDForce(), medianBandwidth(), computeDiversity()
  // ════════════════════════════════════════════════════════════════════

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

    // Pairwise diff (G×G) — diff[j,i] = x_j - x_i
    Eigen::MatrixXd sq_dist(G, G);
    std::vector<Eigen::VectorXd> diff_flat(G * G);

    for (int j = 0; j < G; ++j) {
      for (int i = 0; i < G; ++i) {
        diff_flat[j * G + i] = guide_particles.row(j).transpose() - guide_particles.row(i).transpose();
        sq_dist(j, i) = diff_flat[j * G + i].squaredNorm();
      }
    }

    // 수식 (4): Median heuristic bandwidth
    double h;
    if (params_.svgd_bandwidth > 0.0) {
      h = params_.svgd_bandwidth;
    } else {
      h = medianBandwidth(sq_dist, G);
    }

    // 수식 (3): RBF kernel
    Eigen::MatrixXd kernel = (-sq_dist / (2.0 * h * h)).array().exp().matrix();

    // 수식 (2a, 2b): SVGD force = attractive + repulsive
    Eigen::MatrixXd force = computeSVGDForce(diff_flat, guide_weights, kernel, h, G, D);

    // Guide particle 업데이트: x_g ← x_g + ε · φ*(x_g)
    guide_particles += step_size * force;

    // Clip & re-evaluate (unflatten → clip → re-flatten)
    for (int g = 0; g < G; ++g) {
      Eigen::MatrixXd ctrl(N, nu);
      Eigen::Map<Eigen::VectorXd> flat(ctrl.data(), D);
      flat = guide_particles.row(g).transpose();
      ctrl = dynamics_->clipControls(ctrl);
      Eigen::Map<Eigen::VectorXd> flat2(ctrl.data(), D);
      guide_particles.row(g) = flat2.transpose();
      guide_ctrl_vec[g] = ctrl;
    }

    // Re-rollout & re-cost (guide만, G개)
    auto guide_trajectories = dynamics_->rolloutBatch(
      current_state, guide_ctrl_vec, params_.dt);
    guide_costs = cost_function_->compute(
      guide_trajectories, guide_ctrl_vec, reference_trajectory);
  }

  // Diversity after SVGD (SVGD로 인한 다양성 변화 측정)
  double diversity_after = computeDiversity(guide_ctrl_vec, G, D);

  // ════════════════════════════════════════════════════════════════════
  // Phase 4: Follower resampling — 수식 (5)
  //   각 guide 주변에서 (K-G)/G개씩 follower 리샘플링
  //   x_follower ~ N(x_guide, σ²_resample · I)
  //
  //   Python: follower_noise = sampler.sample(n_f, N, nu) * resample_std
  //           follower_controls = guide[g] + follower_noise
  // ════════════════════════════════════════════════════════════════════

  int n_followers = K - G;
  int followers_per_guide = std::max(1, n_followers / G);

  std::vector<Eigen::MatrixXd> all_controls;
  all_controls.reserve(K);

  // Guide 자체를 먼저 추가
  for (int g = 0; g < G; ++g) {
    all_controls.push_back(guide_ctrl_vec[g]);
  }

  // 각 guide 주변 follower 리샘플링
  double resample_std = params_.svg_resample_std;
  for (int g = 0; g < G; ++g) {
    int n_f;
    if (g < G - 1) {
      n_f = followers_per_guide;
    } else {
      // 마지막 guide에 나머지 할당 (K-G가 G로 나누어떨어지지 않는 경우)
      n_f = n_followers - followers_per_guide * (G - 1);
    }
    if (n_f <= 0) {
      continue;
    }

    // σ_resample 스케일링된 노이즈로 follower 생성
    auto follower_noise = sampler_->sample(n_f, N, nu);
    for (int f = 0; f < n_f; ++f) {
      Eigen::MatrixXd follower_ctrl = guide_ctrl_vec[g] + resample_std * follower_noise[f];
      follower_ctrl = dynamics_->clipControls(follower_ctrl);
      all_controls.push_back(follower_ctrl);
    }
  }

  int K_total = static_cast<int>(all_controls.size());

  // ════════════════════════════════════════════════════════════════════
  // Phase 5: 전체 rollout → cost → weight → U 업데이트
  //   Guide + Follower 전체 K개로 최종 가중 평균 업데이트
  //   U* ← U + Σ_k w_k · (x_k - U)
  // ════════════════════════════════════════════════════════════════════

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

  // effective noise 역산 후 가중 평균
  // U ← U + Σ_k w_k · (perturbed_k - U)
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K_total; ++k) {
    Eigen::MatrixXd effective_noise = all_controls[k] - control_sequence_;
    weighted_noise += weights(k) * effective_noise;
  }
  control_sequence_ += weighted_noise;
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  Eigen::Vector2d u_opt = control_sequence_.row(0).transpose();

  // Weighted average trajectory
  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, 3);
  for (int k = 0; k < K_total; ++k) {
    weighted_traj += weights(k) * all_trajectories[k];
  }

  int best_idx;
  double min_cost = all_costs.minCoeff(&best_idx);
  double ess = computeESS(weights);

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
