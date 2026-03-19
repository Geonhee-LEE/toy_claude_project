// =============================================================================
// Trajectory Library MPPI Controller Plugin
//
// 사전 계산된 7종 제어 시퀀스 프리미티브 라이브러리를 결정적 샘플로 주입.
// Biased-MPPI 패턴 확장: 4종 ancillary → 7-8종 프리미티브 라이브러리.
//
// computeControl() 흐름:
//   1. traj_library_enabled=false → parent::computeControl()
//   2. Warm-start shift
//   3. noise = sampler_->sample(K, N, nu)
//   4. library_.updatePreviousSolution(control_sequence_)
//   5. L = min(samples_per_prim * num_prims, K-1)
//   6. 첫 L개: library primitive + perturbation_scale * noise
//   7. 나머지 K-L개: control_sequence_ + noise (기존 패턴)
//   8. rollout → cost → IT정규화 → weight → update
// =============================================================================

#include "mpc_controller_ros2/trajectory_library_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>
#include <algorithm>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::TrajectoryLibraryMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void TrajectoryLibraryMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  auto node = parent.lock();

  // 라이브러리 생성
  int nu = dynamics_->model().controlDim();
  library_.generate(
    params_.N, nu, params_.dt,
    params_.v_max, params_.v_min, params_.omega_max);

  RCLCPP_INFO(
    node->get_logger(),
    "Trajectory Library MPPI configured: enabled=%d, ratio=%.2f, perturbation=%.2f, "
    "adaptive=%d, num_primitives=%d",
    params_.traj_library_enabled,
    params_.traj_library_ratio,
    params_.traj_library_perturbation,
    params_.traj_library_adaptive,
    library_.numPrimitives());
}

std::pair<Eigen::VectorXd, MPPIInfo> TrajectoryLibraryMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // 비활성 시 base 호출
  if (!params_.traj_library_enabled) {
    return MPPIControllerPlugin::computeControl(current_state, reference_trajectory);
  }

  int N = params_.N;
  int K = params_.K;
  int nu = dynamics_->model().controlDim();
  int nx = dynamics_->model().stateDim();

  // ──── STEP 1: Warm-start (shift control sequence) ────
  for (int t = 0; t < N - 1; ++t) {
    control_sequence_.row(t) = control_sequence_.row(t + 1);
  }
  control_sequence_.row(N - 1) = control_sequence_.row(N - 2);

  // ──── STEP 2: Noise 샘플링 ────
  auto noise_samples = sampler_->sample(K, N, nu);

  // ──── STEP 3: 라이브러리 업데이트 ────
  library_.updatePreviousSolution(control_sequence_);

  const auto& primitives = library_.getPrimitives();
  int num_prims = library_.numPrimitives();

  // ──── STEP 4: L 계산 ────
  int samples_per_prim = params_.traj_library_num_per_primitive;
  if (samples_per_prim <= 0) {
    // auto: ratio 기반 계산
    int total_lib = static_cast<int>(std::floor(params_.traj_library_ratio * K));
    samples_per_prim = (num_prims > 0) ? std::max(1, total_lib / num_prims) : 0;
  }
  int L = samples_per_prim * num_prims;
  // L이 K를 초과하지 않도록 보호
  if (L >= K) {
    samples_per_prim = (num_prims > 0) ? (K - 1) / num_prims : 0;
    L = samples_per_prim * num_prims;
  }
  int K_gaussian = K - L;

  // ──── STEP 5: 샘플 구성 ────
  std::vector<Eigen::MatrixXd> perturbed_controls;
  perturbed_controls.reserve(K);

  std::vector<Eigen::MatrixXd> noise_for_weight;
  noise_for_weight.reserve(K);

  // 최저 비용 프리미티브 추적용
  std::vector<int> sample_to_primitive(K, -1);  // -1 = Gaussian

  // (a) 라이브러리 결정적 샘플 (L개)
  int noise_idx = 0;
  for (int p = 0; p < num_prims; ++p) {
    for (int j = 0; j < samples_per_prim; ++j) {
      Eigen::MatrixXd lib_ctrl = primitives[p].control_sequence;
      // 작은 섭동 추가
      if (params_.traj_library_perturbation > 0.0 && noise_idx < K) {
        lib_ctrl += params_.traj_library_perturbation * noise_samples[noise_idx];
      }
      lib_ctrl = dynamics_->clipControls(lib_ctrl);
      perturbed_controls.push_back(lib_ctrl);
      // noise_for_weight = lib_ctrl - control_sequence_ (밀도비 소거용)
      noise_for_weight.push_back(lib_ctrl - control_sequence_);
      sample_to_primitive[noise_idx] = p;
      ++noise_idx;
    }
  }

  // (b) Gaussian 샘플 (K_gaussian개)
  int K_exploit = static_cast<int>((1.0 - params_.exploration_ratio) * K_gaussian);
  for (int k = 0; k < K_gaussian; ++k) {
    int src_idx = L + k;
    if (src_idx >= K) src_idx = k;  // safety
    Eigen::MatrixXd perturbed;
    if (k < K_exploit) {
      perturbed = control_sequence_ + noise_samples[src_idx];
    } else {
      perturbed = noise_samples[src_idx];
    }
    perturbed = dynamics_->clipControls(perturbed);
    perturbed_controls.push_back(perturbed);
    noise_for_weight.push_back(noise_samples[src_idx]);
  }

  // ──── STEP 6: Batch rollout ────
  auto trajectories = dynamics_->rolloutBatch(
    current_state, perturbed_controls, params_.dt);

  // ──── STEP 7: Cost 계산 ────
  Eigen::VectorXd costs;
  CostBreakdown cost_breakdown;
  if (params_.debug_collision_viz) {
    cost_breakdown = cost_function_->computeDetailed(
      trajectories, perturbed_controls, reference_trajectory);
    costs = cost_breakdown.total_costs;
  } else {
    costs = cost_function_->compute(
      trajectories, perturbed_controls, reference_trajectory);
  }

  // ──── STEP 7.5: IT 정규화 ────
  if (params_.it_alpha < 1.0) {
    Eigen::VectorXd sigma_inv = params_.noise_sigma.cwiseInverse().cwiseAbs2();
    for (int k = 0; k < K; ++k) {
      double it_cost = 0.0;
      for (int t = 0; t < N; ++t) {
        Eigen::VectorXd u_prev_t = control_sequence_.row(t).transpose();
        Eigen::VectorXd u_k_t = perturbed_controls[k].row(t).transpose();
        it_cost += u_prev_t.dot(sigma_inv.cwiseProduct(u_k_t));
      }
      costs(k) += params_.lambda * (1.0 - params_.it_alpha) * it_cost;
    }
  }

  // ──── STEP 8: Weight 계산 ────
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);
  }
  Eigen::VectorXd weights = weight_computation_->compute(costs, current_lambda);

  // ──── STEP 9: control_sequence_ += sum(w[k] * noise_for_weight[k]) ────
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_noise += weights(k) * noise_for_weight[k];
  }
  control_sequence_ += weighted_noise;
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // ──── STEP 10: 적응형 비율 조정 ────
  if (params_.traj_library_adaptive && L > 0) {
    // 라이브러리 샘플 중 최저 비용 확인
    double lib_min_cost = std::numeric_limits<double>::max();
    double gauss_min_cost = std::numeric_limits<double>::max();
    for (int k = 0; k < L && k < costs.size(); ++k) {
      lib_min_cost = std::min(lib_min_cost, costs(k));
    }
    for (int k = L; k < K && k < costs.size(); ++k) {
      gauss_min_cost = std::min(gauss_min_cost, costs(k));
    }
    // 라이브러리가 더 좋으면 비율 증가, 아니면 감소 (간단한 EMA)
    if (lib_min_cost < gauss_min_cost) {
      params_.traj_library_ratio = std::min(0.5, params_.traj_library_ratio * 1.05);
    } else {
      params_.traj_library_ratio = std::max(0.05, params_.traj_library_ratio * 0.95);
    }
  }

  // ──── STEP 11: Extract optimal control ────
  Eigen::VectorXd u_opt = control_sequence_.row(0).transpose();

  // Weighted average trajectory
  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int k = 0; k < K; ++k) {
    weighted_traj += weights(k) * trajectories[k];
  }

  // Best sample
  int best_idx;
  double min_cost = costs.minCoeff(&best_idx);
  double ess = computeESS(weights);

  // 최저 비용 프리미티브 이름
  std::string best_prim_name = "GAUSSIAN";
  if (best_idx < L && best_idx < static_cast<int>(sample_to_primitive.size())) {
    int prim_idx = sample_to_primitive[best_idx];
    if (prim_idx >= 0 && prim_idx < num_prims) {
      best_prim_name = primitives[prim_idx].name;
    }
  }

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
  info.library_best_primitive = best_prim_name;

  if (params_.debug_collision_viz) {
    info.cost_breakdown = cost_breakdown;
  }

  info.colored_noise_used = params_.colored_noise;
  info.adaptive_temp_used = params_.adaptive_temperature;
  info.tube_mppi_used = params_.tube_enabled;

  RCLCPP_DEBUG(
    node_->get_logger(),
    "TrajLib-MPPI: min_cost=%.4f, ESS=%.1f/%d, L=%d, K_gauss=%d, best_prim=%s",
    min_cost, ess, K, L, K_gaussian, best_prim_name.c_str());

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
