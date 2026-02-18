// =============================================================================
// Smooth-MPPI (SMPPI) Controller Plugin
//
// Reference: Kim et al. (2021) "Smooth Model Predictive Path Integral Control
//            without Smoothing" — pytorch_mppi v0.8.0+
//
// 핵심 아이디어 (Input-Lifting / Δu Reparameterization):
//   기존 MPPI는 u space에서 직접 최적화하여 시간 스텝 간 독립적 노이즈가
//   들어가므로 jerky한 제어를 생성. Smooth-MPPI는 Δu (제어 변화량) space에서
//   최적화하고 cumulative sum으로 u를 복원하여 구조적 부드러움을 보장.
//
// 수식:
//   u[t] = u_prev + Σ_{i=0}^{t} Δu[i]              ... (1) Input-Lifting
//   J_jerk = w_jerk · Σ_t R_jerk · ‖Δu[t+1] - Δu[t]‖²  ... (2) Jerk Cost
//   ΔU* ← ΔU + Σ_k w_k · ε_k                        ... (3) Δu-space Update
//
// Python 대응: mpc_controller/controllers/mppi/smooth_mppi.py
// =============================================================================

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
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  // Δu warm-start 시퀀스 초기화 (Python: self.DU = np.zeros((N, nu)))
  delta_u_sequence_ = Eigen::MatrixXd::Zero(params_.N, 2);
  // 이전 스텝의 마지막 적용 제어 — cumsum 기준점 (Python: self._u_prev)
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

  // ──── Step 1: Shift ΔU (warm start) ────
  // Python: self.DU[:-1] = self.DU[1:]; self.DU[-1] = 0.0
  for (int t = 0; t < N - 1; ++t) {
    delta_u_sequence_.row(t) = delta_u_sequence_.row(t + 1);
  }
  delta_u_sequence_.row(N - 1).setZero();

  // ──── Step 2: Δu space에서 노이즈 샘플링 ────
  // ε_k ~ N(0, Σ), k = 1..K  — Δu space에서 샘플링
  // Python: delta_noise = self.sampler.sample(K, N, nu)  → (K, N, nu)
  // C++ 차이점: sampler_->sample()은 vector<MatrixXd>를 반환 (K개의 (N,nu) 행렬)
  auto delta_noise = sampler_->sample(K, N, nu);

  // ──── Step 3: ΔU + ε → perturbed Δu ────
  // Python: perturbed_du = self.DU[np.newaxis, :, :] + delta_noise (broadcasting)
  // C++ 차이점: 개별 행렬 덧셈 (numpy broadcasting 대신 명시적 루프)
  std::vector<Eigen::MatrixXd> perturbed_du;
  perturbed_du.reserve(K);
  for (int k = 0; k < K; ++k) {
    perturbed_du.push_back(delta_u_sequence_ + delta_noise[k]);
  }

  // ──── Step 4: Cumulative sum으로 u 시퀀스 복원 ────
  // 수식 (1): u[t] = u_prev + Σ_{i=0}^{t} Δu[i]
  // Python: u_sequences = u_prev + np.cumsum(perturbed_du, axis=1)
  // C++ 차이점: np.cumsum 대신 명시적 for-loop 누적합
  std::vector<Eigen::MatrixXd> perturbed_controls;
  perturbed_controls.reserve(K);
  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd u_seq(N, nu);
    Eigen::Vector2d cumulative = u_prev_;
    for (int t = 0; t < N; ++t) {
      cumulative += perturbed_du[k].row(t).transpose();
      u_seq.row(t) = cumulative.transpose();
    }
    u_seq = dynamics_->clipControls(u_seq);
    perturbed_controls.push_back(u_seq);
  }

  // ──── Step 5: Batch rollout ────
  // x[t+1] = f(x[t], u[t]) — 동역학 전파 (RK4)
  auto trajectories = dynamics_->rolloutBatch(
    current_state, perturbed_controls, params_.dt);

  // ──── Step 6: Cost 계산 (tracking + terminal + effort) ────
  Eigen::VectorXd costs = cost_function_->compute(
    trajectories, perturbed_controls, reference_trajectory);

  // ──── Step 7: Jerk cost 추가 ────
  // 수식 (2): J_jerk = w · Σ_{t=0}^{N-2} [R_v·(Δu_v[t+1]-Δu_v[t])² + R_ω·(Δu_ω[t+1]-Δu_ω[t])²]
  // ΔΔu := Δu[t+1] - Δu[t] (2차 차분 = jerk에 비례)
  // Python: ddu = perturbed_du[:, 1:, :] - perturbed_du[:, :-1, :] (vectorized)
  // C++ 차이점: 중첩 for-loop으로 element-wise 계산
  if (params_.smooth_action_cost_weight > 0.0 && N > 1) {
    Eigen::Vector2d R_jerk(params_.smooth_R_jerk_v, params_.smooth_R_jerk_omega);

    for (int k = 0; k < K; ++k) {
      double jerk_cost = 0.0;
      for (int t = 0; t < N - 1; ++t) {
        // ΔΔu[t] = Δu[t+1] - Δu[t]
        Eigen::Vector2d ddu = perturbed_du[k].row(t + 1).transpose()
                            - perturbed_du[k].row(t).transpose();
        // ‖ΔΔu‖²_R = ΔΔu^T · R · ΔΔu (대각 가중)
        jerk_cost += ddu(0) * ddu(0) * R_jerk(0) + ddu(1) * ddu(1) * R_jerk(1);
      }
      costs(k) += params_.smooth_action_cost_weight * jerk_cost;
    }
  }

  // ──── Step 8: Softmax 가중치 계산 ────
  // w_k = exp(-cost_k / λ) / Σ_j exp(-cost_j / λ)
  // C++ 차이점: Python의 softmax_weights() 대신 WeightComputation Strategy 패턴 사용
  //           → VanillaMPPIWeights / LogMPPIWeights 등 교체 가능
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);
  }
  Eigen::VectorXd weights = weight_computation_->compute(costs, current_lambda);

  // ──── Step 9: Δu space에서 가중 평균 업데이트 ────
  // 수식 (3): ΔU ← ΔU + Σ_k w_k · ε_k
  // 핵심: u space가 아닌 Δu space에서 업데이트 → 구조적 부드러움 유지
  Eigen::MatrixXd weighted_delta_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_delta_noise += weights(k) * delta_noise[k];
  }
  delta_u_sequence_ += weighted_delta_noise;

  // ──── Step 10: 최적 U 복원 (ΔU → U via cumsum) ────
  // U* = u_prev + cumsum(ΔU*)
  control_sequence_ = Eigen::MatrixXd::Zero(N, nu);
  Eigen::Vector2d cumulative = u_prev_;
  for (int t = 0; t < N; ++t) {
    cumulative += delta_u_sequence_.row(t).transpose();
    control_sequence_.row(t) = cumulative.transpose();
  }
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // ──── Step 11: 최적 제어 추출 & u_prev 업데이트 ────
  Eigen::Vector2d u_opt = control_sequence_.row(0).transpose();
  u_prev_ = u_opt;  // 다음 스텝의 cumsum 기준점

  // Weighted average trajectory
  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, 3);
  for (int k = 0; k < K; ++k) {
    weighted_traj += weights(k) * trajectories[k];
  }

  int best_idx;
  double min_cost = costs.minCoeff(&best_idx);
  double ess = computeESS(weights);

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
