// =============================================================================
// Spline-MPPI Controller Plugin — B-spline 보간 기반 Smooth Sampling
//
// Reference: Yamada et al. (2024) "Spline-Based Model Predictive Path
//            Integral Control" — ICRA 2024
//
// 핵심 아이디어 (Parameterized Sampling):
//   기존 MPPI는 N개 시점 모두에 독립 노이즈를 부여 → 고주파 진동 발생.
//   Spline-MPPI는 P개 제어점(knot)에만 노이즈를 부여하고,
//   B-spline basis matrix로 N개 시점으로 보간 (P << N).
//   → 노이즈 차원이 N·nu → P·nu로 감소하여 자연스럽게 부드러운 제어 생성.
//
// 수식:
//   U_k = B · C_k,   B ∈ R^{N×P}, C_k ∈ R^{P×nu}    ... (1) B-spline 보간
//   C_k = C + ε_k,   ε_k ~ N(0, Σ)                   ... (2) Knot perturbation
//   C* ← C + Σ_k w_k · ε_k                            ... (3) Knot-space update
//
// B-spline basis N_{i,k}(t) — de Boor 재귀 (Cox-de Boor recursion):
//   N_{i,0}(t) = { 1  if t_i ≤ t < t_{i+1}, 0 otherwise }
//   N_{i,k}(t) = (t - t_i)/(t_{i+k} - t_i) · N_{i,k-1}(t)
//              + (t_{i+k+1} - t)/(t_{i+k+1} - t_{i+1}) · N_{i+1,k-1}(t)
//
// Python 대응: mpc_controller/controllers/mppi/spline_mppi.py
// =============================================================================

#include "mpc_controller_ros2/spline_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/utils.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>
#include <random>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::SplineMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

// -----------------------------------------------------------------------------
// B-spline Basis Matrix 계산 — de Boor 재귀 (Cox-de Boor recursion)
//
// Clamped uniform knot vector:
//   t = [0, ..., 0, t_{k+1}, ..., t_{P-1}, 1, ..., 1]
//        \_k+1_/                              \_k+1_/
//   처음/마지막 k+1개를 0/1로 고정하여 곡선이 첫/마지막 제어점을 통과하게 함.
//
// Python 대응: spline_mppi.py:37-99 (_bspline_basis)
// C++ 차이점: numpy 벡터화 대신 명시적 이중 for-loop
// -----------------------------------------------------------------------------
Eigen::MatrixXd SplineMPPIControllerPlugin::computeBSplineBasis(int N, int P, int degree)
{
  int k = degree;
  int n_knots = P + k + 1;  // knot vector 길이 = P + k + 1

  // ── Clamped uniform knot vector 생성 ──
  // t = [0,...,0, internal_knots, 1,...,1]
  Eigen::VectorXd knots = Eigen::VectorXd::Zero(n_knots);
  int n_internal = P - k - 1;
  if (n_internal > 0) {
    // Python: internal = np.linspace(0, 1, n_internal + 2)[1:-1]
    for (int i = 0; i < n_internal; ++i) {
      knots(k + 1 + i) = static_cast<double>(i + 1) / (n_internal + 1);
    }
  }
  for (int i = P; i < n_knots; ++i) {
    knots(i) = 1.0;
  }

  // 평가 지점: t ∈ [0, 1], N개 균등 분할
  Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(N, 0.0, 1.0);
  t(N - 1) = 1.0 - 1e-10;  // 마지막 점 경계 처리 (partition of unity 보장)

  // ── de Boor 재귀: Degree 0 (indicator function) ──
  // N_{i,0}(t) = { 1  if t_i ≤ t < t_{i+1},  0  otherwise }
  int cols0 = n_knots - 1;
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(N, cols0);
  for (int i = 0; i < cols0; ++i) {
    for (int j = 0; j < N; ++j) {
      if (t(j) >= knots(i) && t(j) < knots(i + 1)) {
        B(j, i) = 1.0;
      }
    }
  }

  // ── de Boor 재귀: Degree 1 → k (bottom-up) ──
  // N_{i,d}(t) = [(t - t_i) / (t_{i+d} - t_i)] · N_{i,d-1}(t)
  //            + [(t_{i+d+1} - t) / (t_{i+d+1} - t_{i+1})] · N_{i+1,d-1}(t)
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

  Eigen::MatrixXd basis = B.leftCols(P);

  // ── 행 정규화 (Partition of Unity 보정) ──
  // Σ_i N_{i,k}(t) = 1 이어야 하나, 수치 오차로 약간 벗어날 수 있음
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
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  P_ = params_.spline_num_knots;
  degree_ = params_.spline_degree;

  // 1. B-spline basis 사전 계산 — configure() 시 1회만 (N, P 고정이므로)
  basis_ = computeBSplineBasis(params_.N, P_, degree_);

  // 2. Pseudo-inverse 사전 계산 (LS warm-start용)
  // pinv(B) = (B^T B)^{-1} B^T — Eigen SVD 기반
  basis_pinv_ = basis_.completeOrthogonalDecomposition().pseudoInverse();

  // 3. Knot sigma 결정 (basis 감쇠 자동 보정)
  // B-spline 보간 시 Var(control) = σ² × Σ basis[t,p]² < σ²
  // → 유효 σ가 원본의 ~35%로 감쇠 → amp_factor로 보정
  knot_sigma_ = params_.noise_sigma;
  if (params_.spline_auto_knot_sigma) {
    double mean_row_sq_sum = 0.0;
    for (int i = 0; i < params_.N; ++i) {
      mean_row_sq_sum += basis_.row(i).squaredNorm();
    }
    mean_row_sq_sum /= params_.N;
    double amp_factor = std::sqrt(1.0 / mean_row_sq_sum);
    knot_sigma_ = params_.noise_sigma * amp_factor;
  }

  // Knot warm-start
  u_knots_ = Eigen::MatrixXd::Zero(P_, 2);

  auto node = parent.lock();
  RCLCPP_INFO(
    node->get_logger(),
    "Spline-MPPI plugin configured: num_knots=%d, degree=%d, basis=(%dx%d), "
    "auto_sigma=%s, knot_sigma=[%.3f, %.3f]",
    P_, degree_, static_cast<int>(basis_.rows()), static_cast<int>(basis_.cols()),
    params_.spline_auto_knot_sigma ? "true" : "false",
    knot_sigma_(0), knot_sigma_(1));
}

std::pair<Eigen::Vector2d, MPPIInfo> SplineMPPIControllerPlugin::computeControl(
  const Eigen::Vector3d& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  int N = params_.N;
  int K = params_.K;
  int nu = 2;

  // ──── Step 1: Warm-start — LS 재투영 ────
  // 기존 shift는 knot index ↔ 시간축 불일치 (B-spline 비선형 매핑)
  // → U를 1-step shift 후 knot space에 재투영하여 시간 정렬 보정
  Eigen::MatrixXd u_shifted = control_sequence_;
  for (int t = 0; t < N - 1; ++t) {
    u_shifted.row(t) = u_shifted.row(t + 1);
  }
  u_shifted.row(N - 1).setZero();
  u_knots_ = basis_pinv_ * u_shifted;

  // ──── Step 2: Knot space에서 노이즈 샘플링 ────
  // 수식 (2): ε_k ~ N(0, Σ),  ε_k ∈ R^{P×nu}
  // knot_sigma_: auto 보정 적용된 σ (basis 감쇠 보정)
  static std::mt19937 rng(42);
  std::normal_distribution<double> dist(0.0, 1.0);

  std::vector<Eigen::MatrixXd> knot_noise;
  knot_noise.reserve(K);
  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd noise(P_, nu);
    for (int p = 0; p < P_; ++p) {
      noise(p, 0) = dist(rng) * knot_sigma_(0);
      noise(p, 1) = dist(rng) * knot_sigma_(1);
    }
    knot_noise.push_back(noise);
  }

  // ──── Step 3: B-spline 보간 — Knot → Control ────
  // 수식 (1): U_k = B · C_k,  B ∈ R^{N×P}, C_k ∈ R^{P×nu} → U_k ∈ R^{N×nu}
  // Python: np.einsum("np,kpd->knd", self._basis, perturbed_knots)
  // C++ 차이점: numpy einsum 대신 per-sample Eigen 행렬곱 basis_ * knots
  std::vector<Eigen::MatrixXd> perturbed_controls;
  perturbed_controls.reserve(K);

  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd perturbed_knots = u_knots_ + knot_noise[k];  // (P, nu)
    Eigen::MatrixXd u_interp = basis_ * perturbed_knots;           // (N, nu)
    u_interp = dynamics_->clipControls(u_interp);
    perturbed_controls.push_back(u_interp);
  }

  // ──── Step 4: Batch rollout ────
  auto trajectories = dynamics_->rolloutBatch(
    current_state, perturbed_controls, params_.dt);

  // ──── Step 5: Cost 계산 ────
  Eigen::VectorXd costs = cost_function_->compute(
    trajectories, perturbed_controls, reference_trajectory);

  // ──── Step 6: Softmax 가중치 ────
  double current_lambda = params_.lambda;
  if (params_.adaptive_temperature && adaptive_temp_) {
    Eigen::VectorXd temp_weights = weight_computation_->compute(costs, current_lambda);
    double ess = computeESS(temp_weights);
    current_lambda = adaptive_temp_->update(ess, K);
  }
  Eigen::VectorXd weights = weight_computation_->compute(costs, current_lambda);

  // ──── Step 7: Knot space에서 가중 평균 업데이트 ────
  // 수식 (3): C* ← C + Σ_k w_k · ε_k
  // 핵심: N-차원 u space가 아닌 P-차원 knot space에서 업데이트 (P << N)
  Eigen::MatrixXd weighted_knot_noise = Eigen::MatrixXd::Zero(P_, nu);
  for (int k = 0; k < K; ++k) {
    weighted_knot_noise += weights(k) * knot_noise[k];
  }
  u_knots_ += weighted_knot_noise;

  // ──── Step 8: 최적 U 복원 (B-spline 보간) ────
  // U* = B · C*
  control_sequence_ = basis_ * u_knots_;
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // ──── Step 9: 최적 제어 추출 ────
  Eigen::Vector2d u_opt = control_sequence_.row(0).transpose();

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
    "Spline-MPPI: min_cost=%.4f, ESS=%.1f/%d, knots=%d",
    min_cost, ess, K, P_);

  return {u_opt, info};
}

}  // namespace mpc_controller_ros2
