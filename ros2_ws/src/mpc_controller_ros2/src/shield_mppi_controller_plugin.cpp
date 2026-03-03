#include "mpc_controller_ros2/shield_mppi_controller_plugin.hpp"
#include "mpc_controller_ros2/motion_model_factory.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <cmath>

PLUGINLIB_EXPORT_CLASS(mpc_controller_ros2::ShieldMPPIControllerPlugin, nav2_core::Controller)

namespace mpc_controller_ros2
{

void ShieldMPPIControllerPlugin::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  // 부모 configure 호출 (기존 MPPI 전체 초기화)
  MPPIControllerPlugin::configure(parent, name, tf, costmap_ros);

  // Shield-MPPI 전용 파라미터
  shield_cbf_stride_ = std::max(1, params_.shield_cbf_stride);
  shield_max_iterations_ = std::max(1, params_.shield_max_iterations);
  shield_step_size_ = 0.1;

  RCLCPP_INFO(node_->get_logger(),
    "Shield-MPPI configured (stride=%d, max_iter=%d)",
    shield_cbf_stride_, shield_max_iterations_);
}

std::pair<Eigen::VectorXd, MPPIInfo> ShieldMPPIControllerPlugin::computeControl(
  const Eigen::VectorXd& current_state,
  const Eigen::MatrixXd& reference_trajectory)
{
  // 1. 기본 MPPI 계산 (부모 호출)
  auto [u_opt, info] = MPPIControllerPlugin::computeControl(
    current_state, reference_trajectory);

  // 2. CBF가 비활성화이거나 barrier가 없으면 기본 결과 반환
  if (!params_.cbf_enabled || barrier_set_.empty()) {
    return {u_opt, info};
  }

  int K = params_.K;
  int N = params_.N;
  int nx = dynamics_->model().stateDim();
  int nu = dynamics_->model().controlDim();

  // 3. per-step CBF 투영: 각 샘플의 각 타임스텝에서 투영
  //    trajectory_buffer_와 perturbed_buffer_를 수정하여 안전 궤적 생성
  for (int k = 0; k < K; ++k) {
    Eigen::VectorXd state_k = current_state;

    for (int t = 0; t < N; ++t) {
      if (t % shield_cbf_stride_ == 0) {
        // CBF 투영
        Eigen::VectorXd u_t = perturbed_buffer_[k].row(t).transpose();
        Eigen::VectorXd u_safe = projectControlCBF(state_k, u_t);
        perturbed_buffer_[k].row(t) = u_safe.transpose();
      }

      // 상태 전파
      Eigen::MatrixXd state_mat(1, nx);
      state_mat.row(0) = state_k.transpose();
      Eigen::MatrixXd ctrl_mat(1, nu);
      ctrl_mat.row(0) = perturbed_buffer_[k].row(t);
      Eigen::MatrixXd next_state = dynamics_->model().propagateBatch(
        state_mat, ctrl_mat, params_.dt);
      state_k = next_state.row(0).transpose();

      // 궤적 버퍼 갱신
      trajectory_buffer_[k].row(t + 1) = state_k.transpose();
    }
  }

  // 4. 투영된 궤적으로 비용 재계산
  Eigen::VectorXd costs = cost_function_->compute(
    trajectory_buffer_, perturbed_buffer_, reference_trajectory);

  // 5. 가중치 재계산
  double current_lambda = params_.lambda;
  Eigen::VectorXd weights = weight_computation_->compute(costs, current_lambda);

  // 6. 제어 시퀀스 업데이트
  Eigen::MatrixXd weighted_noise = Eigen::MatrixXd::Zero(N, nu);
  for (int k = 0; k < K; ++k) {
    weighted_noise.noalias() += weights(k) * noise_buffer_[k];
  }
  control_sequence_ += weighted_noise;
  control_sequence_ = dynamics_->clipControls(control_sequence_);

  // 7. 최적 제어 추출
  u_opt = control_sequence_.row(0).transpose();

  // 8. info 갱신
  info.sample_trajectories = trajectory_buffer_;
  info.costs = costs;
  info.sample_weights = weights;

  int best_idx;
  costs.minCoeff(&best_idx);
  info.best_trajectory = trajectory_buffer_[best_idx];

  // 가중 평균 궤적
  Eigen::MatrixXd weighted_traj = Eigen::MatrixXd::Zero(N + 1, nx);
  for (int k = 0; k < K; ++k) {
    weighted_traj.noalias() += weights(k) * trajectory_buffer_[k];
  }
  info.weighted_avg_trajectory = weighted_traj;

  return {u_opt, info};
}

Eigen::VectorXd ShieldMPPIControllerPlugin::projectControlCBF(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& u) const
{
  auto active_barriers = barrier_set_.getActiveBarriers(state);
  if (active_barriers.empty()) {
    return u;
  }

  Eigen::VectorXd u_proj = u;
  int nu = u.size();

  for (int iter = 0; iter < shield_max_iterations_; ++iter) {
    bool all_satisfied = true;

    for (const auto* barrier : active_barriers) {
      double h = barrier->evaluate(state);
      Eigen::VectorXd grad_h = barrier->gradient(state);

      // ḣ = ∇h · f(x,u) = ∇h · x_dot
      Eigen::VectorXd x_dot = computeXdot(state, u_proj);
      double h_dot = grad_h.dot(x_dot);

      // CBF 조건: ḣ + γ·h ≥ 0
      double constraint = h_dot + params_.cbf_gamma * h;

      if (constraint < 0.0) {
        all_satisfied = false;

        // Projected gradient: ∂ḣ/∂u 계산 (유한 차분)
        Eigen::VectorXd dhdot_du(nu);
        constexpr double eps = 1e-4;
        for (int j = 0; j < nu; ++j) {
          Eigen::VectorXd u_plus = u_proj;
          u_plus(j) += eps;
          Eigen::VectorXd x_dot_plus = computeXdot(state, u_plus);
          double h_dot_plus = grad_h.dot(x_dot_plus);
          dhdot_du(j) = (h_dot_plus - h_dot) / eps;
        }

        // 경사 방향으로 제어 보정
        double dhdot_du_norm_sq = dhdot_du.squaredNorm();
        if (dhdot_du_norm_sq > 1e-12) {
          double step = shield_step_size_ * (-constraint) / dhdot_du_norm_sq;
          u_proj += step * dhdot_du;
        }

        // 제어 한계 클리핑
        u_proj = dynamics_->clipControls(
          Eigen::MatrixXd(u_proj.transpose())).row(0).transpose();
      }
    }

    if (all_satisfied) {
      break;
    }
  }

  return u_proj;
}

Eigen::VectorXd ShieldMPPIControllerPlugin::computeXdot(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& u) const
{
  Eigen::MatrixXd s(1, state.size());
  s.row(0) = state.transpose();
  Eigen::MatrixXd c(1, u.size());
  c.row(0) = u.transpose();
  return dynamics_->model().dynamicsBatch(s, c).row(0).transpose();
}

}  // namespace mpc_controller_ros2
