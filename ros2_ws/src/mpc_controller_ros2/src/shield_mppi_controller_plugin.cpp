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

  // 3. 현재 상태에서 active barrier가 없으면 skip (장애물 없는 구간 → 0 오버헤드)
  auto active_barriers = barrier_set_.getActiveBarriers(current_state);
  if (active_barriers.empty()) {
    return {u_opt, info};
  }

  int N = params_.N;
  int nx = dynamics_->model().stateDim();
  int nu = dynamics_->model().controlDim();

  // 4. 최적 제어 시퀀스의 처음 shield_steps 스텝만 CBF 투영
  //    shield_cbf_stride 값을 "투영할 스텝 수"로 재해석:
  //    stride=3 → 첫 3 스텝만 투영 (나머지는 MPPI 비용이 처리)
  int shield_steps = std::min(shield_cbf_stride_, N);
  Eigen::VectorXd state_k = current_state;
  bool any_projected = false;

  for (int t = 0; t < shield_steps; ++t) {
    Eigen::VectorXd u_t = control_sequence_.row(t).transpose();
    Eigen::VectorXd u_safe = projectControlCBF(state_k, u_t);

    if ((u_safe - u_t).squaredNorm() > 1e-12) {
      control_sequence_.row(t) = u_safe.transpose();
      any_projected = true;
    }

    // 상태 전파
    Eigen::MatrixXd state_mat(1, nx);
    state_mat.row(0) = state_k.transpose();
    Eigen::MatrixXd ctrl_mat(1, nu);
    ctrl_mat.row(0) = control_sequence_.row(t);
    state_k = dynamics_->model().propagateBatch(
      state_mat, ctrl_mat, params_.dt).row(0).transpose();
  }

  // 5. 투영된 최적 제어 추출
  u_opt = control_sequence_.row(0).transpose();

  // 6. info에 투영된 궤적 반영 (시각화용)
  if (any_projected) {
    Eigen::MatrixXd projected_traj(N + 1, nx);
    projected_traj.row(0) = current_state.transpose();
    Eigen::VectorXd s = current_state;
    for (int t = 0; t < N; ++t) {
      Eigen::MatrixXd sm(1, nx);
      sm.row(0) = s.transpose();
      Eigen::MatrixXd cm(1, nu);
      cm.row(0) = control_sequence_.row(t);
      s = dynamics_->model().propagateBatch(sm, cm, params_.dt).row(0).transpose();
      projected_traj.row(t + 1) = s.transpose();
    }
    info.weighted_avg_trajectory = projected_traj;
  }

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

  for (int iter = 0; iter < shield_max_iterations_; ++iter) {
    bool all_satisfied = true;

    for (const auto* barrier : active_barriers) {
      double h = barrier->evaluate(state);
      Eigen::VectorXd grad_h = barrier->gradient(state);

      // ḣ = ∇h · f(x,u)
      Eigen::VectorXd x_dot = computeXdot(state, u_proj);
      double h_dot = grad_h.dot(x_dot);

      // CBF 조건: ḣ + γ·h ≥ 0
      double constraint = h_dot + params_.cbf_gamma * h;

      if (constraint < 0.0) {
        all_satisfied = false;

        // 해석적 ∂ḣ/∂u (diff_drive: f = [v·cosθ, v·sinθ, ω])
        int nu_dim = u.size();
        Eigen::VectorXd dhdot_du(nu_dim);

        if (nu_dim == 2 && state.size() >= 3) {
          double theta = state(2);
          dhdot_du(0) = grad_h(0) * std::cos(theta) + grad_h(1) * std::sin(theta);
          dhdot_du(1) = (grad_h.size() > 2) ? grad_h(2) : 0.0;
        } else {
          // fallback: 유한 차분
          constexpr double eps = 1e-4;
          for (int j = 0; j < nu_dim; ++j) {
            Eigen::VectorXd u_plus = u_proj;
            u_plus(j) += eps;
            double h_dot_plus = grad_h.dot(computeXdot(state, u_plus));
            dhdot_du(j) = (h_dot_plus - h_dot) / eps;
          }
        }

        double dhdot_du_norm_sq = dhdot_du.squaredNorm();
        if (dhdot_du_norm_sq > 1e-12) {
          double step = shield_step_size_ * (-constraint) / dhdot_du_norm_sq;
          u_proj += step * dhdot_du;
        }

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
