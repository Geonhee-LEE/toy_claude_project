#include "mpc_controller_ros2/motion_model.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

Linearization MotionModel::getLinearization(
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& control,
  double dt) const
{
  int nx = stateDim();
  int nu = controlDim();

  // 유한차분 fallback
  constexpr double eps = 1e-5;

  // 단일 상태/제어 → 배치(1행)
  Eigen::MatrixXd s0(1, nx), c0(1, nu);
  s0.row(0) = state.transpose();
  c0.row(0) = control.transpose();
  Eigen::VectorXd f0 = propagateBatch(s0, c0, dt).row(0).transpose();

  Eigen::MatrixXd A(nx, nx);
  for (int j = 0; j < nx; ++j) {
    Eigen::MatrixXd s_plus = s0;
    s_plus(0, j) += eps;
    Eigen::VectorXd f_plus = propagateBatch(s_plus, c0, dt).row(0).transpose();
    A.col(j) = (f_plus - f0) / eps;
  }

  Eigen::MatrixXd B(nx, nu);
  for (int j = 0; j < nu; ++j) {
    Eigen::MatrixXd c_plus = c0;
    c_plus(0, j) += eps;
    Eigen::VectorXd f_plus = propagateBatch(s0, c_plus, dt).row(0).transpose();
    B.col(j) = (f_plus - f0) / eps;
  }

  return {A, B};
}

Eigen::MatrixXd MotionModel::propagateBatch(
  const Eigen::MatrixXd& states,
  const Eigen::MatrixXd& controls,
  double dt) const
{
  // RK4 integration
  Eigen::MatrixXd k1 = dynamicsBatch(states, controls);
  Eigen::MatrixXd k2 = dynamicsBatch(states + dt / 2.0 * k1, controls);
  Eigen::MatrixXd k3 = dynamicsBatch(states + dt / 2.0 * k2, controls);
  Eigen::MatrixXd k4 = dynamicsBatch(states + dt * k3, controls);

  Eigen::MatrixXd states_next = states + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

  // Normalize states (angle wrapping etc.)
  normalizeStates(states_next);

  return states_next;
}

std::vector<Eigen::MatrixXd> MotionModel::rolloutBatch(
  const Eigen::VectorXd& x0,
  const std::vector<Eigen::MatrixXd>& control_sequences,
  double dt) const
{
  int K = control_sequences.size();
  int N = control_sequences[0].rows();
  int nx = stateDim();

  std::vector<Eigen::MatrixXd> trajectories(K);
  for (int k = 0; k < K; ++k) {
    trajectories[k].resize(N + 1, nx);
  }

  rolloutBatchInPlace(x0, control_sequences, dt, trajectories);
  return trajectories;
}

void MotionModel::rolloutBatchInPlace(
  const Eigen::VectorXd& x0,
  const std::vector<Eigen::MatrixXd>& control_sequences,
  double dt,
  std::vector<Eigen::MatrixXd>& trajectories_out) const
{
  int K = control_sequences.size();
  if (K == 0) return;

  int N = control_sequences[0].rows();
  int nx = stateDim();
  int nu = controlDim();

  // Ensure output buffer size
  if (static_cast<int>(trajectories_out.size()) != K) {
    trajectories_out.resize(K);
  }
  for (int k = 0; k < K; ++k) {
    if (trajectories_out[k].rows() != N + 1 || trajectories_out[k].cols() != nx) {
      trajectories_out[k].resize(N + 1, nx);
    }
    trajectories_out[k].row(0) = x0.transpose();
  }

  // True Batch: K개 샘플을 동시에 propagate (시간 스텝별)
  // 배치 행렬 (K x nx)로 모아서 propagateBatch 1번 호출
  Eigen::MatrixXd batch_states(K, nx);
  Eigen::MatrixXd batch_controls(K, nu);

  // 초기 상태 설정
  for (int k = 0; k < K; ++k) {
    batch_states.row(k) = x0.transpose();
  }

  for (int t = 0; t < N; ++t) {
    // Gather: 각 샘플의 t번째 제어 입력
    for (int k = 0; k < K; ++k) {
      batch_controls.row(k) = control_sequences[k].row(t);
    }

    // 배치 propagate (K개 동시 RK4 적분)
    batch_states = propagateBatch(batch_states, batch_controls, dt);

    // Scatter: 결과를 각 궤적에 기록
    for (int k = 0; k < K; ++k) {
      trajectories_out[k].row(t + 1) = batch_states.row(k);
    }
  }
}

}  // namespace mpc_controller_ros2
