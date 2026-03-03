#include "mpc_controller_ros2/residual_dynamics_model.hpp"
#include <stdexcept>

namespace mpc_controller_ros2
{

ResidualDynamicsModel::ResidualDynamicsModel(
  std::unique_ptr<MotionModel> nominal,
  std::unique_ptr<EigenMLP> residual_mlp,
  double alpha)
: nominal_(std::move(nominal)),
  residual_mlp_(std::move(residual_mlp)),
  alpha_(std::clamp(alpha, 0.0, 1.0))
{
  if (!nominal_) {
    throw std::invalid_argument("ResidualDynamicsModel: nominal model is null");
  }
  if (!residual_mlp_) {
    throw std::invalid_argument("ResidualDynamicsModel: residual MLP is null");
  }

  // MLP 입력 차원 = nx + nu
  int expected_input = nominal_->stateDim() + nominal_->controlDim();
  if (residual_mlp_->inputDim() != expected_input) {
    throw std::invalid_argument(
      "ResidualDynamicsModel: MLP input dim (" + std::to_string(residual_mlp_->inputDim()) +
      ") != nx+nu (" + std::to_string(expected_input) + ")");
  }

  // MLP 출력 차원 = nx
  if (residual_mlp_->outputDim() != nominal_->stateDim()) {
    throw std::invalid_argument(
      "ResidualDynamicsModel: MLP output dim (" + std::to_string(residual_mlp_->outputDim()) +
      ") != nx (" + std::to_string(nominal_->stateDim()) + ")");
  }
}

Eigen::MatrixXd ResidualDynamicsModel::dynamicsBatch(
  const Eigen::MatrixXd& states,
  const Eigen::MatrixXd& controls) const
{
  // 공칭 동역학
  Eigen::MatrixXd x_dot_nominal = nominal_->dynamicsBatch(states, controls);

  if (alpha_ < 1e-12) {
    return x_dot_nominal;  // alpha=0 → 순수 공칭
  }

  // MLP 입력 특성: [states | controls] → (M, nx+nu)
  int M = states.rows();
  int nx = states.cols();
  int nu = controls.cols();
  Eigen::MatrixXd features(M, nx + nu);
  features.leftCols(nx) = states;
  features.rightCols(nu) = controls;

  // 잔차 예측
  Eigen::MatrixXd x_dot_residual = residual_mlp_->forwardBatch(features);

  // 합산: f_total = f_nominal + alpha * f_residual
  return x_dot_nominal + alpha_ * x_dot_residual;
}

Eigen::MatrixXd ResidualDynamicsModel::clipControls(
  const Eigen::MatrixXd& controls) const
{
  return nominal_->clipControls(controls);
}

void ResidualDynamicsModel::normalizeStates(Eigen::MatrixXd& states) const
{
  nominal_->normalizeStates(states);
}

geometry_msgs::msg::Twist ResidualDynamicsModel::controlToTwist(
  const Eigen::VectorXd& control) const
{
  return nominal_->controlToTwist(control);
}

Eigen::VectorXd ResidualDynamicsModel::twistToControl(
  const geometry_msgs::msg::Twist& twist) const
{
  return nominal_->twistToControl(twist);
}

std::vector<int> ResidualDynamicsModel::angleIndices() const
{
  return nominal_->angleIndices();
}

}  // namespace mpc_controller_ros2
