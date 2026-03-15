#include "mpc_controller_ros2/ensemble_dynamics_model.hpp"
#include <stdexcept>
#include <numeric>

namespace mpc_controller_ros2
{

EnsembleDynamicsModel::EnsembleDynamicsModel(
  std::unique_ptr<MotionModel> nominal,
  std::vector<std::unique_ptr<EigenMLP>> ensemble,
  double alpha)
: nominal_(std::move(nominal)),
  ensemble_(std::move(ensemble)),
  alpha_(std::clamp(alpha, 0.0, 1.0))
{
  if (!nominal_) {
    throw std::invalid_argument("EnsembleDynamicsModel: nominal model is null");
  }
  if (ensemble_.empty()) {
    throw std::invalid_argument("EnsembleDynamicsModel: ensemble is empty");
  }

  int expected_input = nominal_->stateDim() + nominal_->controlDim();
  int expected_output = nominal_->stateDim();

  for (size_t i = 0; i < ensemble_.size(); ++i) {
    if (!ensemble_[i]) {
      throw std::invalid_argument(
        "EnsembleDynamicsModel: ensemble[" + std::to_string(i) + "] is null");
    }
    if (ensemble_[i]->inputDim() != expected_input) {
      throw std::invalid_argument(
        "EnsembleDynamicsModel: ensemble[" + std::to_string(i) +
        "] input dim (" + std::to_string(ensemble_[i]->inputDim()) +
        ") != nx+nu (" + std::to_string(expected_input) + ")");
    }
    if (ensemble_[i]->outputDim() != expected_output) {
      throw std::invalid_argument(
        "EnsembleDynamicsModel: ensemble[" + std::to_string(i) +
        "] output dim (" + std::to_string(ensemble_[i]->outputDim()) +
        ") != nx (" + std::to_string(expected_output) + ")");
    }
  }
}

Eigen::MatrixXd EnsembleDynamicsModel::buildFeatures(
  const Eigen::MatrixXd& states,
  const Eigen::MatrixXd& controls) const
{
  int M = states.rows();
  int nx = states.cols();
  int nu = controls.cols();
  Eigen::MatrixXd features(M, nx + nu);
  features.leftCols(nx) = states;
  features.rightCols(nu) = controls;
  return features;
}

Eigen::MatrixXd EnsembleDynamicsModel::dynamicsBatch(
  const Eigen::MatrixXd& states,
  const Eigen::MatrixXd& controls) const
{
  // 공칭 동역학
  Eigen::MatrixXd x_dot_nominal = nominal_->dynamicsBatch(states, controls);

  if (alpha_ < 1e-12) {
    return x_dot_nominal;
  }

  // 앙상블 평균 잔차
  Eigen::MatrixXd features = buildFeatures(states, controls);
  int M_models = static_cast<int>(ensemble_.size());

  Eigen::MatrixXd mean_residual = ensemble_[0]->forwardBatch(features);
  for (int i = 1; i < M_models; ++i) {
    mean_residual += ensemble_[i]->forwardBatch(features);
  }
  mean_residual /= static_cast<double>(M_models);

  return x_dot_nominal + alpha_ * mean_residual;
}

EnsembleDynamicsModel::PredictionResult EnsembleDynamicsModel::predictWithUncertainty(
  const Eigen::MatrixXd& states,
  const Eigen::MatrixXd& controls) const
{
  Eigen::MatrixXd features = buildFeatures(states, controls);
  int M_models = static_cast<int>(ensemble_.size());
  int M_batch = features.rows();
  int nx = nominal_->stateDim();

  // 각 MLP 예측 수집
  std::vector<Eigen::MatrixXd> predictions(M_models);
  for (int i = 0; i < M_models; ++i) {
    predictions[i] = ensemble_[i]->forwardBatch(features);
  }

  // 평균 계산
  Eigen::MatrixXd mean = Eigen::MatrixXd::Zero(M_batch, nx);
  for (int i = 0; i < M_models; ++i) {
    mean += predictions[i];
  }
  mean /= static_cast<double>(M_models);

  // 분산 계산: Var = E[(x - mean)²]
  Eigen::MatrixXd variance = Eigen::MatrixXd::Zero(M_batch, nx);
  for (int i = 0; i < M_models; ++i) {
    Eigen::MatrixXd diff = predictions[i] - mean;
    variance += diff.cwiseProduct(diff);
  }
  variance /= static_cast<double>(M_models);

  return {mean, variance};
}

Eigen::MatrixXd EnsembleDynamicsModel::clipControls(
  const Eigen::MatrixXd& controls) const
{
  return nominal_->clipControls(controls);
}

void EnsembleDynamicsModel::normalizeStates(Eigen::MatrixXd& states) const
{
  nominal_->normalizeStates(states);
}

geometry_msgs::msg::Twist EnsembleDynamicsModel::controlToTwist(
  const Eigen::VectorXd& control) const
{
  return nominal_->controlToTwist(control);
}

Eigen::VectorXd EnsembleDynamicsModel::twistToControl(
  const geometry_msgs::msg::Twist& twist) const
{
  return nominal_->twistToControl(twist);
}

std::vector<int> EnsembleDynamicsModel::angleIndices() const
{
  return nominal_->angleIndices();
}

}  // namespace mpc_controller_ros2
