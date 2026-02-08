#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/utils.hpp"

namespace mpc_controller_ros2
{

Eigen::VectorXd VanillaMPPIWeights::compute(
  const Eigen::VectorXd& costs, double lambda) const
{
  return softmaxWeights(costs, lambda);
}

Eigen::VectorXd LogMPPIWeights::compute(
  const Eigen::VectorXd& costs, double lambda) const
{
  // Zero-lambda fallback: greedy (최소 비용 샘플만 선택)
  if (lambda < 1e-9) {
    Eigen::VectorXd weights = Eigen::VectorXd::Zero(costs.size());
    int min_idx;
    costs.minCoeff(&min_idx);
    weights(min_idx) = 1.0;
    return weights;
  }

  // Log-space 가중치 계산
  Eigen::VectorXd log_weights = -costs / lambda;

  // Log-space 정규화: log_w -= logSumExp(log_w)
  log_weights.array() -= logSumExp(log_weights);

  // Exp하여 실제 가중치로 변환
  Eigen::VectorXd weights = log_weights.array().exp();

  // 수치 오차 대비 재정규화
  double sum = weights.sum();
  if (sum < 1e-12) {
    return Eigen::VectorXd::Constant(costs.size(), 1.0 / costs.size());
  }
  return weights / sum;
}

}  // namespace mpc_controller_ros2
