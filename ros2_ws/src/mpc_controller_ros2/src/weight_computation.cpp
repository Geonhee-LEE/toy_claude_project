#include "mpc_controller_ros2/weight_computation.hpp"
#include "mpc_controller_ros2/utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

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

// ============================================================================
// Tsallis-MPPI
// ============================================================================

TsallisMPPIWeights::TsallisMPPIWeights(double q)
  : q_(q)
{
}

Eigen::VectorXd TsallisMPPIWeights::compute(
  const Eigen::VectorXd& costs, double lambda) const
{
  // Zero-lambda fallback: greedy
  if (lambda < 1e-9) {
    Eigen::VectorXd weights = Eigen::VectorXd::Zero(costs.size());
    int min_idx;
    costs.minCoeff(&min_idx);
    weights(min_idx) = 1.0;
    return weights;
  }

  // q → 1 극한: Vanilla MPPI에 위임
  if (std::abs(q_ - 1.0) < 1e-8) {
    return softmaxWeights(costs, lambda);
  }

  // 1. Min-centering (q-exp는 translation-invariant 아님)
  Eigen::VectorXd centered = costs.array() - costs.minCoeff();

  // 2. q-exponential 가중치
  Eigen::VectorXd raw = qExponential(-centered / lambda, q_);

  // 3. 정규화
  double sum = raw.sum();
  if (sum < 1e-12) {
    return Eigen::VectorXd::Constant(costs.size(), 1.0 / costs.size());
  }
  return raw / sum;
}

// ============================================================================
// Risk-Aware MPPI (CVaR)
// ============================================================================

RiskAwareMPPIWeights::RiskAwareMPPIWeights(double alpha)
  : alpha_(alpha)
{
}

Eigen::VectorXd RiskAwareMPPIWeights::compute(
  const Eigen::VectorXd& costs, double lambda) const
{
  const int K = static_cast<int>(costs.size());

  // alpha >= 1.0 → Vanilla에 위임 (모든 샘플 사용)
  if (alpha_ >= 1.0 - 1e-9) {
    return softmaxWeights(costs, lambda);
  }

  // n_keep = ceil(alpha * K), 최소 1개
  int n_keep = std::max(1, static_cast<int>(std::ceil(alpha_ * K)));

  // 인덱스 배열 생성 및 비용 기준 정렬
  std::vector<int> indices(K);
  std::iota(indices.begin(), indices.end(), 0);
  std::nth_element(indices.begin(), indices.begin() + n_keep, indices.end(),
    [&costs](int a, int b) { return costs(a) < costs(b); });

  // 선택된 n_keep개의 비용 추출
  Eigen::VectorXd selected_costs(n_keep);
  for (int i = 0; i < n_keep; ++i) {
    selected_costs(i) = costs(indices[i]);
  }

  // 선택된 비용에 대해 softmax 가중치 계산
  Eigen::VectorXd selected_weights = softmaxWeights(selected_costs, lambda);

  // 전체 배열에 재배치 (선택되지 않은 샘플은 가중치 0)
  Eigen::VectorXd weights = Eigen::VectorXd::Zero(K);
  for (int i = 0; i < n_keep; ++i) {
    weights(indices[i]) = selected_weights(i);
  }

  return weights;
}

}  // namespace mpc_controller_ros2
