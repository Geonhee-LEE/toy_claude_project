#include "mpc_controller_ros2/adaptive_temperature.hpp"

namespace mpc_controller_ros2
{

AdaptiveTemperature::AdaptiveTemperature(
  double initial_lambda,
  double target_ess_ratio,
  double adaptation_rate,
  double lambda_min,
  double lambda_max
)
: lambda_(initial_lambda),
  log_lambda_(std::log(initial_lambda)),
  target_ess_ratio_(target_ess_ratio),
  adaptation_rate_(adaptation_rate),
  lambda_min_(lambda_min),
  lambda_max_(lambda_max)
{
  last_info_ = {lambda_, log_lambda_, 0.0, target_ess_ratio_, 0.0};
}

double AdaptiveTemperature::update(double ess, int K)
{
  // ESS 비율 계산
  double ess_ratio = ess / static_cast<double>(K);

  // 적응 규칙:
  // - ESS_ratio < target → λ 증가 (더 많은 탐색)
  // - ESS_ratio > target → λ 감소 (더 많은 활용)
  double delta = adaptation_rate_ * (target_ess_ratio_ - ess_ratio);
  log_lambda_ += delta;

  // λ 범위 제한 (log 스케일에서)
  double log_min = std::log(lambda_min_);
  double log_max = std::log(lambda_max_);
  log_lambda_ = std::clamp(log_lambda_, log_min, log_max);

  // λ 계산
  lambda_ = std::exp(log_lambda_);

  // 디버그 정보 저장
  last_info_ = {lambda_, log_lambda_, ess_ratio, target_ess_ratio_, delta};

  return lambda_;
}

void AdaptiveTemperature::setLambda(double lambda)
{
  lambda = std::clamp(lambda, lambda_min_, lambda_max_);
  lambda_ = lambda;
  log_lambda_ = std::log(lambda);
}

void AdaptiveTemperature::setParameters(
  double target_ess_ratio,
  double adaptation_rate,
  double lambda_min,
  double lambda_max
)
{
  target_ess_ratio_ = std::clamp(target_ess_ratio, 0.01, 0.99);
  adaptation_rate_ = std::clamp(adaptation_rate, 0.001, 1.0);
  lambda_min_ = std::max(0.001, lambda_min);
  lambda_max_ = std::max(lambda_min_ * 2.0, lambda_max);
}

void AdaptiveTemperature::reset(double initial_lambda)
{
  lambda_ = std::clamp(initial_lambda, lambda_min_, lambda_max_);
  log_lambda_ = std::log(lambda_);
  last_info_ = {lambda_, log_lambda_, 0.0, target_ess_ratio_, 0.0};
}

}  // namespace mpc_controller_ros2
