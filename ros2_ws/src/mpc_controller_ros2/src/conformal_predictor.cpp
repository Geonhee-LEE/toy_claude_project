#include "mpc_controller_ros2/conformal_predictor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace mpc_controller_ros2
{

ConformalPredictor::ConformalPredictor()
: params_(), current_margin_(params_.initial_margin)
{
}

ConformalPredictor::ConformalPredictor(const Params& params)
: params_(params), current_margin_(params.initial_margin)
{
}

void ConformalPredictor::update(double prediction_error)
{
  total_observations_++;

  // 현재 마진 내에 있는지 확인 (커버리지 추적)
  if (prediction_error <= current_margin_) {
    covered_count_++;
  }

  // 슬라이딩 윈도우에 추가
  error_window_.push_back(prediction_error);
  if (static_cast<int>(error_window_.size()) > params_.window_size) {
    error_window_.pop_front();
  }

  // 마진 재계산
  recomputeMargin();
}

double ConformalPredictor::getMargin() const
{
  return current_margin_;
}

double ConformalPredictor::getCoverage() const
{
  if (total_observations_ == 0) {
    return 1.0;  // 관측 전 → 완전 커버리지 가정
  }
  return static_cast<double>(covered_count_) / total_observations_;
}

void ConformalPredictor::reset()
{
  error_window_.clear();
  current_margin_ = params_.initial_margin;
  total_observations_ = 0;
  covered_count_ = 0;
}

void ConformalPredictor::recomputeMargin()
{
  if (error_window_.empty()) {
    current_margin_ = params_.initial_margin;
    return;
  }

  // 가중 분위수 계산 (최근 오차에 높은 가중치)
  int n = static_cast<int>(error_window_.size());
  std::vector<std::pair<double, double>> weighted_errors;  // (error, weight)
  weighted_errors.reserve(n);

  double weight = 1.0;
  // 최신 항목이 뒤에 있으므로 역순으로 가중치 할당
  for (int i = n - 1; i >= 0; --i) {
    weighted_errors.push_back({error_window_[i], weight});
    weight *= params_.decay_rate;
  }

  // 오차 기준 정렬
  std::sort(weighted_errors.begin(), weighted_errors.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  // 가중 분위수: coverage_probability에 해당하는 위치
  double total_weight = 0.0;
  for (const auto& [err, w] : weighted_errors) {
    total_weight += w;
  }

  double target_weight = params_.coverage_probability * total_weight;
  double cumulative = 0.0;
  double quantile_margin = weighted_errors.back().first;  // 최대값 폴백

  for (const auto& [err, w] : weighted_errors) {
    cumulative += w;
    if (cumulative >= target_weight) {
      quantile_margin = err;
      break;
    }
  }

  // 클리핑
  current_margin_ = std::clamp(quantile_margin, params_.min_margin, params_.max_margin);
}

}  // namespace mpc_controller_ros2
