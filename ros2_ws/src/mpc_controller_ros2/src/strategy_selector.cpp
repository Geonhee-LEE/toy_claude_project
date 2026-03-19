// =============================================================================
// StrategySelector: EMA 스무딩 + 히스테리시스 기반 MPPI 전략 자동 선택
//
// 우선순위: SAFE > RECOVERY > AGGRESSIVE > PRECISE > CRUISE
// 히스테리시스: 후보가 N 사이클 연속 유지해야 전환 (chattering 방지)
// =============================================================================

#include "mpc_controller_ros2/strategy_selector.hpp"
#include <algorithm>
#include <cmath>

namespace mpc_controller_ros2
{

StrategySelector::StrategySelector(
  double safety_threshold,
  double recovery_threshold,
  double fast_threshold,
  double precision_dist,
  int hysteresis_count,
  double smoothing_alpha)
: safety_threshold_(safety_threshold),
  recovery_threshold_(recovery_threshold),
  fast_threshold_(fast_threshold),
  precision_dist_(precision_dist),
  hysteresis_count_(std::max(1, hysteresis_count)),
  alpha_(std::clamp(smoothing_alpha, 0.0, 1.0))
{
}

MPPIStrategy StrategySelector::update(
  double speed, double v_max,
  double min_obs_dist,
  double tracking_error,
  double goal_dist)
{
  // ──── Step 1: EMA 스무딩 ────
  if (!initialized_) {
    smoothed_.speed = speed;
    smoothed_.min_obs_dist = min_obs_dist;
    smoothed_.tracking_error = tracking_error;
    smoothed_.goal_dist = goal_dist;
    initialized_ = true;
  } else {
    smoothed_.speed = alpha_ * speed + (1.0 - alpha_) * smoothed_.speed;
    smoothed_.min_obs_dist = alpha_ * min_obs_dist + (1.0 - alpha_) * smoothed_.min_obs_dist;
    smoothed_.tracking_error = alpha_ * tracking_error + (1.0 - alpha_) * smoothed_.tracking_error;
    smoothed_.goal_dist = alpha_ * goal_dist + (1.0 - alpha_) * smoothed_.goal_dist;
  }

  // ──── Step 2: 후보 전략 결정 (우선순위 순) ────
  MPPIStrategy candidate = selectCandidate(v_max);

  // ──── Step 3: 히스테리시스 ────
  if (candidate == candidate_strategy_) {
    candidate_count_++;
  } else {
    candidate_strategy_ = candidate;
    candidate_count_ = 1;
  }

  if (candidate_count_ >= hysteresis_count_) {
    active_strategy_ = candidate_strategy_;
  }

  return active_strategy_;
}

MPPIStrategy StrategySelector::selectCandidate(double v_max) const
{
  // 우선순위: SAFE > RECOVERY > AGGRESSIVE > PRECISE > CRUISE
  if (smoothed_.min_obs_dist < safety_threshold_) {
    return MPPIStrategy::SAFE;
  }
  if (smoothed_.tracking_error > recovery_threshold_) {
    return MPPIStrategy::RECOVERY;
  }
  double speed_ratio = (v_max > 0.0) ? (smoothed_.speed / v_max) : 0.0;
  if (speed_ratio > fast_threshold_) {
    return MPPIStrategy::AGGRESSIVE;
  }
  if (smoothed_.goal_dist < precision_dist_) {
    return MPPIStrategy::PRECISE;
  }
  return MPPIStrategy::CRUISE;
}

void StrategySelector::reset()
{
  initialized_ = false;
  active_strategy_ = MPPIStrategy::CRUISE;
  candidate_strategy_ = MPPIStrategy::CRUISE;
  candidate_count_ = 0;
  smoothed_ = ContextMetrics{};
}

}  // namespace mpc_controller_ros2
