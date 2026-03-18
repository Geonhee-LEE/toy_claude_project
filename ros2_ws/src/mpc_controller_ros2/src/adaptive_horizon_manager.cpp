// =============================================================================
// AdaptiveHorizonManager — RH-MPPI 동적 예측 horizon 관리
//
// 환경 컨텍스트(속도, 장애물, 추적 오차)에 기반한 가중 평균 + EMA 스무딩으로
// effective horizon N을 [N_min, N_max] 범위에서 적응적으로 결정.
//
// 수식:
//   combined = (w_s·speed_f + w_o·obs_f + w_e·err_f) / (w_s + w_o + w_e)
//   raw_N = N_min + (N_max - N_min) * combined
//   smoothed_N = α·raw_N + (1-α)·prev_smoothed_N
// =============================================================================

#include "mpc_controller_ros2/adaptive_horizon_manager.hpp"

namespace mpc_controller_ros2
{

AdaptiveHorizonManager::AdaptiveHorizonManager(
  int N_min, int N_max,
  double speed_weight, double obstacle_weight, double error_weight,
  double obs_dist_threshold, double error_threshold,
  double smoothing_alpha)
: N_min_(N_min),
  N_max_(N_max),
  speed_weight_(speed_weight),
  obstacle_weight_(obstacle_weight),
  error_weight_(error_weight),
  obs_dist_threshold_(obs_dist_threshold),
  error_threshold_(error_threshold),
  smoothing_alpha_(smoothing_alpha),
  prev_smoothed_N_(static_cast<double>(N_min)),
  initialized_(false)
{
}

int AdaptiveHorizonManager::computeEffectiveN(
  double speed, double v_max,
  double min_obs_dist, double tracking_error)
{
  // 속도 팩터: 고속 → 1 (긴 horizon), 저속 → 0 (짧은 horizon)
  double speed_factor = (v_max > 0.0) ?
    std::clamp(speed / v_max, 0.0, 1.0) : 0.0;

  // 장애물 팩터: 먼 장애물 → 1 (긴 horizon), 가까운 장애물 → 0 (짧은 horizon)
  double obstacle_factor = (obs_dist_threshold_ > 0.0) ?
    std::clamp(min_obs_dist / obs_dist_threshold_, 0.0, 1.0) : 1.0;

  // 오차 팩터: 작은 오차 → 1 (긴 horizon), 큰 오차 → 0 (짧은 horizon)
  double error_factor = (error_threshold_ > 0.0) ?
    std::clamp(1.0 - tracking_error / error_threshold_, 0.0, 1.0) : 1.0;

  // 가중 평균
  double total_weight = speed_weight_ + obstacle_weight_ + error_weight_;
  double combined = (total_weight > 0.0) ?
    (speed_weight_ * speed_factor +
     obstacle_weight_ * obstacle_factor +
     error_weight_ * error_factor) / total_weight
    : 0.5;

  // Raw N 계산
  double raw_N = static_cast<double>(N_min_) +
    (static_cast<double>(N_max_) - static_cast<double>(N_min_)) * combined;

  // EMA 스무딩
  double smoothed_N;
  if (!initialized_) {
    smoothed_N = raw_N;
    initialized_ = true;
  } else {
    smoothed_N = smoothing_alpha_ * raw_N +
                 (1.0 - smoothing_alpha_) * prev_smoothed_N_;
  }
  prev_smoothed_N_ = smoothed_N;

  // 반올림 + 클램핑
  int effective_N = static_cast<int>(std::round(smoothed_N));
  effective_N = std::clamp(effective_N, N_min_, N_max_);

  return effective_N;
}

void AdaptiveHorizonManager::reset()
{
  prev_smoothed_N_ = static_cast<double>(N_min_);
  initialized_ = false;
}

}  // namespace mpc_controller_ros2
