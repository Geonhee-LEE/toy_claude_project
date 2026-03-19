#ifndef MPC_CONTROLLER_ROS2__STRATEGY_SELECTOR_HPP_
#define MPC_CONTROLLER_ROS2__STRATEGY_SELECTOR_HPP_

#include <string>

namespace mpc_controller_ros2
{

/**
 * @brief MPPI 전략 열거형
 *
 * 우선순위: SAFE > RECOVERY > AGGRESSIVE > PRECISE > CRUISE
 */
enum class MPPIStrategy
{
  CRUISE = 0,       // 기본 — 특수 기능 없이 baseline
  PRECISE,          // 목표 근접 — 피드백 보정 + Q 증폭
  AGGRESSIVE,       // 고속 주행 — LP 필터 + 비용 튜닝
  RECOVERY,         // 큰 추적 오차 — 탐색 강화 + 짧은 horizon
  SAFE              // 장애물 근접 — CBF 활성화 (최우선)
};

inline std::string strategyToString(MPPIStrategy s)
{
  switch (s) {
    case MPPIStrategy::CRUISE:     return "CRUISE";
    case MPPIStrategy::PRECISE:    return "PRECISE";
    case MPPIStrategy::AGGRESSIVE: return "AGGRESSIVE";
    case MPPIStrategy::RECOVERY:   return "RECOVERY";
    case MPPIStrategy::SAFE:       return "SAFE";
    default:                       return "UNKNOWN";
  }
}

/**
 * @brief 런타임 컨텍스트 메트릭 (EMA 스무딩 적용 후)
 */
struct ContextMetrics
{
  double speed{0.0};           // 현재 속도 (m/s)
  double min_obs_dist{10.0};   // 최근접 장애물 거리 (m)
  double tracking_error{0.0};  // 추적 오차 (m)
  double goal_dist{10.0};      // 목표까지 거리 (m)
};

/**
 * @brief EMA 스무딩 + 히스테리시스 기반 전략 선택기
 *
 * 알고리즘:
 *   1. raw 메트릭 → EMA 스무딩 (jitter 방지)
 *   2. 스무딩된 메트릭으로 후보 전략 결정 (우선순위 순)
 *   3. 히스테리시스: 후보가 hysteresis_count 연속 유지해야 전환
 */
class StrategySelector
{
public:
  StrategySelector(
    double safety_threshold,       // SAFE 전환 장애물 거리 (m)
    double recovery_threshold,     // RECOVERY 전환 추적 오차 (m)
    double fast_threshold,         // AGGRESSIVE 속도 비율 (v/v_max)
    double precision_dist,         // PRECISE 목표 근접 거리 (m)
    int hysteresis_count,          // 전환 히스테리시스 (cycles)
    double smoothing_alpha         // EMA 계수 (0~1)
  );

  /**
   * @brief 컨텍스트로부터 전략 업데이트
   * @param speed 현재 속도 (m/s)
   * @param v_max 최대 속도 (m/s)
   * @param min_obs_dist 최근접 장애물 거리 (m)
   * @param tracking_error 추적 오차 (m)
   * @param goal_dist 목표까지 거리 (m)
   * @return 현재 활성 전략
   */
  MPPIStrategy update(
    double speed, double v_max,
    double min_obs_dist,
    double tracking_error,
    double goal_dist
  );

  MPPIStrategy currentStrategy() const { return active_strategy_; }
  const ContextMetrics& smoothedMetrics() const { return smoothed_; }
  void reset();

private:
  MPPIStrategy selectCandidate(double v_max) const;

  double safety_threshold_;
  double recovery_threshold_;
  double fast_threshold_;
  double precision_dist_;
  int hysteresis_count_;
  double alpha_;

  ContextMetrics smoothed_;
  bool initialized_{false};

  MPPIStrategy active_strategy_{MPPIStrategy::CRUISE};
  MPPIStrategy candidate_strategy_{MPPIStrategy::CRUISE};
  int candidate_count_{0};
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__STRATEGY_SELECTOR_HPP_
