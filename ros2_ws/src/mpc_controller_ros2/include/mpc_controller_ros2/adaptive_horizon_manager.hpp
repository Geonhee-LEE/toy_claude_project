#ifndef MPC_CONTROLLER_ROS2__ADAPTIVE_HORIZON_MANAGER_HPP_
#define MPC_CONTROLLER_ROS2__ADAPTIVE_HORIZON_MANAGER_HPP_

#include <algorithm>
#include <cmath>

namespace mpc_controller_ros2
{

/**
 * @brief 동적 예측 horizon N 관리자 (RH-MPPI)
 *
 * 환경 컨텍스트(속도, 장애물 거리, 추적 오차)에 따라
 * effective horizon N을 [N_min, N_max] 범위에서 적응적으로 조정.
 *
 * 알고리즘:
 *   speed_factor    = clamp(speed / v_max, 0, 1)
 *   obstacle_factor = clamp(min_obs_dist / obs_dist_threshold, 0, 1)
 *   error_factor    = clamp(1.0 - tracking_error / error_threshold, 0, 1)
 *   combined = weighted_avg(speed_f, obstacle_f, error_f)
 *   raw_N = N_min + (N_max - N_min) * combined
 *   smoothed_N = alpha * raw_N + (1-alpha) * prev_smoothed_N   (EMA)
 */
class AdaptiveHorizonManager
{
public:
  AdaptiveHorizonManager(
    int N_min, int N_max,
    double speed_weight, double obstacle_weight, double error_weight,
    double obs_dist_threshold, double error_threshold,
    double smoothing_alpha);

  /**
   * @brief 환경 컨텍스트로부터 effective N 계산
   * @param speed 현재 속도 (m/s, >=0)
   * @param v_max 최대 속도 (m/s)
   * @param min_obs_dist 가장 가까운 장애물까지 거리 (m)
   * @param tracking_error 현재 추적 오차 (m)
   * @return effective horizon N ∈ [N_min, N_max]
   */
  int computeEffectiveN(
    double speed, double v_max,
    double min_obs_dist, double tracking_error);

  /** @brief 현재 스무딩된 N 값 (디버그용) */
  double getSmoothedN() const { return prev_smoothed_N_; }

  /** @brief EMA 상태 리셋 */
  void reset();

private:
  int N_min_;
  int N_max_;
  double speed_weight_;
  double obstacle_weight_;
  double error_weight_;
  double obs_dist_threshold_;
  double error_threshold_;
  double smoothing_alpha_;

  double prev_smoothed_N_;
  bool initialized_{false};
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__ADAPTIVE_HORIZON_MANAGER_HPP_
