#ifndef MPC_CONTROLLER_ROS2__CONFORMAL_PREDICTOR_HPP_
#define MPC_CONTROLLER_ROS2__CONFORMAL_PREDICTOR_HPP_

#include <deque>

namespace mpc_controller_ros2
{

/**
 * @brief Adaptive Conformal Prediction (ACP) 기반 동적 안전 마진
 *
 * 실시간 예측 오차를 관측하여 통계적 보장 하에 안전 마진을 적응적으로 조정합니다.
 *
 * 알고리즘 (Gibbs & Candes, 2021):
 *   1. 예측 오차 e_t = ||x_actual - x_predicted|| 관측
 *   2. 슬라이딩 윈도우에 오차 저장
 *   3. 목표 커버리지 확률에 해당하는 분위수 계산 → 마진
 *   4. 가중 감쇄(decay)로 최근 오차에 높은 가중치
 *
 * BarrierFunctionSet과 통합:
 *   margin = base_margin + conformal_predictor.getMargin()
 *   barrier_set.updateSafetyMargin(margin)
 */
class ConformalPredictor
{
public:
  struct Params {
    double coverage_probability{0.95};  // 목표 커버리지 (1-α)
    int window_size{100};               // 슬라이딩 윈도우 크기
    double initial_margin{0.3};         // 초기 마진 (m)
    double min_margin{0.05};            // 최소 마진 (m)
    double max_margin{1.0};             // 최대 마진 (m)
    double decay_rate{0.99};            // ACP 가중치 감쇄율
  };

  ConformalPredictor();
  explicit ConformalPredictor(const Params& params);

  /** @brief 새로운 예측 오차 관측 */
  void update(double prediction_error);

  /** @brief 보정된 안전 마진 반환 (m) */
  double getMargin() const;

  /** @brief 현재 실측 커버리지 비율 (진단용) */
  double getCoverage() const;

  /** @brief 관측 횟수 */
  int numObservations() const { return total_observations_; }

  /** @brief 리셋 */
  void reset();

private:
  Params params_;
  std::deque<double> error_window_;  // 슬라이딩 윈도우
  double current_margin_;            // 현재 계산된 마진
  int total_observations_{0};
  int covered_count_{0};             // 마진 내 관측 수

  void recomputeMargin();
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__CONFORMAL_PREDICTOR_HPP_
