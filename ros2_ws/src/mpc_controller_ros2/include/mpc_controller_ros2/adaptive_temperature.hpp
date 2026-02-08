#ifndef MPC_CONTROLLER_ROS2__ADAPTIVE_TEMPERATURE_HPP_
#define MPC_CONTROLLER_ROS2__ADAPTIVE_TEMPERATURE_HPP_

#include <cmath>
#include <algorithm>

namespace mpc_controller_ros2
{

/**
 * @brief ESS 기반 Adaptive Temperature 조정기
 *
 * Effective Sample Size (ESS)를 기반으로 MPPI의 temperature 파라미터 λ를 자동 조정.
 *
 * 알고리즘:
 *   ess_ratio = ESS / K
 *   log(λ) += adaptation_rate * (target_ratio - ess_ratio)
 *
 * 동작 원리:
 *   - ESS 낮음 (weight collapse) → λ 증가 → 탐색 강화
 *   - ESS 높음 (uniform weights)  → λ 감소 → 활용 강화
 *
 * 참조:
 *   - Williams et al., "Information Theoretic MPC for Model-Based RL"
 *   - Python M2 구현: mpc_controller/controllers/mppi/adaptive_temperature.py
 */
class AdaptiveTemperature
{
public:
  /**
   * @brief 생성자
   * @param initial_lambda 초기 λ 값
   * @param target_ess_ratio 목표 ESS 비율 (0~1, 일반적으로 0.3~0.7)
   * @param adaptation_rate 적응 속도 (0.01~0.5)
   * @param lambda_min 최소 λ
   * @param lambda_max 최대 λ
   */
  explicit AdaptiveTemperature(
    double initial_lambda = 10.0,
    double target_ess_ratio = 0.5,
    double adaptation_rate = 0.1,
    double lambda_min = 0.1,
    double lambda_max = 100.0
  );

  /**
   * @brief λ 업데이트
   * @param ess 현재 Effective Sample Size
   * @param K 총 샘플 수
   * @return 업데이트된 λ
   */
  double update(double ess, int K);

  /**
   * @brief 현재 λ 반환
   */
  double getLambda() const { return lambda_; }

  /**
   * @brief λ 직접 설정
   */
  void setLambda(double lambda);

  /**
   * @brief 파라미터 재설정
   */
  void setParameters(
    double target_ess_ratio,
    double adaptation_rate,
    double lambda_min,
    double lambda_max
  );

  /**
   * @brief 초기 상태로 리셋
   */
  void reset(double initial_lambda);

  /**
   * @brief 디버그 정보 출력
   */
  struct AdaptiveInfo {
    double lambda;
    double log_lambda;
    double ess_ratio;
    double target_ratio;
    double delta;
  };
  AdaptiveInfo getInfo() const { return last_info_; }

private:
  double lambda_;           // 현재 λ
  double log_lambda_;       // log(λ) - 안정적인 업데이트를 위해
  double target_ess_ratio_; // 목표 ESS 비율
  double adaptation_rate_;  // 적응 속도
  double lambda_min_;       // 최소 λ
  double lambda_max_;       // 최대 λ

  AdaptiveInfo last_info_;  // 마지막 업데이트 정보
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__ADAPTIVE_TEMPERATURE_HPP_
