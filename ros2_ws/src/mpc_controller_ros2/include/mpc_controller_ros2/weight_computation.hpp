#ifndef MPC_CONTROLLER_ROS2__WEIGHT_COMPUTATION_HPP_
#define MPC_CONTROLLER_ROS2__WEIGHT_COMPUTATION_HPP_

#include <Eigen/Dense>
#include <string>

namespace mpc_controller_ros2
{

/**
 * @brief MPPI 가중치 계산 Strategy 인터페이스
 *
 * Vanilla, Log-MPPI 등 다양한 가중치 계산 방식을 플러그 가능하게 추상화.
 * 향후 Tsallis, Risk-Aware 등도 동일 패턴으로 추가 가능.
 */
class WeightComputation
{
public:
  virtual ~WeightComputation() = default;

  /**
   * @brief 비용 벡터로부터 정규화된 가중치 계산
   * @param costs 비용 벡터 (K개 샘플)
   * @param lambda Temperature 파라미터
   * @return 정규화된 가중치 벡터 (합 = 1)
   */
  virtual Eigen::VectorXd compute(const Eigen::VectorXd& costs, double lambda) const = 0;

  /**
   * @brief Strategy 이름 반환 (로깅용)
   */
  virtual std::string name() const = 0;
};

/**
 * @brief Vanilla MPPI 가중치 (기존 softmaxWeights 래핑)
 *
 * weights[k] = exp(-costs[k]/lambda) / Sigma exp(-costs[i]/lambda)
 * max-shift trick 적용으로 수치 안정성 보장.
 */
class VanillaMPPIWeights : public WeightComputation
{
public:
  Eigen::VectorXd compute(const Eigen::VectorXd& costs, double lambda) const override;
  std::string name() const override { return "VanillaMPPI"; }
};

/**
 * @brief Log-MPPI 가중치 (log-space 정규화)
 *
 * log_w_k = -S_k / lambda
 * log_w_k -= logSumExp(log_w)  // log-space 정규화
 * w_k = exp(log_w_k)
 *
 * Vanilla와 수학적으로 동일하나, log-space 가중치 조작이 필요한
 * 후속 확장(importance sampling 보정 등)의 기반 클래스로 활용.
 */
class LogMPPIWeights : public WeightComputation
{
public:
  Eigen::VectorXd compute(const Eigen::VectorXd& costs, double lambda) const override;
  std::string name() const override { return "LogMPPI"; }
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__WEIGHT_COMPUTATION_HPP_
