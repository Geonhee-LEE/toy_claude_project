#ifndef MPC_CONTROLLER_ROS2__HALTON_SAMPLER_HPP_
#define MPC_CONTROLLER_ROS2__HALTON_SAMPLER_HPP_

#include "mpc_controller_ros2/sampling.hpp"

namespace mpc_controller_ros2
{

/**
 * @brief Halton 저불일치 시퀀스 기반 노이즈 샘플러
 *
 * MDPI Drones 2026 기반: Gaussian 노이즈 대신 Halton 시퀀스로
 * 제어 공간을 균일하게 커버하여 적은 샘플(K)로도 빠른 수렴 달성.
 *
 * 알고리즘:
 *   1. Van der Corput 시퀀스 -> [0,1] 균일 저불일치 샘플
 *   2. 역정규 CDF (Rational Approximation) -> N(0,1) 변환
 *   3. sigma 스케일링 -> 목표 분포
 *   4. OU 프로세스 시간 상관 (선택적)
 *
 * 핵심 수식:
 *   H_b(n) = sum d_i * b^{-(i+1)}, where n = sum d_i * b^i  (Van der Corput)
 *   Phi^{-1}(p) ~ rational approximation (Abramowitz & Stegun)
 */
class HaltonSampler : public BaseSampler
{
public:
  /**
   * @param sigma (nu,) 노이즈 표준편차
   * @param beta OU 시간 상관 계수 (0=완전 상관, inf=독립, 기본=2.0)
   * @param sequence_offset Halton 시퀀스 시작 오프셋 (burn-in, 기본=100)
   */
  explicit HaltonSampler(
    const Eigen::VectorXd& sigma,
    double beta = 2.0,
    int sequence_offset = 100);

  std::vector<Eigen::MatrixXd> sample(int K, int N, int nu) override;
  void sampleInPlace(std::vector<Eigen::MatrixXd>& out, int K, int N, int nu) override;

  /** @brief Van der Corput 시퀀스 값 (base b, index n) */
  static double haltonValue(int index, int base);

  /** @brief 역정규 CDF (Rational Approximation, Abramowitz & Stegun) */
  static double inverseNormalCDF(double p);

  /** @brief 시퀀스 카운터 리셋 */
  void reset();

private:
  /** @brief 차원 dim에 대한 소수 base 반환 */
  static int primeForDimension(int dim);

  Eigen::VectorXd sigma_;
  double beta_;
  int sequence_offset_;
  int sample_counter_{0};

  // 소수 테이블 (최대 20차원)
  static constexpr int PRIMES[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                                    31, 37, 41, 43, 47, 53, 59, 61, 67, 71};
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__HALTON_SAMPLER_HPP_
