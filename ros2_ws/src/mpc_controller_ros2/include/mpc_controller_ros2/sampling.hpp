#ifndef MPC_CONTROLLER_ROS2__SAMPLING_HPP_
#define MPC_CONTROLLER_ROS2__SAMPLING_HPP_

#include <Eigen/Dense>
#include <random>
#include <vector>

namespace mpc_controller_ros2
{

/**
 * @brief 노이즈 샘플러 베이스 클래스
 */
class BaseSampler
{
public:
  virtual ~BaseSampler() = default;

  /**
   * @brief 노이즈 샘플 생성
   * @param K 샘플 수
   * @param N 예측 horizon
   * @param nu 제어 입력 차원
   * @return 노이즈 샘플 벡터 [K개, 각각 N x nu 행렬]
   */
  virtual std::vector<Eigen::MatrixXd> sample(int K, int N, int nu) = 0;
};

/**
 * @brief Gaussian 노이즈 샘플러
 */
class GaussianSampler : public BaseSampler
{
public:
  explicit GaussianSampler(const Eigen::VectorXd& sigma, unsigned int seed = 42);

  std::vector<Eigen::MatrixXd> sample(int K, int N, int nu) override;

  void resetSeed(unsigned int seed);

private:
  Eigen::VectorXd sigma_;
  std::mt19937 rng_;
  std::normal_distribution<double> dist_;
};

/**
 * @brief Colored Noise 샘플러 (Ornstein-Uhlenbeck 프로세스)
 *
 * 시간 연관성 있는 노이즈 생성:
 *   ε[t+1] = decay · ε[t] + diffusion · w[t]
 *   where decay = exp(-beta * dt)
 *         diffusion = sigma * sqrt(1 - decay^2)
 *
 * beta가 클수록 → 백색 노이즈에 가까움
 * beta가 작을수록 → 시간 연관 강함 (부드러운 샘플)
 */
class ColoredNoiseSampler : public BaseSampler
{
public:
  explicit ColoredNoiseSampler(
    const Eigen::VectorXd& sigma,
    double beta = 2.0,
    unsigned int seed = 42
  );

  std::vector<Eigen::MatrixXd> sample(int K, int N, int nu) override;

  void resetSeed(unsigned int seed);

private:
  Eigen::VectorXd sigma_;
  double beta_;
  std::mt19937 rng_;
  std::normal_distribution<double> dist_;
};

}  // namespace mpc_controller_ros2

#endif  // MPC_CONTROLLER_ROS2__SAMPLING_HPP_
