#include "mpc_controller_ros2/sampling.hpp"
#include <cmath>

namespace mpc_controller_ros2
{

// ============================================================================
// GaussianSampler
// ============================================================================

GaussianSampler::GaussianSampler(const Eigen::VectorXd& sigma, unsigned int seed)
: sigma_(sigma), rng_(seed), dist_(0.0, 1.0)
{
}

std::vector<Eigen::MatrixXd> GaussianSampler::sample(int K, int N, int nu)
{
  std::vector<Eigen::MatrixXd> samples;
  samples.reserve(K);

  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd noise(N, nu);
    for (int t = 0; t < N; ++t) {
      for (int i = 0; i < nu; ++i) {
        noise(t, i) = dist_(rng_) * sigma_(i);
      }
    }
    samples.push_back(noise);
  }

  return samples;
}

void GaussianSampler::resetSeed(unsigned int seed)
{
  rng_.seed(seed);
}

// ============================================================================
// ColoredNoiseSampler
// ============================================================================

ColoredNoiseSampler::ColoredNoiseSampler(
  const Eigen::VectorXd& sigma,
  double beta,
  unsigned int seed
)
: sigma_(sigma), beta_(beta), rng_(seed), dist_(0.0, 1.0)
{
}

std::vector<Eigen::MatrixXd> ColoredNoiseSampler::sample(int K, int N, int nu)
{
  // OU process discretization parameters
  // dt = 1.0 (unit time - actual dt is handled in dynamics)
  double dt = 1.0;
  double decay = std::exp(-beta_ * dt);

  // Maintain stationary variance sigma^2
  Eigen::VectorXd diffusion = sigma_ * std::sqrt(1.0 - decay * decay);

  std::vector<Eigen::MatrixXd> samples;
  samples.reserve(K);

  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd noise(N, nu);

    // Initial sample from stationary distribution
    for (int i = 0; i < nu; ++i) {
      noise(0, i) = dist_(rng_) * sigma_(i);
    }

    // OU process evolution
    for (int t = 1; t < N; ++t) {
      for (int i = 0; i < nu; ++i) {
        double w = dist_(rng_);
        noise(t, i) = decay * noise(t - 1, i) + diffusion(i) * w;
      }
    }

    samples.push_back(noise);
  }

  return samples;
}

void ColoredNoiseSampler::resetSeed(unsigned int seed)
{
  rng_.seed(seed);
}

}  // namespace mpc_controller_ros2
