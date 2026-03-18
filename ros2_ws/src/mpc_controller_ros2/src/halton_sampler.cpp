// =============================================================================
// Halton Sampler — 저불일치 시퀀스 기반 MPPI 노이즈 샘플러
//
// MDPI Drones 2026 기반: Van der Corput 시퀀스 + 역정규 CDF로
// 제어 공간을 균일하게 커버. Gaussian 대비 적은 K로도 빠른 수렴.
//
// 핵심 수식:
//   H_b(n) = sum d_i * b^{-(i+1)}       ... Van der Corput (base b)
//   Phi^{-1}(p) ~ rational approx       ... Abramowitz & Stegun 26.2.23
//   OU: x[t] = decay*x[t-1] + diff*z[t] ... 시간 상관 (선택적)
// =============================================================================

#include "mpc_controller_ros2/halton_sampler.hpp"
#include <cmath>
#include <algorithm>

namespace mpc_controller_ros2
{

// ============================================================================
// 생성자
// ============================================================================

HaltonSampler::HaltonSampler(
  const Eigen::VectorXd& sigma,
  double beta,
  int sequence_offset)
: sigma_(sigma),
  beta_(beta),
  sequence_offset_(sequence_offset),
  sample_counter_(0)
{
}

// ============================================================================
// Van der Corput 시퀀스
// ============================================================================

double HaltonSampler::haltonValue(int index, int base)
{
  double result = 0.0;
  double f = 1.0 / static_cast<double>(base);
  int i = index;
  while (i > 0) {
    result += (i % base) * f;
    f /= static_cast<double>(base);
    i /= base;
  }
  return result;
}

// ============================================================================
// 역정규 CDF (Abramowitz & Stegun, formula 26.2.23)
// ============================================================================

double HaltonSampler::inverseNormalCDF(double p)
{
  // Clamp to avoid log(0) or log(negative)
  constexpr double eps = 1e-10;
  p = std::clamp(p, eps, 1.0 - eps);

  // Rational approximation coefficients (Abramowitz & Stegun 26.2.23)
  constexpr double c0 = 2.515517;
  constexpr double c1 = 0.802853;
  constexpr double c2 = 0.010328;
  constexpr double d1 = 1.432788;
  constexpr double d2 = 0.189269;
  constexpr double d3 = 0.001308;

  // Use symmetry: if p >= 0.5, compute for 1-p and negate
  bool negate = (p < 0.5);
  double pp = negate ? p : (1.0 - p);

  double t = std::sqrt(-2.0 * std::log(pp));
  double x = t - (c0 + c1 * t + c2 * t * t) /
                 (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

  return negate ? -x : x;
}

// ============================================================================
// 소수 테이블 접근
// ============================================================================

int HaltonSampler::primeForDimension(int dim)
{
  return PRIMES[dim % 20];
}

// ============================================================================
// sample
// ============================================================================

std::vector<Eigen::MatrixXd> HaltonSampler::sample(int K, int N, int nu)
{
  std::vector<Eigen::MatrixXd> samples;
  samples.reserve(K);

  for (int k = 0; k < K; ++k) {
    Eigen::MatrixXd noise(N, nu);

    for (int t = 0; t < N; ++t) {
      for (int d = 0; d < nu; ++d) {
        int base = primeForDimension(d);
        int idx = k * N + t + sequence_offset_ + sample_counter_;
        double h = haltonValue(idx, base);
        double z = inverseNormalCDF(h);
        noise(t, d) = z * sigma_(d);
      }
    }

    // OU 시간 상관 적용 (선택적)
    if (beta_ > 0.0 && beta_ < 1000.0) {
      double decay = std::exp(-beta_);
      double diffusion_factor = std::sqrt(1.0 - decay * decay);
      for (int t = 1; t < N; ++t) {
        for (int d = 0; d < nu; ++d) {
          noise(t, d) = decay * noise(t - 1, d) + diffusion_factor * noise(t, d);
        }
      }
    }

    samples.push_back(std::move(noise));
  }

  sample_counter_ += K * N;
  return samples;
}

// ============================================================================
// sampleInPlace
// ============================================================================

void HaltonSampler::sampleInPlace(std::vector<Eigen::MatrixXd>& out, int K, int N, int nu)
{
  // Resize only if needed
  if (static_cast<int>(out.size()) != K) {
    out.resize(K);
  }

  for (int k = 0; k < K; ++k) {
    if (out[k].rows() != N || out[k].cols() != nu) {
      out[k].resize(N, nu);
    }

    for (int t = 0; t < N; ++t) {
      for (int d = 0; d < nu; ++d) {
        int base = primeForDimension(d);
        int idx = k * N + t + sequence_offset_ + sample_counter_;
        double h = haltonValue(idx, base);
        double z = inverseNormalCDF(h);
        out[k](t, d) = z * sigma_(d);
      }
    }

    // OU 시간 상관 적용 (선택적)
    if (beta_ > 0.0 && beta_ < 1000.0) {
      double decay = std::exp(-beta_);
      double diffusion_factor = std::sqrt(1.0 - decay * decay);
      for (int t = 1; t < N; ++t) {
        for (int d = 0; d < nu; ++d) {
          out[k](t, d) = decay * out[k](t - 1, d) + diffusion_factor * out[k](t, d);
        }
      }
    }
  }

  sample_counter_ += K * N;
}

// ============================================================================
// reset
// ============================================================================

void HaltonSampler::reset()
{
  sample_counter_ = 0;
}

}  // namespace mpc_controller_ros2
