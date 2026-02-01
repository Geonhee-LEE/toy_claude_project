"""MPPI 샘플링 모듈 테스트."""

import numpy as np
import pytest

from mpc_controller.controllers.mppi.sampling import GaussianSampler, ColoredNoiseSampler


class TestGaussianSampler:
    """GaussianSampler 테스트."""

    @pytest.fixture
    def sampler(self):
        sigma = np.array([0.5, 0.3])
        return GaussianSampler(sigma, seed=42)

    def test_output_shape(self, sampler):
        K, N, nu = 100, 10, 2
        noise = sampler.sample(K, N, nu)
        assert noise.shape == (K, N, nu)

    def test_mean_near_zero(self, sampler):
        K, N, nu = 10000, 10, 2
        noise = sampler.sample(K, N, nu)
        mean = np.mean(noise, axis=0)
        np.testing.assert_allclose(mean, 0.0, atol=0.1)

    def test_std_matches_sigma(self, sampler):
        K, N, nu = 10000, 10, 2
        noise = sampler.sample(K, N, nu)
        std_v = np.std(noise[:, :, 0])
        std_omega = np.std(noise[:, :, 1])
        np.testing.assert_allclose(std_v, 0.5, atol=0.05)
        np.testing.assert_allclose(std_omega, 0.3, atol=0.05)

    def test_reproducibility(self):
        sigma = np.array([0.5, 0.3])
        sampler1 = GaussianSampler(sigma, seed=123)
        sampler2 = GaussianSampler(sigma, seed=123)

        noise1 = sampler1.sample(10, 5, 2)
        noise2 = sampler2.sample(10, 5, 2)

        np.testing.assert_array_equal(noise1, noise2)

    def test_different_seeds_different_samples(self):
        sigma = np.array([0.5, 0.3])
        sampler1 = GaussianSampler(sigma, seed=1)
        sampler2 = GaussianSampler(sigma, seed=2)

        noise1 = sampler1.sample(10, 5, 2)
        noise2 = sampler2.sample(10, 5, 2)

        assert not np.allclose(noise1, noise2)

    def test_reset_seed(self):
        sigma = np.array([0.5, 0.3])
        sampler = GaussianSampler(sigma, seed=42)

        noise1 = sampler.sample(10, 5, 2)
        sampler.reset_seed(42)
        noise2 = sampler.sample(10, 5, 2)

        np.testing.assert_array_equal(noise1, noise2)


class TestColoredNoiseSampler:
    """ColoredNoiseSampler 테스트."""

    @pytest.fixture
    def sampler(self):
        sigma = np.array([0.5, 0.3])
        return ColoredNoiseSampler(sigma, beta=2.0, seed=42)

    def test_output_shape(self, sampler):
        K, N, nu = 100, 10, 2
        noise = sampler.sample(K, N, nu)
        assert noise.shape == (K, N, nu)

    def test_mean_near_zero(self, sampler):
        K, N, nu = 10000, 20, 2
        noise = sampler.sample(K, N, nu)
        mean = np.mean(noise, axis=0)
        np.testing.assert_allclose(mean, 0.0, atol=0.15)

    def test_temporal_autocorrelation(self):
        """시간축 자기상관이 양수인지 검증 (colored noise 특성)."""
        sigma = np.array([1.0, 1.0])
        sampler = ColoredNoiseSampler(sigma, beta=0.5, seed=42)
        K, N, nu = 5000, 30, 2

        noise = sampler.sample(K, N, nu)

        # lag-1 자기상관 계산 (차원 0 기준)
        x = noise[:, :-1, 0].flatten()
        y = noise[:, 1:, 0].flatten()
        corr = np.corrcoef(x, y)[0, 1]

        # colored noise는 양의 자기상관을 가져야 함
        assert corr > 0.1, f"lag-1 autocorrelation = {corr:.4f} (expected > 0.1)"

    def test_high_beta_low_correlation(self):
        """β가 크면 자기상관이 낮아야 함 (백색 노이즈에 수렴)."""
        sigma = np.array([1.0, 1.0])
        low_beta = ColoredNoiseSampler(sigma, beta=0.3, seed=42)
        high_beta = ColoredNoiseSampler(sigma, beta=10.0, seed=42)
        K, N, nu = 5000, 30, 2

        noise_low = low_beta.sample(K, N, nu)
        noise_high = high_beta.sample(K, N, nu)

        corr_low = np.corrcoef(
            noise_low[:, :-1, 0].flatten(),
            noise_low[:, 1:, 0].flatten(),
        )[0, 1]
        corr_high = np.corrcoef(
            noise_high[:, :-1, 0].flatten(),
            noise_high[:, 1:, 0].flatten(),
        )[0, 1]

        assert corr_low > corr_high, (
            f"low_beta corr={corr_low:.4f} should > high_beta corr={corr_high:.4f}"
        )

    def test_reproducibility(self):
        sigma = np.array([0.5, 0.3])
        s1 = ColoredNoiseSampler(sigma, beta=2.0, seed=123)
        s2 = ColoredNoiseSampler(sigma, beta=2.0, seed=123)

        noise1 = s1.sample(10, 5, 2)
        noise2 = s2.sample(10, 5, 2)
        np.testing.assert_array_equal(noise1, noise2)

    def test_reset_seed(self):
        sigma = np.array([0.5, 0.3])
        sampler = ColoredNoiseSampler(sigma, beta=2.0, seed=42)
        noise1 = sampler.sample(10, 5, 2)
        sampler.reset_seed(42)
        noise2 = sampler.sample(10, 5, 2)
        np.testing.assert_array_equal(noise1, noise2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
