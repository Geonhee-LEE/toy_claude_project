"""MPPI 샘플링 모듈 테스트."""

import numpy as np
import pytest

from mpc_controller.controllers.mppi.sampling import GaussianSampler


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
