"""MPPI 노이즈 샘플링 모듈."""

from typing import Optional

import numpy as np


class BaseSampler:
    """샘플링 기본 클래스."""

    def sample(self, K: int, N: int, nu: int) -> np.ndarray:
        """노이즈 샘플 생성.

        Args:
            K: 샘플 수
            N: 호라이즌 길이
            nu: 제어 입력 차원

        Returns:
            (K, N, nu) 노이즈 배열
        """
        raise NotImplementedError


class GaussianSampler(BaseSampler):
    """가우시안 노이즈 샘플러.

    각 제어 차원별 독립 가우시안 노이즈 생성.
    noise ~ N(0, diag(sigma^2))
    """

    def __init__(
        self,
        sigma: np.ndarray,
        seed: Optional[int] = None,
    ):
        """
        Args:
            sigma: (nu,) 각 제어 차원의 표준편차
            seed: 랜덤 시드 (재현성)
        """
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def sample(self, K: int, N: int, nu: int) -> np.ndarray:
        """가우시안 노이즈 샘플 생성.

        Args:
            K: 샘플 수
            N: 호라이즌 길이
            nu: 제어 입력 차원

        Returns:
            (K, N, nu) 노이즈 배열
        """
        noise = self.rng.standard_normal((K, N, nu))
        noise *= self.sigma[np.newaxis, np.newaxis, :]
        return noise

    def reset_seed(self, seed: int) -> None:
        """랜덤 시드 재설정."""
        self.rng = np.random.default_rng(seed)
