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


class ColoredNoiseSampler(BaseSampler):
    """시간 연관성 있는 colored noise 샘플러.

    Ornstein-Uhlenbeck 프로세스 기반:
        ε[t+1] = (1 - β·dt) · ε[t] + σ · √(2β·dt) · w[t]

    β가 클수록 → 빠르게 decorrelate (백색 노이즈에 가까움)
    β가 작을수록 → 강한 시간 연관 (부드러운 샘플)

    정상 분포 조건: 분산이 σ² 에 수렴하도록 diffusion 계수를 설정.
    """

    def __init__(
        self,
        sigma: np.ndarray,
        beta: float = 2.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            sigma: (nu,) 각 제어 차원의 표준편차
            beta: 역상관 속도 (correlation length ≈ 1/beta)
            seed: 랜덤 시드 (재현성)
        """
        self.sigma = sigma
        self.beta = beta
        self.rng = np.random.default_rng(seed)

    def sample(self, K: int, N: int, nu: int) -> np.ndarray:
        """Colored noise 샘플 생성.

        Args:
            K: 샘플 수
            N: 호라이즌 길이
            nu: 제어 입력 차원

        Returns:
            (K, N, nu) 노이즈 배열 — 시간축 자기상관 존재
        """
        # 정확한 OU 프로세스 이산화: decay = exp(-beta * dt)
        # 이 공식은 beta*dt 값에 관계없이 항상 decay ∈ (0, 1) 보장
        dt = 1.0  # 단위 시간 간격 (실제 dt는 dynamics에서 처리)
        decay = np.exp(-self.beta * dt)
        # 정상 분포 분산 σ² 유지 조건: diffusion = sigma * sqrt(1 - decay^2)
        diffusion = self.sigma * np.sqrt(1.0 - decay ** 2)

        noise = np.empty((K, N, nu))
        # 초기 샘플: 정상 분포에서 추출
        noise[:, 0, :] = self.rng.standard_normal((K, nu)) * self.sigma
        for t in range(1, N):
            w = self.rng.standard_normal((K, nu))
            noise[:, t, :] = decay * noise[:, t - 1, :] + diffusion * w
        return noise

    def reset_seed(self, seed: int) -> None:
        """랜덤 시드 재설정."""
        self.rng = np.random.default_rng(seed)
