"""MPPI 파라미터 데이터클래스."""

from dataclasses import dataclass

import numpy as np


@dataclass
class MPPIParams:
    """MPPI 튜닝 파라미터.

    Attributes:
        N: 예측 호라이즌 스텝 수
        dt: 시간 간격 [s]
        K: 샘플 수
        lambda_: 온도 파라미터 (softmax 스케일링)
        noise_sigma: 제어 입력 노이즈 표준편차 [v_sigma, omega_sigma]
        Q: 상태 가중 행렬 [x, y, theta]
        R: 제어 가중 행렬 [v, omega]
        Qf: 터미널 상태 가중 행렬
    """

    # Horizon
    N: int = 30
    dt: float = 0.05

    # Sampling
    K: int = 1024
    lambda_: float = 10.0

    # Noise standard deviation [v, omega]
    noise_sigma: np.ndarray | None = None

    # State weights [x, y, theta]
    Q: np.ndarray | None = None

    # Control weights [v, omega]
    R: np.ndarray | None = None

    # Terminal state weights
    Qf: np.ndarray | None = None

    # Control rate weights (u_t - u_{t-1}) penalty — None = 비활성
    R_rate: np.ndarray | None = None

    # Adaptive temperature (ESS 기반 λ 자동 조정)
    adaptive_temperature: bool = False
    adaptive_temp_config: dict | None = None

    # Colored noise sampling (Ornstein-Uhlenbeck 프로세스)
    colored_noise: bool = False
    noise_beta: float = 2.0

    def __post_init__(self):
        if self.noise_sigma is None:
            self.noise_sigma = np.array([0.3, 0.3])
        if self.Q is None:
            self.Q = np.diag([10.0, 10.0, 1.0])
        if self.R is None:
            self.R = np.diag([0.01, 0.01])
        if self.Qf is None:
            self.Qf = np.diag([50.0, 50.0, 5.0])
