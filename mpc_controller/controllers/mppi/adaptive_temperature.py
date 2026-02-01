"""ESS 기반 적응형 온도 파라미터 (λ) 조정.

Effective Sample Size(ESS) 비율에 따라 λ를 자동으로 조정하여
탐색(exploration)과 활용(exploitation) 간의 균형을 유지한다.

ESS가 낮으면 → 소수 샘플에 가중치 집중 → λ 증가로 탐색 강화
ESS가 높으면 → 가중치 균등 분산 → λ 감소로 최적 샘플 집중
"""

import numpy as np


class AdaptiveTemperature:
    """ESS 비율에 따라 λ를 자동 조정.

    log-space에서 조정하여 양수를 보장하고,
    목표 ESS 비율과의 편차에 비례하여 λ를 업데이트한다.
    """

    def __init__(
        self,
        initial_lambda: float = 10.0,
        target_ess_ratio: float = 0.5,
        adaptation_rate: float = 0.1,
        lambda_min: float = 1.0,
        lambda_max: float = 100.0,
    ):
        """
        Args:
            initial_lambda: 초기 온도 값
            target_ess_ratio: 목표 ESS/K 비율 (0~1)
            adaptation_rate: 조정 속도 (클수록 빠른 적응)
            lambda_min: λ 하한
            lambda_max: λ 상한
        """
        self.log_lambda = np.log(initial_lambda)
        self.target_ess_ratio = target_ess_ratio
        self.adaptation_rate = adaptation_rate
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def update(self, ess: float, K: int) -> float:
        """ESS 관측값으로 λ 업데이트.

        Args:
            ess: 현재 Effective Sample Size
            K: 총 샘플 수

        Returns:
            업데이트된 λ 값
        """
        ess_ratio = ess / K
        # ess_ratio < target → log_lambda 증가 → λ 증가 (탐색 강화)
        # ess_ratio > target → log_lambda 감소 → λ 감소 (활용 강화)
        self.log_lambda += self.adaptation_rate * (self.target_ess_ratio - ess_ratio)
        # 클램핑
        lambda_val = np.exp(self.log_lambda)
        lambda_val = np.clip(lambda_val, self.lambda_min, self.lambda_max)
        self.log_lambda = np.log(lambda_val)
        return float(lambda_val)

    @property
    def lambda_(self) -> float:
        """현재 λ 값."""
        return float(np.exp(self.log_lambda))
