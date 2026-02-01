"""Log-MPPI 컨트롤러 — log-space softmax 가중치 참조 구현.

가중치 계산을 log-space에서 수행하는 MPPI 변형.

주의: 현재 Vanilla MPPI의 softmax_weights()가 이미 max-shift trick을
적용하고 있어, Log-MPPI와 수학적으로 동일한 결과를 낸다.
따라서 **제어 성능 차이는 없으며**, log-space 가중치 조작이 필요한
후속 확장(importance sampling 보정 등)의 기반 클래스로 활용 가능.
"""

import numpy as np

from mpc_controller.controllers.mppi.base_mppi import MPPIController
from mpc_controller.controllers.mppi.utils import log_sum_exp


class LogMPPIController(MPPIController):
    """Log-space MPPI — 수치 안정성 개선.

    softmax 가중치를 log-space에서 계산하여
    극단적 cost 범위(1e-15~1e15)에서도 NaN/Inf 방지.

    알고리즘:
      log_w_k = -S_k / λ
      log_w_k -= log_sum_exp(log_w)   # log-space 정규화
      w_k = exp(log_w_k)
    """

    def _compute_weights(self, costs: np.ndarray) -> np.ndarray:
        """Log-space softmax 가중치 계산.

        Args:
            costs: (K,) 비용 배열

        Returns:
            (K,) 정규화된 가중치 배열 (NaN/Inf 안전)
        """
        lambda_ = self._get_current_lambda()
        log_weights = -costs / lambda_
        log_weights -= log_sum_exp(log_weights)  # log-space 정규화
        return np.exp(log_weights)
