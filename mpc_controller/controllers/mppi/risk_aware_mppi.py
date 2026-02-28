"""Risk-Aware MPPI 컨트롤러 — CVaR 가중치 절단.

Yin et al. (2023) "Risk-Aware MPPI" 기반 구현.

CVaR (Conditional Value at Risk) 접근으로, 비용이 높은 (위험한) 샘플을 제거하고
최저 비용 상위 alpha 비율의 샘플만으로 가중치를 계산한다.

  alpha=1.0 → risk-neutral (Vanilla MPPI와 동일, 전체 K개 사용)
  alpha=0.5 → 하위 50% 비용 샘플만 사용
  alpha→0   → 극단적 risk-averse (최선 소수만 사용)

┌───────────────────────────────────────┐
│ costs (K=8): [3, 7, 1, 9, 2, 8, 4, 6]│
│ alpha=0.5 → n_keep=4                  │
│ 최저 4개: [3, 1, 2, 4] (idx 0,2,4,6) │
│ weights:  [w, 0, w, 0, w, 0, w, 0]   │
│           └─ softmax on selected ─┘   │
└───────────────────────────────────────┘
"""

import numpy as np

from mpc_controller.controllers.mppi.base_mppi import MPPIController
from mpc_controller.controllers.mppi.utils import softmax_weights


class RiskAwareMPPIController(MPPIController):
    """Risk-Aware MPPI — CVaR 기반 가중치 절단.

    cvar_alpha=1.0 → Vanilla (전체 샘플), <1 → risk-averse (저비용 샘플만 사용).
    """

    def _compute_weights(self, costs: np.ndarray) -> np.ndarray:
        """CVaR 가중치 계산 — 최저 비용 ceil(alpha*K)개만 softmax.

        Args:
            costs: (K,) 비용 배열

        Returns:
            (K,) 정규화된 가중치 배열 (제거된 샘플은 0)
        """
        alpha = self.params.cvar_alpha
        lambda_ = self._get_current_lambda()
        K = len(costs)

        # alpha >= 1.0이면 전체 샘플 사용 (Vanilla 동일)
        if alpha >= 1.0:
            return softmax_weights(costs, lambda_)

        n_keep = max(1, int(np.ceil(alpha * K)))

        # n_keep >= K이면 전체 사용
        if n_keep >= K:
            return softmax_weights(costs, lambda_)

        # 최저 비용 n_keep개 선택 (argpartition으로 O(K) 선택)
        keep_indices = np.argpartition(costs, n_keep)[:n_keep]

        # 선택된 샘플만으로 softmax 가중치 계산
        selected_weights = softmax_weights(costs[keep_indices], lambda_)

        # 전체 가중치 배열 구성 (나머지는 0)
        weights = np.zeros(K)
        weights[keep_indices] = selected_weights
        return weights

    def _get_gpu_weight_fn(self):
        """GPU CVaR 가중치 함수."""
        from mpc_controller.controllers.mppi.gpu_weights import make_cvar_weights
        return make_cvar_weights(self.params.cvar_alpha, self.params.K)
