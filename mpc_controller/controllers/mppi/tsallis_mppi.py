"""Tsallis-MPPI 컨트롤러 — Tsallis 엔트로피 기반 일반화 MPPI.

Yin et al. (2021) "Variational Inference MPC using Tsallis Divergence"

q-exponential 가중치를 사용하여 탐색/집중 균형을 제어한다.
  q=1.0 → Shannon 엔트로피 (Vanilla MPPI와 동일)
  q>1.0 → heavy-tail (탐색 범위 확대)
  q<1.0 → light-tail (최적 해 주변 집중)
"""

import numpy as np

from mpc_controller.controllers.mppi.base_mppi import MPPIController
from mpc_controller.controllers.mppi.utils import q_exponential


class TsallisMPPIController(MPPIController):
    """Tsallis 엔트로피 기반 MPPI.

    q=1.0 → Vanilla (Shannon), q>1 → heavy-tail (탐색↑), q<1 → light-tail (집중↑)
    """

    def _compute_weights(self, costs: np.ndarray) -> np.ndarray:
        """Tsallis q-exponential 가중치 계산.

        비용을 min-centering 후 q-exponential 적용.
        q-exponential은 translation-invariant가 아니므로
        centering 없이는 절대 비용 크기가 가중치를 지배한다.

        Args:
            costs: (K,) 비용 배열

        Returns:
            (K,) 정규화된 가중치 배열
        """
        lambda_ = self._get_current_lambda()
        q = self.params.tsallis_q
        centered = costs - np.min(costs)  # min-centering (best cost → 0)
        raw = q_exponential(-centered / lambda_, q)
        total = np.sum(raw)
        if total == 0.0:
            return np.ones_like(costs) / len(costs)
        return raw / total
