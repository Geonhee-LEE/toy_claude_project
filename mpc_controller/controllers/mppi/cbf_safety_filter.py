"""CBF Safety Filter — Post-hoc QP 기반 안전 보장 필터.

MPPI가 산출한 u_mppi를 최소한으로 수정하여 CBF 제약을 만족:

  min_u  ½||u - u_mppi||²
  s.t.   ḣ_i(x,u) + γ·h_i(x) ≥ 0   ∀ active obstacle i
         u_min ≤ u ≤ u_max

QP 실패 시 u_mppi를 그대로 반환 (graceful degradation).
"""

import logging
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from mpc_controller.controllers.mppi.barrier_function import BarrierFunctionSet

logger = logging.getLogger(__name__)


class CBFSafetyFilter:
    """Post-hoc CBF safety filter (QP via scipy SLSQP).

    MPPI 출력 u_mppi를 입력으로 받아,
    CBF 제약을 만족하는 가장 가까운 u_safe를 반환.
    """

    def __init__(
        self,
        barrier_set: BarrierFunctionSet,
        gamma: float = 1.0,
        dt: float = 0.05,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
    ):
        """
        Args:
            barrier_set: 다중 장애물 barrier 함수 집합
            gamma: class-K 함수 계수 (CBF decay rate)
            dt: 시간 간격 [s]
            u_min: 제어 입력 하한 (nu,). None이면 제한 없음.
            u_max: 제어 입력 상한 (nu,). None이면 제한 없음.
        """
        self.barrier_set = barrier_set
        self.gamma = gamma
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max

    def filter(
        self,
        state: np.ndarray,
        u_mppi: np.ndarray,
        dynamics_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, dict]:
        """Safety filter 적용.

        Args:
            state: (nx,) 현재 상태
            u_mppi: (nu,) MPPI 출력 제어
            dynamics_fn: 연속시간 동역학 f(state, control) -> state_dot

        Returns:
            (u_safe, info) 튜플:
              - u_safe: (nu,) 안전 보장 제어
              - info: 디버그 정보
        """
        active_barriers = self.barrier_set.get_active_barriers(state)

        info = {
            "num_active_barriers": len(active_barriers),
            "filter_applied": False,
            "qp_success": True,
            "barrier_values": [],
            "constraint_margins": [],
        }

        # 활성 장애물이 없으면 u_mppi 그대로 반환
        if len(active_barriers) == 0:
            return u_mppi.copy(), info

        # 현재 barrier 값 기록
        h_values = np.array([b.evaluate(state) for b in active_barriers])
        info["barrier_values"] = h_values.tolist()

        # 이미 안전한 제어인지 확인 (빠른 경로)
        if self._is_safe(state, u_mppi, active_barriers, dynamics_fn):
            info["constraint_margins"] = self._compute_margins(
                state, u_mppi, active_barriers, dynamics_fn
            ).tolist()
            return u_mppi.copy(), info

        # QP 풀기
        info["filter_applied"] = True
        nu = len(u_mppi)

        # 목적함수: ½||u - u_mppi||²
        def objective(u):
            diff = u - u_mppi
            return 0.5 * np.dot(diff, diff)

        def objective_grad(u):
            return u - u_mppi

        # CBF 제약: ḣ(x,u) + γ·h(x) ≥ 0
        constraints = []
        for barrier in active_barriers:
            h_val = barrier.evaluate(state)
            grad_h = barrier.gradient(state)

            def cbf_constraint(u, _h=h_val, _grad=grad_h):
                # ḣ = ∇h · f(x,u)
                x_dot = dynamics_fn(state, u)
                h_dot = np.dot(_grad, x_dot)
                return h_dot + self.gamma * _h

            constraints.append({
                "type": "ineq",
                "fun": cbf_constraint,
            })

        # 제어 입력 범위
        bounds = None
        if self.u_min is not None and self.u_max is not None:
            bounds = list(zip(self.u_min, self.u_max))

        # SLSQP 최적화
        try:
            result = minimize(
                objective,
                u_mppi.copy(),
                jac=objective_grad,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 100, "ftol": 1e-8},
            )

            if result.success:
                u_safe = result.x
                info["qp_success"] = True
            else:
                logger.warning(
                    f"CBF QP failed: {result.message}. Falling back to u_mppi."
                )
                u_safe = u_mppi.copy()
                info["qp_success"] = False
        except Exception as e:
            logger.warning(f"CBF QP exception: {e}. Falling back to u_mppi.")
            u_safe = u_mppi.copy()
            info["qp_success"] = False

        info["constraint_margins"] = self._compute_margins(
            state, u_safe, active_barriers, dynamics_fn
        ).tolist()

        return u_safe, info

    def _is_safe(
        self,
        state: np.ndarray,
        u: np.ndarray,
        active_barriers: list,
        dynamics_fn: Callable,
    ) -> bool:
        """현재 제어가 모든 CBF 제약을 만족하는지 확인."""
        margins = self._compute_margins(state, u, active_barriers, dynamics_fn)
        return np.all(margins >= -1e-6)

    def _compute_margins(
        self,
        state: np.ndarray,
        u: np.ndarray,
        active_barriers: list,
        dynamics_fn: Callable,
    ) -> np.ndarray:
        """각 barrier에 대한 CBF 제약 마진 계산.

        margin_i = ḣ_i(x,u) + γ·h_i(x)
        margin ≥ 0 이면 안전.
        """
        x_dot = dynamics_fn(state, u)
        margins = np.zeros(len(active_barriers))
        for i, barrier in enumerate(active_barriers):
            h_val = barrier.evaluate(state)
            grad_h = barrier.gradient(state)
            h_dot = np.dot(grad_h, x_dot)
            margins[i] = h_dot + self.gamma * h_val
        return margins
