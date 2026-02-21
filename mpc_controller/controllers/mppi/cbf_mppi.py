"""CBF-MPPI 컨트롤러 — Hybrid 안전성 보장 MPPI.

Zeng et al. (2021) 기반 Hybrid 접근:
1. MPPI 비용에 CBFCost 추가 (soft constraint → 안전 방향 유도)
2. Post-hoc QP safety filter (hard constraint → 최종 안전 보장)

cbf_params.enabled=False이면 부모 MPPIController와 100% 동일 동작.

┌─────────────────────────────────────────┐
│  MPPI + CBFCost → u_mppi (soft 유도)    │
│       ↓                                  │
│  CBF Safety Filter (QP) → u_safe (hard) │
│  min ||u - u_mppi||²                     │
│  s.t. ḣ(x,u) + γ·h(x) ≥ 0             │
└─────────────────────────────────────────┘
"""

import logging
from typing import Optional, Tuple

import numpy as np

from mpc_controller.models.differential_drive import RobotParams
from mpc_controller.controllers.mppi.mppi_params import MPPIParams, CBFParams
from mpc_controller.controllers.mppi.base_mppi import MPPIController
from mpc_controller.controllers.mppi.barrier_function import BarrierFunctionSet
from mpc_controller.controllers.mppi.cbf_safety_filter import CBFSafetyFilter
from mpc_controller.controllers.mppi.cost_functions import CBFCost

logger = logging.getLogger(__name__)


class CBFMPPIController(MPPIController):
    """Hybrid CBF-MPPI 컨트롤러.

    MPPIController 상속. cbf_params.enabled=False이면 부모와 동일 동작.
    """

    def __init__(
        self,
        robot_params: RobotParams | None = None,
        mppi_params: MPPIParams | None = None,
        seed: Optional[int] = None,
        obstacles: Optional[np.ndarray] = None,
        cbf_params: CBFParams | None = None,
    ):
        """
        Args:
            robot_params: 로봇 물리 파라미터
            mppi_params: MPPI 튜닝 파라미터
            seed: 랜덤 시드
            obstacles: (M, 3) 장애물 배열 [x, y, radius]
            cbf_params: CBF 파라미터 (None이면 기본값, enabled=False)
        """
        super().__init__(
            robot_params=robot_params,
            mppi_params=mppi_params,
            seed=seed,
            obstacles=obstacles,
        )

        self.cbf_params = cbf_params or CBFParams()
        self.cbf_enabled = self.cbf_params.enabled

        if not self.cbf_enabled:
            self._barrier_set = None
            self._safety_filter = None
            self._cbf_cost = None
            return

        # Barrier function 집합
        self._barrier_set = BarrierFunctionSet(
            obstacles=obstacles,
            robot_radius=self.cbf_params.robot_radius,
            safety_margin=self.cbf_params.safety_margin,
            activation_distance=self.cbf_params.activation_distance,
        )

        # Soft CBF cost 추가 (MPPI 비용 함수에)
        self._cbf_cost = CBFCost(
            barrier_set=self._barrier_set,
            weight=self.cbf_params.cost_weight,
            gamma=self.cbf_params.gamma,
            dt=self.params.dt,
        )
        self.cost.add(self._cbf_cost)

        # Post-hoc safety filter (QP)
        self._safety_filter = None
        if self.cbf_params.use_safety_filter:
            rp = self.robot_params
            u_min = np.array([-rp.max_velocity, -rp.max_omega])
            u_max = np.array([rp.max_velocity, rp.max_omega])
            self._safety_filter = CBFSafetyFilter(
                barrier_set=self._barrier_set,
                gamma=self.cbf_params.gamma,
                dt=self.params.dt,
                u_min=u_min,
                u_max=u_max,
            )

    def compute_control(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """최적 제어 입력 계산.

        cbf_enabled=False이면 부모 MPPIController.compute_control() 그대로 호출.

        Args:
            current_state: 현재 로봇 상태 [x, y, theta]
            reference_trajectory: 참조 궤적 (N+1, 3)

        Returns:
            (control, info) 튜플
        """
        if not self.cbf_enabled:
            return super().compute_control(current_state, reference_trajectory)

        # 1. MPPI (CBFCost 포함) → u_mppi
        u_mppi, info = super().compute_control(
            current_state, reference_trajectory
        )

        # 2. Post-hoc safety filter → u_safe
        if self._safety_filter is not None:
            u_safe, filter_info = self._safety_filter.filter(
                current_state,
                u_mppi,
                self._dynamics_fn,
            )
        else:
            u_safe = u_mppi.copy()
            filter_info = {
                "num_active_barriers": 0,
                "filter_applied": False,
                "qp_success": True,
                "barrier_values": [],
                "constraint_margins": [],
            }

        # 3. info 확장
        info["cbf_enabled"] = True
        info["u_mppi"] = u_mppi.copy()
        info["u_safe"] = u_safe.copy()
        info["cbf_filter_info"] = filter_info
        info["barrier_values"] = self._barrier_set.evaluate_all(
            current_state
        ).tolist()
        info["min_barrier_value"] = (
            float(min(info["barrier_values"]))
            if info["barrier_values"]
            else float("inf")
        )

        return u_safe, info

    def _dynamics_fn(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """Safety filter용 연속시간 동역학 래퍼.

        BatchDynamicsWrapper._dynamics_batch 단일 상태 래핑.
        """
        return self.dynamics._dynamics_batch(
            state[np.newaxis, :], control[np.newaxis, :]
        )[0]

    def set_obstacles(self, obstacles: np.ndarray) -> None:
        """장애물 목록 업데이트 (부모 + barrier set 갱신).

        Args:
            obstacles: (M, 3) 장애물 배열 [x, y, radius]
        """
        super().set_obstacles(obstacles)

        if self.cbf_enabled and self._barrier_set is not None:
            self._barrier_set.set_obstacles(obstacles)

            # CBFCost 재추가 (부모의 set_obstacles가 cost 재구성)
            if self._cbf_cost is not None:
                self._cbf_cost = CBFCost(
                    barrier_set=self._barrier_set,
                    weight=self.cbf_params.cost_weight,
                    gamma=self.cbf_params.gamma,
                    dt=self.params.dt,
                )
                self.cost.add(self._cbf_cost)
