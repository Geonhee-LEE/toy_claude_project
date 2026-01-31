"""Tube-MPPI 컨트롤러 — 외란 강건 MPPI.

Williams et al. (2018) "Robust Sampling Based Model Predictive Path Integral Control"

핵심 원리:
  u_applied = u_nominal(MPPI) + K_fb·(x_nom - x_act)

  1. 명목 상태(x_nom)에서 MPPI로 u_nominal 계산 (외란 가정 없음)
  2. 실제 상태(x_act)와 명목 상태의 편차를 ancillary 피드백으로 보정
  3. 명목 상태는 외란 없이 모델로 전파
  4. "Tube" = ancillary가 실제 상태를 명목 궤적 주변 bounded 영역에 유지

tube_enabled=False이면 부모 MPPIController와 100% 동일 동작 (호환성 보장).
"""

import logging
from typing import Optional, Tuple

import numpy as np

from mpc_controller.models.differential_drive import RobotParams
from mpc_controller.controllers.mppi.mppi_params import MPPIParams
from mpc_controller.controllers.mppi.base_mppi import MPPIController
from mpc_controller.controllers.mppi.ancillary_controller import AncillaryController
from mpc_controller.controllers.mppi.utils import normalize_angle_batch

logger = logging.getLogger(__name__)


class TubeMPPIController(MPPIController):
    """Tube-MPPI 컨트롤러.

    MPPIController 상속. tube_enabled=False이면 부모와 동일 동작.
    """

    def __init__(
        self,
        robot_params: RobotParams | None = None,
        mppi_params: MPPIParams | None = None,
        seed: Optional[int] = None,
        obstacles: Optional[np.ndarray] = None,
    ):
        super().__init__(
            robot_params=robot_params,
            mppi_params=mppi_params,
            seed=seed,
            obstacles=obstacles,
        )

        self.tube_enabled = self.params.tube_enabled

        # Ancillary 컨트롤러 초기화
        self.ancillary = AncillaryController(
            K_fb=self.params.tube_K_fb,
            max_correction=self.params.tube_max_correction,
        )

        # 명목 상태 (첫 호출 시 초기화)
        self._x_nominal: Optional[np.ndarray] = None

        # Tube 폭 (시각화용, 불변)
        self._tube_width = self.ancillary.compute_tube_width(
            self.params.tube_disturbance_bound
        )

    def compute_control(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """최적 제어 입력 계산.

        tube_enabled=False이면 부모 MPPIController.compute_control() 그대로 호출.

        Args:
            current_state: 현재 로봇 상태 [x, y, theta]
            reference_trajectory: 참조 궤적 (N+1, 3)

        Returns:
            (control, info) 튜플
        """
        if not self.tube_enabled:
            return super().compute_control(current_state, reference_trajectory)

        # 1. 명목 상태 초기화 (첫 호출 또는 리셋 후)
        if self._x_nominal is None:
            self._x_nominal = current_state.copy()

        # 편차가 threshold 초과 시 명목 상태 리셋 (발산 방지)
        deviation = np.linalg.norm(
            self._x_nominal[:2] - current_state[:2]
        )
        if deviation > self.params.tube_nominal_reset_threshold:
            logger.warning(
                f"Tube-MPPI: nominal state deviation {deviation:.3f}m > "
                f"threshold {self.params.tube_nominal_reset_threshold:.3f}m, "
                f"resetting nominal state"
            )
            self._x_nominal = current_state.copy()

        # 2. MPPI: 명목 상태 기준으로 u_nominal 계산
        u_nominal, info = super().compute_control(
            self._x_nominal, reference_trajectory
        )

        # 3. Ancillary 보정
        du = self.ancillary.compute_correction(
            self._x_nominal, current_state
        )

        # 4. 최종 제어 = u_nominal + du
        u_applied = u_nominal + du

        # 제어 한계 클리핑
        u_applied[0] = np.clip(
            u_applied[0],
            -self.robot_params.max_velocity,
            self.robot_params.max_velocity,
        )
        u_applied[1] = np.clip(
            u_applied[1],
            -self.robot_params.max_omega,
            self.robot_params.max_omega,
        )

        # 5. 명목 상태 전파 (외란 없는 모델)
        self._x_nominal = self.dynamics.propagate_batch(
            self._x_nominal[np.newaxis, :],
            u_nominal[np.newaxis, :],
            self.params.dt,
        )[0]

        # 6. info 확장
        nominal_trajectory = self._compute_nominal_trajectory(
            self._x_nominal, info.get("predicted_controls")
        )
        tube_boundary = self._compute_tube_boundary(
            nominal_trajectory, self._tube_width
        )

        info["tube_enabled"] = True
        info["nominal_state"] = self._x_nominal.copy()
        info["feedback_correction"] = du.copy()
        info["tube_width"] = self._tube_width
        info["nominal_trajectory"] = nominal_trajectory
        info["tube_boundary"] = tube_boundary
        info["deviation"] = deviation

        return u_applied, info

    def _compute_nominal_trajectory(
        self, x_nominal: np.ndarray, controls: Optional[np.ndarray]
    ) -> np.ndarray:
        """명목 상태에서 예측 제어열로 궤적 전파.

        Args:
            x_nominal: 현재 명목 상태 (3,)
            controls: 예측 제어열 (N, 2). None이면 명목 상태만 반환.

        Returns:
            (N+1, 3) 명목 궤적
        """
        if controls is None:
            return x_nominal[np.newaxis, :].copy()

        N = controls.shape[0]
        traj = np.zeros((N + 1, 3))
        traj[0] = x_nominal.copy()

        for t in range(N):
            traj[t + 1] = self.dynamics.propagate_batch(
                traj[t][np.newaxis, :],
                controls[t][np.newaxis, :],
                self.params.dt,
            )[0]

        return traj

    def _compute_tube_boundary(
        self, nominal_trajectory: np.ndarray, tube_width: float
    ) -> dict:
        """명목 궤적의 법선 방향으로 tube 경계 생성.

        시각화용: matplotlib fill_between 또는 RVIZ LINE_STRIP.

        Args:
            nominal_trajectory: (N+1, 3) 명목 궤적
            tube_width: tube 반경 [m]

        Returns:
            {"upper": (N+1, 2), "lower": (N+1, 2)} — 경계 좌표
        """
        N_plus_1 = nominal_trajectory.shape[0]
        upper = np.zeros((N_plus_1, 2))
        lower = np.zeros((N_plus_1, 2))

        for i in range(N_plus_1):
            x, y, theta = nominal_trajectory[i]
            # 법선 방향 (궤적 진행 방향의 수직)
            nx = -np.sin(theta)
            ny = np.cos(theta)

            upper[i] = [x + tube_width * nx, y + tube_width * ny]
            lower[i] = [x - tube_width * nx, y - tube_width * ny]

        return {"upper": upper, "lower": lower}

    def reset(self) -> None:
        """제어열 및 명목 상태 초기화."""
        super().reset()
        self._x_nominal = None
