"""배치 동역학 래퍼 - K개 샘플 벡터화 처리."""

import numpy as np

from mpc_controller.models.differential_drive import DifferentialDriveModel, RobotParams
from mpc_controller.controllers.mppi.utils import normalize_angle_batch


class BatchDynamicsWrapper:
    """DifferentialDriveModel의 배치 벡터화 래퍼.

    K개 샘플을 for-loop 없이 NumPy broadcasting으로 동시 처리.

    State: [x, y, theta]  (nx=3)
    Control: [v, omega]   (nu=2)
    """

    def __init__(self, robot_params: RobotParams | None = None):
        self.params = robot_params or RobotParams()
        self.nx = DifferentialDriveModel.STATE_DIM  # 3
        self.nu = DifferentialDriveModel.CONTROL_DIM  # 2

    def _dynamics_batch(
        self, states: np.ndarray, controls: np.ndarray
    ) -> np.ndarray:
        """연속시간 동역학 (배치).

        Args:
            states: (K, 3) [x, y, theta]
            controls: (K, 2) [v, omega]

        Returns:
            (K, 3) 상태 미분
        """
        theta = states[:, 2]
        v = controls[:, 0]
        omega = controls[:, 1]

        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = omega

        return np.stack([x_dot, y_dot, theta_dot], axis=-1)

    def propagate_batch(
        self, states: np.ndarray, controls: np.ndarray, dt: float
    ) -> np.ndarray:
        """RK4 적분으로 한 스텝 전파 (배치).

        Args:
            states: (K, 3) 현재 상태
            controls: (K, 2) 제어 입력
            dt: 시간 간격

        Returns:
            (K, 3) 다음 상태
        """
        k1 = self._dynamics_batch(states, controls)
        k2 = self._dynamics_batch(states + dt / 2 * k1, controls)
        k3 = self._dynamics_batch(states + dt / 2 * k2, controls)
        k4 = self._dynamics_batch(states + dt * k3, controls)

        states_next = states + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        states_next[:, 2] = normalize_angle_batch(states_next[:, 2])

        return states_next

    def rollout_batch(
        self,
        x0: np.ndarray,
        control_sequences: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """K개 제어 시퀀스를 동시 rollout.

        Args:
            x0: (3,) 초기 상태
            control_sequences: (K, N, 2) 제어 시퀀스
            dt: 시간 간격

        Returns:
            (K, N+1, 3) 궤적 배열
        """
        K, N, _ = control_sequences.shape

        trajectories = np.zeros((K, N + 1, self.nx))
        trajectories[:, 0, :] = x0[np.newaxis, :]

        for t in range(N):
            trajectories[:, t + 1, :] = self.propagate_batch(
                trajectories[:, t, :],
                control_sequences[:, t, :],
                dt,
            )

        return trajectories

    def clip_controls(self, controls: np.ndarray) -> np.ndarray:
        """제어 입력을 로봇 한계 내로 클리핑.

        Args:
            controls: (..., 2) 제어 입력

        Returns:
            클리핑된 제어 입력
        """
        clipped = controls.copy()
        clipped[..., 0] = np.clip(
            clipped[..., 0],
            -self.params.max_velocity,
            self.params.max_velocity,
        )
        clipped[..., 1] = np.clip(
            clipped[..., 1],
            -self.params.max_omega,
            self.params.max_omega,
        )
        return clipped
