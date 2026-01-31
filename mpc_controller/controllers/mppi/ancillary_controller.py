"""Ancillary 피드백 보정기 — Tube-MPPI의 핵심 구성 요소.

Williams et al. (2018) "Robust Sampling Based MPPI" 기반.
명목 궤적과 실제 상태의 편차를 body frame으로 변환 후
피드백 게인으로 보정량을 계산한다.

u_applied = u_nominal + K_fb @ error_body
"""

import numpy as np

from mpc_controller.controllers.mppi.utils import normalize_angle_batch


class AncillaryController:
    """상태 피드백 보정기 (ancillary controller).

    오차를 로봇 body frame으로 변환하여 전방/횡방향/각도 성분을
    독립적으로 보정한다.

    Attributes:
        K_fb: (nu, nx) 피드백 게인 행렬. nu=2, nx=3.
        max_correction: (nu,) 보정량 클리핑 한계.
    """

    # 기본 피드백 게인 — 보수적 설정
    # [dv]      [k_forward  0           0      ] [e_forward ]
    # [domega] = [0          k_lateral   k_angle] [e_lateral ]
    #                                              [e_angle   ]
    DEFAULT_K_FB = np.array([
        [0.8, 0.0, 0.0],   # v 보정: 전방 오차에만 반응
        [0.0, 0.5, 1.0],   # omega 보정: 횡방향 + 각도 오차
    ])

    DEFAULT_MAX_CORRECTION = np.array([0.3, 0.5])  # [dv_max, domega_max]

    def __init__(
        self,
        K_fb: np.ndarray | None = None,
        max_correction: np.ndarray | None = None,
    ):
        """
        Args:
            K_fb: (2, 3) 피드백 게인 행렬. None이면 기본값 사용.
            max_correction: (2,) [dv_max, domega_max]. None이면 기본값 사용.
        """
        self.K_fb = K_fb if K_fb is not None else self.DEFAULT_K_FB.copy()
        self.max_correction = (
            max_correction if max_correction is not None
            else self.DEFAULT_MAX_CORRECTION.copy()
        )

    def _compute_body_frame_error(
        self, nominal_state: np.ndarray, actual_state: np.ndarray
    ) -> np.ndarray:
        """세계 좌표 오차를 로봇 body frame으로 변환.

        Args:
            nominal_state: [x_nom, y_nom, theta_nom]
            actual_state: [x_act, y_act, theta_act]

        Returns:
            (3,) [e_forward, e_lateral, e_angle] — body frame 오차
        """
        dx = nominal_state[0] - actual_state[0]
        dy = nominal_state[1] - actual_state[1]
        theta = actual_state[2]

        # 세계→body 회전
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        e_forward = cos_t * dx + sin_t * dy
        e_lateral = -sin_t * dx + cos_t * dy

        # 각도 오차 (정규화)
        e_angle = float(normalize_angle_batch(
            np.array(nominal_state[2] - actual_state[2])
        ))

        return np.array([e_forward, e_lateral, e_angle])

    def compute_correction(
        self, nominal_state: np.ndarray, actual_state: np.ndarray
    ) -> np.ndarray:
        """피드백 보정량 계산.

        Args:
            nominal_state: [x_nom, y_nom, theta_nom] 명목 상태
            actual_state: [x_act, y_act, theta_act] 실제 상태

        Returns:
            (2,) [dv, domega] 보정량 (클리핑 적용)
        """
        error_body = self._compute_body_frame_error(nominal_state, actual_state)
        correction = self.K_fb @ error_body

        # 클리핑
        correction = np.clip(
            correction,
            -self.max_correction,
            self.max_correction,
        )
        return correction

    def compute_tube_width(self, disturbance_bound: float) -> float:
        """튜브 반경 추정 (시각화용).

        K_fb의 최소 특이값 기반으로 tube 폭을 추정한다.
        tube_width ≈ disturbance_bound / sigma_min(K_fb)

        클수록 보정 능력이 약해 tube가 넓어진다.

        Args:
            disturbance_bound: 예상 외란 크기 (예: 프로세스 노이즈 std)

        Returns:
            tube 반경 스칼라 [m]
        """
        singular_values = np.linalg.svd(self.K_fb, compute_uv=False)
        sigma_min = np.min(singular_values)
        if sigma_min < 1e-6:
            return disturbance_bound * 10.0  # 게인이 거의 0이면 큰 tube
        return disturbance_bound / sigma_min
