"""AncillaryController 단위 테스트."""

import numpy as np
import pytest

from mpc_controller.controllers.mppi.ancillary_controller import AncillaryController


class TestAncillaryBasic:
    """기본 동작 테스트."""

    def setup_method(self):
        self.ctrl = AncillaryController()

    def test_zero_error_zero_correction(self):
        """오차 0 → 보정 0."""
        state = np.array([1.0, 2.0, 0.5])
        du = self.ctrl.compute_correction(state, state)
        np.testing.assert_allclose(du, [0.0, 0.0], atol=1e-10)

    def test_forward_error_positive_v_correction(self):
        """명목이 실제보다 앞에 있을 때 → dv > 0 (가속)."""
        nominal = np.array([1.0, 0.0, 0.0])  # 전방 1m
        actual = np.array([0.0, 0.0, 0.0])   # 원점, theta=0 (전방=+x)
        du = self.ctrl.compute_correction(nominal, actual)
        assert du[0] > 0, f"전방 오차에 대해 dv > 0 기대, got {du[0]}"

    def test_lateral_error_omega_correction(self):
        """명목이 왼쪽에 있을 때 → domega 보정."""
        nominal = np.array([0.0, 1.0, 0.0])  # 왼쪽 1m
        actual = np.array([0.0, 0.0, 0.0])   # 원점, theta=0
        du = self.ctrl.compute_correction(nominal, actual)
        # 횡방향 오차 → omega 보정 (왼쪽으로 가야 하므로 domega > 0)
        assert du[1] != 0, "횡방향 오차에 대해 domega != 0 기대"

    def test_angle_error_wrapping(self):
        """pi 경계 정규화 테스트."""
        nominal = np.array([0.0, 0.0, 3.0])
        actual = np.array([0.0, 0.0, -3.0])
        # 3.0 - (-3.0) = 6.0 → normalize → 약 -0.28
        du = self.ctrl.compute_correction(nominal, actual)
        # 각도차가 크지 않아야 함 (정규화 됨)
        assert abs(du[1]) < 2.0, f"각도 정규화 실패: domega={du[1]}"

    def test_correction_clipping(self):
        """max_correction 클리핑."""
        max_corr = np.array([0.1, 0.2])
        ctrl = AncillaryController(max_correction=max_corr)
        # 매우 큰 오차
        nominal = np.array([10.0, 10.0, 3.0])
        actual = np.array([0.0, 0.0, 0.0])
        du = ctrl.compute_correction(nominal, actual)
        assert abs(du[0]) <= max_corr[0] + 1e-10
        assert abs(du[1]) <= max_corr[1] + 1e-10

    def test_custom_gain_matrix(self):
        """사용자 정의 K_fb."""
        K_fb = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.5],
        ])
        ctrl = AncillaryController(K_fb=K_fb)
        nominal = np.array([1.0, 0.0, 0.0])
        actual = np.array([0.0, 0.0, 0.0])
        du = ctrl.compute_correction(nominal, actual)
        # K_fb[0,0]=1.0 → dv = 1.0 * e_forward = 1.0
        # 클리핑 적용 가능
        assert du[0] > 0

    def test_output_shape(self):
        """출력 형태 (2,)."""
        du = self.ctrl.compute_correction(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        assert du.shape == (2,)


class TestTubeWidth:
    """tube 폭 계산 테스트."""

    def test_tube_width_positive(self):
        """tube 폭은 양수."""
        ctrl = AncillaryController()
        width = ctrl.compute_tube_width(0.1)
        assert width > 0

    def test_tube_width_scales_with_disturbance(self):
        """외란이 크면 tube 폭도 커야 함."""
        ctrl = AncillaryController()
        w1 = ctrl.compute_tube_width(0.05)
        w2 = ctrl.compute_tube_width(0.20)
        assert w2 > w1

    def test_tube_width_zero_gain(self):
        """게인이 0이면 tube 폭이 매우 커야 함."""
        K_fb = np.zeros((2, 3))
        ctrl = AncillaryController(K_fb=K_fb)
        width = ctrl.compute_tube_width(0.1)
        assert width >= 1.0  # 큰 값


class TestBodyFrameTransformation:
    """좌표 변환 정확성 테스트."""

    def test_forward_aligned(self):
        """theta=0: 전방=+x, 횡방향=+y."""
        ctrl = AncillaryController()
        nominal = np.array([1.0, 0.0, 0.0])
        actual = np.array([0.0, 0.0, 0.0])
        error = ctrl._compute_body_frame_error(nominal, actual)
        # e_forward ≈ 1.0, e_lateral ≈ 0, e_angle ≈ 0
        np.testing.assert_allclose(error[0], 1.0, atol=1e-10)
        np.testing.assert_allclose(error[1], 0.0, atol=1e-10)
        np.testing.assert_allclose(error[2], 0.0, atol=1e-10)

    def test_lateral_aligned(self):
        """theta=0: 횡방향은 +y."""
        ctrl = AncillaryController()
        nominal = np.array([0.0, 1.0, 0.0])
        actual = np.array([0.0, 0.0, 0.0])
        error = ctrl._compute_body_frame_error(nominal, actual)
        np.testing.assert_allclose(error[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(error[1], 1.0, atol=1e-10)

    def test_rotated_frame(self):
        """theta=pi/2: 전방=+y, 횡방향=-x."""
        ctrl = AncillaryController()
        # 로봇이 +y 방향을 향하고 있을 때, 명목이 +y 방향 1m 앞
        nominal = np.array([0.0, 1.0, np.pi / 2])
        actual = np.array([0.0, 0.0, np.pi / 2])
        error = ctrl._compute_body_frame_error(nominal, actual)
        np.testing.assert_allclose(error[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(error[1], 0.0, atol=1e-6)

    def test_pure_angle_error(self):
        """위치 동일, 각도만 다름."""
        ctrl = AncillaryController()
        nominal = np.array([0.0, 0.0, 0.3])
        actual = np.array([0.0, 0.0, 0.0])
        error = ctrl._compute_body_frame_error(nominal, actual)
        np.testing.assert_allclose(error[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(error[1], 0.0, atol=1e-10)
        np.testing.assert_allclose(error[2], 0.3, atol=1e-10)
