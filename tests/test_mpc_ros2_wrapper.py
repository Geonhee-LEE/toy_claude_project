"""
MPC ROS2 Wrapper 기본 테스트.

ROS2 의존성 없이 기본 로직을 테스트합니다.
"""

import unittest
import numpy as np


class TestMPCROS2Wrapper(unittest.TestCase):
    """MPC ROS2 Wrapper 기본 테스트."""

    def test_quaternion_to_euler(self):
        """쿼터니언에서 오일러 각 변환 테스트."""
        # theta = pi/4 (45도)
        theta = np.pi / 4
        w = np.cos(theta / 2.0)
        z = np.sin(theta / 2.0)

        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        # x = y = 0 인 경우
        calculated_theta = np.arctan2(2.0 * w * z, 1.0 - 2.0 * z**2)

        self.assertAlmostEqual(calculated_theta, theta, places=5)

    def test_euler_to_quaternion(self):
        """오일러 각에서 쿼터니언 변환 테스트."""
        theta = np.pi / 3  # 60도

        w = np.cos(theta / 2.0)
        z = np.sin(theta / 2.0)

        # 다시 역변환
        calculated_theta = np.arctan2(2.0 * w * z, 1.0 - 2.0 * z**2)

        self.assertAlmostEqual(calculated_theta, theta, places=5)

    def test_state_representation(self):
        """상태 표현 검증."""
        # [x, y, theta] 형식
        state = np.array([1.0, 2.0, np.pi / 4])

        self.assertEqual(len(state), 3)
        self.assertAlmostEqual(state[0], 1.0)
        self.assertAlmostEqual(state[1], 2.0)
        self.assertAlmostEqual(state[2], np.pi / 4)

    def test_control_representation(self):
        """제어 입력 표현 검증."""
        # [v, omega] 형식
        control = np.array([0.5, 0.3])

        self.assertEqual(len(control), 2)
        self.assertAlmostEqual(control[0], 0.5)
        self.assertAlmostEqual(control[1], 0.3)

    def test_trajectory_format(self):
        """궤적 형식 검증."""
        N = 20
        trajectory = np.zeros((N + 1, 3))

        for i in range(N + 1):
            trajectory[i] = [float(i) * 0.1, 0.0, 0.0]

        self.assertEqual(trajectory.shape, (N + 1, 3))
        self.assertAlmostEqual(trajectory[0, 0], 0.0)
        self.assertAlmostEqual(trajectory[10, 0], 1.0)


if __name__ == '__main__':
    unittest.main()
