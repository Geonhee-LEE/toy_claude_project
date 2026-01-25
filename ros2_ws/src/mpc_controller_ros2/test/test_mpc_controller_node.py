"""
MPC Controller Node 테스트.

ROS2 노드의 기본 기능을 테스트합니다.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion


class TestMPCControllerNode(unittest.TestCase):
    """MPC Controller Node 테스트 클래스."""

    @classmethod
    def setUpClass(cls):
        """테스트 클래스 초기화."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 정리."""
        rclpy.shutdown()

    def setUp(self):
        """각 테스트 전 초기화."""
        # MPC 컨트롤러 모킹
        self.mock_mpc_controller = MagicMock()
        self.mock_mpc_controller.params.dt = 0.1
        self.mock_mpc_controller.params.N = 20

    def test_odom_to_state(self):
        """Odometry 메시지를 상태로 변환하는 테스트."""
        with patch('mpc_controller.ros2.mpc_controller_node.MPCController',
                   return_value=self.mock_mpc_controller):
            from mpc_controller.ros2.mpc_controller_node import MPCControllerNode

            node = MPCControllerNode()

            # 테스트용 Odometry 메시지 생성
            odom = Odometry()
            odom.pose.pose.position.x = 1.0
            odom.pose.pose.position.y = 2.0
            odom.pose.pose.position.z = 0.0

            # theta = pi/4 (45도)를 쿼터니언으로 표현
            theta = np.pi / 4
            odom.pose.pose.orientation.w = np.cos(theta / 2.0)
            odom.pose.pose.orientation.x = 0.0
            odom.pose.pose.orientation.y = 0.0
            odom.pose.pose.orientation.z = np.sin(theta / 2.0)

            # 변환 테스트
            state = node._odom_to_state(odom)

            # 검증
            self.assertAlmostEqual(state[0], 1.0, places=5)
            self.assertAlmostEqual(state[1], 2.0, places=5)
            self.assertAlmostEqual(state[2], np.pi / 4, places=5)

            node.destroy_node()

    def test_path_to_trajectory(self):
        """Path 메시지를 궤적으로 변환하는 테스트."""
        with patch('mpc_controller.ros2.mpc_controller_node.MPCController',
                   return_value=self.mock_mpc_controller):
            from mpc_controller.ros2.mpc_controller_node import MPCControllerNode

            node = MPCControllerNode()

            # 테스트용 Path 메시지 생성
            path = Path()

            for i in range(5):
                pose_stamped = PoseStamped()
                pose_stamped.pose.position.x = float(i)
                pose_stamped.pose.position.y = float(i * 2)

                theta = np.pi / 6 * i
                pose_stamped.pose.orientation.w = np.cos(theta / 2.0)
                pose_stamped.pose.orientation.z = np.sin(theta / 2.0)

                path.poses.append(pose_stamped)

            # 변환 테스트
            trajectory = node._path_to_trajectory(path)

            # 검증
            self.assertEqual(len(trajectory), 5)
            self.assertEqual(trajectory.shape[1], 3)  # [x, y, theta]

            for i in range(5):
                self.assertAlmostEqual(trajectory[i, 0], float(i), places=5)
                self.assertAlmostEqual(trajectory[i, 1], float(i * 2), places=5)

            node.destroy_node()

    def test_publish_control(self):
        """제어 명령 발행 테스트."""
        with patch('mpc_controller.ros2.mpc_controller_node.MPCController',
                   return_value=self.mock_mpc_controller):
            from mpc_controller.ros2.mpc_controller_node import MPCControllerNode

            node = MPCControllerNode()

            # 제어 입력
            control = np.array([0.5, 0.3])  # [v, omega]

            # Mock publisher
            published_msg = []

            def mock_publish(msg):
                published_msg.append(msg)

            node.cmd_vel_pub.publish = mock_publish

            # 발행 테스트
            node._publish_control(control)

            # 검증
            self.assertEqual(len(published_msg), 1)
            msg = published_msg[0]
            self.assertIsInstance(msg, Twist)
            self.assertAlmostEqual(msg.linear.x, 0.5, places=5)
            self.assertAlmostEqual(msg.angular.z, 0.3, places=5)

            node.destroy_node()

    def test_control_loop_no_data(self):
        """데이터가 없을 때 제어 루프 테스트."""
        with patch('mpc_controller.ros2.mpc_controller_node.MPCController',
                   return_value=self.mock_mpc_controller):
            from mpc_controller.ros2.mpc_controller_node import MPCControllerNode

            node = MPCControllerNode()

            # 데이터 없이 control_loop 호출 (경고만 출력되고 에러 없어야 함)
            try:
                node.control_loop()
            except Exception as e:
                self.fail(f"control_loop raised exception: {e}")

            node.destroy_node()

    def test_control_loop_with_data(self):
        """데이터가 있을 때 제어 루프 테스트."""
        # MPC 컨트롤러 모킹
        mock_control = np.array([0.5, 0.2])
        mock_info = {
            'predicted_trajectory': np.array([[0, 0, 0], [0.1, 0.1, 0.1]]),
            'predicted_controls': np.array([[0.5, 0.2]]),
            'cost': 10.5,
            'solve_time': 0.015,
            'solver_status': 'Solve_Succeeded',
            'soft_constraints': {}
        }
        self.mock_mpc_controller.compute_control.return_value = (mock_control, mock_info)

        with patch('mpc_controller.ros2.mpc_controller_node.MPCController',
                   return_value=self.mock_mpc_controller):
            from mpc_controller.ros2.mpc_controller_node import MPCControllerNode

            node = MPCControllerNode()

            # Odometry 설정
            odom = Odometry()
            odom.pose.pose.position.x = 0.0
            odom.pose.pose.position.y = 0.0
            odom.pose.pose.orientation.w = 1.0
            node.current_odom = odom

            # Path 설정
            path = Path()
            for i in range(25):  # N+1 보다 많은 포인트
                pose_stamped = PoseStamped()
                pose_stamped.pose.position.x = float(i) * 0.1
                pose_stamped.pose.position.y = 0.0
                pose_stamped.pose.orientation.w = 1.0
                path.poses.append(pose_stamped)
            node.reference_path = path

            # Mock publishers
            published_cmd_vel = []
            published_traj = []

            def mock_cmd_vel_publish(msg):
                published_cmd_vel.append(msg)

            def mock_traj_publish(msg):
                published_traj.append(msg)

            node.cmd_vel_pub.publish = mock_cmd_vel_publish
            node.predicted_traj_pub.publish = mock_traj_publish

            # 제어 루프 실행
            try:
                node.control_loop()
            except Exception as e:
                self.fail(f"control_loop raised exception: {e}")

            # 검증: MPC가 호출되었는지
            self.assertTrue(self.mock_mpc_controller.compute_control.called)

            # 검증: cmd_vel이 발행되었는지
            self.assertEqual(len(published_cmd_vel), 1)
            self.assertAlmostEqual(published_cmd_vel[0].linear.x, 0.5, places=5)

            # 검증: 예측 궤적이 발행되었는지
            self.assertEqual(len(published_traj), 1)

            node.destroy_node()


if __name__ == '__main__':
    unittest.main()
