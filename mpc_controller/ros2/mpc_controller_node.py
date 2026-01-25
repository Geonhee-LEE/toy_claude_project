"""
MPC Controller ROS2 Node Wrapper.

ROS2 노드로 구현된 MPC 컨트롤러.
Odometry와 참조 경로를 구독하여 최적 제어 명령을 발행합니다.
"""

from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray

from mpc_controller.controllers.mpc import MPCController, MPCParams
from mpc_controller.models.differential_drive import RobotParams


class MPCControllerNode(Node):
    """
    MPC 컨트롤러 ROS2 노드.

    구독:
        - /odom (nav_msgs/Odometry): 현재 로봇 위치
        - /reference_path (nav_msgs/Path): 참조 경로

    발행:
        - /cmd_vel (geometry_msgs/Twist): 제어 명령
        - /predicted_trajectory (nav_msgs/Path): 예측 궤적
        - /mpc_markers (visualization_msgs/MarkerArray): RVIZ 시각화 마커

    파라미터:
        MPC 관련:
            - mpc.N (int): 예측 구간 길이
            - mpc.dt (float): 시간 간격
            - mpc.Q (list[float]): 상태 가중치 [x, y, theta]
            - mpc.R (list[float]): 제어 가중치 [v, omega]
            - mpc.Qf (list[float]): 종료 상태 가중치
            - mpc.Rd (list[float]): 제어 변화율 가중치

        로봇 관련:
            - robot.max_velocity (float): 최대 선속도
            - robot.max_omega (float): 최대 각속도
            - robot.wheel_base (float): 휠베이스
    """

    def __init__(self):
        super().__init__('mpc_controller_node')

        # 파라미터 선언 및 로드
        self._declare_parameters()
        robot_params = self._load_robot_params()
        mpc_params = self._load_mpc_params()

        # MPC 컨트롤러 초기화
        self.mpc_controller = MPCController(
            robot_params=robot_params,
            mpc_params=mpc_params,
            enable_soft_constraints=True
        )

        # 상태 변수
        self.current_odom: Optional[Odometry] = None
        self.reference_path: Optional[Path] = None
        self.last_control_time = self.get_clock().now()

        # QoS 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 구독자 설정
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile
        )

        self.path_sub = self.create_subscription(
            Path,
            '/reference_path',
            self.path_callback,
            qos_profile
        )

        # 발행자 설정
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            qos_profile
        )

        self.predicted_traj_pub = self.create_publisher(
            Path,
            '/predicted_trajectory',
            qos_profile
        )

        self.markers_pub = self.create_publisher(
            MarkerArray,
            '/mpc_markers',
            qos_profile
        )

        # 제어 루프 타이머 (MPC dt와 동일하게 설정)
        self.control_timer = self.create_timer(
            mpc_params.dt,
            self.control_loop
        )

        self.get_logger().info('MPC Controller Node 초기화 완료')
        self.get_logger().info(f'예측 구간: {mpc_params.N}, 시간 간격: {mpc_params.dt}s')

    def _declare_parameters(self):
        """ROS2 파라미터 선언."""
        # MPC 파라미터
        self.declare_parameter('mpc.N', 20)
        self.declare_parameter('mpc.dt', 0.1)
        self.declare_parameter('mpc.Q', [10.0, 10.0, 1.0])
        self.declare_parameter('mpc.R', [0.1, 0.1])
        self.declare_parameter('mpc.Qf', [100.0, 100.0, 10.0])
        self.declare_parameter('mpc.Rd', [0.5, 0.5])

        # 로봇 파라미터
        self.declare_parameter('robot.max_velocity', 1.0)
        self.declare_parameter('robot.max_omega', 1.5)
        self.declare_parameter('robot.wheel_base', 0.5)
        self.declare_parameter('robot.max_acceleration', 0.5)
        self.declare_parameter('robot.max_alpha', 1.0)

    def _load_robot_params(self) -> RobotParams:
        """ROS2 파라미터에서 로봇 파라미터 로드."""
        return RobotParams(
            wheel_base=self.get_parameter('robot.wheel_base').value,
            max_velocity=self.get_parameter('robot.max_velocity').value,
            max_omega=self.get_parameter('robot.max_omega').value,
            max_acceleration=self.get_parameter('robot.max_acceleration').value,
            max_alpha=self.get_parameter('robot.max_alpha').value
        )

    def _load_mpc_params(self) -> MPCParams:
        """ROS2 파라미터에서 MPC 파라미터 로드."""
        Q = np.diag(self.get_parameter('mpc.Q').value)
        R = np.diag(self.get_parameter('mpc.R').value)
        Qf = np.diag(self.get_parameter('mpc.Qf').value)
        Rd = np.diag(self.get_parameter('mpc.Rd').value)

        return MPCParams(
            N=self.get_parameter('mpc.N').value,
            dt=self.get_parameter('mpc.dt').value,
            Q=Q,
            R=R,
            Qf=Qf,
            Rd=Rd
        )

    def odom_callback(self, msg: Odometry):
        """Odometry 메시지 콜백."""
        self.current_odom = msg

    def path_callback(self, msg: Path):
        """참조 경로 메시지 콜백."""
        self.reference_path = msg
        self.get_logger().info(f'참조 경로 수신: {len(msg.poses)}개 포인트')

    def control_loop(self):
        """
        주기적 제어 루프.

        Odometry와 참조 경로를 기반으로 MPC를 실행하고
        최적 제어 명령을 발행합니다.
        """
        # 데이터 유효성 확인
        if self.current_odom is None:
            self.get_logger().warn('Odometry 데이터 미수신', throttle_duration_sec=1.0)
            return

        if self.reference_path is None or len(self.reference_path.poses) == 0:
            self.get_logger().warn('참조 경로 미수신', throttle_duration_sec=1.0)
            return

        # 현재 상태 추출 [x, y, theta]
        current_state = self._odom_to_state(self.current_odom)

        # 참조 경로를 MPC 입력 형식으로 변환
        reference_trajectory = self._path_to_trajectory(self.reference_path)

        # 참조 경로가 예측 구간보다 짧으면 경고
        N = self.mpc_controller.params.N
        if len(reference_trajectory) < N + 1:
            self.get_logger().warn(
                f'참조 경로가 너무 짧음 (필요: {N+1}, 현재: {len(reference_trajectory)})',
                throttle_duration_sec=1.0
            )
            # 마지막 점을 반복하여 확장
            last_point = reference_trajectory[-1]
            while len(reference_trajectory) < N + 1:
                reference_trajectory = np.vstack([reference_trajectory, last_point])

        # MPC 계산
        try:
            control, info = self.mpc_controller.compute_control(
                current_state,
                reference_trajectory[:N+1]  # N+1개 포인트만 사용
            )

            # 제어 명령 발행
            self._publish_control(control)

            # 예측 궤적 발행
            self._publish_predicted_trajectory(info['predicted_trajectory'])

            # RVIZ 마커 발행
            self._publish_markers(info)

            # 로깅
            solve_time_ms = info['solve_time'] * 1000
            self.get_logger().info(
                f'MPC 제어: v={control[0]:.3f} m/s, ω={control[1]:.3f} rad/s, '
                f'풀이시간={solve_time_ms:.2f}ms',
                throttle_duration_sec=1.0
            )

        except Exception as e:
            self.get_logger().error(f'MPC 계산 실패: {str(e)}')

    def _odom_to_state(self, odom: Odometry) -> np.ndarray:
        """
        Odometry 메시지를 MPC 상태로 변환.

        Args:
            odom: Odometry 메시지

        Returns:
            상태 [x, y, theta]
        """
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y

        # 쿼터니언을 오일러 각으로 변환
        q = odom.pose.pose.orientation
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        theta = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2)
        )

        return np.array([x, y, theta])

    def _path_to_trajectory(self, path: Path) -> np.ndarray:
        """
        Path 메시지를 MPC 참조 궤적으로 변환.

        Args:
            path: Path 메시지

        Returns:
            참조 궤적 (N, 3) [x, y, theta]
        """
        trajectory = []

        for pose_stamped in path.poses:
            pose = pose_stamped.pose
            x = pose.position.x
            y = pose.position.y

            # 쿼터니언을 오일러 각으로 변환
            q = pose.orientation
            theta = np.arctan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y**2 + q.z**2)
            )

            trajectory.append([x, y, theta])

        return np.array(trajectory)

    def _publish_control(self, control: np.ndarray):
        """
        제어 명령 발행.

        Args:
            control: [v, omega]
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = float(control[0])
        cmd_vel.angular.z = float(control[1])

        self.cmd_vel_pub.publish(cmd_vel)

    def _publish_predicted_trajectory(self, predicted_states: np.ndarray):
        """
        예측 궤적 발행.

        Args:
            predicted_states: 예측된 상태들 (N+1, 3)
        """
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'odom'

        for state in predicted_states:
            pose_stamped = PoseStamped()
            pose_stamped.header = path.header

            pose_stamped.pose.position.x = float(state[0])
            pose_stamped.pose.position.y = float(state[1])
            pose_stamped.pose.position.z = 0.0

            # theta를 쿼터니언으로 변환
            theta = state[2]
            pose_stamped.pose.orientation.w = np.cos(theta / 2.0)
            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = np.sin(theta / 2.0)

            path.poses.append(pose_stamped)

        self.predicted_traj_pub.publish(path)

    def _publish_markers(self, info: dict):
        """
        RVIZ 시각화 마커 발행.

        Args:
            info: MPC 정보 딕셔너리
        """
        marker_array = MarkerArray()

        # 예측 궤적 라인 마커
        traj_marker = Marker()
        traj_marker.header.stamp = self.get_clock().now().to_msg()
        traj_marker.header.frame_id = 'odom'
        traj_marker.ns = 'predicted_trajectory'
        traj_marker.id = 0
        traj_marker.type = Marker.LINE_STRIP
        traj_marker.action = Marker.ADD

        traj_marker.scale.x = 0.05  # 선 두께
        traj_marker.color.r = 0.0
        traj_marker.color.g = 1.0
        traj_marker.color.b = 0.0
        traj_marker.color.a = 0.8

        for state in info['predicted_trajectory']:
            from geometry_msgs.msg import Point
            point = Point()
            point.x = float(state[0])
            point.y = float(state[1])
            point.z = 0.0
            traj_marker.points.append(point)

        marker_array.markers.append(traj_marker)

        # Soft constraint violation 시각화
        soft_info = info.get('soft_constraints', {})
        if soft_info.get('has_violations', False):
            violation_marker = Marker()
            violation_marker.header.stamp = self.get_clock().now().to_msg()
            violation_marker.header.frame_id = 'odom'
            violation_marker.ns = 'constraint_violations'
            violation_marker.id = 1
            violation_marker.type = Marker.TEXT_VIEW_FACING
            violation_marker.action = Marker.ADD

            # 현재 위치에 표시
            if self.current_odom:
                violation_marker.pose.position.x = self.current_odom.pose.pose.position.x
                violation_marker.pose.position.y = self.current_odom.pose.pose.position.y
                violation_marker.pose.position.z = 1.0  # 위쪽에 표시

            violation_marker.scale.z = 0.2  # 텍스트 크기
            violation_marker.color.r = 1.0
            violation_marker.color.g = 0.0
            violation_marker.color.b = 0.0
            violation_marker.color.a = 1.0

            max_vel_viol = soft_info.get('max_velocity_violation', 0.0)
            max_acc_viol = soft_info.get('max_acceleration_violation', 0.0)
            violation_marker.text = f'제약 위반\nVel:{max_vel_viol:.3f}\nAcc:{max_acc_viol:.3f}'

            marker_array.markers.append(violation_marker)

        self.markers_pub.publish(marker_array)


def main(args=None):
    """메인 함수."""
    rclpy.init(args=args)

    node = MPCControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
