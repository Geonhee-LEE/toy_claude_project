#!/usr/bin/env python3
"""
SafetyMonitorNode - 실제 로봇 필수 안전 레이어

이중 안전 메커니즘:
  SafetyMonitor timeout → zero cmd_vel 발행
  SwerveKinematics timeout → 모터 직접 정지

┌─────────────────────────────────────────┐
│  SafetyMonitorNode 기능                  │
├─────────────────────────────────────────┤
│  1. cmd_vel Watchdog (0.5s timeout)     │
│  2. JointState 통신 감시 (1.0s timeout) │
│  3. 속도 이상 감지 (max_linear_speed)   │
│  4. /emergency_stop 서비스 (SetBool)    │
│  5. /diagnostics 발행 (DiagnosticArray) │
└─────────────────────────────────────────┘

실행:
    ros2 run mpc_controller_ros2 safety_monitor.py
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue


class SafetyMonitorNode(Node):
    """실제 로봇 하드웨어 안전 모니터."""

    def __init__(self, **kwargs):
        super().__init__('safety_monitor', **kwargs)

        # === 파라미터 ===
        self.declare_parameter('cmd_vel_timeout', 0.5)
        self.declare_parameter('joint_state_timeout', 1.0)
        self.declare_parameter('max_linear_speed', 2.0)
        self.declare_parameter('max_angular_speed', 3.0)
        self.declare_parameter('monitor_rate', 20.0)
        self.declare_parameter('enable_speed_limit', True)

        self.cmd_vel_timeout = self.get_parameter('cmd_vel_timeout').value
        self.joint_state_timeout = self.get_parameter('joint_state_timeout').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.monitor_rate = self.get_parameter('monitor_rate').value
        self.enable_speed_limit = self.get_parameter('enable_speed_limit').value

        # === 상태 변수 ===
        self._estop_active = False
        self._last_cmd_vel_time = self.get_clock().now()
        self._last_joint_state_time = self.get_clock().now()
        self._cmd_vel_received = False
        self._joint_state_received = False
        self._current_linear_speed = 0.0
        self._current_angular_speed = 0.0

        # === 서브스크립션 ===
        self._cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel_nav', self._cmd_vel_callback, 10)
        self._joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, 10)
        self._odom_sub = self.create_subscription(
            Odometry, '/swerve_controller/odom', self._odom_callback, 10)

        # === 퍼블리셔 ===
        self._stop_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        self._diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)

        # === 서비스 ===
        self._estop_srv = self.create_service(
            SetBool, '/emergency_stop', self._estop_callback)

        # === 모니터 타이머 ===
        timer_period = 1.0 / self.monitor_rate
        self._monitor_timer = self.create_timer(timer_period, self._monitor_callback)

        self.get_logger().info(
            f'SafetyMonitor 시작 — '
            f'cmd_vel_timeout={self.cmd_vel_timeout}s, '
            f'joint_state_timeout={self.joint_state_timeout}s, '
            f'max_linear={self.max_linear_speed}m/s, '
            f'max_angular={self.max_angular_speed}rad/s'
        )

    # ──────────── 콜백 ────────────

    def _cmd_vel_callback(self, msg: Twist):
        """cmd_vel 수신 시 watchdog 타이머 리셋."""
        self._last_cmd_vel_time = self.get_clock().now()
        self._cmd_vel_received = True

    def _joint_state_callback(self, msg: JointState):
        """JointState 수신 시 통신 상태 업데이트."""
        self._last_joint_state_time = self.get_clock().now()
        self._joint_state_received = True

    def _odom_callback(self, msg: Odometry):
        """Odometry에서 현재 속도 추출."""
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self._current_linear_speed = math.sqrt(vx * vx + vy * vy)
        self._current_angular_speed = abs(msg.twist.twist.angular.z)

    def _estop_callback(self, request: SetBool.Request,
                        response: SetBool.Response) -> SetBool.Response:
        """Emergency Stop 서비스 핸들러."""
        if request.data:
            self._estop_active = True
            self._publish_stop()
            response.success = True
            response.message = 'E-Stop 활성화 — 로봇 정지'
            self.get_logger().warn('E-Stop 활성화')
        else:
            self._estop_active = False
            response.success = True
            response.message = 'E-Stop 해제 — 로봇 제어 복구'
            self.get_logger().info('E-Stop 해제')
        return response

    # ──────────── 모니터 루프 ────────────

    def _monitor_callback(self):
        """주기적 안전 검사 (monitor_rate Hz)."""
        now = self.get_clock().now()
        status = DiagnosticStatus.OK
        messages = []

        # 1. E-Stop 활성 상태면 정지 명령 지속
        if self._estop_active:
            self._publish_stop()
            self._publish_diagnostics(DiagnosticStatus.ERROR, 'E-Stop 활성')
            return

        # 2. cmd_vel watchdog
        if self._cmd_vel_received:
            cmd_vel_elapsed = (now - self._last_cmd_vel_time).nanoseconds / 1e9
            if cmd_vel_elapsed > self.cmd_vel_timeout:
                self._publish_stop()
                status = DiagnosticStatus.WARN
                messages.append(
                    f'cmd_vel timeout ({cmd_vel_elapsed:.2f}s > {self.cmd_vel_timeout}s)')
                self.get_logger().warn(
                    f'cmd_vel watchdog: {cmd_vel_elapsed:.2f}s 동안 미수신 → 정지 명령')

        # 3. JointState 통신 감시
        if self._joint_state_received:
            js_elapsed = (now - self._last_joint_state_time).nanoseconds / 1e9
            if js_elapsed > self.joint_state_timeout:
                self._estop_active = True
                self._publish_stop()
                status = DiagnosticStatus.ERROR
                messages.append(
                    f'JointState timeout ({js_elapsed:.2f}s > '
                    f'{self.joint_state_timeout}s) → E-Stop')
                self.get_logger().error(
                    f'JointState 통신 단절: {js_elapsed:.2f}s → E-Stop 자동 활성화')

        # 4. 속도 이상 감지
        if self.enable_speed_limit:
            if self._current_linear_speed > self.max_linear_speed:
                self._estop_active = True
                self._publish_stop()
                status = DiagnosticStatus.ERROR
                messages.append(
                    f'선속도 초과 ({self._current_linear_speed:.2f} > '
                    f'{self.max_linear_speed}) → E-Stop')
                self.get_logger().error(
                    f'선속도 초과: {self._current_linear_speed:.2f}m/s → E-Stop')

            if self._current_angular_speed > self.max_angular_speed:
                self._estop_active = True
                self._publish_stop()
                status = DiagnosticStatus.ERROR
                messages.append(
                    f'각속도 초과 ({self._current_angular_speed:.2f} > '
                    f'{self.max_angular_speed}) → E-Stop')
                self.get_logger().error(
                    f'각속도 초과: {self._current_angular_speed:.2f}rad/s → E-Stop')

        # 5. Diagnostics 발행
        msg_str = '; '.join(messages) if messages else '정상'
        self._publish_diagnostics(status, msg_str)

    # ──────────── 유틸리티 ────────────

    def _publish_stop(self):
        """zero Twist 발행 (로봇 정지)."""
        self._stop_pub.publish(Twist())

    def _publish_diagnostics(self, level: int, message: str):
        """DiagnosticArray 발행."""
        diag = DiagnosticArray()
        diag.header.stamp = self.get_clock().now().to_msg()

        status = DiagnosticStatus()
        status.level = level
        status.name = 'SafetyMonitor'
        status.message = message
        status.hardware_id = 'swerve_robot'
        status.values = [
            KeyValue(key='estop_active', value=str(self._estop_active)),
            KeyValue(key='cmd_vel_received', value=str(self._cmd_vel_received)),
            KeyValue(key='joint_state_received', value=str(self._joint_state_received)),
            KeyValue(key='linear_speed', value=f'{self._current_linear_speed:.3f}'),
            KeyValue(key='angular_speed', value=f'{self._current_angular_speed:.3f}'),
        ]

        diag.status.append(status)
        self._diag_pub.publish(diag)

    @property
    def estop_active(self) -> bool:
        """E-Stop 활성 상태 조회 (테스트용)."""
        return self._estop_active

    @estop_active.setter
    def estop_active(self, value: bool):
        """E-Stop 상태 설정 (테스트용)."""
        self._estop_active = value


def main(args=None):
    rclpy.init(args=args)
    node = SafetyMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
