#!/usr/bin/env python3
"""SafetyMonitorNode 테스트 (13개 테스트)

테스트 항목:
  - 초기화: 노드 생성, 커스텀 파라미터
  - E-Stop 서비스: 활성화, 해제, zero Twist 발행
  - Watchdog: cmd_vel timeout, joint_state timeout, 타이머 리셋
  - 속도 제한: 정상 범위, 초과 시 E-Stop
  - Diagnostics: 발행 확인, OK/ERROR 상태
"""

import math
import time
import pytest

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus

from mpc_controller_ros2.safety_monitor import SafetyMonitorNode


@pytest.fixture(scope='module')
def rclpy_init():
    """모듈 레벨 rclpy 초기화/종료."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def safety_monitor(rclpy_init):
    """SafetyMonitorNode 인스턴스 생성."""
    node = SafetyMonitorNode()
    yield node
    node.destroy_node()


@pytest.fixture
def test_node(rclpy_init):
    """테스트용 보조 노드."""
    node = Node('test_helper_node')
    yield node
    node.destroy_node()


def spin_for(executor, duration_s=0.1, step=0.01):
    """executor를 duration_s 동안 spin."""
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        executor.spin_once(timeout_sec=step)


# ============================================================
# 초기화 테스트
# ============================================================

class TestNodeCreation:
    def test_node_creation(self, safety_monitor):
        """노드 생성 + 기본 파라미터 확인."""
        assert safety_monitor.get_name() == 'safety_monitor'
        assert safety_monitor.cmd_vel_timeout == 0.5
        assert safety_monitor.joint_state_timeout == 1.0
        assert safety_monitor.max_linear_speed == 2.0
        assert safety_monitor.max_angular_speed == 3.0
        assert safety_monitor.monitor_rate == 20.0
        assert safety_monitor.enable_speed_limit is True

    def test_custom_parameters(self, rclpy_init):
        """커스텀 파라미터 반영 확인."""
        node = SafetyMonitorNode(
            parameter_overrides=[
                rclpy.parameter.Parameter('cmd_vel_timeout', value=1.0),
                rclpy.parameter.Parameter('max_linear_speed', value=5.0),
                rclpy.parameter.Parameter('monitor_rate', value=10.0),
            ]
        )
        assert node.cmd_vel_timeout == 1.0
        assert node.max_linear_speed == 5.0
        assert node.monitor_rate == 10.0
        node.destroy_node()


# ============================================================
# E-Stop 서비스 테스트
# ============================================================

class TestEStop:
    def test_estop_activate(self, safety_monitor):
        """E-Stop 활성화."""
        req = SetBool.Request()
        req.data = True
        resp = SetBool.Response()
        result = safety_monitor._estop_callback(req, resp)
        assert result.success is True
        assert safety_monitor.estop_active is True

    def test_estop_release(self, safety_monitor):
        """E-Stop 해제."""
        # 먼저 활성화
        safety_monitor.estop_active = True

        req = SetBool.Request()
        req.data = False
        resp = SetBool.Response()
        result = safety_monitor._estop_callback(req, resp)
        assert result.success is True
        assert safety_monitor.estop_active is False

    def test_estop_publishes_zero_cmd(self, safety_monitor, test_node):
        """E-Stop 시 zero Twist 발행 확인."""
        received_msgs = []

        def _cb(msg):
            received_msgs.append(msg)

        sub = test_node.create_subscription(Twist, '/cmd_vel_nav', _cb, 10)
        executor = SingleThreadedExecutor()
        executor.add_node(safety_monitor)
        executor.add_node(test_node)

        # E-Stop 활성화
        safety_monitor.estop_active = True
        safety_monitor._publish_stop()
        spin_for(executor, 0.2)

        assert len(received_msgs) > 0
        last = received_msgs[-1]
        assert last.linear.x == 0.0
        assert last.linear.y == 0.0
        assert last.angular.z == 0.0

        test_node.destroy_subscription(sub)
        executor.shutdown()


# ============================================================
# Watchdog 테스트
# ============================================================

class TestWatchdog:
    def test_cmd_vel_timeout_triggers_stop(self, safety_monitor, test_node):
        """cmd_vel timeout → 정지 명령."""
        received_msgs = []
        sub = test_node.create_subscription(
            Twist, '/cmd_vel_nav', lambda m: received_msgs.append(m), 10)
        executor = SingleThreadedExecutor()
        executor.add_node(safety_monitor)
        executor.add_node(test_node)

        # cmd_vel 수신 마크 (과거 시간으로 설정하여 timeout 유발)
        safety_monitor._cmd_vel_received = True
        # 과거 시간으로 설정 (1초 전)
        safety_monitor._last_cmd_vel_time = (
            safety_monitor.get_clock().now()
            - rclpy.duration.Duration(seconds=1.0)
        )

        # 모니터 콜백 강제 실행
        safety_monitor._monitor_callback()
        spin_for(executor, 0.2)

        # 정지 명령이 발행되어야 함
        assert len(received_msgs) > 0

        test_node.destroy_subscription(sub)
        executor.shutdown()

    def test_joint_state_timeout_triggers_estop(self, safety_monitor):
        """joint_state timeout → E-Stop 자동 활성화."""
        safety_monitor._joint_state_received = True
        safety_monitor._last_joint_state_time = (
            safety_monitor.get_clock().now()
            - rclpy.duration.Duration(seconds=2.0)
        )

        safety_monitor._monitor_callback()
        assert safety_monitor.estop_active is True

    def test_cmd_vel_received_resets_timer(self, safety_monitor):
        """cmd_vel 수신 시 타이머 리셋 확인."""
        old_time = safety_monitor._last_cmd_vel_time
        time.sleep(0.05)

        msg = Twist()
        safety_monitor._cmd_vel_callback(msg)

        new_time = safety_monitor._last_cmd_vel_time
        assert new_time > old_time
        assert safety_monitor._cmd_vel_received is True


# ============================================================
# 속도 제한 테스트
# ============================================================

class TestSpeedLimit:
    def test_speed_limit_normal(self, safety_monitor):
        """정상 범위 속도 → E-Stop 미발생."""
        safety_monitor._current_linear_speed = 1.0
        safety_monitor._current_angular_speed = 1.0
        safety_monitor._estop_active = False

        safety_monitor._monitor_callback()
        assert safety_monitor.estop_active is False

    def test_linear_speed_exceeded_triggers_estop(self, safety_monitor):
        """선속도 초과 시 E-Stop 활성화."""
        safety_monitor._current_linear_speed = 3.0  # > max_linear_speed (2.0)
        safety_monitor._estop_active = False

        safety_monitor._monitor_callback()
        assert safety_monitor.estop_active is True

    def test_angular_speed_exceeded_triggers_estop(self, rclpy_init):
        """각속도 초과 시 E-Stop 활성화."""
        node = SafetyMonitorNode()
        node._current_angular_speed = 4.0  # > max_angular_speed (3.0)
        node._estop_active = False

        node._monitor_callback()
        assert node.estop_active is True
        node.destroy_node()


# ============================================================
# Diagnostics 테스트
# ============================================================

class TestDiagnostics:
    def test_diagnostics_published(self, safety_monitor, test_node):
        """DiagnosticArray 발행 확인."""
        received = []
        sub = test_node.create_subscription(
            DiagnosticArray, '/diagnostics', lambda m: received.append(m), 10)
        executor = SingleThreadedExecutor()
        executor.add_node(safety_monitor)
        executor.add_node(test_node)

        safety_monitor._publish_diagnostics(DiagnosticStatus.OK, '정상')
        spin_for(executor, 0.2)

        assert len(received) > 0
        assert received[-1].status[0].name == 'SafetyMonitor'

        test_node.destroy_subscription(sub)
        executor.shutdown()

    def test_diagnostics_ok_state(self, safety_monitor, test_node):
        """정상 상태 → DiagnosticStatus.OK."""
        received = []
        sub = test_node.create_subscription(
            DiagnosticArray, '/diagnostics', lambda m: received.append(m), 10)
        executor = SingleThreadedExecutor()
        executor.add_node(safety_monitor)
        executor.add_node(test_node)

        safety_monitor._publish_diagnostics(DiagnosticStatus.OK, '정상')
        spin_for(executor, 0.2)

        assert len(received) > 0
        assert received[-1].status[0].level == DiagnosticStatus.OK

        test_node.destroy_subscription(sub)
        executor.shutdown()

    def test_diagnostics_error_state(self, safety_monitor, test_node):
        """E-Stop → DiagnosticStatus.ERROR."""
        received = []
        sub = test_node.create_subscription(
            DiagnosticArray, '/diagnostics', lambda m: received.append(m), 10)
        executor = SingleThreadedExecutor()
        executor.add_node(safety_monitor)
        executor.add_node(test_node)

        safety_monitor._publish_diagnostics(DiagnosticStatus.ERROR, 'E-Stop 활성')
        spin_for(executor, 0.2)

        assert len(received) > 0
        # 타이머 콜백이 OK를 발행할 수 있으므로, ERROR 메시지가 포함되어 있는지 확인
        has_error = any(
            msg.status[0].level == DiagnosticStatus.ERROR
            for msg in received if msg.status
        )
        assert has_error, f'ERROR diagnostics not found in {len(received)} messages'

        test_node.destroy_subscription(sub)
        executor.shutdown()
