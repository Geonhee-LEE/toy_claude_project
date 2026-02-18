#!/usr/bin/env python3
"""
Gazebo differential robot 모션 테스트 스크립트

로봇을 다양한 패턴으로 움직여서 cmd_vel 및 odometry가 정상 작동하는지 확인합니다.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
import math


class RobotMotionTester(Node):
    def __init__(self):
        super().__init__('robot_motion_tester')

        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.current_odom = None
        self.start_position = None

        self.get_logger().info('로봇 모션 테스터 시작')

    def odom_callback(self, msg):
        self.current_odom = msg
        if self.start_position is None:
            self.start_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def send_velocity(self, linear_x, angular_z, duration):
        """
        지정된 속도 명령을 duration 초 동안 전송
        """
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z

        start_time = time.time()
        rate = self.create_rate(20)  # 20 Hz

        while time.time() - start_time < duration:
            self.cmd_vel_pub.publish(cmd)
            rate.sleep()

        # 정지
        self.stop()

    def stop(self):
        """로봇 정지"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        time.sleep(0.5)

    def test_forward(self):
        """전진 테스트"""
        self.get_logger().info('=== 테스트 1: 전진 (0.5 m/s, 3초) ===')
        self.send_velocity(0.5, 0.0, 3.0)
        self.print_odometry()

    def test_backward(self):
        """후진 테스트"""
        self.get_logger().info('=== 테스트 2: 후진 (-0.3 m/s, 2초) ===')
        self.send_velocity(-0.3, 0.0, 2.0)
        self.print_odometry()

    def test_rotate_left(self):
        """좌회전 테스트"""
        self.get_logger().info('=== 테스트 3: 좌회전 (0.5 rad/s, 3초) ===')
        self.send_velocity(0.0, 0.5, 3.0)
        self.print_odometry()

    def test_rotate_right(self):
        """우회전 테스트"""
        self.get_logger().info('=== 테스트 4: 우회전 (-0.5 rad/s, 3초) ===')
        self.send_velocity(0.0, -0.5, 3.0)
        self.print_odometry()

    def test_arc(self):
        """원호 주행 테스트"""
        self.get_logger().info('=== 테스트 5: 원호 주행 (v=0.3, ω=0.3, 5초) ===')
        self.send_velocity(0.3, 0.3, 5.0)
        self.print_odometry()

    def print_odometry(self):
        """현재 odometry 정보 출력"""
        if self.current_odom is None:
            self.get_logger().warn('Odometry 데이터 없음')
            return

        pos = self.current_odom.pose.pose.position
        vel = self.current_odom.twist.twist

        if self.start_position is not None:
            dx = pos.x - self.start_position[0]
            dy = pos.y - self.start_position[1]
            distance = math.sqrt(dx**2 + dy**2)
            self.get_logger().info(f'이동 거리: {distance:.2f} m')

        self.get_logger().info(
            f'위치: x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f}'
        )
        self.get_logger().info(
            f'속도: linear={vel.linear.x:.2f}, angular={vel.angular.z:.2f}'
        )
        self.get_logger().info('---')

    def run_all_tests(self):
        """모든 테스트 실행"""
        self.get_logger().info('Odometry 초기화 대기 중...')

        # Odometry 수신 대기
        while self.current_odom is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('테스트 시작!')
        time.sleep(1.0)

        self.test_forward()
        time.sleep(1.0)

        self.test_backward()
        time.sleep(1.0)

        self.test_rotate_left()
        time.sleep(1.0)

        self.test_rotate_right()
        time.sleep(1.0)

        self.test_arc()

        self.get_logger().info('=== 모든 테스트 완료 ===')
        self.stop()


def main(args=None):
    rclpy.init(args=args)

    tester = RobotMotionTester()

    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        tester.get_logger().info('테스트 중단')
    finally:
        tester.stop()
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
