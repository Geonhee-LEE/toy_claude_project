#!/usr/bin/env python3
"""
로봇이 실제로 움직이는지 검증하는 스크립트
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
import math


class MotionVerifier(Node):
    def __init__(self):
        super().__init__('motion_verifier')

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.current_odom = None
        self.initial_pos = None

    def odom_callback(self, msg):
        self.current_odom = msg

    def wait_for_odom(self, timeout=5.0):
        """Odometry 수신 대기"""
        start = time.time()
        while self.current_odom is None and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.current_odom is not None

    def get_position(self):
        """현재 위치 반환"""
        if self.current_odom is None:
            return None
        pos = self.current_odom.pose.pose.position
        return (pos.x, pos.y, pos.z)

    def send_cmd_vel(self, linear_x, angular_z, duration):
        """속도 명령 전송"""
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z

        start_time = time.time()
        rate = self.create_rate(20)  # 20 Hz

        while (time.time() - start_time) < duration:
            self.cmd_vel_pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.0)
            rate.sleep()

        # 정지
        self.stop()

    def stop(self):
        """로봇 정지"""
        cmd = Twist()
        for _ in range(10):
            self.cmd_vel_pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.05)

    def calculate_distance(self, pos1, pos2):
        """두 위치 간 거리 계산"""
        if pos1 is None or pos2 is None:
            return None
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return math.sqrt(dx**2 + dy**2)

    def run_test(self):
        """모션 테스트 실행"""
        self.get_logger().info('='*50)
        self.get_logger().info('로봇 모션 검증 테스트 시작')
        self.get_logger().info('='*50)

        # Odometry 대기
        self.get_logger().info('Odometry 대기 중...')
        if not self.wait_for_odom(timeout=10.0):
            self.get_logger().error('Odometry 수신 실패!')
            return False

        # 초기 위치 저장
        self.initial_pos = self.get_position()
        self.get_logger().info(f'초기 위치: x={self.initial_pos[0]:.4f}, y={self.initial_pos[1]:.4f}')

        time.sleep(1.0)

        # 테스트 1: 전진
        self.get_logger().info('\n[테스트 1] 전진 (0.5 m/s, 3초)')
        self.send_cmd_vel(0.5, 0.0, 3.0)
        time.sleep(0.5)

        pos_after_forward = self.get_position()
        distance_moved = self.calculate_distance(self.initial_pos, pos_after_forward)

        self.get_logger().info(f'전진 후 위치: x={pos_after_forward[0]:.4f}, y={pos_after_forward[1]:.4f}')
        self.get_logger().info(f'이동 거리: {distance_moved:.4f} m')

        if distance_moved > 0.5:  # 최소 0.5m 이동 예상
            self.get_logger().info('✓ 전진 테스트 통과!')
            result = True
        else:
            self.get_logger().warn(f'✗ 전진 테스트 실패! (이동 거리: {distance_moved:.4f}m < 0.5m)')
            self.get_logger().warn('  → 바퀴가 헛돌고 있거나 마찰이 부족합니다.')
            result = False

        time.sleep(1.0)

        # 테스트 2: 회전
        self.get_logger().info('\n[테스트 2] 제자리 회전 (0.5 rad/s, 2초)')
        pos_before_rot = self.get_position()
        self.send_cmd_vel(0.0, 0.5, 2.0)
        time.sleep(0.5)

        pos_after_rot = self.get_position()
        rot_distance = self.calculate_distance(pos_before_rot, pos_after_rot)

        self.get_logger().info(f'회전 중 이동 거리: {rot_distance:.4f} m')

        if rot_distance < 0.2:  # 제자리 회전이므로 0.2m 이내 이동
            self.get_logger().info('✓ 회전 테스트 통과!')
        else:
            self.get_logger().warn(f'✗ 회전 테스트 실패! (이동 거리: {rot_distance:.4f}m > 0.2m)')
            self.get_logger().warn('  → 회전 중 미끄러지고 있습니다.')
            result = False

        # 최종 결과
        self.get_logger().info('\n' + '='*50)
        if result:
            self.get_logger().info('✓ 로봇이 정상적으로 움직입니다!')
        else:
            self.get_logger().warn('✗ 로봇 모션에 문제가 있습니다.')
            self.get_logger().warn('다음을 확인하세요:')
            self.get_logger().warn('  1. 바퀴 마찰 계수 (mu, mu2)')
            self.get_logger().warn('  2. 캐스터 높이 및 마찰')
            self.get_logger().warn('  3. 로봇 무게 및 관성')
            self.get_logger().warn('  4. 바퀴-지면 contact 설정')
        self.get_logger().info('='*50)

        return result


def main(args=None):
    rclpy.init(args=args)

    verifier = MotionVerifier()

    try:
        success = verifier.run_test()
        if not success:
            verifier.get_logger().info('\nGazebo GUI에서 로봇을 직접 확인하세요.')
    except KeyboardInterrupt:
        verifier.get_logger().info('테스트 중단')
    finally:
        verifier.stop()
        verifier.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
