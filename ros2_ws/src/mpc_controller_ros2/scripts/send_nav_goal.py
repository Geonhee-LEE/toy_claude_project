#!/usr/bin/env python3
"""
nav2 Goal 전송 스크립트

사용 방법:
    # 기본 목표 (5m 전진)
    ros2 run mpc_controller_ros2 send_nav_goal.py

    # 커스텀 목표
    ros2 run mpc_controller_ros2 send_nav_goal.py --x 8.0 --y 2.0 --yaw 1.57
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
import argparse
import math


class Nav2GoalSender(Node):
    def __init__(self):
        super().__init__('nav2_goal_sender')

        self._action_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        self.get_logger().info('nav2 Goal Sender 초기화 완료')

    def send_goal(self, x, y, yaw):
        """
        목표 위치로 이동 명령 전송

        Args:
            x: 목표 x 좌표 (m)
            y: 목표 y 좌표 (m)
            yaw: 목표 방향 (rad)
        """
        self.get_logger().info('nav2 action server 대기 중...')

        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('nav2 action server를 찾을 수 없습니다!')
            self.get_logger().error('다음을 확인하세요:')
            self.get_logger().error('  1. nav2 스택이 실행 중인지')
            self.get_logger().error('  2. bt_navigator 노드가 활성화되었는지')
            return False

        # Goal message 생성
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # Yaw를 quaternion으로 변환
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f'목표 전송: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} rad')

        # Goal 전송
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal이 거부되었습니다!')
            return False

        self.get_logger().info('Goal이 수락되었습니다. 로봇이 이동 중...')

        # 결과 대기
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        status = result_future.result().status

        if status == 4:  # SUCCEEDED
            self.get_logger().info('✓ 목표 도착 완료!')
            return True
        else:
            self.get_logger().warn(f'✗ 목표 도달 실패 (status: {status})')
            return False

    def feedback_callback(self, feedback_msg):
        """주행 중 피드백 출력"""
        feedback = feedback_msg.feedback
        current_pose = feedback.current_pose.pose

        distance_remaining = feedback.distance_remaining
        eta = feedback.estimated_time_remaining

        self.get_logger().info(
            f'진행 중... 남은 거리: {distance_remaining:.2f}m, '
            f'예상 시간: {eta.sec}초',
            throttle_duration_sec=2.0  # 2초마다 출력
        )


def main():
    parser = argparse.ArgumentParser(description='nav2 목표 전송')
    parser.add_argument('--x', type=float, default=5.0, help='목표 x 좌표 (m)')
    parser.add_argument('--y', type=float, default=0.0, help='목표 y 좌표 (m)')
    parser.add_argument('--yaw', type=float, default=0.0, help='목표 방향 (rad)')

    args = parser.parse_args()

    rclpy.init()

    goal_sender = Nav2GoalSender()

    try:
        success = goal_sender.send_goal(args.x, args.y, args.yaw)

        if success:
            goal_sender.get_logger().info('='*50)
            goal_sender.get_logger().info('테스트 완료!')
            goal_sender.get_logger().info('='*50)

    except KeyboardInterrupt:
        goal_sender.get_logger().info('사용자 중단')
    finally:
        goal_sender.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
