#!/usr/bin/env python3
"""
보행자 동적 장애물 제어기
Gazebo VelocityControl 플러그인에 cmd_vel 왕복 속도를 퍼블리시하여
보행자가 직선 왕복 운동을 수행하도록 합니다.

사용법:
  ros2 run mpc_controller_ros2 pedestrian_controller.py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math


class PedestrianConfig:
    def __init__(self, name: str, topic: str, axis: str,
                 speed: float, period: float):
        self.name = name
        self.topic = topic
        self.axis = axis      # 'x' or 'y'
        self.speed = speed    # m/s
        self.period = period  # 왕복 주기 (초)


class PedestrianController(Node):
    def __init__(self):
        super().__init__('pedestrian_controller')

        self.pedestrians = [
            PedestrianConfig(
                'ped1', '/model/pedestrian_1/cmd_vel',
                'y', 0.4, 8.0),
            PedestrianConfig(
                'ped2', '/model/pedestrian_2/cmd_vel',
                'x', 0.6, 6.0),
            PedestrianConfig(
                'ped3', '/model/pedestrian_3/cmd_vel',
                'y', 0.8, 5.0),
            PedestrianConfig(
                'ped4', '/model/pedestrian_4/cmd_vel',
                'x', 0.5, 7.0),
        ]

        self.pubs_ = []
        for ped in self.pedestrians:
            pub = self.create_publisher(Twist, ped.topic, 10)
            self.pubs_.append(pub)

        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            f'Pedestrian controller started: {len(self.pedestrians)} pedestrians')

    def timer_callback(self):
        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9

        for i, ped in enumerate(self.pedestrians):
            twist = Twist()

            # 사인파로 왕복 운동 (부드러운 방향 전환)
            phase = 2.0 * math.pi * elapsed / ped.period
            velocity = ped.speed * math.sin(phase)

            if ped.axis == 'x':
                twist.linear.x = velocity
            else:
                twist.linear.y = velocity

            self.pubs_[i].publish(twist)


def main():
    rclpy.init()
    node = PedestrianController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
