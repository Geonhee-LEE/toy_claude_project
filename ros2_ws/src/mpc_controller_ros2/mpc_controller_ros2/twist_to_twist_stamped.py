#!/usr/bin/env python3
"""
Twist → TwistStamped 변환 노드

nav2 controller_server (Twist) → diff_drive_controller (TwistStamped) 연결용
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped


class TwistToTwistStamped(Node):
    def __init__(self):
        super().__init__('twist_to_twist_stamped')

        # Parameters
        self.declare_parameter('input_topic', '/cmd_vel_in')
        self.declare_parameter('output_topic', '/cmd_vel_out')
        self.declare_parameter('frame_id', 'base_link')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.frame_id = self.get_parameter('frame_id').value

        # Subscriber (Twist)
        self.sub = self.create_subscription(
            Twist,
            input_topic,
            self.twist_callback,
            10
        )

        # Publisher (TwistStamped)
        self.pub = self.create_publisher(
            TwistStamped,
            output_topic,
            10
        )

        self.get_logger().info(f'Converting {input_topic} (Twist) → {output_topic} (TwistStamped)')

    def twist_callback(self, msg: Twist):
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = self.frame_id
        stamped.twist = msg
        self.pub.publish(stamped)


def main():
    rclpy.init()
    node = TwistToTwistStamped()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
