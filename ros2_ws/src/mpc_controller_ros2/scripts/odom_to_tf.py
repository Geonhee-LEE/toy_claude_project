#!/usr/bin/env python3
"""
Odom 메시지를 TF로 변환하는 노드

Gazebo Harmonic의 DiffDrive 플러그인이 tf_topic 절대경로를
지원하지 않아 별도로 TF를 발행합니다.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class OdomToTfNode(Node):
    def __init__(self):
        super().__init__('odom_to_tf')

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Odom subscriber
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.get_logger().info('OdomToTf node started')

    def odom_callback(self, msg: Odometry):
        # Create transform message
        t = TransformStamped()

        t.header.stamp = msg.header.stamp
        t.header.frame_id = msg.header.frame_id  # odom
        t.child_frame_id = msg.child_frame_id    # base_link

        # Translation
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        # Rotation
        t.transform.rotation = msg.pose.pose.orientation

        # Broadcast transform
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomToTfNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
