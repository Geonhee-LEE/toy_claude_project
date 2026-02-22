#!/usr/bin/env python3
"""
Odom 메시지를 TF로 변환하는 노드

용도:
  1. Gazebo DiffDrive 플러그인: tf_topic 절대경로 미지원 대응
  2. Gazebo OdometryPublisher 플러그인: ground truth odom → TF 변환 (swerve)

remapping으로 입력 토픽 변경 가능:
  remappings=[('odom', '/model/swerve_robot/odometry')]
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class OdomToTfNode(Node):
    def __init__(self):
        super().__init__('odom_to_tf')

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Odom subscriber (RELIABLE QoS for bridge compatibility)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            qos,
        )

        self.get_logger().info('OdomToTf node started')

    def odom_callback(self, msg: Odometry):
        t = TransformStamped()

        t.header.stamp = msg.header.stamp
        # Fallback: Gazebo OdometryPublisher may leave frame_id empty
        t.header.frame_id = msg.header.frame_id if msg.header.frame_id else 'odom'
        t.child_frame_id = msg.child_frame_id if msg.child_frame_id else 'base_link'

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation

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
