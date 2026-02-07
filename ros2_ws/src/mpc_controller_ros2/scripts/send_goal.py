#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import sys


class GoalPublisher(Node):
    def __init__(self):
        super().__init__('goal_publisher')
        self.path_pub = self.create_publisher(Path, '/plan', 10)

        # Wait for subscribers
        self.get_logger().info('Waiting for controller to subscribe...')
        while self.path_pub.get_subscription_count() == 0:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('Controller subscribed, sending path...')

    def publish_straight_path(self, target_x=8.0, target_y=0.0):
        """직선 경로 생성 및 퍼블리시"""
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'

        # Generate waypoints from (0,0) to target
        num_points = 50
        for i in range(num_points + 1):
            pose = PoseStamped()
            pose.header = path.header

            # Linear interpolation
            alpha = i / num_points
            pose.pose.position.x = alpha * target_x
            pose.pose.position.y = alpha * target_y
            pose.pose.position.z = 0.0

            # Orientation (facing forward)
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0

            path.poses.append(pose)

        self.path_pub.publish(path)
        self.get_logger().info(f'Published straight path to ({target_x}, {target_y}) with {len(path.poses)} waypoints')

    def publish_curve_path(self, radius=3.0, angle_deg=90.0):
        """곡선 경로 생성 및 퍼블리시"""
        import math

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'

        num_points = 50
        angle_rad = math.radians(angle_deg)

        for i in range(num_points + 1):
            pose = PoseStamped()
            pose.header = path.header

            # Arc trajectory
            theta = (i / num_points) * angle_rad
            pose.pose.position.x = radius * math.sin(theta)
            pose.pose.position.y = radius * (1.0 - math.cos(theta))
            pose.pose.position.z = 0.0

            # Orientation tangent to arc
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(theta / 2.0)
            pose.pose.orientation.w = math.cos(theta / 2.0)

            path.poses.append(pose)

        self.path_pub.publish(path)
        self.get_logger().info(f'Published curved path (R={radius}m, angle={angle_deg}°) with {len(path.poses)} waypoints')

    def publish_obstacle_avoidance_path(self):
        """장애물 회피 경로 (S자 곡선)"""
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'

        # S-curve to avoid obstacles at (3,2) and (5,-1)
        waypoints = [
            (0.0, 0.0),
            (1.0, 0.5),
            (2.0, 1.0),
            (3.0, 0.5),   # Avoid obstacle_1 at (3,2)
            (4.0, 0.0),
            (5.0, 0.5),   # Avoid obstacle_2 at (5,-1)
            (6.0, 1.0),
            (7.0, 0.5),
            (8.0, 0.0),
            (9.0, 0.0),
        ]

        for (x, y) in waypoints:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self.path_pub.publish(path)
        self.get_logger().info(f'Published obstacle avoidance path with {len(path.poses)} waypoints')


def main(args=None):
    rclpy.init(args=args)
    node = GoalPublisher()

    # Check command line arguments
    if len(sys.argv) > 1:
        path_type = sys.argv[1]

        if path_type == 'straight':
            target_x = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
            target_y = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
            node.publish_straight_path(target_x, target_y)
        elif path_type == 'curve':
            radius = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
            angle = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0
            node.publish_curve_path(radius, angle)
        elif path_type == 'obstacle':
            node.publish_obstacle_avoidance_path()
        else:
            node.get_logger().error(f'Unknown path type: {path_type}')
            node.get_logger().info('Usage: send_goal.py [straight|curve|obstacle] [args...]')
    else:
        # Default: straight path
        node.get_logger().info('No path type specified, using default straight path')
        node.publish_straight_path()

    # Keep node alive briefly to ensure message is sent
    rclpy.spin_once(node, timeout_sec=1.0)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
