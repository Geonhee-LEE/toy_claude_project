"""
Swerve Drive Kinematics Node

IK (Twist → joint commands) + FK (joint_states → odom) + TF broadcast

아키텍처:
  nav2 MPPI → /cmd_vel_nav (Twist: vx, vy, omega)
      → [SwerveKinematicsNode]
          ├→ /steer_position_controller/commands (Float64MultiArray)
          └→ /wheel_velocity_controller/commands (Float64MultiArray)
      ← /joint_states (JointState)
          → /swerve_controller/odom (Odometry) + odom→base_link TF

사용법:
  ros2 run mpc_controller_ros2 swerve_kinematics_node
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster


# Wheel order: FL, FR, RL, RR
STEER_JOINTS = [
    'fl_steer_joint', 'fr_steer_joint',
    'rl_steer_joint', 'rr_steer_joint',
]
WHEEL_JOINTS = [
    'fl_wheel_joint', 'fr_wheel_joint',
    'rl_wheel_joint', 'rr_wheel_joint',
]


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class SwerveKinematicsNode(Node):
    """Swerve drive IK/FK + odometry + TF broadcast node."""

    def __init__(self):
        super().__init__('swerve_kinematics_node')

        # Parameters
        self.declare_parameter('wheel_radius', 0.08)
        self.declare_parameter('wheel_x', 0.25)
        self.declare_parameter('wheel_y', 0.22)
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('cmd_vel_timeout', 0.5)
        self.declare_parameter('use_gazebo_odom', False)

        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.wheel_x = self.get_parameter('wheel_x').value
        self.wheel_y = self.get_parameter('wheel_y').value
        publish_rate = self.get_parameter('publish_rate').value
        self.cmd_vel_timeout = self.get_parameter('cmd_vel_timeout').value
        self.use_gazebo_odom = self.get_parameter('use_gazebo_odom').value

        # Wheel positions relative to base_link center: (Lx, Ly)
        #   FL: (+x, +y), FR: (+x, -y), RL: (-x, +y), RR: (-x, -y)
        self.wheel_positions = [
            (+self.wheel_x, +self.wheel_y),  # FL
            (+self.wheel_x, -self.wheel_y),  # FR
            (-self.wheel_x, +self.wheel_y),  # RL
            (-self.wheel_x, -self.wheel_y),  # RR
        ]

        # Odometry state
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0
        self.last_odom_time = None

        # Last steer angles (for flip optimization persistence)
        self.last_steer_angles = [0.0, 0.0, 0.0, 0.0]

        # Last cmd_vel time (for timeout safety)
        self.last_cmd_vel_time = None

        # Joint state cache
        self.steer_positions = [0.0, 0.0, 0.0, 0.0]
        self.wheel_velocities = [0.0, 0.0, 0.0, 0.0]

        # QoS
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel_nav', self._cmd_vel_callback, 10)

        # Joint state subscriber: FK에만 필요 (Gazebo odom 사용 시 불필요)
        if not self.use_gazebo_odom:
            self.joint_state_sub = self.create_subscription(
                JointState, '/joint_states', self._joint_state_callback, qos)

        # Publishers (IK: always active)
        self.steer_pub = self.create_publisher(
            Float64MultiArray,
            '/steer_position_controller/commands', 10)
        self.wheel_pub = self.create_publisher(
            Float64MultiArray,
            '/wheel_velocity_controller/commands', 10)

        # FK odom + TF: only when NOT using Gazebo ground truth
        if not self.use_gazebo_odom:
            self.odom_pub = self.create_publisher(
                Odometry, '/swerve_controller/odom', 10)
            self.tf_broadcaster = TransformBroadcaster(self)
            timer_period = 1.0 / publish_rate
            self.odom_timer = self.create_timer(timer_period, self._odom_timer_callback)
        else:
            self.get_logger().info(
                'use_gazebo_odom=True → FK/odom/TF 비활성화 (IK만 동작)')

        # Timeout timer (check at 10 Hz)
        self.timeout_timer = self.create_timer(0.1, self._timeout_check)

        self.get_logger().info(
            f'SwerveKinematicsNode started: '
            f'wheel_radius={self.wheel_radius}, '
            f'wheel_x={self.wheel_x}, wheel_y={self.wheel_y}')

    def _cmd_vel_callback(self, msg: Twist):
        """IK: Twist → 4x steer angles + 4x wheel speeds."""
        self.last_cmd_vel_time = self.get_clock().now()

        vx = msg.linear.x
        vy = msg.linear.y
        omega = msg.angular.z

        steer_angles = []
        wheel_speeds = []

        for i, (lx, ly) in enumerate(self.wheel_positions):
            # Velocity at wheel i due to body motion
            vx_i = vx - omega * ly
            vy_i = vy + omega * lx

            speed = math.hypot(vx_i, vy_i)

            if speed < 1e-4:
                # Near-zero velocity: keep previous steer angle, zero speed
                steer_angles.append(self.last_steer_angles[i])
                wheel_speeds.append(0.0)
                continue

            angle = math.atan2(vy_i, vx_i)

            # Flip optimization: if |angle_diff| > pi/2, reverse direction
            angle_diff = normalize_angle(angle - self.last_steer_angles[i])
            if abs(angle_diff) > math.pi / 2.0:
                angle = normalize_angle(angle + math.pi)
                speed = -speed

            steer_angles.append(angle)
            wheel_speeds.append(speed / self.wheel_radius)

        # Update last steer angles
        self.last_steer_angles = steer_angles[:]

        # Publish commands
        steer_msg = Float64MultiArray()
        steer_msg.data = steer_angles
        self.steer_pub.publish(steer_msg)

        wheel_msg = Float64MultiArray()
        wheel_msg.data = wheel_speeds
        self.wheel_pub.publish(wheel_msg)

    def _joint_state_callback(self, msg: JointState):
        """Cache joint states for FK computation."""
        for i, name in enumerate(STEER_JOINTS):
            if name in msg.name:
                idx = msg.name.index(name)
                if idx < len(msg.position):
                    self.steer_positions[i] = msg.position[idx]

        for i, name in enumerate(WHEEL_JOINTS):
            if name in msg.name:
                idx = msg.name.index(name)
                if idx < len(msg.velocity):
                    self.wheel_velocities[i] = msg.velocity[idx]

    def _odom_timer_callback(self):
        """FK: joint_states → body velocity → odom integration + TF broadcast."""
        now = self.get_clock().now()

        if self.last_odom_time is None:
            self.last_odom_time = now
            return

        dt = (now - self.last_odom_time).nanoseconds * 1e-9
        if dt <= 0.0 or dt > 1.0:
            self.last_odom_time = now
            return

        # FK: compute body-frame velocities from wheel states
        vx_sum = 0.0
        vy_sum = 0.0

        for i in range(4):
            v_i = self.wheel_velocities[i] * self.wheel_radius
            vx_sum += v_i * math.cos(self.steer_positions[i])
            vy_sum += v_i * math.sin(self.steer_positions[i])

        body_vx = vx_sum / 4.0
        body_vy = vy_sum / 4.0

        # Omega from lateral velocity differences
        #   omega ≈ (sum of cross products) / (sum of r²)
        #   For each wheel: omega_contrib = (vy_i * lx_i - vx_i * ly_i) / (lx_i² + ly_i²)
        omega_sum = 0.0
        r_sq_sum = 0.0
        for i, (lx, ly) in enumerate(self.wheel_positions):
            v_i = self.wheel_velocities[i] * self.wheel_radius
            vx_i = v_i * math.cos(self.steer_positions[i])
            vy_i = v_i * math.sin(self.steer_positions[i])
            r_sq = lx * lx + ly * ly
            if r_sq > 1e-6:
                # Cross product: (vy_i - body_vy) * lx - (vx_i - body_vx) * ly
                # Simplified: omega contribution from each wheel
                omega_sum += (-vy_i * (-ly) + vx_i * (-lx))
                # More accurate: use the difference from body center
                pass
            r_sq_sum += r_sq

        # Simple omega estimation: average of per-wheel omega estimates
        omega_estimates = []
        for i, (lx, ly) in enumerate(self.wheel_positions):
            v_i = self.wheel_velocities[i] * self.wheel_radius
            vx_i = v_i * math.cos(self.steer_positions[i])
            vy_i = v_i * math.sin(self.steer_positions[i])
            r_sq = lx * lx + ly * ly
            if r_sq > 1e-6:
                # omega = (vy_i * lx - vx_i * ly) / r²
                # But we need to subtract body translation:
                # omega = ((vy_i - body_vy)*lx - (vx_i - body_vx)*ly) / r²
                omega_i = ((vy_i - body_vy) * lx - (vx_i - body_vx) * ly) / r_sq
                omega_estimates.append(omega_i)

        body_omega = sum(omega_estimates) / len(omega_estimates) if omega_estimates else 0.0

        # Integrate odometry (world frame)
        cos_theta = math.cos(self.odom_theta)
        sin_theta = math.sin(self.odom_theta)
        self.odom_x += (body_vx * cos_theta - body_vy * sin_theta) * dt
        self.odom_y += (body_vx * sin_theta + body_vy * cos_theta) * dt
        self.odom_theta += body_omega * dt
        self.odom_theta = normalize_angle(self.odom_theta)

        self.last_odom_time = now

        # Publish Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = now.to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        odom_msg.pose.pose.position.x = self.odom_x
        odom_msg.pose.pose.position.y = self.odom_y
        odom_msg.pose.pose.position.z = 0.0

        # Quaternion from yaw
        cy = math.cos(self.odom_theta * 0.5)
        sy = math.sin(self.odom_theta * 0.5)
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = sy
        odom_msg.pose.pose.orientation.w = cy

        odom_msg.twist.twist.linear.x = body_vx
        odom_msg.twist.twist.linear.y = body_vy
        odom_msg.twist.twist.angular.z = body_omega

        self.odom_pub.publish(odom_msg)

        # Broadcast TF: odom → base_link
        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.odom_x
        t.transform.translation.y = self.odom_y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = sy
        t.transform.rotation.w = cy

        self.tf_broadcaster.sendTransform(t)

    def _timeout_check(self):
        """Safety: zero wheel speeds if no cmd_vel received within timeout."""
        if self.last_cmd_vel_time is None:
            return

        elapsed = (self.get_clock().now() - self.last_cmd_vel_time).nanoseconds * 1e-9
        if elapsed > self.cmd_vel_timeout:
            wheel_msg = Float64MultiArray()
            wheel_msg.data = [0.0, 0.0, 0.0, 0.0]
            self.wheel_pub.publish(wheel_msg)
            # Reset to prevent repeated zero publishes
            self.last_cmd_vel_time = None


def main(args=None):
    rclpy.init(args=args)
    node = SwerveKinematicsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
