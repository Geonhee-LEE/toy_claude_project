#!/usr/bin/env python3
"""
Gazebo Harmonic + ros2_control + nav2 + MPPI 통합 launch 파일

ros2_control을 통해 odom과 TF가 발행됩니다.

실행 방법:
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    TimerAction,
    SetEnvironmentVariable,
    LogInfo,
)
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    # Package directory
    pkg_dir = get_package_share_directory('mpc_controller_ros2')

    # Paths
    world_file = os.path.join(pkg_dir, 'worlds', 'mppi_test_simple.world')
    urdf_file = os.path.join(pkg_dir, 'urdf', 'differential_robot_ros2_control.urdf')
    controller_config = os.path.join(pkg_dir, 'config', 'diff_drive_controller.yaml')
    nav2_params_file = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
    rviz_config = os.path.join(pkg_dir, 'config', 'mpc_rviz.rviz')

    # Process xacro with controller config path
    robot_description = xacro.process_file(
        urdf_file,
        mappings={'controller_config': controller_config}
    ).toxml()

    # Set Gazebo paths
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.path.join(pkg_dir, 'models')
    )

    gz_plugin_path = '/opt/ros/jazzy/lib'
    set_gz_plugin_path = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value=gz_plugin_path
    )

    # ========== 1. Gazebo Harmonic ==========
    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-v4', world_file],
        output='screen',
        additional_env={
            'GZ_SIM_SYSTEM_PLUGIN_PATH': gz_plugin_path,
            'GZ_SIM_RESOURCE_PATH': os.path.join(pkg_dir, 'models'),
        }
    )

    # ========== 2. Robot State Publisher ==========
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }]
    )

    # ========== 3. Spawn Robot ==========
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_robot',
        output='screen',
        arguments=[
            '-name', 'differential_robot',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.15',
        ],
    )

    # ========== 4. ros_gz_bridge ==========
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        ]
    )

    # ========== 5. Controller Spawners ==========
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    diff_drive_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'diff_drive_controller',
            '--controller-manager', '/controller_manager',
            '--param-file', controller_config
        ],
        output='screen',
    )

    # ========== 6. Map Server ==========
    map_file = os.path.join(pkg_dir, 'maps', 'empty_map.yaml')
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'yaml_filename': map_file
        }]
    )

    # ========== 7. AMCL ==========
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[nav2_params_file]
    )

    # ========== 8. nav2 Nodes ==========
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[nav2_params_file],
        remappings=[('cmd_vel', '/cmd_vel_nav')]
    )

    # Twist → TwistStamped 변환 (inline Python)
    twist_to_stamped_cmd = '''
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, TwistStamped

class TwistStamper(Node):
    def __init__(self):
        super().__init__("twist_stamper")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(TwistStamped, "/diff_drive_controller/cmd_vel", qos)
        self.sub = self.create_subscription(Twist, "/cmd_vel_nav", self.cb, 10)
    def cb(self, msg):
        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "base_link"
        out.twist = msg
        self.pub.publish(out)

rclpy.init()
rclpy.spin(TwistStamper())
'''
    twist_stamper = ExecuteProcess(
        cmd=['python3', '-c', twist_to_stamped_cmd],
        output='screen'
    )

    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[nav2_params_file]
    )

    behavior_server = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[nav2_params_file]
    )

    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[nav2_params_file]
    )

    # velocity_smoother 제거 - controller_server가 직접 diff_drive_controller로 발행

    # Lifecycle manager for localization (map_server, amcl)
    lifecycle_manager_localization = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'autostart': True},
            {'bond_timeout': 4.0},
            {'node_names': ['map_server', 'amcl']}
        ]
    )

    # Lifecycle manager for navigation
    lifecycle_manager_navigation = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'autostart': True},
            {'bond_timeout': 4.0},
            {'node_names': [
                'controller_server',
                'planner_server',
                'behavior_server',
                'bt_navigator'
            ]}
        ]
    )

    # ========== 7. RVIZ ==========
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else []
    )

    # ========== Launch Description ==========
    return LaunchDescription([
        set_gz_resource_path,
        set_gz_plugin_path,

        # 1. Gazebo
        gz_sim,

        # 2. Robot State Publisher
        robot_state_publisher,

        # 3. Bridge
        bridge,

        # 4. Spawn robot (5s delay)
        TimerAction(
            period=5.0,
            actions=[
                LogInfo(msg='Spawning robot...'),
                spawn_robot
            ]
        ),

        # 5. Controllers (8s delay)
        TimerAction(
            period=8.0,
            actions=[
                LogInfo(msg='Starting controllers...'),
                joint_state_broadcaster_spawner,
                diff_drive_controller_spawner,
            ]
        ),

        # 6. Localization (map_server, amcl) - 10s delay
        TimerAction(
            period=10.0,
            actions=[
                LogInfo(msg='Starting localization (map_server, amcl)...'),
                map_server,
                amcl,
                lifecycle_manager_localization,
            ]
        ),

        # 7. Navigation nodes - 15s delay (after localization is ready)
        TimerAction(
            period=15.0,
            actions=[
                LogInfo(msg='Starting navigation...'),
                twist_stamper,
                controller_server,
                planner_server,
                behavior_server,
                bt_navigator,
                lifecycle_manager_navigation,
            ]
        ),

        # 8. RVIZ (20s delay)
        TimerAction(
            period=20.0,
            actions=[
                LogInfo(msg='Starting RVIZ...'),
                rviz
            ]
        ),
    ])
