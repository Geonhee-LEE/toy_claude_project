#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directories
    pkg_dir = get_package_share_directory('mpc_controller_ros2')

    # Paths
    world_file = os.path.join(pkg_dir, 'worlds', 'mppi_test_simple.world')
    urdf_file = os.path.join(pkg_dir, 'urdf', 'differential_robot_simple.urdf')

    # Read URDF
    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    # Gazebo Harmonic (gz sim) 실행
    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-v4', world_file],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': True
        }]
    )

    # Spawn robot in Gazebo (Harmonic 방식)
    spawn_robot = ExecuteProcess(
        cmd=[
            'gz', 'service', '-s', '/world/mppi_test_world/create',
            '--reqtype', 'gz.msgs.EntityFactory',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000',
            '--req',
            f'sdf_filename: "{urdf_file}", name: "differential_robot", pose: {{position: {{x: 0.0, y: 0.0, z: 0.15}}}}'
        ],
        output='screen'
    )

    # Bridge: Gazebo → ROS2
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
        ]
    )

    # RVIZ
    rviz_config = os.path.join(pkg_dir, 'config', 'mpc_rviz.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else []
    )

    return LaunchDescription([
        gz_sim,
        robot_state_publisher,
        bridge,
        TimerAction(
            period=3.0,
            actions=[spawn_robot]
        ),
        TimerAction(
            period=5.0,
            actions=[rviz]
        )
    ])
