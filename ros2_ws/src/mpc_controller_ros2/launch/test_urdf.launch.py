#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('mpc_controller_ros2')
    urdf_file = os.path.join(pkg_dir, 'urdf', 'differential_robot_simple.urdf')

    print(f"URDF file path: {urdf_file}")
    print(f"File exists: {os.path.exists(urdf_file)}")

    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    print(f"URDF content length: {len(robot_desc)}")
    print(f"First 200 chars: {robot_desc[:200]}")

    # Robot state publisher만 실행
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': False
        }]
    )

    return LaunchDescription([
        robot_state_publisher
    ])
