#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directories
    pkg_dir = get_package_share_directory('mpc_controller_ros2')
    gazebo_ros_pkg_dir = get_package_share_directory('gazebo_ros')

    # Paths
    world_file = os.path.join(pkg_dir, 'worlds', 'mppi_test_world.world')
    urdf_file = os.path.join(pkg_dir, 'urdf', 'differential_robot.urdf')

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_pkg_dir, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_file, 'verbose': 'true'}.items()
    )

    # Read URDF
    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )

    # Spawn robot in Gazebo
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_robot',
        output='screen',
        arguments=[
            '-entity', 'differential_robot',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.1',
            '-Y', '0.0'
        ]
    )

    # RVIZ
    rviz_config = os.path.join(pkg_dir, 'config', 'mpc_rviz.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else []
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        TimerAction(
            period=3.0,
            actions=[spawn_robot]
        ),
        TimerAction(
            period=5.0,
            actions=[rviz]
        )
    ])
