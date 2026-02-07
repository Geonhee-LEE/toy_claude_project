#!/usr/bin/env python3
"""
Gazebo Harmonic + ros2_control + diff_drive_controller launch 파일

ros2_control을 통해 odom과 TF가 자동으로 발행됩니다.

실행 방법:
    ros2 launch mpc_controller_ros2 gazebo_ros2_control.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    TimerAction,
    SetEnvironmentVariable,
    RegisterEventHandler,
    DeclareLaunchArgument,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    # Package directory
    pkg_dir = get_package_share_directory('mpc_controller_ros2')

    # Paths
    world_file = os.path.join(pkg_dir, 'worlds', 'mppi_test_simple.world')
    urdf_file = os.path.join(pkg_dir, 'urdf', 'differential_robot_ros2_control.urdf')
    controller_config = os.path.join(pkg_dir, 'config', 'diff_drive_controller.yaml')
    rviz_config = os.path.join(pkg_dir, 'config', 'mpc_rviz.rviz')

    # Process xacro
    robot_description = xacro.process_file(urdf_file).toxml()

    # Set Gazebo resource path
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.path.join(pkg_dir, 'models')
    )

    # Set Gazebo system plugin path for gz_ros2_control (시스템 패키지 사용)
    gz_plugin_path = '/opt/ros/jazzy/lib'
    set_gz_plugin_path = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value=gz_plugin_path
    )

    # Set LD_LIBRARY_PATH to include ros2_control libraries
    set_ld_library_path = SetEnvironmentVariable(
        name='LD_LIBRARY_PATH',
        value='/opt/ros/jazzy/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
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

    # ========== 5. Controller Manager + Controllers ==========
    # Controller spawners
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    diff_drive_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_drive_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    # ========== 6. RVIZ ==========
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
        set_ld_library_path,

        # 1. Gazebo
        gz_sim,

        # 2. Robot State Publisher (immediately)
        robot_state_publisher,

        # 3. Bridge (immediately)
        bridge,

        # 4. Spawn robot (wait for Gazebo)
        TimerAction(
            period=5.0,
            actions=[spawn_robot]
        ),

        # 5. Controllers (wait for robot spawn)
        TimerAction(
            period=8.0,
            actions=[
                joint_state_broadcaster_spawner,
                diff_drive_controller_spawner,
            ]
        ),

        # 6. RVIZ (last)
        TimerAction(
            period=12.0,
            actions=[rviz]
        ),
    ])
