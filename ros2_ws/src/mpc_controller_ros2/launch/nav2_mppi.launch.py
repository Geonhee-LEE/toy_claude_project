#!/usr/bin/env python3
"""
nav2 + MPPI 컨트롤러 launch 파일 (Gazebo 별도 실행 후 사용)

사용법:
    # 1단계: 먼저 Gazebo 실행
    ros2 launch mpc_controller_ros2 gazebo_harmonic_test.launch.py

    # 2단계: Gazebo가 완전히 시작된 후 (로봇이 보이면) nav2 실행
    ros2 launch mpc_controller_ros2 nav2_mppi.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    # Package directory
    pkg_dir = get_package_share_directory('mpc_controller_ros2')

    # Paths
    nav2_params_file = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
    rviz_config = os.path.join(pkg_dir, 'config', 'mpc_rviz.rviz')

    # ========== No map/AMCL needed ==========
    # Using odom as global frame (no localization)
    # TF tree: odom -> base_link (from Gazebo DiffDrive)

    # ========== nav2 Nodes ==========
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[nav2_params_file],
        remappings=[('cmd_vel', '/cmd_vel')]
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

    velocity_smoother = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        output='screen',
        parameters=[nav2_params_file],
        remappings=[
            ('cmd_vel', 'cmd_vel_nav'),
            ('cmd_vel_smoothed', '/cmd_vel')
        ]
    )

    # ========== Lifecycle Manager ==========
    lifecycle_manager_navigation = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'autostart': True},
            {'node_names': [
                'controller_server',
                'planner_server',
                'behavior_server',
                'bt_navigator',
                'velocity_smoother'
            ]}
        ]
    )

    # ========== RVIZ ==========
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else []
    )

    # ========== Launch Description ==========
    # AMCL 없이 odom을 global_frame으로 사용
    # TF tree: odom -> base_link (Gazebo DiffDrive에서 publish)
    return LaunchDescription([
        # 1. nav2 nodes (clock 동기화를 위해 2초 대기)
        TimerAction(
            period=2.0,
            actions=[
                controller_server,
                planner_server,
                behavior_server,
                bt_navigator,
                velocity_smoother,
                lifecycle_manager_navigation,
            ]
        ),

        # 2. RVIZ (4초 후)
        TimerAction(
            period=4.0,
            actions=[rviz]
        ),
    ])
