#!/usr/bin/env python3
"""
nav2 + MPPI 컨트롤러 launch 파일 (Gazebo 별도 실행 후 사용)

사용법:
    # 1단계: 먼저 Gazebo 실행
    ros2 launch mpc_controller_ros2 gazebo_harmonic_test.launch.py

    # 2단계: Gazebo가 완전히 시작된 후 (로봇이 보이면) nav2 실행
    ros2 launch mpc_controller_ros2 nav2_mppi.launch.py

주의: 처음 시작 시 "jump back in time" 경고가 몇 번 나타날 수 있습니다.
      이는 정상이며, 시스템이 동기화되면 사라집니다.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction, ExecuteProcess, LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    # Package directory
    pkg_dir = get_package_share_directory('mpc_controller_ros2')

    # Paths
    nav2_params_file = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
    rviz_config = os.path.join(pkg_dir, 'config', 'mpc_rviz.rviz')

    # ========== Clock 대기 ==========
    # Gazebo /clock 토픽이 발행될 때까지 대기
    wait_for_clock = ExecuteProcess(
        cmd=['ros2', 'topic', 'echo', '/clock', '--once', '--no-arr'],
        output='screen',
        name='wait_for_clock'
    )

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
        # 0. /clock 토픽 대기 (Gazebo 시작 확인)
        wait_for_clock,
        LogInfo(msg='Gazebo /clock detected. Starting nav2 nodes...'),

        # 1. nav2 nodes (clock 동기화를 위해 5초 대기)
        TimerAction(
            period=5.0,
            actions=[
                controller_server,
                planner_server,
                behavior_server,
                bt_navigator,
                velocity_smoother,
                lifecycle_manager_navigation,
            ]
        ),

        # 2. RVIZ (8초 후)
        TimerAction(
            period=8.0,
            actions=[rviz]
        ),
    ])
