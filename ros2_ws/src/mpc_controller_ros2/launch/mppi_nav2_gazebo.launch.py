#!/usr/bin/env python3
"""
Gazebo Harmonic + nav2 + MPPI 컨트롤러 통합 launch 파일

실행 방법:
    ros2 launch mpc_controller_ros2 mppi_nav2_gazebo.launch.py

Goal 전송:
    ros2 run mpc_controller_ros2 send_nav_goal.py --x 5.0 --y 0.0
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
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Package directories
    pkg_dir = get_package_share_directory('mpc_controller_ros2')

    # Paths
    world_file = os.path.join(pkg_dir, 'worlds', 'mppi_test_simple.world')
    model_path = os.path.join(pkg_dir, 'models')
    sdf_file = os.path.join(pkg_dir, 'models', 'differential_robot', 'model.sdf')
    nav2_params_file = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
    rviz_config = os.path.join(pkg_dir, 'config', 'mpc_rviz.rviz')

    # Set Gazebo resource path
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=model_path
    )

    # ========== 1. Gazebo Harmonic ==========
    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-v4', world_file],
        output='screen',
        additional_env={'GZ_SIM_RESOURCE_PATH': model_path}
    )

    # Spawn robot in Gazebo
    spawn_robot = ExecuteProcess(
        cmd=[
            'gz', 'service', '-s', '/world/mppi_test_world/create',
            '--reqtype', 'gz.msgs.EntityFactory',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '5000',
            '--req',
            f'sdf_filename: "{sdf_file}", name: "differential_robot", pose: {{position: {{x: 0.0, y: 0.0, z: 0.15}}}}'
        ],
        output='screen'
    )

    # ========== 2. ROS-Gazebo Bridge ==========
    # 브릿지 방향: [ = GZ→ROS2, ] = ROS2→GZ, @ = 양방향
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=[
            # GZ → ROS2 (단방향)
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            # ROS2 → GZ (단방향)
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
        ]
    )

    # ========== 3. nav2 Nodes ==========
    # Using odom as global_frame (no map_server, no AMCL)
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

    # ========== 4. Lifecycle Manager ==========
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
                'bt_navigator',
                'velocity_smoother'
            ]}
        ]
    )

    # ========== 5. RVIZ ==========
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else []
    )

    # ========== 6. Odom to TF ==========
    # Gazebo DiffDrive 플러그인의 TF 브릿지 문제 해결
    odom_to_tf = Node(
        package='mpc_controller_ros2',
        executable='odom_to_tf.py',
        name='odom_to_tf',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # ========== Launch Description ==========
    return LaunchDescription([
        # Environment
        set_gz_resource_path,

        # 1. Gazebo first
        gz_sim,

        # 2. Bridge (start immediately with Gazebo)
        bridge,

        # 3. Spawn robot (wait for Gazebo to initialize)
        TimerAction(
            period=5.0,
            actions=[
                LogInfo(msg='Spawning robot...'),
                spawn_robot
            ]
        ),

        # 4. Odom to TF (로봇 스폰 후)
        TimerAction(
            period=7.0,
            actions=[
                LogInfo(msg='Starting odom_to_tf...'),
                odom_to_tf
            ]
        ),

        # 5. nav2 nodes (wait for TF to be available)
        TimerAction(
            period=10.0,
            actions=[
                LogInfo(msg='Starting nav2 nodes...'),
                controller_server,
                planner_server,
                behavior_server,
                bt_navigator,
                velocity_smoother,
                lifecycle_manager_navigation,
            ]
        ),

        # 6. RVIZ (last)
        TimerAction(
            period=15.0,
            actions=[
                LogInfo(msg='Starting RVIZ...'),
                rviz
            ]
        ),
    ])
