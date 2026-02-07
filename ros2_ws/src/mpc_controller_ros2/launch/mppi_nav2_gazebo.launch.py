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
    GroupAction,
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
    map_yaml_file = os.path.join(pkg_dir, 'maps', 'empty_map.yaml')
    rviz_config = os.path.join(pkg_dir, 'config', 'mpc_rviz.rviz')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

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
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ]
    )

    # ========== 3. Static Transform (map -> odom) ==========
    static_tf_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_pub_map_odom',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'map', '--child-frame-id', 'odom']
    )

    # ========== 4. Map Server ==========
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'yaml_filename': map_yaml_file},
        ]
    )

    # ========== 5. nav2 Nodes ==========
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

    # ========== 6. Lifecycle Managers ==========
    lifecycle_manager_localization = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'autostart': True},
            {'node_names': ['map_server']}
        ]
    )

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
        # Environment
        set_gz_resource_path,

        # 1. Gazebo first (must start before nav2)
        gz_sim,

        # 2. Bridge (starts with Gazebo)
        bridge,

        # 3. Spawn robot after Gazebo starts
        TimerAction(
            period=5.0,
            actions=[spawn_robot]
        ),

        # 4. Static TF (wait for clock)
        TimerAction(
            period=6.0,
            actions=[static_tf_map_odom]
        ),

        # 5. Map server (wait for clock)
        TimerAction(
            period=7.0,
            actions=[
                map_server,
                lifecycle_manager_localization,
            ]
        ),

        # 6. nav2 nodes (wait for map_server)
        TimerAction(
            period=10.0,
            actions=[
                controller_server,
                planner_server,
                behavior_server,
                bt_navigator,
                velocity_smoother,
                lifecycle_manager_navigation,
            ]
        ),

        # 7. RVIZ (last)
        TimerAction(
            period=12.0,
            actions=[rviz]
        ),
    ])
