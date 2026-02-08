#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directories
    pkg_dir = get_package_share_directory('mpc_controller_ros2')

    # Paths
    world_file = os.path.join(pkg_dir, 'worlds', 'mppi_test_simple.world')
    model_path = os.path.join(pkg_dir, 'models')
    sdf_file = os.path.join(pkg_dir, 'models', 'differential_robot', 'model.sdf')

    # Set Gazebo resource path
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=model_path
    )

    # Read SDF for robot_state_publisher (optional, for TF publishing)
    with open(sdf_file, 'r') as f:
        robot_desc = f.read()

    # Gazebo Harmonic (gz sim) 실행
    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-v4', world_file],
        output='screen',
        additional_env={'GZ_SIM_RESOURCE_PATH': model_path}
    )

    # Robot state publisher (URDF 대신 SDF 사용)
    # 참고: robot_state_publisher는 URDF만 지원하므로, TF는 Gazebo 플러그인이 담당
    # robot_state_publisher = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     name='robot_state_publisher',
    #     output='screen',
    #     parameters=[{
    #         'robot_description': robot_desc,
    #         'use_sim_time': True
    #     }]
    # )

    # Spawn robot in Gazebo (SDF 파일 직접 사용)
    spawn_robot = ExecuteProcess(
        cmd=[
            'gz', 'service', '-s', '/world/mppi_test_world/create',
            '--reqtype', 'gz.msgs.EntityFactory',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '3000',
            '--req',
            f'sdf_filename: "{sdf_file}", name: "differential_robot", pose: {{position: {{x: 0.0, y: 0.0, z: 0.15}}}}'
        ],
        output='screen'
    )

    # Bridge: Gazebo ↔ ROS2
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

    # Odom to TF node (Gazebo DiffDrive TF 브릿지 문제 해결)
    odom_to_tf = Node(
        package='mpc_controller_ros2',
        executable='odom_to_tf.py',
        name='odom_to_tf',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        set_gz_resource_path,
        gz_sim,
        bridge,
        TimerAction(
            period=5.0,
            actions=[spawn_robot]
        ),
        # Odom to TF (로봇 스폰 후 시작)
        TimerAction(
            period=6.0,
            actions=[odom_to_tf]
        ),
        # RVIZ는 nav2_mppi.launch.py에서 실행
        # TimerAction(
        #     period=7.0,
        #     actions=[rviz]
        # )
    ])
