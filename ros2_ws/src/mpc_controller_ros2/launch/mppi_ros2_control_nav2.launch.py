#!/usr/bin/env python3
"""
Gazebo Harmonic + ros2_control + nav2 + MPPI 통합 launch 파일

ros2_control을 통해 odom과 TF가 발행됩니다.

실행 방법:
    # 커스텀 MPPI (기본)
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py

    # nav2 기본 MPPI
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=nav2

    # Maze 환경 + nav2 MPPI
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py \
        world:=maze_world.world map:=maze_map.yaml controller:=nav2

    # Corridor 환경 (충돌 방지 테스트)
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py \
        world:=corridor_world.world map:=corridor_map.yaml

컨트롤러 전환:
    controller:=custom  → 커스텀 MPPI (mpc_controller_ros2::MPPIControllerPlugin)
    controller:=nav2    → nav2 기본 MPPI (nav2_mppi_controller::MPPIController)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    TimerAction,
    SetEnvironmentVariable,
    LogInfo,
    DeclareLaunchArgument,
    OpaqueFunction,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import xacro


def launch_setup(context, *args, **kwargs):
    """OpaqueFunction callback - controller arg를 평가하여 적절한 파라미터 파일 선택"""

    # Package directory
    pkg_dir = get_package_share_directory('mpc_controller_ros2')

    # Resolve controller argument
    controller_type = LaunchConfiguration('controller').perform(context)

    # 컨트롤러별 파라미터 파일 선택
    if controller_type == 'nav2':
        controller_params_file = os.path.join(
            pkg_dir, 'config', 'nav2_params_nav2_mppi.yaml'
        )
        controller_label = 'nav2 기본 MPPI (nav2_mppi_controller::MPPIController)'
    else:
        controller_params_file = os.path.join(
            pkg_dir, 'config', 'nav2_params_custom_mppi.yaml'
        )
        controller_label = '커스텀 MPPI (mpc_controller_ros2::MPPIControllerPlugin)'

    # 공통 파라미터 파일 (AMCL, costmap, planner, behavior 등)
    nav2_params_file = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')

    # Static paths
    urdf_file = os.path.join(pkg_dir, 'urdf', 'differential_robot_ros2_control.urdf')
    controller_config = os.path.join(pkg_dir, 'config', 'diff_drive_controller.yaml')
    rviz_config = os.path.join(pkg_dir, 'config', 'mpc_rviz.rviz')

    # Dynamic paths
    world_name = LaunchConfiguration('world').perform(context)
    map_name = LaunchConfiguration('map').perform(context)
    world_file = os.path.join(pkg_dir, 'worlds', world_name)
    map_file = os.path.join(pkg_dir, 'maps', map_name)

    # Process xacro with controller config path
    robot_description = xacro.process_file(
        urdf_file,
        mappings={'controller_config': controller_config}
    ).toxml()

    gz_plugin_path = '/opt/ros/jazzy/lib'

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
    # controller_server: 컨트롤러별 전용 파라미터 파일 사용
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[controller_params_file],
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

    # ========== 9. RVIZ ==========
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else []
    )

    # ========== Launch Nodes ==========
    return [
        LogInfo(msg=f'[MPPI Controller] {controller_label}'),
        LogInfo(msg=f'[MPPI Controller] params: {controller_params_file}'),

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

        # 6. Localization nodes (map_server, amcl) - 10s delay
        TimerAction(
            period=10.0,
            actions=[
                LogInfo(msg='Starting localization nodes (map_server, amcl)...'),
                map_server,
                amcl,
            ]
        ),

        # 7. Localization lifecycle manager - 13s delay
        TimerAction(
            period=13.0,
            actions=[
                LogInfo(msg='Activating localization lifecycle...'),
                lifecycle_manager_localization,
            ]
        ),

        # 8. Navigation nodes - 18s delay
        TimerAction(
            period=18.0,
            actions=[
                LogInfo(msg=f'Starting navigation nodes ({controller_type} MPPI)...'),
                twist_stamper,
                controller_server,
                planner_server,
                behavior_server,
                bt_navigator,
            ]
        ),

        # 9. Navigation lifecycle manager - 21s delay
        TimerAction(
            period=21.0,
            actions=[
                LogInfo(msg='Activating navigation lifecycle...'),
                lifecycle_manager_navigation,
            ]
        ),

        # 10. RVIZ (25s delay)
        TimerAction(
            period=25.0,
            actions=[
                LogInfo(msg='Starting RVIZ...'),
                rviz
            ]
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'world',
            default_value='maze_world.world',
            description='World file name (e.g., mppi_test_simple.world, corridor_world.world)'
        ),
        DeclareLaunchArgument(
            'map',
            default_value='maze_map.yaml',
            description='Map file name (e.g., empty_map.yaml, corridor_map.yaml)'
        ),
        DeclareLaunchArgument(
            'controller',
            default_value='custom',
            description='MPPI controller type: "custom" (mpc_controller_ros2) or "nav2" (nav2_mppi_controller)'
        ),

        # Environment variables
        SetEnvironmentVariable(
            name='GZ_SIM_RESOURCE_PATH',
            value=os.path.join(
                get_package_share_directory('mpc_controller_ros2'), 'models'
            )
        ),
        SetEnvironmentVariable(
            name='GZ_SIM_SYSTEM_PLUGIN_PATH',
            value='/opt/ros/jazzy/lib'
        ),

        # OpaqueFunction으로 controller arg 기반 분기
        OpaqueFunction(function=launch_setup),
    ])
