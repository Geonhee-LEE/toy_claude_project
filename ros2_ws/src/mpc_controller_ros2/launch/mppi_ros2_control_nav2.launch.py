#!/usr/bin/env python3
"""
Gazebo Harmonic + ros2_control + nav2 + MPPI 통합 launch 파일

ros2_control을 통해 odom과 TF가 발행됩니다.

실행 방법:
    # 커스텀 MPPI (기본)
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py

    # Headless 모드 (GUI 없이 시뮬레이션만)
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py headless:=true

    # nav2 기본 MPPI
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=nav2

    # Maze 환경 + nav2 MPPI
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py \
        world:=maze_world.world map:=maze_map.yaml controller:=nav2

    # Corridor 환경 (충돌 방지 테스트)
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py \
        world:=corridor_world.world map:=corridor_map.yaml

    # Tsallis-MPPI (q-exponential 가중치)
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=tsallis

    # Risk-Aware MPPI (CVaR 가중치 절단)
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=risk_aware

    # SVMPC (Stein Variational MPC)
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=svmpc

    # Smooth-MPPI (Δu space 최적화)
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=smooth

    # Spline-MPPI (B-spline 보간)
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=spline

    # SVG-MPPI (Guide particle SVGD)
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=svg

컨트롤러 전환:
    controller:=custom      → 커스텀 MPPI (mpc_controller_ros2::MPPIControllerPlugin)
    controller:=log         → Log-MPPI (mpc_controller_ros2::LogMPPIControllerPlugin)
    controller:=tsallis     → Tsallis-MPPI (mpc_controller_ros2::TsallisMPPIControllerPlugin)
    controller:=risk_aware  → Risk-Aware MPPI (mpc_controller_ros2::RiskAwareMPPIControllerPlugin)
    controller:=svmpc       → SVMPC (mpc_controller_ros2::SVMPCControllerPlugin)
    controller:=smooth      → Smooth-MPPI (mpc_controller_ros2::SmoothMPPIControllerPlugin)
    controller:=spline      → Spline-MPPI (mpc_controller_ros2::SplineMPPIControllerPlugin)
    controller:=svg         → SVG-MPPI (mpc_controller_ros2::SVGMPPIControllerPlugin)
    controller:=nav2        → nav2 기본 MPPI (nav2_mppi_controller::MPPIController)
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

    # Resolve arguments
    controller_type = LaunchConfiguration('controller').perform(context)
    headless = LaunchConfiguration('headless').perform(context).lower() == 'true'

    # 컨트롤러별 파라미터 파일 선택
    if controller_type == 'nav2':
        controller_params_file = os.path.join(
            pkg_dir, 'config', 'nav2_params_nav2_mppi.yaml'
        )
        controller_label = 'nav2 기본 MPPI (nav2_mppi_controller::MPPIController)'
    elif controller_type == 'log':
        controller_params_file = os.path.join(
            pkg_dir, 'config', 'nav2_params_log_mppi.yaml'
        )
        controller_label = 'Log-MPPI (mpc_controller_ros2::LogMPPIControllerPlugin)'
    elif controller_type == 'tsallis':
        controller_params_file = os.path.join(
            pkg_dir, 'config', 'nav2_params_tsallis_mppi.yaml'
        )
        controller_label = 'Tsallis-MPPI (mpc_controller_ros2::TsallisMPPIControllerPlugin)'
    elif controller_type == 'risk_aware':
        controller_params_file = os.path.join(
            pkg_dir, 'config', 'nav2_params_risk_aware_mppi.yaml'
        )
        controller_label = 'Risk-Aware MPPI (mpc_controller_ros2::RiskAwareMPPIControllerPlugin)'
    elif controller_type == 'svmpc':
        controller_params_file = os.path.join(
            pkg_dir, 'config', 'nav2_params_svmpc.yaml'
        )
        controller_label = 'SVMPC (mpc_controller_ros2::SVMPCControllerPlugin)'
    elif controller_type == 'smooth':
        controller_params_file = os.path.join(
            pkg_dir, 'config', 'nav2_params_smooth_mppi.yaml'
        )
        controller_label = 'Smooth-MPPI (mpc_controller_ros2::SmoothMPPIControllerPlugin)'
    elif controller_type == 'spline':
        controller_params_file = os.path.join(
            pkg_dir, 'config', 'nav2_params_spline_mppi.yaml'
        )
        controller_label = 'Spline-MPPI (mpc_controller_ros2::SplineMPPIControllerPlugin)'
    elif controller_type == 'svg':
        controller_params_file = os.path.join(
            pkg_dir, 'config', 'nav2_params_svg_mppi.yaml'
        )
        controller_label = 'SVG-MPPI (mpc_controller_ros2::SVGMPPIControllerPlugin)'
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
    gz_cmd = ['gz', 'sim', '-r', '-v4']
    if headless:
        gz_cmd.extend(['-s', '--headless-rendering'])  # Server only + EGL rendering
    gz_cmd.append(world_file)

    gz_env = {
        'GZ_SIM_SYSTEM_PLUGIN_PATH': gz_plugin_path,
        'GZ_SIM_RESOURCE_PATH': os.path.join(pkg_dir, 'models'),
    }

    gz_sim = ExecuteProcess(
        cmd=gz_cmd,
        output='screen',
        additional_env=gz_env,
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
    # gz_ros2_control 플러그인이 URDF의 <parameters> 경로에서
    # diff_drive_controller.yaml을 읽어 joint_state_broadcaster와
    # diff_drive_controller를 자동 로드/설정/활성화합니다.
    # 별도 spawner 없이 gz_ros2_control에 위임합니다.

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
    # controller_server: 공통 파라미터(local_costmap 등) + 컨트롤러별 전용 파라미터
    # nav2의 controller_server는 내부적으로 local_costmap 노드를 생성하므로
    # 반드시 local_costmap 파라미터가 포함된 공통 파일도 함께 전달해야 함
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[nav2_params_file, controller_params_file],
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
    nodes = [
        LogInfo(msg=f'[MPPI Controller] {controller_label}'),
        LogInfo(msg=f'[MPPI Controller] headless: {headless}'),
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

        # 5. Controllers: gz_ros2_control이 자동 활성화하므로 spawner 불필요
        # 컨트롤러 활성화 대기를 위해 로그만 출력
        TimerAction(
            period=8.0,
            actions=[
                LogInfo(msg='Controllers auto-started by gz_ros2_control plugin'),
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

    ]

    # 10. RVIZ (25s delay) - headless 모드에서는 비활성화
    if not headless:
        nodes.append(
            TimerAction(
                period=25.0,
                actions=[
                    LogInfo(msg='Starting RVIZ...'),
                    rviz
                ]
            )
        )

    return nodes


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
            description='MPPI controller type: "custom", "log", "tsallis", "risk_aware", "svmpc", "smooth", "spline", "svg", or "nav2"'
        ),
        DeclareLaunchArgument(
            'headless',
            default_value='false',
            description='Run in headless mode (no Gazebo GUI, no RVIZ)'
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
