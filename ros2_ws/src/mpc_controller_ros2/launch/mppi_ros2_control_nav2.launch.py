#!/usr/bin/env python3
"""
Gazebo Harmonic + ros2_control + nav2 + MPPI 통합 launch 파일

ros2_control을 통해 odom과 TF가 발행됩니다.
nav2 노드는 non-composition 모드로 실행하며, bond_timeout=0.0으로 bond를
비활성화하고 `ros2 lifecycle set` subprocess로 직접 활성화합니다.

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

    # Swerve Drive MPPI (홀로노믹 3축)
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=swerve

    # Non-Coaxial Swerve Drive MPPI
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=non_coaxial

컨트롤러 전환:
    controller:=custom       → 커스텀 MPPI (mpc_controller_ros2::MPPIControllerPlugin)
    controller:=log          → Log-MPPI (mpc_controller_ros2::LogMPPIControllerPlugin)
    controller:=tsallis      → Tsallis-MPPI (mpc_controller_ros2::TsallisMPPIControllerPlugin)
    controller:=risk_aware   → Risk-Aware MPPI (mpc_controller_ros2::RiskAwareMPPIControllerPlugin)
    controller:=svmpc        → SVMPC (mpc_controller_ros2::SVMPCControllerPlugin)
    controller:=smooth       → Smooth-MPPI (mpc_controller_ros2::SmoothMPPIControllerPlugin)
    controller:=spline       → Spline-MPPI (mpc_controller_ros2::SplineMPPIControllerPlugin)
    controller:=svg          → SVG-MPPI (mpc_controller_ros2::SVGMPPIControllerPlugin)
    controller:=swerve       → Swerve Drive MPPI (motion_model=swerve)
    controller:=non_coaxial  → Non-Coaxial Swerve MPPI (motion_model=non_coaxial_swerve)
    controller:=non_coaxial_60deg → Non-Coaxial Swerve MPPI 60° (max_steering_angle=π/3)
    controller:=nav2         → nav2 기본 MPPI (nav2_mppi_controller::MPPIController)
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
    GroupAction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
import xacro


def launch_setup(context, *args, **kwargs):
    """OpaqueFunction callback - controller arg를 평가하여 적절한 파라미터 파일 선택"""

    # Package directory
    pkg_dir = get_package_share_directory('mpc_controller_ros2')

    # Resolve arguments
    controller_type = LaunchConfiguration('controller').perform(context)
    headless = LaunchConfiguration('headless').perform(context).lower() == 'true'

    # 컨트롤러별 파라미터 파일 선택
    controller_map = {
        'nav2': ('nav2_params_nav2_mppi.yaml',
                 'nav2 기본 MPPI (nav2_mppi_controller::MPPIController)'),
        'log': ('nav2_params_log_mppi.yaml',
                'Log-MPPI (mpc_controller_ros2::LogMPPIControllerPlugin)'),
        'tsallis': ('nav2_params_tsallis_mppi.yaml',
                    'Tsallis-MPPI (mpc_controller_ros2::TsallisMPPIControllerPlugin)'),
        'risk_aware': ('nav2_params_risk_aware_mppi.yaml',
                       'Risk-Aware MPPI (mpc_controller_ros2::RiskAwareMPPIControllerPlugin)'),
        'svmpc': ('nav2_params_svmpc.yaml',
                  'SVMPC (mpc_controller_ros2::SVMPCControllerPlugin)'),
        'smooth': ('nav2_params_smooth_mppi.yaml',
                   'Smooth-MPPI (mpc_controller_ros2::SmoothMPPIControllerPlugin)'),
        'spline': ('nav2_params_spline_mppi.yaml',
                   'Spline-MPPI (mpc_controller_ros2::SplineMPPIControllerPlugin)'),
        'svg': ('nav2_params_svg_mppi.yaml',
                'SVG-MPPI (mpc_controller_ros2::SVGMPPIControllerPlugin)'),
        'swerve': ('nav2_params_swerve_mppi.yaml',
                   'Swerve MPPI (motion_model=swerve)'),
        'non_coaxial': ('nav2_params_non_coaxial_mppi.yaml',
                        'Non-Coaxial Swerve MPPI (motion_model=non_coaxial_swerve)'),
        'non_coaxial_60deg': ('nav2_params_non_coaxial_60deg_mppi.yaml',
                              'Non-Coaxial Swerve MPPI 60° (max_steering_angle=π/3)'),
    }
    if controller_type in controller_map:
        params_name, controller_label = controller_map[controller_type]
    else:
        params_name = 'nav2_params_custom_mppi.yaml'
        controller_label = '커스텀 MPPI (mpc_controller_ros2::MPPIControllerPlugin)'

    controller_params_file = os.path.join(pkg_dir, 'config', params_name)

    # Swerve drive 판별
    is_swerve = controller_type in ['swerve', 'non_coaxial', 'non_coaxial_60deg']

    # URDF / ros2_control config / nav2 공통 파라미터 분기
    if is_swerve:
        urdf_file = os.path.join(pkg_dir, 'urdf', 'swerve_robot.urdf')
        controller_config = os.path.join(pkg_dir, 'config', 'swerve_drive_controller.yaml')
        nav2_params_file = os.path.join(pkg_dir, 'config', 'nav2_params_swerve.yaml')
        robot_name = 'swerve_robot'
        spawn_z = '0.20'
    else:
        urdf_file = os.path.join(pkg_dir, 'urdf', 'differential_robot_ros2_control.urdf')
        controller_config = os.path.join(pkg_dir, 'config', 'diff_drive_controller.yaml')
        nav2_params_file = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
        robot_name = 'differential_robot'
        spawn_z = '0.15'

    # Static paths
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
            '-name', robot_name,
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', spawn_z,
        ],
    )

    # ========== 4. ros_gz_bridge ==========
    bridge_args = [
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
    ]
    if is_swerve:
        # Gazebo ground truth odom → ROS2 (OdometryPublisher 플러그인)
        bridge_args.append(
            f'/model/{robot_name}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry'
        )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=bridge_args,
    )

    # ========== 5. Controller Activation ==========
    # gz_ros2_control이 컨트롤러를 자동 로드/설정하므로, spawner와 레이스 컨디션 발생 가능.
    # ExecuteProcess로 controller_manager 준비 대기 + spawner + switch_controllers 폴백.
    def _activate_cmd(ctrl_name, extra_args=''):
        """단일 컨트롤러 활성화 명령 (spawner 시도 → switch 폴백)."""
        return (
            f'echo "[controllers] Activating {ctrl_name}..."; '
            f'ros2 run controller_manager spawner {ctrl_name}'
            f' -c /controller_manager --controller-manager-timeout 30'
            f' {extra_args} 2>&1 || '
            f'ros2 control switch_controllers'
            f' --activate {ctrl_name} -c /controller_manager 2>&1 || true; '
        )

    wait_for_cm = (
        'echo "[controllers] Waiting for controller_manager..."; '
        'for i in $(seq 1 30); do '
        '  ros2 service list 2>/dev/null | grep -q "/controller_manager/list_controllers" && '
        '  echo "[controllers] controller_manager ready" && break; '
        '  sleep 1; '
        'done; '
    )

    if is_swerve:
        activate_cmd = (
            wait_for_cm
            + _activate_cmd('joint_state_broadcaster')
            + _activate_cmd('steer_position_controller')
            + _activate_cmd('wheel_velocity_controller')
            + 'echo "[controllers] All swerve controllers activated"; '
            + 'ros2 control list_controllers -c /controller_manager 2>&1 || true'
        )
    else:
        activate_cmd = (
            wait_for_cm
            + _activate_cmd('joint_state_broadcaster')
            + _activate_cmd('diff_drive_controller',
                            f'--param-file {controller_config}')
            + 'echo "[controllers] All diff_drive controllers activated"; '
            + 'ros2 control list_controllers -c /controller_manager 2>&1 || true'
        )

    activate_controllers = ExecuteProcess(
        cmd=['bash', '-c', activate_cmd],
        output='screen',
    )

    # ========== 6. cmd_vel relay ==========
    if is_swerve:
        # SwerveKinematicsNode: IK 전용 (Gazebo ground truth odom 사용 시 FK/odom/TF 비활성화)
        cmd_vel_relay = Node(
            package='mpc_controller_ros2',
            executable='swerve_kinematics_node.py',
            name='swerve_kinematics_node',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'wheel_radius': 0.08,
                'wheel_x': 0.25,
                'wheel_y': 0.22,
                'publish_rate': 50.0,
                'cmd_vel_timeout': 0.5,
                'use_gazebo_odom': True,
            }],
        )

        # Gazebo ground truth odom → odom→base_link TF broadcast
        odom_to_tf = Node(
            package='mpc_controller_ros2',
            executable='odom_to_tf.py',
            name='odom_to_tf',
            output='screen',
            parameters=[{'use_sim_time': True}],
            remappings=[
                ('odom', f'/model/{robot_name}/odometry'),
            ],
        )
    else:
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
        cmd_vel_relay = ExecuteProcess(
            cmd=['python3', '-c', twist_to_stamped_cmd],
            output='screen'
        )

    # ========== 7. Nav2 Nodes + Lifecycle Bringup ==========
    # nav2_lifecycle_manager의 service call이 비결정적으로 실패하는 문제 우회:
    # - nav2 노드들은 별도 프로세스로 실행 (non-composition)
    # - nav2_lifecycle_bringup.py 스크립트가 `ros2 lifecycle set` subprocess로
    #   configure/activate 수행 (각 호출이 별도 DDS participant → 안정적)
    # - 동일 스크립트가 bond heartbeat를 publish하여 노드 self-deactivation 방지

    # --- Localization nodes ---
    # bond_heartbeat_period: 0.0 → bond 생성 건너뜀 (lifecycle_manager 없이 직접 활성화)
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'yaml_filename': map_file,
            'bond_heartbeat_period': 0.0,
        }],
    )
    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[nav2_params_file, {
            'use_sim_time': True,
            'bond_heartbeat_period': 0.0,
        }],
    )

    # --- Navigation nodes ---
    controller_server_node = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[nav2_params_file, controller_params_file, {
            'use_sim_time': True,
            'bond_heartbeat_period': 0.0,
        }],
        remappings=[('cmd_vel', '/cmd_vel_nav')],
    )
    planner_server_node = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[nav2_params_file, {
            'use_sim_time': True,
            'bond_heartbeat_period': 0.0,
        }],
    )
    behavior_server_node = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[nav2_params_file, {
            'use_sim_time': True,
            'bond_heartbeat_period': 0.0,
        }],
    )
    bt_navigator_node = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[nav2_params_file, {
            'use_sim_time': True,
            'bond_heartbeat_period': 0.0,
        }],
    )

    # --- Lifecycle bringup (bond-free) ---
    # bond_timeout=0.0으로 bond 비활성화 → subprocess lifecycle set으로 직접 활성화
    nav2_node_names = 'map_server,amcl,controller_server,planner_server,behavior_server,bt_navigator'
    lifecycle_bringup = Node(
        package='mpc_controller_ros2',
        executable='nav2_lifecycle_bringup.py',
        name='nav2_lifecycle_bringup',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'node_names': nav2_node_names,
            'max_retries': 10,
            'check_interval': 10.0,
        }],
    )

    # ========== 8. RVIZ ==========
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

        # stdout 버퍼링 활성화 (nav2 표준 설정)
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),

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

        # 5. Controllers (12s delay — Gazebo spawn 후 gz_ros2_control 안정화 대기)
        TimerAction(
            period=12.0,
            actions=[
                LogInfo(msg='Activating ros2_control controllers...'),
                activate_controllers,
                cmd_vel_relay,
                # Swerve: Gazebo ground truth odom → TF (AMCL/costmap에 필요)
                *([odom_to_tf] if is_swerve else []),
            ]
        ),

        # 6. Nav2 Localization nodes (20s delay)
        TimerAction(
            period=20.0,
            actions=[
                LogInfo(msg='Starting nav2 localization nodes...'),
                map_server_node,
                amcl_node,
            ]
        ),

        # 7. Nav2 Navigation nodes + lifecycle bringup (30s delay)
        TimerAction(
            period=30.0,
            actions=[
                LogInfo(msg=f'Starting nav2 navigation nodes ({controller_type} MPPI)...'),
                controller_server_node,
                planner_server_node,
                behavior_server_node,
                bt_navigator_node,
                # Lifecycle bringup: subprocess로 configure/activate + bond 유지
                lifecycle_bringup,
            ]
        ),

    ]

    # 8. RVIZ (60s delay) - headless 모드에서는 비활성화
    if not headless:
        nodes.append(
            TimerAction(
                period=60.0,
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
            description='MPPI controller type: "custom", "log", "tsallis", "risk_aware", '
                        '"svmpc", "smooth", "spline", "svg", "swerve", "non_coaxial", '
                        '"non_coaxial_60deg", or "nav2"'
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
