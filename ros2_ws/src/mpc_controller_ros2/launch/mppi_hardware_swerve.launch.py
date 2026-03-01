#!/usr/bin/env python3
"""
Swerve Drive 하드웨어 배포 Launch 파일

Gazebo 시뮬레이션 계층을 제거하고, 실제 하드웨어 드라이버를 사용합니다.
nav2 MPPI 컨트롤러 플러그인은 변경 없이 그대로 사용됩니다.

┌─────────────────────────────────────────────────────┐
│  Sim (기존)              →    Hardware (M6)          │
├─────────────────────────────────────────────────────┤
│  gz sim                  →    제거                   │
│  ros_gz_bridge           →    제거                   │
│  GazeboSimSystem         →    사용자 SystemInterface │
│  Gazebo OdometryPublisher→    SwerveKinematics FK    │
│  use_sim_time: true      →    false                  │
│  없음                    →    SafetyMonitorNode      │
│                                                      │
│  [변경 없음] MPPI Plugin, nav2, MotionModel, TF     │
└─────────────────────────────────────────────────────┘

Launch 시퀀스:
  T=0s:  robot_state_publisher + controller_manager + safety_monitor
  T=3s:  controller spawners (joint_state, steer, wheel)
  T=5s:  swerve_kinematics_node (FK/odom/TF)
  T=8s:  nav2 localization (map_server, amcl)
  T=12s: nav2 navigation + lifecycle_bringup
  T=20s: rviz (선택)

실행 방법:
    # Swerve MPPI 하드웨어
    ros2 launch mpc_controller_ros2 mppi_hardware_swerve.launch.py \\
        controller:=swerve map:=my_map.yaml

    # Non-Coaxial Swerve
    ros2 launch mpc_controller_ros2 mppi_hardware_swerve.launch.py \\
        controller:=non_coaxial

    # RVIZ 포함
    ros2 launch mpc_controller_ros2 mppi_hardware_swerve.launch.py \\
        controller:=swerve rviz:=true

    # Emergency Stop
    ros2 service call /emergency_stop std_srvs/srv/SetBool "{data: true}"
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    TimerAction,
    LogInfo,
    DeclareLaunchArgument,
    OpaqueFunction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def launch_setup(context, *args, **kwargs):
    """OpaqueFunction callback — controller arg로 파라미터 파일 선택."""

    pkg_dir = get_package_share_directory('mpc_controller_ros2')

    # Resolve arguments
    controller_type = LaunchConfiguration('controller').perform(context)
    map_name = LaunchConfiguration('map').perform(context)
    hardware_params_name = LaunchConfiguration('hardware_params').perform(context)
    scan_topic = LaunchConfiguration('scan_topic').perform(context)
    show_rviz = LaunchConfiguration('rviz').perform(context).lower() == 'true'

    # 컨트롤러별 파라미터 파일 선택
    controller_map = {
        'swerve': (
            'nav2_params_swerve_hardware_mppi.yaml',
            'Swerve MPPI (Hardware)',
        ),
        'non_coaxial': (
            'nav2_params_non_coaxial_hardware_mppi.yaml',
            'Non-Coaxial Swerve MPPI (Hardware)',
        ),
    }
    if controller_type in controller_map:
        params_name, controller_label = controller_map[controller_type]
    else:
        params_name = 'nav2_params_swerve_hardware_mppi.yaml'
        controller_label = f'Swerve MPPI (Hardware, fallback from {controller_type})'

    controller_params_file = os.path.join(pkg_dir, 'config', params_name)
    nav2_params_file = os.path.join(pkg_dir, 'config', 'nav2_params_swerve_hardware.yaml')
    hardware_params_file = os.path.join(pkg_dir, 'config', hardware_params_name)
    map_file = os.path.join(pkg_dir, 'maps', map_name)

    # URDF (Gazebo 플러그인 없는 하드웨어용)
    urdf_file = os.path.join(pkg_dir, 'urdf', 'swerve_robot.urdf')
    rviz_config = os.path.join(pkg_dir, 'config', 'mpc_rviz.rviz')

    # ros2_control config (하드웨어 파라미터에서 가져옴)
    controller_config = hardware_params_file

    # Process xacro
    robot_description = xacro.process_file(
        urdf_file,
        mappings={'controller_config': controller_config}
    ).toxml()

    # ========== 1. Robot State Publisher (T=0s) ==========
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': False,
        }]
    )

    # ========== 2. ros2_control Controller Manager (T=0s) ==========
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        name='controller_manager',
        output='screen',
        parameters=[
            hardware_params_file,
            {'use_sim_time': False},
        ],
        remappings=[
            ('~/robot_description', '/robot_description'),
        ],
    )

    # ========== 3. Safety Monitor (T=0s) ==========
    safety_monitor = Node(
        package='mpc_controller_ros2',
        executable='safety_monitor.py',
        name='safety_monitor',
        output='screen',
        parameters=[hardware_params_file, {'use_sim_time': False}],
    )

    # ========== 4. Controller Spawners (T=3s) ==========
    def _activate_cmd(ctrl_name):
        """단일 컨트롤러 활성화 명령."""
        return (
            f'echo "[controllers] Activating {ctrl_name}..."; '
            f'ros2 run controller_manager spawner {ctrl_name}'
            f' -c /controller_manager --controller-manager-timeout 30'
            f' 2>&1 || '
            f'ros2 control switch_controllers'
            f' --activate {ctrl_name} -c /controller_manager 2>&1 || true; '
        )

    activate_cmd = (
        'echo "[controllers] Waiting for controller_manager..."; '
        'for i in $(seq 1 30); do '
        '  ros2 service list 2>/dev/null | grep -q "/controller_manager/list_controllers" && '
        '  echo "[controllers] controller_manager ready" && break; '
        '  sleep 1; '
        'done; '
        + _activate_cmd('joint_state_broadcaster')
        + _activate_cmd('steer_position_controller')
        + _activate_cmd('wheel_velocity_controller')
        + 'echo "[controllers] All swerve controllers activated"; '
        + 'ros2 control list_controllers -c /controller_manager 2>&1 || true'
    )

    activate_controllers = ExecuteProcess(
        cmd=['bash', '-c', activate_cmd],
        output='screen',
    )

    # ========== 5. Swerve Kinematics (T=5s) ==========
    # FK 기반 오도메트리 + odom→base_link TF (use_gazebo_odom=false)
    swerve_kinematics = Node(
        package='mpc_controller_ros2',
        executable='swerve_kinematics_node.py',
        name='swerve_kinematics_node',
        output='screen',
        parameters=[hardware_params_file, {'use_sim_time': False}],
    )

    # ========== 6. Nav2 Localization (T=8s) ==========
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': False,
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
            'use_sim_time': False,
            'bond_heartbeat_period': 0.0,
            'scan_topic': scan_topic,
        }],
    )

    # ========== 7. Nav2 Navigation (T=12s) ==========
    controller_server_node = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[nav2_params_file, controller_params_file, {
            'use_sim_time': False,
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
            'use_sim_time': False,
            'bond_heartbeat_period': 0.0,
        }],
    )
    behavior_server_node = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[nav2_params_file, {
            'use_sim_time': False,
            'bond_heartbeat_period': 0.0,
        }],
    )
    bt_navigator_node = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[nav2_params_file, {
            'use_sim_time': False,
            'bond_heartbeat_period': 0.0,
        }],
    )

    # Lifecycle bringup (bond-free)
    nav2_node_names = 'map_server,amcl,controller_server,planner_server,behavior_server,bt_navigator'
    lifecycle_bringup = Node(
        package='mpc_controller_ros2',
        executable='nav2_lifecycle_bringup.py',
        name='nav2_lifecycle_bringup',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'node_names': nav2_node_names,
            'max_retries': 10,
            'check_interval': 10.0,
        }],
    )

    # ========== 8. RVIZ (T=20s, 선택) ==========
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{'use_sim_time': False}],
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else []
    )

    # ========== Launch Sequence ==========
    nodes = [
        LogInfo(msg=f'[Hardware MPPI] {controller_label}'),
        LogInfo(msg=f'[Hardware MPPI] map: {map_file}'),
        LogInfo(msg=f'[Hardware MPPI] params: {controller_params_file}'),
        LogInfo(msg=f'[Hardware MPPI] scan_topic: {scan_topic}'),

        # T=0s: robot_state_publisher + controller_manager + safety_monitor
        robot_state_publisher,
        ros2_control_node,
        safety_monitor,

        # T=3s: controller spawners
        TimerAction(
            period=3.0,
            actions=[
                LogInfo(msg='Activating ros2_control controllers...'),
                activate_controllers,
            ]
        ),

        # T=5s: swerve kinematics (FK/odom/TF)
        TimerAction(
            period=5.0,
            actions=[
                LogInfo(msg='Starting swerve kinematics (FK odom + TF)...'),
                swerve_kinematics,
            ]
        ),

        # T=8s: nav2 localization
        TimerAction(
            period=8.0,
            actions=[
                LogInfo(msg='Starting nav2 localization nodes...'),
                map_server_node,
                amcl_node,
            ]
        ),

        # T=12s: nav2 navigation + lifecycle bringup
        TimerAction(
            period=12.0,
            actions=[
                LogInfo(msg=f'Starting nav2 navigation ({controller_type} MPPI)...'),
                controller_server_node,
                planner_server_node,
                behavior_server_node,
                bt_navigator_node,
                lifecycle_bringup,
            ]
        ),
    ]

    # T=20s: RVIZ (선택)
    if show_rviz:
        nodes.append(
            TimerAction(
                period=20.0,
                actions=[
                    LogInfo(msg='Starting RVIZ...'),
                    rviz,
                ]
            )
        )

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'controller',
            default_value='swerve',
            description='Controller type: "swerve" or "non_coaxial"'
        ),
        DeclareLaunchArgument(
            'map',
            default_value='maze_map.yaml',
            description='Map YAML file name (in maps/ directory)'
        ),
        DeclareLaunchArgument(
            'hardware_params',
            default_value='swerve_hardware_params.yaml',
            description='Hardware params YAML file name (in config/ directory)'
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='false',
            description='Enable RVIZ visualization'
        ),
        DeclareLaunchArgument(
            'scan_topic',
            default_value='/scan',
            description='LiDAR scan topic'
        ),

        OpaqueFunction(function=launch_setup),
    ])
