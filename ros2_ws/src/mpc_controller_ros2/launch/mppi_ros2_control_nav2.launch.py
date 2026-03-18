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

    # Narrow Passage 환경 (0.8m 통과폭 테스트)
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py \
        world:=narrow_passage_world.world map:=narrow_passage_map.yaml

    # Random Forest 환경 (장애물 회피 테스트)
    ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py \
        world:=random_forest_world.world map:=random_forest_map.yaml

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
    controller:=ackermann    → Ackermann MPPI (motion_model=ackermann, bicycle model)
    controller:=nav2         → nav2 기본 MPPI (nav2_mppi_controller::MPPIController)
    controller:=stress_test  → Stress Test MPPI (고속 v_max=1.5 + CBF + 동적 장애물)
    controller:=shield       → Shield-MPPI (per-step CBF + BR-MPPI + Conformal)
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
    RegisterEventHandler,
    EmitEvent,
)
from launch.events import Shutdown as ShutdownEvent
from launch.event_handlers import OnProcessExit, OnShutdown
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
    nav2_stress = LaunchConfiguration('nav2_stress').perform(context).lower() == 'true'

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
        'biased': ('nav2_params_biased_mppi.yaml',
                   'Biased-MPPI (mpc_controller_ros2::BiasedMPPIControllerPlugin)'),
        'dial': ('nav2_params_dial_mppi.yaml',
                 'DIAL-MPPI (mpc_controller_ros2::DialMPPIControllerPlugin)'),
        'dial_swerve': ('nav2_params_dial_swerve_mppi.yaml',
                        'DIAL-MPPI Swerve (motion_model=swerve)'),
        'dial_non_coaxial': ('nav2_params_dial_non_coaxial_mppi.yaml',
                             'DIAL-MPPI Non-Coaxial (motion_model=non_coaxial_swerve)'),
        'stress_test': ('nav2_params_stress_test.yaml',
                        'Stress Test MPPI (고속 + CBF + 동적 장애물)'),
        'ackermann': ('nav2_params_ackermann_mppi.yaml',
                      'Ackermann MPPI (motion_model=ackermann, bicycle model)'),
        'shield': ('nav2_params_shield_mppi.yaml',
                   'Shield-MPPI (per-step CBF + BR-MPPI + Conformal)'),
        'ilqr_mppi': ('nav2_params_ilqr_mppi.yaml',
                      'iLQR-MPPI (iLQR warm-start + MPPI sampling)'),
        'cs_mppi': ('nav2_params_cs_mppi.yaml',
                    'CS-MPPI (Covariance Steering, CoVO-MPC CoRL 2023)'),
        'pi_mppi': ('nav2_params_pi_mppi.yaml',
                    'pi-MPPI (Projection MPPI, ADMM QP, RA-L 2025)'),
        'hybrid_swerve': ('nav2_params_hybrid_swerve_mppi.yaml',
                          'MPPI-H Hybrid Swerve (IROS 2024, Low-D↔4D)'),
        'hybrid_non_coaxial': ('nav2_params_hybrid_non_coaxial_mppi.yaml',
                               'MPPI-H Hybrid Non-Coaxial (IROS 2024, Low-D↔4D)'),
        'log_swerve': ('nav2_params_log_swerve_mppi.yaml',
                       'Log-MPPI Swerve (log-space weights + swerve)'),
        'tsallis_swerve': ('nav2_params_tsallis_swerve_mppi.yaml',
                           'Tsallis-MPPI Swerve (q-exponential + swerve)'),
        'smooth_swerve': ('nav2_params_smooth_swerve_mppi.yaml',
                          'Smooth-MPPI Swerve (Δu space + jerk + swerve)'),
        'biased_swerve': ('nav2_params_biased_swerve_mppi.yaml',
                          'Biased-MPPI Swerve (ancillary + swerve, RA-L 2024)'),
        'shield_swerve': ('nav2_params_shield_swerve_mppi.yaml',
                          'Shield-MPPI Swerve (CBF + BR-MPPI + swerve)'),
        'cs_swerve': ('nav2_params_cs_swerve_mppi.yaml',
                      'CS-MPPI Swerve (Covariance Steering + swerve)'),
        'pi_swerve': ('nav2_params_pi_swerve_mppi.yaml',
                      'pi-MPPI Swerve (ADMM QP projection + swerve)'),
        'svg_swerve': ('nav2_params_svg_swerve_mppi.yaml',
                       'SVG-MPPI Swerve (Guide SVGD + swerve)'),
        'ilqr_swerve': ('nav2_params_ilqr_swerve_mppi.yaml',
                        'iLQR-MPPI Swerve (iLQR warm-start + swerve)'),
        'adaptive_shield': ('nav2_params_adaptive_shield_mppi.yaml',
                            'Adaptive Shield-MPPI (distance/velocity adaptive CBF)'),
        'clf_cbf': ('nav2_params_clf_cbf_mppi.yaml',
                    'CLF-CBF-MPPI (unified CLF+CBF QP safety filter)'),
        'predictive_safety': ('nav2_params_predictive_safety_mppi.yaml',
                              'Predictive Safety MPPI (N-step CBF projection)'),
        'lp': ('nav2_params_lp_mppi.yaml',
               'LP-MPPI (Low-Pass filtering for smooth control)'),
        'halton': ('nav2_params_halton_mppi.yaml',
                   'Halton-MPPI (low-discrepancy sequence sampling)'),
        'feedback': ('nav2_params_feedback_mppi.yaml',
                     'Feedback-MPPI (Riccati time-varying feedback gains)'),
        'tube_mppi': ('nav2_params_tube_mppi.yaml',
                      'Tube-MPPI (nominal state MPPI + body frame feedback)'),
    }
    if controller_type in controller_map:
        params_name, controller_label = controller_map[controller_type]
    else:
        params_name = 'nav2_params_custom_mppi.yaml'
        controller_label = '커스텀 MPPI (mpc_controller_ros2::MPPIControllerPlugin)'

    controller_params_file = os.path.join(pkg_dir, 'config', params_name)

    # Swerve drive 판별
    is_swerve = controller_type in [
        'swerve', 'non_coaxial', 'non_coaxial_60deg',
        'dial_swerve', 'dial_non_coaxial',
        'hybrid_swerve', 'hybrid_non_coaxial',
        'log_swerve', 'tsallis_swerve', 'smooth_swerve',
        'biased_swerve', 'shield_swerve', 'cs_swerve',
        'pi_swerve', 'svg_swerve', 'ilqr_swerve',
    ]

    # Ackermann 판별
    is_ackermann = controller_type in ['ackermann']

    # URDF / ros2_control config / nav2 공통 파라미터 분기
    if is_ackermann:
        urdf_file = os.path.join(pkg_dir, 'urdf', 'ackermann_robot.urdf')
        controller_config = os.path.join(pkg_dir, 'config', 'ackermann_steering_controller.yaml')
        nav2_params_file = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
        robot_name = 'ackermann_robot'
        spawn_z = '0.15'
    elif is_swerve:
        urdf_file = os.path.join(pkg_dir, 'urdf', 'swerve_robot.urdf')
        controller_config = os.path.join(pkg_dir, 'config', 'swerve_drive_controller.yaml')
        swerve_nav2 = 'nav2_params_swerve_stress.yaml' if nav2_stress else 'nav2_params_swerve.yaml'
        nav2_params_file = os.path.join(pkg_dir, 'config', swerve_nav2)
        robot_name = 'swerve_robot'
        spawn_z = '0.20'
    else:
        urdf_file = os.path.join(pkg_dir, 'urdf', 'differential_robot_ros2_control.urdf')
        # stress_test: 고속 diff_drive 설정 (v_max=1.5 대응)
        if controller_type == 'stress_test':
            controller_config = os.path.join(pkg_dir, 'config', 'diff_drive_controller_high_speed.yaml')
        else:
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
        sigterm_timeout='5',
        sigkill_timeout='3',
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
    if is_swerve or is_ackermann:
        # Gazebo ground truth odom → ROS2 (OdometryPublisher 플러그인)
        bridge_args.append(
            f'/model/{robot_name}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry'
        )
    if is_swerve:
        # IMU + Depth Camera bridges (swerve robot only)
        bridge_args.append('/imu@sensor_msgs/msg/Imu[gz.msgs.IMU')
        bridge_args.append('/depth@sensor_msgs/msg/Image[gz.msgs.Image')
        bridge_args.append('/depth/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked')

    # 동적 장애물 cmd_vel bridge (stress_test 월드 사용 시)
    if controller_type == 'stress_test':
        for obs_name in ['dynamic_obstacle_slow', 'dynamic_obstacle_medium',
                         'dynamic_obstacle_fast', 'dynamic_obstacle_cross']:
            bridge_args.append(
                f'/model/{obs_name}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist'
            )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        sigterm_timeout='3',
        sigkill_timeout='2',
        parameters=[{
            'use_sim_time': True,
            # /clock QoS → reliable (best_effort의 UDP 순서 역전 방지)
            # → TF "jump back in time" 해결
            'qos_overrides': {
                '/clock': {
                    'publisher': {
                        'reliability': 'reliable',
                        'history': 'keep_last',
                        'depth': 10,
                    }
                }
            },
        }],
        arguments=bridge_args,
    )

    # ========== 5. Controller Activation ==========
    # 공식 Jazzy gz_ros2_control 패턴:
    #   OnProcessExit(spawn_robot) → JSB spawner → DD/Swerve spawner
    #
    # spawner가 내부적으로 controller_manager 서비스를 대기하므로
    # 별도 wait 스크립트가 필요하지 않음.
    # configure_controller 실패 시 bash retry loop로 폴백.

    def _make_spawner_cmd(ctrl_name, extra_args=''):
        """spawner를 bash retry loop로 감싼 명령어 생성.

        gz_ros2_control이 URDF <parameters>에서 컨트롤러를 자동 로드하면
        spawner는 "already loaded"로 parameter 설정을 건너뛰고 configure가 실패.
        → 매 시도마다 unload 후 clean load로 해결.
        """
        return [
            'bash', '-c',
            f'for attempt in 1 2 3 4 5 6 7 8 9 10; do '
            f'  echo "[controllers] Spawning {ctrl_name} (attempt $attempt)..."; '
            # 자동 로드된 컨트롤러 unload (spawner가 clean load 가능하도록)
            f'  ros2 control unload_controller {ctrl_name}'
            f'    -c /controller_manager 2>/dev/null || true; '
            f'  sleep 0.5; '
            f'  ros2 run controller_manager spawner {ctrl_name}'
            f'    -c /controller_manager'
            f'    --controller-manager-timeout 120'
            f'    {extra_args} 2>&1 && '
            f'  echo "[controllers] {ctrl_name} activated" && exit 0; '
            f'  echo "[controllers] {ctrl_name} attempt $attempt failed, retrying in 3s..."; '
            f'  sleep 3; '
            f'done; '
            f'echo "[controllers] WARNING: {ctrl_name} failed after 10 attempts"; '
            f'exit 1; '
        ]

    def _on_spawner_failure(ctrl_name):
        """spawner 실패 시 Shutdown 이벤트를 발생시키는 이벤트 핸들러 생성."""
        def _check_exit(event, context):
            retcode = event.returncode
            if retcode != 0:
                return [
                    LogInfo(msg=f'[FATAL] {ctrl_name} spawner failed (exit code {retcode}) — shutting down'),
                    EmitEvent(event=ShutdownEvent(reason=f'{ctrl_name} spawner failed after 10 retries')),
                ]
            return []
        return _check_exit

    jsb_spawner = ExecuteProcess(
        cmd=_make_spawner_cmd('joint_state_broadcaster'),
        output='screen',
    )

    if is_ackermann:
        ackermann_spawner = ExecuteProcess(
            cmd=_make_spawner_cmd('ackermann_steering_controller',
                                  f'--param-file {controller_config}'),
            output='screen',
        )
    elif is_swerve:
        steer_spawner = ExecuteProcess(
            cmd=_make_spawner_cmd('steer_position_controller',
                                  f'--param-file {controller_config}'),
            output='screen',
        )
        wheel_spawner = ExecuteProcess(
            cmd=_make_spawner_cmd('wheel_velocity_controller',
                                  f'--param-file {controller_config}'),
            output='screen',
        )
    else:
        dd_spawner = ExecuteProcess(
            cmd=_make_spawner_cmd('diff_drive_controller',
                                  f'--param-file {controller_config}'),
            output='screen',
        )

    # ========== 6. cmd_vel relay ==========
    if is_ackermann:
        # Ackermann: Twist → TwistStamped relay for ackermann_steering_controller/reference
        ack_twist_to_stamped_cmd = '''
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, TwistStamped

class TwistStamper(Node):
    def __init__(self):
        super().__init__("ackermann_twist_stamper")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(TwistStamped, "/ackermann_steering_controller/reference", qos)
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
            cmd=['python3', '-c', ack_twist_to_stamped_cmd],
            output='screen'
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
    elif is_swerve:
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
    # sigterm/sigkill timeout: 좀비 프로세스 방지 (Shutdown 시 확실히 종료)
    _node_kill_args = {
        'sigterm_timeout': '5',
        'sigkill_timeout': '3',
    }

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
        **_node_kill_args,
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
        **_node_kill_args,
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
        **_node_kill_args,
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
        **_node_kill_args,
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
        **_node_kill_args,
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
        **_node_kill_args,
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
        **_node_kill_args,
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

    # ========== Nav2 노드 그룹 (이벤트 체인에서 사용) ==========
    nav2_start_actions = [
        LogInfo(msg=f'Controllers ready — starting nav2 ({controller_type} MPPI)...'),
        map_server_node,
        amcl_node,
        controller_server_node,
        planner_server_node,
        behavior_server_node,
        bt_navigator_node,
        lifecycle_bringup,
    ]

    # ========== Launch Nodes ==========
    #
    # 이벤트 기반 실행 순서 (하드코딩 타이머 제거):
    #   1. Gazebo + RSP + Bridge + RVIZ (즉시)
    #   2. spawn_robot (2s 딜레이 — Gazebo 월드 로드 최소 대기)
    #   3. spawn_robot 종료 → JSB spawner + cmd_vel relay
    #   4. JSB 종료 → DD/Swerve/Ackermann spawner
    #   5. 마지막 controller spawner 종료 → nav2 전체 노드 + lifecycle bringup
    #
    # Before: 하드코딩 타이머 (5s + 20s + 30s + 60s) → ~60초
    # After:  이벤트 체인 → ~10초 (각 단계 완료 즉시 다음 시작)

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

        # 4. Spawn robot (2s delay — Gazebo 월드 로드 최소 대기)
        TimerAction(
            period=2.0,
            actions=[
                LogInfo(msg='Spawning robot...'),
                spawn_robot,
            ]
        ),

        # 5. Controller event chain: spawn_robot → JSB → DD/Swerve/Ackermann → nav2
        #    spawner 내부 bash retry loop가 controller_manager 대기 + 재시도 처리
        #    실패 시 (exit code != 0) → Shutdown 이벤트 발생
        RegisterEventHandler(
            OnProcessExit(
                target_action=spawn_robot,
                on_exit=[
                    LogInfo(msg='Robot spawned — activating controllers...'),
                    jsb_spawner,
                    # cmd_vel relay는 컨트롤러와 독립 — 병렬 시작
                    cmd_vel_relay,
                    *([odom_to_tf] if (is_swerve or is_ackermann) else []),
                ],
            )
        ),

        # JSB spawner 실패 감지
        RegisterEventHandler(
            OnProcessExit(
                target_action=jsb_spawner,
                on_exit=_on_spawner_failure('joint_state_broadcaster'),
            )
        ),
    ]

    # cleanup 명령: 모든 관련 프로세스 강제 종료
    # trap '' TERM: launch의 Shutdown SIGTERM이 cleanup bash 자체를 죽이는 것 방지
    _cleanup_cmd = (
        'trap "" TERM; '
        'pkill -9 -f "gz sim" 2>/dev/null; '
        'pkill -9 -f "parameter_bridge" 2>/dev/null; '
        'pkill -9 -f "twist_stamper" 2>/dev/null; '
        'pkill -9 -f "swerve_kinematics_node" 2>/dev/null; '
        'pkill -9 -f "odom_to_tf" 2>/dev/null; '
        'pkill -9 -f "nav2_lifecycle_bringup" 2>/dev/null; '
        'pkill -9 -f "robot_state_publisher" 2>/dev/null; '
        'pkill -9 -f "map_server --ros" 2>/dev/null; '
        'pkill -9 -f "amcl --ros" 2>/dev/null; '
        'pkill -9 -f "controller_server --ros" 2>/dev/null; '
        'pkill -9 -f "planner_server --ros" 2>/dev/null; '
        'pkill -9 -f "behavior_server --ros" 2>/dev/null; '
        'pkill -9 -f "bt_navigator --ros" 2>/dev/null; '
        'pkill -9 -f "rviz2" 2>/dev/null; '
        'pkill -9 -f "ros2 lifecycle" 2>/dev/null; '
        'echo "[Shutdown] Cleanup complete"'
    )

    def _make_cleanup_action():
        return ExecuteProcess(
            cmd=['bash', '-c', _cleanup_cmd],
            output='screen',
            sigterm_timeout='2',
            sigkill_timeout='2',
        )

    # Gazebo 종료 → cleanup + Shutdown (좀비 프로세스 방지)
    nodes.append(
        RegisterEventHandler(
            OnProcessExit(
                target_action=gz_sim,
                on_exit=[
                    LogInfo(msg='[Shutdown] Gazebo exited — cleaning up all nodes'),
                    _make_cleanup_action(),
                    EmitEvent(event=ShutdownEvent(reason='Gazebo process exited')),
                ],
            )
        )
    )

    # Shutdown 시 남은 프로세스 강제 정리
    nodes.append(
        RegisterEventHandler(
            OnShutdown(
                on_shutdown=[
                    LogInfo(msg='[Shutdown] Launch shutting down — cleaning up...'),
                    _make_cleanup_action(),
                ],
            )
        )
    )

    # RVIZ 즉시 시작 (headless 제외) — 토픽 구독이므로 대기 불필요
    if not headless:
        nodes.append(rviz)
        # RVIZ 종료 → cleanup + Shutdown
        nodes.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=rviz,
                    on_exit=[
                        LogInfo(msg='[Shutdown] RVIZ closed — cleaning up all nodes'),
                        _make_cleanup_action(),
                        EmitEvent(event=ShutdownEvent(reason='RVIZ closed')),
                    ],
                )
            )
        )

    if is_ackermann:
        nodes.extend([
            RegisterEventHandler(
                OnProcessExit(
                    target_action=jsb_spawner,
                    on_exit=[ackermann_spawner],
                )
            ),
            # Ackermann spawner 실패 감지
            RegisterEventHandler(
                OnProcessExit(
                    target_action=ackermann_spawner,
                    on_exit=_on_spawner_failure('ackermann_steering_controller'),
                )
            ),
            # 마지막 controller → nav2 전체 시작
            RegisterEventHandler(
                OnProcessExit(
                    target_action=ackermann_spawner,
                    on_exit=[
                        LogInfo(msg='Ackermann steering controller activated'),
                        *nav2_start_actions,
                    ],
                )
            ),
        ])
    elif is_swerve:
        nodes.extend([
            RegisterEventHandler(
                OnProcessExit(
                    target_action=jsb_spawner,
                    on_exit=[steer_spawner],
                )
            ),
            # Steer spawner 실패 감지
            RegisterEventHandler(
                OnProcessExit(
                    target_action=steer_spawner,
                    on_exit=_on_spawner_failure('steer_position_controller'),
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=steer_spawner,
                    on_exit=[wheel_spawner],
                )
            ),
            # Wheel spawner 실패 감지
            RegisterEventHandler(
                OnProcessExit(
                    target_action=wheel_spawner,
                    on_exit=_on_spawner_failure('wheel_velocity_controller'),
                )
            ),
            # 마지막 controller → nav2 전체 시작
            RegisterEventHandler(
                OnProcessExit(
                    target_action=wheel_spawner,
                    on_exit=[
                        LogInfo(msg='All swerve controllers activated'),
                        *nav2_start_actions,
                    ],
                )
            ),
        ])
    else:
        nodes.extend([
            RegisterEventHandler(
                OnProcessExit(
                    target_action=jsb_spawner,
                    on_exit=[dd_spawner],
                )
            ),
            # DD spawner 실패 감지
            RegisterEventHandler(
                OnProcessExit(
                    target_action=dd_spawner,
                    on_exit=_on_spawner_failure('diff_drive_controller'),
                )
            ),
            # 마지막 controller → nav2 전체 시작
            RegisterEventHandler(
                OnProcessExit(
                    target_action=dd_spawner,
                    on_exit=[
                        LogInfo(msg='All diff_drive controllers activated'),
                        *nav2_start_actions,
                    ],
                )
            ),
        ])

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
            default_value='swerve',
            description='MPPI controller type: "custom", "log", "tsallis", "risk_aware", '
                        '"svmpc", "smooth", "spline", "svg", "biased", "swerve", '
                        '"non_coaxial", "non_coaxial_60deg", "ackermann", "shield", "adaptive_shield", "clf_cbf", "predictive_safety", "stress_test", or "nav2"'
        ),
        DeclareLaunchArgument(
            'headless',
            default_value='false',
            description='Run in headless mode (no Gazebo GUI, no RVIZ)'
        ),
        DeclareLaunchArgument(
            'nav2_stress',
            default_value='false',
            description='Use stress-test nav2 params (minimal inflation, slow planner)'
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
