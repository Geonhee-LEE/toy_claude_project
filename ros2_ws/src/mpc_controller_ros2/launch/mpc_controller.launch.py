"""
MPC Controller Launch File.

MPC 컨트롤러 노드를 실행하는 launch 파일입니다.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch 설정을 생성합니다."""

    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('mpc_controller_ros2'),
            'config',
            'mpc_params.yaml'
        ]),
        description='Path to MPC parameters YAML file'
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RVIZ for visualization if true'
    )

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([
            FindPackageShare('mpc_controller_ros2'),
            'config',
            'mpc_rviz.rviz'
        ]),
        description='Path to RVIZ config file'
    )

    # MPC Controller Node
    mpc_controller_node = Node(
        package='mpc_controller_ros2',
        executable='mpc_controller_node',
        name='mpc_controller_node',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        remappings=[
            # 필요시 토픽 리매핑 설정
            # ('/odom', '/robot/odom'),
            # ('/reference_path', '/planner/path'),
        ]
    )

    # RVIZ Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )

    return LaunchDescription([
        use_sim_time_arg,
        config_file_arg,
        use_rviz_arg,
        rviz_config_arg,
        mpc_controller_node,
        rviz_node,
    ])
