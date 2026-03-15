#!/bin/bash
# cleanup_launch.sh — MPPI launch 관련 모든 프로세스 강제 종료
#
# 사용법:
#   ros2_ws/src/mpc_controller_ros2/scripts/cleanup_launch.sh
#
# launch 종료 후 좀비 프로세스가 남을 때 사용

PATTERNS=(
    "gz sim"
    "ros2 launch mpc_controller_ros2"
    "parameter_bridge"
    "twist_stamper"
    "robot_state_publisher"
    "nav2_lifecycle_bringup"
    "map_server --ros"
    "amcl --ros"
    "controller_server --ros"
    "planner_server --ros"
    "behavior_server --ros"
    "bt_navigator --ros"
    "swerve_kinematics_node"
    "odom_to_tf"
    "rviz2"
    "ros2 lifecycle"
)

echo "[cleanup] Sending SIGTERM..."
for p in "${PATTERNS[@]}"; do
    pkill -TERM -f "$p" 2>/dev/null
done

sleep 2

echo "[cleanup] Sending SIGKILL to survivors..."
for p in "${PATTERNS[@]}"; do
    pkill -9 -f "$p" 2>/dev/null
done

sleep 1

REMAINING=$(ps aux | grep -E "(gz sim|ros2 launch|parameter_bridge|twist_stamper|nav2|controller_server|planner_server|amcl|map_server|behavior_server|bt_navigator|rviz2|swerve_kinematics|odom_to_tf|lifecycle_bringup)" | grep -v grep | grep -v daemon | grep -v cleanup_launch | wc -l)
echo "[cleanup] Done. Remaining processes: $REMAINING"
