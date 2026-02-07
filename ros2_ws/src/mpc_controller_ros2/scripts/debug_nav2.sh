#!/bin/bash
# nav2 디버깅 스크립트
# 사용법: ./debug_nav2.sh [command]
# 예: ./debug_nav2.sh topics

source /opt/ros/jazzy/setup.bash
source /home/geonhee/toy_claude_project/ros2_ws/install/setup.bash

case "$1" in
    topics)
        echo "=== ROS2 토픽 목록 ==="
        ros2 topic list
        ;;
    tf)
        echo "=== TF 트리 ==="
        ros2 run tf2_tools view_frames
        ;;
    tf-echo)
        echo "=== map → odom TF ==="
        ros2 run tf2_ros tf2_echo map odom
        ;;
    scan)
        echo "=== /scan 토픽 확인 ==="
        ros2 topic echo /scan --once
        ;;
    map)
        echo "=== /map 토픽 확인 ==="
        ros2 topic info /map -v
        ;;
    odom)
        echo "=== /odom 토픽 확인 ==="
        ros2 topic echo /odom --once
        ;;
    lifecycle)
        echo "=== Lifecycle 노드 상태 ==="
        ros2 lifecycle list /map_server 2>/dev/null || echo "map_server not found"
        ros2 lifecycle list /amcl 2>/dev/null || echo "amcl not found"
        ros2 lifecycle list /controller_server 2>/dev/null || echo "controller_server not found"
        ;;
    nodes)
        echo "=== 활성 노드 ==="
        ros2 node list
        ;;
    gz-topics)
        echo "=== Gazebo 토픽 ==="
        gz topic -l
        ;;
    all)
        echo "=== 전체 상태 점검 ==="
        echo ""
        echo "1. ROS2 노드:"
        ros2 node list
        echo ""
        echo "2. TF 프레임:"
        ros2 run tf2_ros tf2_echo map odom 2>&1 | head -5 &
        sleep 2
        kill %1 2>/dev/null
        echo ""
        echo "3. 토픽 발행 확인:"
        echo "   /scan: $(ros2 topic hz /scan 2>&1 | head -1 &)"
        echo "   /map: $(ros2 topic info /map 2>&1 | head -1)"
        echo "   /odom: $(ros2 topic hz /odom 2>&1 | head -1 &)"
        sleep 2
        ;;
    *)
        echo "사용법: $0 {topics|tf|tf-echo|scan|map|odom|lifecycle|nodes|gz-topics|all}"
        echo ""
        echo "명령어 설명:"
        echo "  topics     - ROS2 토픽 목록"
        echo "  tf         - TF 트리 PDF 생성"
        echo "  tf-echo    - map→odom TF 실시간 확인"
        echo "  scan       - LiDAR 스캔 데이터 확인"
        echo "  map        - 맵 토픽 정보"
        echo "  odom       - 오도메트리 데이터 확인"
        echo "  lifecycle  - 노드 lifecycle 상태"
        echo "  nodes      - 활성 노드 목록"
        echo "  gz-topics  - Gazebo 토픽 목록"
        echo "  all        - 전체 상태 점검"
        ;;
esac
