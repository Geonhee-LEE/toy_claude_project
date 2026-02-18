#!/bin/bash

# nav2 + MPPI 빠른 테스트 스크립트

echo "=========================================="
echo "nav2 + MPPI Controller 빠른 테스트"
echo "=========================================="

source install/setup.bash

echo ""
echo "1. Launch 파일 존재 확인..."
if [ -f "install/mpc_controller_ros2/share/mpc_controller_ros2/launch/mppi_nav2_gazebo.launch.py" ]; then
    echo "  ✓ mppi_nav2_gazebo.launch.py 존재"
else
    echo "  ✗ Launch 파일 없음!"
    exit 1
fi

echo ""
echo "2. nav2 파라미터 파일 확인..."
if [ -f "install/mpc_controller_ros2/share/mpc_controller_ros2/config/nav2_params.yaml" ]; then
    echo "  ✓ nav2_params.yaml 존재"
else
    echo "  ✗ 파라미터 파일 없음!"
    exit 1
fi

echo ""
echo "3. 로봇 모델 파일 확인..."
if [ -f "install/mpc_controller_ros2/share/mpc_controller_ros2/models/differential_robot/model.sdf" ]; then
    echo "  ✓ model.sdf 존재"
else
    echo "  ✗ 모델 파일 없음!"
    exit 1
fi

echo ""
echo "4. Goal 전송 스크립트 확인..."
if [ -f "install/mpc_controller_ros2/lib/mpc_controller_ros2/send_nav_goal.py" ]; then
    echo "  ✓ send_nav_goal.py 설치됨"
else
    echo "  ✗ 스크립트 없음!"
    exit 1
fi

echo ""
echo "5. 필수 ROS2 패키지 확인..."

REQUIRED_PKGS=(
    "nav2_controller"
    "nav2_planner"
    "nav2_bt_navigator"
    "nav2_behaviors"
    "nav2_lifecycle_manager"
    "nav2_map_server"
    "nav2_velocity_smoother"
    "ros_gz_bridge"
)

for pkg in "${REQUIRED_PKGS[@]}"; do
    if ros2 pkg list | grep -q "^${pkg}$"; then
        echo "  ✓ $pkg"
    else
        echo "  ✗ $pkg 설치 필요!"
        echo "    sudo apt install ros-jazzy-${pkg//_/-}"
    fi
done

echo ""
echo "=========================================="
echo "검증 완료!"
echo "=========================================="
echo ""
echo "실행 방법:"
echo "  ros2 launch mpc_controller_ros2 mppi_nav2_gazebo.launch.py"
echo ""
echo "Goal 전송:"
echo "  ros2 run mpc_controller_ros2 send_nav_goal.py --x 5.0 --y 0.0"
echo ""
echo "자세한 사용법:"
echo "  cat src/mpc_controller_ros2/MPPI_NAV2_GUIDE.md"
echo ""
