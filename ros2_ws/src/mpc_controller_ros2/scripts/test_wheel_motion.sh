#!/bin/bash

# Gazebo에서 로봇 바퀴 모션 테스트 스크립트

echo "==================================================="
echo "로봇 바퀴 모션 테스트"
echo "==================================================="

# ROS2 환경 설정
source install/setup.bash

# Gazebo 실행 (백그라운드)
echo "1. Gazebo Harmonic 실행 중..."
ros2 launch mpc_controller_ros2 gazebo_harmonic_test.launch.py > /tmp/gazebo_wheel_test.log 2>&1 &
LAUNCH_PID=$!
sleep 8

# 로봇 스폰
echo "2. 로봇 스폰 중..."
SDF_FILE="install/mpc_controller_ros2/share/mpc_controller_ros2/models/differential_robot/model.sdf"
gz service -s /world/mppi_test_world/create \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --timeout 3000 \
  --req "sdf_filename: \"$SDF_FILE\", name: \"differential_robot\", pose: {position: {x: 0.0, y: 0.0, z: 0.15}}"

sleep 3

# 초기 위치 확인
echo ""
echo "3. 초기 위치 확인..."
timeout 2 ros2 topic echo /odom --once | grep -A3 "position:"

# 테스트 1: 전진
echo ""
echo "==================================================="
echo "테스트 1: 전진 (0.5 m/s, 5초)"
echo "==================================================="
ros2 topic pub --times 100 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}}" > /dev/null 2>&1 &
PUB_PID=$!
sleep 5
kill $PUB_PID 2>/dev/null

# 정지 및 위치 확인
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}" > /dev/null 2>&1
sleep 1
echo "전진 후 위치:"
timeout 2 ros2 topic echo /odom --once | grep -A3 "position:"

# 테스트 2: 회전
echo ""
echo "==================================================="
echo "테스트 2: 제자리 회전 (0.5 rad/s, 3초)"
echo "==================================================="
ros2 topic pub --times 60 /cmd_vel geometry_msgs/msg/Twist "{angular: {z: 0.5}}" > /dev/null 2>&1 &
PUB_PID=$!
sleep 3
kill $PUB_PID 2>/dev/null

ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}" > /dev/null 2>&1
sleep 1
echo "회전 후 위치:"
timeout 2 ros2 topic echo /odom --once | grep -A3 "position:"

# 테스트 3: 원호 주행
echo ""
echo "==================================================="
echo "테스트 3: 원호 주행 (v=0.3, ω=0.3, 5초)"
echo "==================================================="
ros2 topic pub --times 100 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}, angular: {z: 0.3}}" > /dev/null 2>&1 &
PUB_PID=$!
sleep 5
kill $PUB_PID 2>/dev/null

ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}" > /dev/null 2>&1
sleep 1
echo "원호 주행 후 위치:"
timeout 2 ros2 topic echo /odom --once | grep -A3 "position:"

# 테스트 4: 후진
echo ""
echo "==================================================="
echo "테스트 4: 후진 (-0.3 m/s, 3초)"
echo "==================================================="
ros2 topic pub --times 60 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: -0.3}}" > /dev/null 2>&1 &
PUB_PID=$!
sleep 3
kill $PUB_PID 2>/dev/null

ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}" > /dev/null 2>&1
sleep 1
echo "후진 후 위치:"
timeout 2 ros2 topic echo /odom --once | grep -A3 "position:"

# 최종 정지
echo ""
echo "==================================================="
echo "모든 테스트 완료!"
echo "==================================================="
echo ""
echo "Gazebo GUI에서 로봇이 실제로 움직였는지 확인하세요."
echo ""
echo "종료하려면 Ctrl+C를 누르세요..."
echo "Gazebo 로그: /tmp/gazebo_wheel_test.log"
echo ""

# 사용자 입력 대기
read -p "Enter를 누르면 Gazebo가 종료됩니다: "

# Gazebo 종료
pkill -f "gz sim"
kill $LAUNCH_PID 2>/dev/null

echo "테스트 완료 및 종료"
