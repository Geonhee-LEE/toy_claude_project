# Gazebo 설치 가이드 (ROS2 Jazzy)

## 에러 원인
```
package 'gazebo_ros' not found
```

ROS2 Jazzy는 **Gazebo Harmonic (새 버전)**을 사용합니다. 기존 Gazebo Classic이 아닙니다.

## 설치 방법

### 1. Gazebo Harmonic 및 ROS2 연동 패키지 설치

```bash
# 1단계: Gazebo Harmonic 설치
sudo apt update
sudo apt install -y gz-harmonic

# 2단계: ROS2 Jazzy용 Gazebo 연동 패키지 설치
sudo apt install -y \
  ros-jazzy-ros-gz \
  ros-jazzy-ros-gz-sim \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-ros-gz-image

# 3단계: Nav2 관련 패키지 설치
sudo apt install -y \
  ros-jazzy-nav2-bringup \
  ros-jazzy-navigation2 \
  ros-jazzy-nav2-map-server \
  ros-jazzy-nav2-lifecycle-manager

# 4단계: 추가 유틸리티
sudo apt install -y \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-joint-state-publisher
```

### 2. 설치 확인

```bash
# Gazebo Harmonic 설치 확인
gz sim --version

# ROS2 패키지 확인
ros2 pkg list | grep ros_gz
```

**예상 출력:**
```
ros_gz
ros_gz_bridge
ros_gz_image
ros_gz_sim
```

### 3. 환경 변수 설정 (필요시)

```bash
# .bashrc에 추가
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Launch 파일 수정 필요

ROS2 Jazzy는 Gazebo Harmonic을 사용하므로, launch 파일에서 다음과 같이 수정해야 합니다:

### 변경 전 (Gazebo Classic)
```python
from launch.launch_description_sources import PythonLaunchDescriptionSource
gazebo_ros_pkg_dir = get_package_share_directory('gazebo_ros')

gazebo = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(gazebo_ros_pkg_dir, 'launch', 'gazebo.launch.py')
    )
)
```

### 변경 후 (Gazebo Harmonic)
```python
from launch_ros.actions import Node

# Gazebo Harmonic (gz sim)을 직접 실행
gz_sim = Node(
    package='ros_gz_sim',
    executable='create',
    arguments=['-r', '-v4', world_file],
    output='screen'
)
```

## 설치 후 실행

```bash
cd ~/toy_claude_project/ros2_ws
source install/setup.bash

# Gazebo 테스트
ros2 launch mpc_controller_ros2 mppi_navigation.launch.py
```

## 트러블슈팅

### 문제 1: `gz sim` 명령이 없음
**해결:**
```bash
sudo apt install -y gz-harmonic
gz sim --version
```

### 문제 2: ROS2 패키지를 찾을 수 없음
**해결:**
```bash
sudo apt install -y ros-jazzy-ros-gz ros-jazzy-ros-gz-sim
ros2 pkg list | grep ros_gz
```

### 문제 3: 기존 Gazebo Classic과 충돌
**해결:**
```bash
# Gazebo Classic 제거 (주의!)
sudo apt remove gazebo11 gazebo-classic

# Gazebo Harmonic만 사용
sudo apt install -y gz-harmonic
```

## 대안: Gazebo Classic 사용 (권장하지 않음)

ROS2 Jazzy는 공식적으로 Gazebo Harmonic을 지원하지만, 기존 Gazebo Classic을 사용하려면:

```bash
# Gazebo Classic 설치
sudo apt install -y gazebo

# ROS2 - Gazebo Classic 브릿지
sudo apt install -y ros-jazzy-gazebo-ros-pkgs
```

하지만 **Gazebo Harmonic 사용을 강력 권장**합니다.
