# MPC Controller ROS2 Wrapper

ROS2 노드로 구현된 Model Predictive Control (MPC) 컨트롤러입니다.

## 개요

이 패키지는 CasADi 기반 MPC 컨트롤러를 ROS2 인터페이스로 래핑합니다.
Differential drive 로봇의 경로 추종 제어를 수행합니다.

## 아키텍처

```
┌─────────────────────────────────────────┐
│      MPC Controller ROS2 Node           │
├─────────────────────────────────────────┤
│  Subscribers:                           │
│    • /odom (nav_msgs/Odometry)          │
│    • /reference_path (nav_msgs/Path)    │
│                                         │
│  Publishers:                            │
│    • /cmd_vel (geometry_msgs/Twist)     │
│    • /predicted_trajectory              │
│    • /mpc_markers                       │
│                                         │
│  Core:                                  │
│    • MPCController (CasADi-based)       │
│    • DifferentialDriveModel             │
└─────────────────────────────────────────┘
```

## 설치

### 1. 의존성 설치

```bash
# ROS2 패키지 의존성
sudo apt install ros-${ROS_DISTRO}-geometry-msgs \
                 ros-${ROS_DISTRO}-nav-msgs \
                 ros-${ROS_DISTRO}-visualization-msgs

# Python 의존성
pip install numpy casadi scipy
```

### 2. 빌드

```bash
cd ros2_ws
colcon build --packages-select mpc_controller_ros2
source install/setup.bash
```

## 사용법

### 1. 노드 실행

```bash
ros2 launch mpc_controller_ros2 mpc_controller.launch.py
```

### 2. 파라미터 커스터마이징

```bash
ros2 launch mpc_controller_ros2 mpc_controller.launch.py \
  config_file:=/path/to/custom_params.yaml
```

### 3. 시뮬레이션 모드

```bash
ros2 launch mpc_controller_ros2 mpc_controller.launch.py \
  use_sim_time:=true
```

## 토픽

### 구독 (Subscriptions)

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/odom` | `nav_msgs/Odometry` | 현재 로봇 위치 (x, y, θ) |
| `/reference_path` | `nav_msgs/Path` | 참조 경로 |

### 발행 (Publications)

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/cmd_vel` | `geometry_msgs/Twist` | 제어 명령 (v, ω) |
| `/predicted_trajectory` | `nav_msgs/Path` | MPC 예측 궤적 |
| `/mpc_markers` | `visualization_msgs/MarkerArray` | RVIZ 시각화 마커 |

## 파라미터

### MPC 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `mpc.N` | int | 20 | 예측 구간 |
| `mpc.dt` | float | 0.1 | 시간 간격 [s] |
| `mpc.Q` | list[float] | [10, 10, 1] | 상태 가중치 [x, y, θ] |
| `mpc.R` | list[float] | [0.1, 0.1] | 제어 가중치 [v, ω] |
| `mpc.Qf` | list[float] | [100, 100, 10] | 종료 상태 가중치 |
| `mpc.Rd` | list[float] | [0.5, 0.5] | 제어 변화율 가중치 |

### 로봇 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `robot.max_velocity` | float | 1.0 | 최대 선속도 [m/s] |
| `robot.max_omega` | float | 1.5 | 최대 각속도 [rad/s] |
| `robot.wheel_base` | float | 0.5 | 휠베이스 [m] |
| `robot.max_acceleration` | float | 0.5 | 최대 선가속도 [m/s²] |
| `robot.max_alpha` | float | 1.0 | 최대 각가속도 [rad/s²] |

## 예제

### 1. 간단한 경로 발행

```bash
# 터미널 1: MPC 노드 실행
ros2 launch mpc_controller_ros2 mpc_controller.launch.py

# 터미널 2: 테스트 경로 발행
ros2 topic pub /reference_path nav_msgs/Path '{
  header: {frame_id: "odom"},
  poses: [
    {pose: {position: {x: 0, y: 0}, orientation: {w: 1}}},
    {pose: {position: {x: 1, y: 0}, orientation: {w: 1}}},
    {pose: {position: {x: 2, y: 0}, orientation: {w: 1}}}
  ]
}'
```

### 2. RVIZ 시각화

```bash
# RVIZ 실행
rviz2

# 다음 토픽을 추가:
# - /predicted_trajectory (Path)
# - /mpc_markers (MarkerArray)
# - /reference_path (Path)
```

## 테스트

```bash
# 단위 테스트 실행
colcon test --packages-select mpc_controller_ros2
colcon test-result --verbose
```

## 제약 조건

- **소프트 제약**: MPC는 속도 및 가속도 제약을 soft constraint로 처리
- **하드 제약**: 제어 입력 범위는 hard constraint로 적용

## 성능

- **솔버**: IPOPT (Interior Point OPTimizer)
- **평균 풀이 시간**: 10-30ms (N=20 기준)
- **실시간성**: 10Hz 제어 주기 지원

## 문제 해결

### 1. MPC 풀이 실패

- `Q`, `R`, `Qf` 가중치 조정
- 예측 구간 `N` 감소
- 참조 경로 품질 확인

### 2. 제약 위반 경고

- `robot.max_velocity`, `robot.max_omega` 증가
- Soft constraint 가중치 조정

### 3. 진동하는 제어

- `Rd` (제어 변화율 가중치) 증가
- `dt` (시간 간격) 조정

## 라이센스

MIT License

## 참고 자료

- [CasADi Documentation](https://web.casadi.org/)
- [ROS2 Documentation](https://docs.ros.org/)
- [MPC Theory](https://en.wikipedia.org/wiki/Model_predictive_control)
