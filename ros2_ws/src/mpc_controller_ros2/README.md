# MPC Controller ROS2

ROS2 Jazzy용 Model Predictive Control 패키지입니다.

## 개요

이 패키지는 두 가지 컨트롤러를 제공합니다:

1. **MPPI Controller (nav2 플러그인)** - C++ 기반 샘플링 MPC
2. **MPC Controller (Python 노드)** - CasADi/IPOPT 기반 최적화 MPC

```
┌─────────────────────────────────────────────────────────────────┐
│                    mpc_controller_ros2                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐      ┌─────────────────────────────┐  │
│  │   MPPI Controller   │      │      MPC Controller         │  │
│  │   (C++ nav2 plugin) │      │      (Python Node)          │  │
│  ├─────────────────────┤      ├─────────────────────────────┤  │
│  │ • nav2 통합         │      │ • Standalone 노드            │  │
│  │ • 샘플링 기반       │      │ • CasADi 최적화              │  │
│  │ • 실시간 (<50ms)    │      │ • 고정밀 제어                │  │
│  └─────────────────────┘      └─────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## MPPI Controller (nav2 Plugin)

### 특징

- **nav2_core::Controller** 플러그인으로 nav2 스택과 완전 통합
- **C++ Eigen 기반** 고성능 연산 (K=512 샘플, N=30 호라이즌에서 <50ms)
- **RK4 동역학** 정밀한 differential drive 모델
- **비용 함수 계층화** StateTracking, Terminal, ControlEffort, ControlRate, Obstacle
- **동적 파라미터** 런타임 중 튜닝 가능
- **RVIZ 시각화** 예측 궤적, 샘플 궤적, 장애물 마커

### 빠른 시작 (Gazebo Harmonic)

```bash
# 터미널 1: Gazebo 실행
ros2 launch mpc_controller_ros2 gazebo_harmonic_test.launch.py

# 터미널 2: nav2 + MPPI 실행 (Gazebo가 완전히 시작된 후)
ros2 launch mpc_controller_ros2 nav2_mppi.launch.py

# 터미널 3: 목표 지점 전송
ros2 run mpc_controller_ros2 send_nav_goal.py --x 5.0 --y 0.0
```

### 또는 통합 실행

```bash
# 단일 명령으로 Gazebo + nav2 + MPPI 실행
ros2 launch mpc_controller_ros2 mppi_nav2_gazebo.launch.py
```

### MPPI 파라미터 (nav2_params.yaml)

```yaml
FollowPath:
  plugin: "mpc_controller_ros2::MPPIControllerPlugin"

  # 예측 호라이즌
  N: 30              # 예측 스텝 수
  dt: 0.1            # 시간 간격 [s]

  # 샘플링
  K: 512             # 샘플 수
  lambda: 10.0       # 온도 (낮을수록 탐욕적)

  # 노이즈
  noise_sigma_v: 0.3     # 선속도 노이즈 표준편차
  noise_sigma_omega: 0.4 # 각속도 노이즈 표준편차

  # 제어 제한
  v_max: 0.5         # 최대 선속도 [m/s]
  v_min: -0.2        # 최소 선속도 [m/s]
  omega_max: 1.0     # 최대 각속도 [rad/s]
  omega_min: -1.0    # 최소 각속도 [rad/s]

  # 상태 추적 비용
  Q_x: 15.0          # x 위치 가중치
  Q_y: 15.0          # y 위치 가중치
  Q_theta: 2.0       # 방향 가중치

  # 종료 비용
  Qf_x: 30.0
  Qf_y: 30.0
  Qf_theta: 4.0

  # 제어 비용
  R_v: 0.2           # 선속도 제어 비용
  R_omega: 0.2       # 각속도 제어 비용
  R_rate_v: 1.5      # 선속도 변화율 비용
  R_rate_omega: 1.5  # 각속도 변화율 비용

  # 장애물 회피
  obstacle_weight: 150.0
  safety_distance: 0.6
```

### 동적 파라미터 조정

```bash
# 온도 조정 (탐색 vs 탐욕)
ros2 param set /controller_server FollowPath.lambda 15.0

# 노이즈 조정
ros2 param set /controller_server FollowPath.noise_sigma_v 0.5

# 상태 추적 가중치 조정
ros2 param set /controller_server FollowPath.Q_x 20.0
```

### 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                  MPPIControllerPlugin                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐    ┌────────────────┐    ┌───────────┐  │
│  │ BatchDynamics  │    │  CostFunctions │    │  Sampler  │  │
│  │  (RK4 적분)    │    │  (5종 비용)     │    │ (Gaussian)│  │
│  └───────┬────────┘    └───────┬────────┘    └─────┬─────┘  │
│          │                     │                   │        │
│          └─────────────────────┼───────────────────┘        │
│                                ▼                            │
│                    ┌──────────────────────┐                 │
│                    │   MPPI Algorithm     │                 │
│                    │ 1. 제어열 shift       │                 │
│                    │ 2. K개 샘플 생성      │                 │
│                    │ 3. 배치 rollout       │                 │
│                    │ 4. 비용 계산          │                 │
│                    │ 5. softmax 가중치    │                 │
│                    │ 6. 가중 평균 업데이트 │                 │
│                    └──────────────────────┘                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## MPC Controller (Python Node)

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

---

## 테스트

### C++ 단위 테스트

```bash
# 전체 테스트 실행
colcon test --packages-select mpc_controller_ros2
colcon test-result --verbose

# 개별 테스트 실행
./build/mpc_controller_ros2/test_batch_dynamics
./build/mpc_controller_ros2/test_cost_functions
./build/mpc_controller_ros2/test_sampling
./build/mpc_controller_ros2/test_mppi_algorithm
```

### 테스트 커버리지

| 테스트 | 항목 수 | 상태 |
|--------|---------|------|
| test_batch_dynamics | 8 | ✅ |
| test_cost_functions | 15 | ✅ |
| test_sampling | 8 | ✅ |
| test_mppi_algorithm | 7 | ✅ |
| **총계** | **38** | **PASSED** |

---

## 개발 이력

### M4 마일스톤: ROS2 nav2 통합

| Phase | 내용 | 상태 |
|-------|------|------|
| 1 | 기초 인프라 (CMake, utils) | ✅ |
| 2 | 동역학 & 샘플링 | ✅ |
| 3 | 비용 함수 계층 | ✅ |
| 4 | MPPI 알고리즘 | ✅ |
| 5 | nav2 인터페이스 | ✅ |
| 6 | 동적 파라미터 | ✅ |
| 7 | RVIZ 시각화 | ✅ |
| 8 | Gazebo 통합 | ✅ |
| 9 | 테스트 | ✅ |
| 10 | 문서화 | ✅ |

---

## 참고 자료

- [nav2 Controller Plugin Tutorial](https://docs.nav2.org/plugin_tutorials/docs/writing_new_nav2controller_plugin.html)
- [MPPI Paper](https://ieeexplore.ieee.org/document/7487277)
- [CasADi Documentation](https://web.casadi.org/)
- [ROS2 Documentation](https://docs.ros.org/)
- [Eigen Quick Reference](https://eigen.tuxfamily.org/dox/group__QuickRefPage.html)
