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
- **C++ Eigen 기반** 고성능 연산 (K=1024 샘플, N=30 호라이즌에서 <50ms)
- **다모델 지원** DiffDrive / Swerve / NonCoaxialSwerve (MotionModel 추상화)
- **8종 MPPI 플러그인** Vanilla, Log, Tsallis, CVaR, SVMPC, Smooth, Spline, SVG-MPPI
- **비용 함수 계층화** StateTracking, Terminal, ControlEffort, ControlRate, CostmapObstacle, PreferForward
- **M2 고도화** Colored Noise, Adaptive Temperature, Tube-MPPI
- **동적 파라미터** 런타임 중 튜닝 가능 (min_lookahead, costmap costs 포함)
- **RVIZ 시각화** 예측 궤적, 샘플 궤적, 장애물 마커, Collision debug heatmap

### MotionModel 추상화

```
┌─ MotionModel 인터페이스 ─────────────────────────────────┐
│                                                           │
│  MotionModelFactory::create(model_name, params)           │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  DiffDrive   │  │   Swerve     │  │ NonCoaxial     │  │
│  │  nx=3, nu=2  │  │  nx=3, nu=3  │  │ Swerve         │  │
│  │  (v, omega)  │  │ (vx,vy,omega)│  │ nx=4, nu=3     │  │
│  │              │  │  홀로노믹     │  │ (v,omega,d_dot)│  │
│  └──────────────┘  └──────────────┘  └────────────────┘  │
│                                                           │
│  YAML: motion_model: "diff_drive" | "swerve"             │
│                       | "non_coaxial_swerve"              │
└───────────────────────────────────────────────────────────┘
```

### 플러그인 계층 구조

```
MPPIControllerPlugin (base, Vanilla MPPI)
├── LogMPPIControllerPlugin     (log-space softmax)
├── TsallisMPPIControllerPlugin (q-exponential)
├── RiskAwareMPPIControllerPlugin (CVaR)
├── SmoothMPPIControllerPlugin  (du space + jerk cost)
├── SplineMPPIControllerPlugin  (B-spline basis)
└── SVMPCControllerPlugin       (SVGD)
    └── SVGMPPIControllerPlugin (Guide + follower)
```

### 빠른 시작 (Gazebo Harmonic + ros2_control + nav2)

**권장: 단일 명령으로 전체 스택 실행**

```bash
cd ~/toy_claude_project/ros2_ws
source install/setup.bash

# DiffDrive (기본)
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py

# Swerve Drive (홀로노믹)
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=swerve

# Non-Coaxial Swerve
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=non_coaxial
```

**시작 순서 (자동 TimerAction 관리):**

```
┌─────────────────────────────────────────────────────────────────┐
│  0s  │ Gazebo Harmonic + Robot State Publisher + ros_gz_bridge │
│  5s  │ Robot Spawn                                              │
│  8s  │ Controllers (joint_state_broadcaster, diff_drive)        │
│ 10s  │ Localization (map_server, amcl)                          │
│ 15s  │ Navigation (controller_server, planner, bt_navigator)    │
│ 20s  │ RVIZ                                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 목표 지점 설정

**RVIZ에서:**
1. **2D Pose Estimate** 클릭 → 초기 위치 설정
2. **Nav2 Goal** 클릭 → 목표 지점 설정

**명령줄:**
```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 5.0, y: 2.0, z: 0.0}, orientation: {w: 1.0}}}}"
```

### 토픽 구조 (ros2_control 통합)

```
                    ┌──────────────────┐
                    │  nav2            │
                    │  controller_     │
                    │  server          │
                    └────────┬─────────┘
                             │ cmd_vel (geometry_msgs/Twist)
                             │ remapped to:
                             ▼
    ┌──────────────────────────────────────────────────────┐
    │  /diff_drive_controller/cmd_vel_unstamped            │
    └────────────────────────┬─────────────────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  diff_drive_     │
                    │  controller      │
                    │  (ros2_control)  │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
       /odom (nav_msgs/Odometry)     TF: odom → base_link
```

### TF 트리

```
map
 └── odom (AMCL 발행)
      └── base_link (diff_drive_controller 발행)
           ├── left_wheel
           ├── right_wheel
           ├── caster_wheel
           └── lidar_link
```

### 별도 실행 (터미널 분리)

```bash
# 터미널 1: Gazebo + ros2_control
ros2 launch mpc_controller_ros2 gazebo_ros2_control.launch.py

# 터미널 2: nav2 + MPPI (Gazebo 완전 시작 후)
ros2 launch mpc_controller_ros2 nav2_mppi.launch.py
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

### 디버깅

**토픽 확인:**
```bash
# cmd_vel 체인 확인
ros2 topic echo /diff_drive_controller/cmd_vel_unstamped

# odom 확인
ros2 topic echo /odom

# 라이다 스캔 확인
ros2 topic echo /scan
```

**TF 확인:**
```bash
ros2 run tf2_tools view_frames
ros2 run tf2_ros tf2_echo map odom
ros2 run tf2_ros tf2_echo odom base_link
```

**Controller 상태:**
```bash
ros2 control list_controllers
# 예상: joint_state_broadcaster [active], diff_drive_controller [active]
```

**nav2 라이프사이클:**
```bash
ros2 lifecycle get /controller_server
ros2 lifecycle get /amcl
```

### 트러블슈팅

| 문제 | 원인 | 해결 |
|------|------|------|
| "Control loop missed rate" | cmd_vel 토픽 연결 문제 | `ros2 topic info -v` 확인 |
| TF map→odom 없음 | AMCL 라이다 미수신 | `/scan` 토픽 확인 |
| 로봇 미동작 | Controller 비활성 | `ros2 control list_controllers` |
| Twist/TwistStamped 충돌 | 메시지 타입 불일치 | `use_stamped_vel: false` 확인 |

### 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                  MPPIControllerPlugin                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │ MotionModel    │  │ CostFunctions  │  │   Sampler     │  │
│  │ (DiffDrive/    │  │ (6종 비용 +    │  │ (Gaussian /   │  │
│  │  Swerve/       │  │  CostmapObs)   │  │  ColoredNoise)│  │
│  │  NonCoaxial)   │  │                │  │               │  │
│  └───────┬────────┘  └───────┬────────┘  └───────┬───────┘  │
│          │                   │                   │          │
│  ┌───────┴────────┐         │    ┌───────────────┴───────┐  │
│  │ BatchDynamics  │         │    │ WeightComputation     │  │
│  │ (RK4 벡터화)   │         │    │ (Vanilla/Log/Tsallis/ │  │
│  └───────┬────────┘         │    │  RiskAware)           │  │
│          │                  │    └───────────┬───────────┘  │
│          └──────────────────┼───────────────┘              │
│                             ▼                              │
│                 ┌──────────────────────┐                   │
│                 │   MPPI Algorithm     │                   │
│                 │ 1. 제어열 shift       │                   │
│                 │ 2. K개 샘플 생성      │                   │
│                 │ 3. 배치 rollout       │                   │
│                 │ 4. 비용 계산          │                   │
│                 │ 5. 가중치 (Strategy)  │                   │
│                 │ 6. 가중 평균 업데이트 │                   │
│                 └──────────────────────┘                   │
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
| test_cost_functions | 24 | ✅ |
| test_sampling | 8 | ✅ |
| test_mppi_algorithm | 7 | ✅ |
| test_adaptive_temperature | 9 | ✅ |
| test_tube_mppi | 13 | ✅ |
| test_weight_computation | 30 | ✅ |
| test_svmpc | 13 | ✅ |
| test_m35_plugins | 18 | ✅ |
| test_motion_model | 36 | ✅ |
| **총계** | **166** | **PASSED** |

---

## 개발 이력

### 마일스톤 진행 현황

| 마일스톤 | 내용 | 상태 |
|----------|------|------|
| M1 | Vanilla MPPI (Python) | ✅ |
| M2 | Colored Noise, Adaptive Temp, Tube-MPPI | ✅ |
| M3 | Log, Tsallis, CVaR, SVMPC | ✅ |
| M3.5 | Smooth, Spline, SVG-MPPI | ✅ |
| M4 | ROS2 nav2 통합 + Gazebo | ✅ |
| M5a | C++ SOTA 변형 (Log/Tsallis/CVaR/SVMPC) | ✅ |
| M5b | C++ M2 고도화 (Colored/Adaptive/Tube) | ✅ |
| M3.5 C++ | Smooth/Spline/SVG-MPPI C++ 포팅 | ✅ |
| Phase A | MotionModel 추상화 (DiffDrive/Swerve/NonCoaxial) | ✅ |
| Phase B | Goal 수렴 + 장애물 회피 튜닝 | ✅ |
| Phase C | Swerve 오실레이션 진단 + MPPI 옵티마이저 수렴 수정 | ✅ |

---

## 참고 자료

- [nav2 Controller Plugin Tutorial](https://docs.nav2.org/plugin_tutorials/docs/writing_new_nav2controller_plugin.html)
- [MPPI Paper](https://ieeexplore.ieee.org/document/7487277)
- [CasADi Documentation](https://web.casadi.org/)
- [ROS2 Documentation](https://docs.ros.org/)
- [Eigen Quick Reference](https://eigen.tuxfamily.org/dox/group__QuickRefPage.html)
