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
│  │ • 실시간 (1.9ms)    │      │ • 고정밀 제어                │  │
│  └─────────────────────┘      └─────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## MPPI Controller (nav2 Plugin)

### 특징

- **nav2_core::Controller** 플러그인으로 nav2 스택과 완전 통합
- **C++ Eigen 기반** 고성능 연산 (K=1024, N=30에서 3.9ms — 258Hz)
- **다모델 지원** DiffDrive / Swerve / NonCoaxialSwerve / Ackermann (MotionModel 추상화)
- **17종 MPPI 플러그인** Vanilla, Log, Tsallis, CVaR, SVMPC, Smooth, Spline, SVG-MPPI, Biased-MPPI, DIAL-MPPI, Shield-MPPI, Adaptive Shield-MPPI, iLQR-MPPI, CS-MPPI, π-MPPI, MPPI-H (Hybrid Swerve)
- **안전성 고도화** CBF Safety Filter, BR-MPPI, Conformal Predictor(ACP), Shield-MPPI, C3BF(Collision Cone CBF), Adaptive Shield(거리/속도 적응형 α), Horizon-Weighted CBF
- **학습 기반 동역학** Residual Dynamics + Ensemble Dynamics (M개 MLP 앙상블, 불확실성 추정)
- **비용 함수 계층화** StateTracking, Terminal, ControlEffort, ControlRate, CostmapObstacle, PreferForward
- **M2 고도화** Colored Noise, Adaptive Temperature, Tube-MPPI
- **동적 파라미터** 런타임 중 튜닝 가능 (min_lookahead, costmap costs 포함)
- **RVIZ 시각화** 예측 궤적, 샘플 궤적, 장애물 마커, Collision debug heatmap

### MotionModel 추상화

```
┌─ MotionModel 인터페이스 ───────────────────────────────────────────────────┐
│                                                                             │
│  MotionModelFactory::create(model_name, params)                             │
│                                                                             │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────┐ ┌────────────────┐  │
│  │  DiffDrive   │ │   Swerve     │ │ NonCoaxial     │ │  Ackermann     │  │
│  │  nx=3, nu=2  │ │  nx=3, nu=3  │ │ Swerve         │ │  nx=4, nu=2    │  │
│  │  (v, omega)  │ │ (vx,vy,omega)│ │ nx=4, nu=3     │ │  (v, d_dot)    │  │
│  │              │ │  홀로노믹     │ │ (v,omega,d_dot)│ │  θ̇=v·tan(δ)/L │  │
│  └──────────────┘ └──────────────┘ └────────────────┘ └────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  ResidualDynamicsModel (Decorator)                                   │  │
│  │  f_total = f_nominal + α·MLP([x, u])     ← Sim-to-Real 보정        │  │
│  │  EigenMLP: 바이너리 로드, Z-score 정규화, ReLU, BLAS 배치           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  EnsembleDynamicsModel (Decorator)                                   │  │
│  │  f_total = f_nominal + α·mean(MLP_1..M([x, u]))                     │  │
│  │  PredictionResult: mean + variance → UncertaintyAwareCost           │  │
│  │  M개 부트스트랩 MLP 앙상블 (불확실성 추정, Sim-to-Real)             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  YAML: motion_model: "diff_drive" | "swerve"                               │
│                     | "non_coaxial_swerve" | "ackermann"                    │
│        residual_enabled: true  +  residual_weights_path: "model.bin"        │
│        ensemble_enabled: true  +  ensemble_weights_dir: "ensemble/"         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 플러그인 계층 구조

```
MPPIControllerPlugin (base, Vanilla MPPI)
├── LogMPPIControllerPlugin     (log-space softmax)
├── TsallisMPPIControllerPlugin (q-exponential)
├── RiskAwareMPPIControllerPlugin (CVaR)
├── SmoothMPPIControllerPlugin  (du space + jerk cost)
├── SplineMPPIControllerPlugin  (B-spline basis)
├── BiasedMPPIControllerPlugin  (Ancillary biased sampling, RA-L 2024)
├── DialMPPIControllerPlugin    (Diffusion annealing, ICRA 2025)
├── ShieldMPPIControllerPlugin  (per-step CBF 투영, 10Hz@K=256)
│   └── AdaptiveShieldMPPIControllerPlugin (거리/속도 적응형 α, α(d,v)=α_min+(α_max-α_min)·exp(-k_d·d)·(1+k_v·v))
├── IlqrMPPIControllerPlugin   (iLQR warm-start + MPPI)
├── CSMPPIControllerPlugin     (Covariance Steering, CoVO-MPC CoRL 2023)
├── PiMPPIControllerPlugin     (ADMM QP 투영, π-MPPI RA-L 2025)
├── HybridSwerveMPPIControllerPlugin (Low-D↔4D 전환, MPPI-H IROS 2024)
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

# Ackermann (Bicycle model, 전륜 조향)
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=ackermann

# Adaptive Shield-MPPI (거리/속도 적응형 CBF)
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=adaptive_shield
```

**시작 순서 (OnProcessExit 이벤트 체인):**

```
┌─────────────────────────────────────────────────────────────────┐
│  0s   │ Gazebo Harmonic + Robot State Publisher + ros_gz_bridge │
│  5s   │ Robot Spawn (gz service -s /world/*/create)            │
│       │     ↓ OnProcessExit                                     │
│  ~8s  │ joint_state_broadcaster (unload → spawn 패턴)           │
│       │     ↓ OnProcessExit                                     │
│  ~11s │ diff_drive_controller (unload → spawn 패턴)             │
│ 10s   │ Localization (map_server, amcl)                         │
│ 15s   │ Navigation (controller_server, planner, bt_navigator)   │
│ 20s   │ RVIZ                                                    │
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
│  │  Swerve/       │  │  CostmapObs +  │  │  ColoredNoise)│  │
│  │  NonCoaxial/   │  │  BarrierRate)  │  │               │  │
│  │  Ackermann/    │  │                │  │               │  │
│  │  +Residual)    │  │                │  │               │  │
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
│                 └──────────┬───────────┘                   │
│                            ▼                                │
│                 ┌──────────────────────┐                   │
│                 │   Safety Layer       │                   │
│                 │ • CBF Safety Filter  │                   │
│                 │ • Shield-MPPI (투영) │                   │
│                 │ • Adaptive Shield    │                   │
│                 │ • C3BF (Collision    │                   │
│                 │   Cone CBF)          │                   │
│                 │ • Conformal ACP      │                   │
│                 │ • Horizon-Weighted   │                   │
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

### MPPI Controller 벤치마크 (Release -O2 -march=native)

```
┌──────────────────────────────────────────────────────────────────┐
│  MPPI Pipeline Benchmark (DiffDrive, N=30)                       │
├──────────┬───────────┬───────────┬──────────┬───────────────────┤
│ K        │ Pipeline  │ Frequency │ Rollout  │ Cost              │
├──────────┼───────────┼───────────┼──────────┼───────────────────┤
│ 256      │ 0.92ms    │ 1,091 Hz  │ 567μs    │ 194μs             │
│ 512      │ 1.88ms    │ 532 Hz    │ 1.15ms   │ 407μs             │
│ 1024     │ 3.88ms    │ 258 Hz    │ 2.43ms   │ 805μs             │
├──────────┼───────────┼───────────┼──────────┼───────────────────┤
│ Swerve   │ 1.95ms    │ 512 Hz    │ (K=512)  │                   │
│ NonCoax  │ 2.80ms    │ 357 Hz    │ (K=512)  │                   │
│ Ackermann│ ~1.9ms    │ ~530 Hz   │ (K=512)  │ nu=2, nx=4        │
└──────────┴───────────┴───────────┴──────────┴───────────────────┘

최적화 기법:
  - True Batch Rollout: K×N×4 → N×4 propagateBatch 호출 (512배 감소)
  - InPlace 패턴: 힙 할당 1536/call → 0/call
  - 대각 Q/R 특수화: cwiseAbs2().dot() 벡터 연산
  - -march=native: Eigen AVX2 SIMD 자동 활성화
```

벤치마크 실행:
```bash
colcon build --packages-select mpc_controller_ros2 --cmake-args -DCMAKE_BUILD_TYPE=Release
./build/mpc_controller_ros2/bench_mppi_pipeline --K 512 --N 30
./build/mpc_controller_ros2/bench_mppi_pipeline --scaling
```

### Swerve MPPI 컨트롤러 벤치마크 (Narrow Passage, 장애물 회피)

8종 Swerve 호환 MPPI 컨트롤러의 **goal 도달 + 장애물 회피** 성능을 평가합니다.
두 조건으로 테스트하여 글로벌 플래너의 보호 효과와 컨트롤러 자체 회피력을 분리합니다.

**테스트 환경:**
```
World: narrow_passage_world (0.8m 폭 좁은 통로 + pillar 장애물)
Goal:  (5.0, -3.5) — 좁은 통로 통과 필수 경로
모델:  Swerve Drive (홀로노믹, nu=3)
```

**조건 비교:**
```
Normal:  inflation_radius=0.35, robot_radius=0.3, planner=20Hz
Stress:  inflation_radius=0.10, robot_radius=0.1, planner=1Hz
         → 글로벌 플래너 보호 최소화, 컨트롤러 반응적 회피 평가
```

#### 결과 비교 (Normal vs Stress)

```
┌──────────────────┬─────────────────────────────────────┬─────────────────────────────────────┐
│ Controller       │ Normal (inflation=0.35, planner 20Hz)│ Stress (inflation=0.10, planner 1Hz)│
├──────────────────┼─────────────────────────────────────┼─────────────────────────────────────┤
│ swerve (vanilla) │ OK  55.2s |  28 near |  0 coll     │ NG 119.7s | 448 near |  87 coll    │
│ non_coaxial      │ OK  45.2s |   0 near |  0 coll     │ NG 119.9s | 119 near |   0 coll    │
│ dial_swerve      │ OK  80.4s |  50 near |  0 coll     │ NG 120.0s | 415 near |  39 coll    │
│ log_swerve       │ NG  14.6s |   1 near |  0 coll     │ OK  88.2s | 235 near |  64 coll    │
│ smooth_swerve    │ OK  24.4s |  35 near |  6 coll     │ NG  65.6s |  46 near |   2 coll    │
│ biased_swerve    │ OK  63.9s | 102 near | 14 coll     │ OK  20.3s | 117 near |  21 coll    │
│ cs_swerve        │ OK  19.3s |  49 near | 18 coll     │ NG  59.5s | 270 near |  61 coll    │
│ pi_swerve        │ OK  15.4s |  12 near |  0 coll     │ OK  82.5s | 400 near |  51 coll    │
└──────────────────┴─────────────────────────────────────┴─────────────────────────────────────┘

near = 0.3m 미만 접근 횟수, coll = 0.15m 미만 충돌 횟수
OK = goal 도달, NG = timeout 또는 미도달
```

#### 성공률

```
Normal:  7/8 PASS (87.5%)
Stress:  3/8 PASS (37.5%) — biased, log, pi만 성공
```

#### 글로벌 플래너 의존도 분석 (충돌 증가량)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 의존도   │ Controller      │ 충돌 변화      │ 해석                          │
├──────────┼─────────────────┼────────────────┼───────────────────────────────┤
│ 극심     │ swerve(vanilla) │  0 →  87 (+87) │ 자체 회피력 거의 없음         │
│ 극심     │ log_swerve      │  0 →  64 (+64) │ goal 도달하나 위험한 궤적     │
│ 심함     │ pi_swerve       │  0 →  51 (+51) │ 제약 만족 우선, 회피 부족     │
│ 심함     │ cs_swerve       │ 18 →  61 (+43) │ 공분산 조향 장애물 근접 불안정│
│ 심함     │ dial_swerve     │  0 →  39 (+39) │ 어닐링이 좁은 통로에서 발산   │
│ 경미     │ biased_swerve   │ 14 →  21  (+7) │ 보조 컨트롤러가 실질 회피     │
│ 감소     │ smooth_swerve   │  6 →   2  (-4) │ 직접 경로가 오히려 안전       │
│ 없음     │ non_coaxial     │  0 →   0  (±0) │ 유일한 자체 충돌 방지         │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### 핵심 발견

```
1. 글로벌 플래너 마스킹 효과
   ┌─────────────────────────────────────────────────────────┐
   │ Normal에서 충돌 0인 컨트롤러 5종 중 4종이               │
   │ Stress에서 39~87회 충돌 → 실제 회피력이 아닌            │
   │ 글로벌 플래너의 안전 경로 생성에 의존했음               │
   └─────────────────────────────────────────────────────────┘

2. 역전 현상
   - log_swerve: Normal NG → Stress OK
     → 글로벌 플래너의 보수적 경로가 오히려 log 가중치 탐색 방해
   - smooth_swerve: 충돌 6→2 감소
     → 저 inflation 직접 경로가 벽 접근 횟수 자체를 줄임

3. 자체 회피력 종합 순위
   ┌─────────────────────────────────────────────────────────┐
   │ 1위: biased_swerve — PASS + 최단(20s) + 충돌 적음(21)  │
   │ 2위: pi_swerve     — PASS + 안정적 + 충돌 중간(51)     │
   │ 3위: log_swerve    — PASS + 도달 가능 + 충돌 많음(64)  │
   │ 4위: non_coaxial   — 충돌 0 (최고 안전) + goal 미도달  │
   │ 5위: smooth_swerve — 충돌 최소(2) + goal 미도달        │
   └─────────────────────────────────────────────────────────┘
```

#### 이론 vs 실제 (nu=2 → nu=3 차원 확장 영향)

| 변형 | 이론적 swerve 적합도 | 실측 결과 | 원인 |
|------|----------------------|-----------|------|
| Biased-MPPI | 높음 (구조 무관, ancillary 피드백) | 1위 (20.3s, goal+회피) | 보조 컨트롤러가 3축 독립 보정 |
| Log-MPPI | 높음 (가중치 함수만 변경) | 3위 (88.2s, goal OK) | log-sum-exp 꼬리 분포 활용 |
| π-MPPI | 중간 (QP가 nu=3으로 확장) | 2위 (82.5s, goal OK) | ADMM 수렴 가능하나 회피 약화 |
| Smooth-MPPI | 중간 (Δu 3차원 확장) | 충돌 최소(2) 하지만 goal NG | jerk 억제가 목표 수렴도 억제 |
| CS-MPPI | 낮음 (선형화 3×3) | 61 충돌 + goal NG | 선형화 오차 + 공분산 발산 |
| DIAL-MPPI | 중간 (어닐링 범용) | 39 충돌 + goal NG | 좁은 통로에서 어닐링 불안정 |

#### 벤치마크 실행 방법

```bash
# Normal 조건 (글로벌 플래너 보호 있음)
bash ros2_ws/src/mpc_controller_ros2/scripts/swerve_controller_benchmark.sh

# Stress 조건 (글로벌 플래너 보호 최소화)
NAV2_STRESS=true bash ros2_ws/src/mpc_controller_ros2/scripts/swerve_controller_benchmark.sh

# 결과 확인
cat ~/swerve_benchmark_results/benchmark_*.txt
```

### MPC Controller (Python, CasADi/IPOPT)

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
| test_cost_functions | 38 | ✅ |
| test_sampling | 8 | ✅ |
| test_mppi_algorithm | 7 | ✅ |
| test_adaptive_temperature | 9 | ✅ |
| test_tube_mppi | 13 | ✅ |
| test_weight_computation | 30 | ✅ |
| test_svmpc | 13 | ✅ |
| test_m35_plugins | 18 | ✅ |
| test_motion_model | 68 | ✅ |
| test_cbf | 22 | ✅ |
| test_trajectory_stability | 25 | ✅ |
| test_biased_mppi | 15 | ✅ |
| test_dial_mppi | 17 | ✅ |
| test_residual_dynamics | 15 | ✅ |
| test_safety_enhancement | 24 | ✅ |
| test_ilqr_solver | 12 | ✅ |
| test_ilqr_mppi | 8 | ✅ |
| test_cs_mppi | 16 | ✅ |
| test_pi_mppi | 16 | ✅ |
| test_hybrid_swerve_mppi | 18 | ✅ |
| test_ensemble_dynamics | 14 | ✅ |
| test_advanced_cbf | 16 | ✅ |
| **총계** | **430** | **PASSED** |

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
| Phase A | MotionModel 추상화 (DiffDrive/Swerve/NonCoaxial/Ackermann) | ✅ |
| Phase B | Goal 수렴 + 장애물 회피 튜닝 | ✅ |
| Phase C | Swerve 오실레이션 진단 + MPPI 옵티마이저 수렴 수정 | ✅ |
| MPPI-CBF | Control Barrier Function 통합 (Python + C++) | ✅ |
| 궤적 안정화 | SG Filter + IT 정규화 | ✅ |
| Biased-MPPI | Ancillary biased sampling C++ nav2 플러그인 (PR #123) | ✅ |
| DIAL-MPPI | Diffusion annealing C++ nav2 플러그인 (PR #125) | ✅ |
| DIAL-MPPI 최적화 | AnnealingResult 재사용 + Swerve/NonCoaxial 튜닝 (PR #129) | ✅ |
| 성능 최적화 | True Batch + InPlace + SIMD + 대각 Q (PR #132) | ✅ |
| Ackermann | Bicycle model MotionModel C++ (PR #138) | ✅ |
| Residual Dynamics | EigenMLP + ResidualDynamicsModel Decorator (PR #140) | ✅ |
| Safety Enhancement | BR-MPPI + ConformalPredictor + Shield-MPPI (PR #140) | ✅ |
| iLQR-MPPI | iLQR warm-start + MPPI 파이프라인 (PR #142) | ✅ |
| CS-MPPI | Covariance Steering, CoVO-MPC (PR #150) | ✅ |
| π-MPPI | ADMM QP 투영 필터, hard rate/accel bounds (PR #152) | ✅ |
| MPPI-H | Hybrid Swerve, Low-D↔4D 전환 (PR #153) | ✅ |
| Swerve 벤치마크 | 8종 컨트롤러 Normal/Stress 비교 분석 | ✅ |
| Learning MPPI + Advanced CBF | Ensemble Dynamics, C3BF, Adaptive Shield, Horizon-Weighted CBF (PR #159) | ✅ |

---

## 참고 자료

- [nav2 Controller Plugin Tutorial](https://docs.nav2.org/plugin_tutorials/docs/writing_new_nav2controller_plugin.html)
- [MPPI Paper](https://ieeexplore.ieee.org/document/7487277)
- [CasADi Documentation](https://web.casadi.org/)
- [ROS2 Documentation](https://docs.ros.org/)
- [Eigen Quick Reference](https://eigen.tuxfamily.org/dox/group__QuickRefPage.html)
