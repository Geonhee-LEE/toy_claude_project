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
│  │ • 20종 플러그인      │      │ • CasADi 최적화              │  │
│  │ • 실시간 (1.9ms)    │      │ • 고정밀 제어                │  │
│  │ • 494 gtest         │      │                              │  │
│  └─────────────────────┘      └─────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## MPPI Controller (nav2 Plugin)

### 특징

- **nav2_core::Controller** 플러그인으로 nav2 스택과 완전 통합
- **C++ Eigen 기반** 고성능 연산 (K=512, N=30에서 1.88ms — 532Hz)
- **다모델 지원** DiffDrive / Swerve / NonCoaxialSwerve / Ackermann (MotionModel 추상화)
- **20종 MPPI 플러그인** (아래 플러그인 선택 가이드 참조)
- **5단계 안전 스택** CBF → Shield → Adaptive Shield → CLF-CBF-QP → Predictive Safety
- **학습 기반 동역학** Residual + Ensemble MLP (불확실성 추정 + 온라인 핫 스왑)
- **M2 고도화** Colored Noise, Adaptive Temperature, Tube-MPPI
- **RVIZ 시각화** 예측 궤적, 샘플 궤적, 장애물 마커, Collision debug heatmap

### 플러그인 계층 구조 (20종)

```
MPPIControllerPlugin (base, Vanilla MPPI)
│
├── 가중치 변형 ─────────────────────────────────────────────
│   ├── LogMPPIControllerPlugin       (log-space softmax)
│   ├── TsallisMPPIControllerPlugin   (q-exponential)
│   └── RiskAwareMPPIControllerPlugin (CVaR tail-risk)
│
├── 샘플링/보간 변형 ───────────────────────────────────────
│   ├── SmoothMPPIControllerPlugin    (du space + jerk cost)
│   ├── SplineMPPIControllerPlugin    (B-spline basis, P knots)
│   ├── BiasedMPPIControllerPlugin    (Ancillary biased, RA-L 2024)
│   └── DialMPPIControllerPlugin      (Diffusion annealing, ICRA 2025)
│
├── 안전성 강화 ─────────────────────────────────────────────
│   └── ShieldMPPIControllerPlugin    (per-step CBF 투영)
│       ├── AdaptiveShieldMPPI        (거리/속도 적응형 α)
│       ├── CLFCBFMPPIControllerPlugin(CLF-CBF-QP 통합 필터)
│       └── PredictiveSafetyMPPI      (N-step forward CBF)
│
├── 고급 최적화 ─────────────────────────────────────────────
│   ├── IlqrMPPIControllerPlugin      (iLQR warm-start)
│   ├── CSMPPIControllerPlugin        (Covariance Steering, CoRL 2023)
│   ├── PiMPPIControllerPlugin        (ADMM QP 투영, RA-L 2025)
│   └── HybridSwerveMPPIControllerPlugin (Low-D↔4D, IROS 2024)
│
└── 다중 분포 ───────────────────────────────────────────────
    └── SVMPCControllerPlugin          (SVGD loop)
        └── SVGMPPIControllerPlugin    (Guide + follower, ICRA 2024)
```

### 플러그인 선택 가이드

**어떤 플러그인을 써야 할까요?**

```
시작 ──► 안전성이 최우선? ──Yes──► 장애물 속도 추정 가능?
 │                                  │Yes              │No
 │                           AdaptiveShield      PredictiveSafety
 │                           (d,v 적응 α)        (N-step CBF)
 │
 │No
 ▼
 제어 품질(부드러움)이 중요? ──Yes──► 하드 제약 필요?
 │                                   │Yes         │No
 │                                π-MPPI       Smooth-MPPI
 │                               (rate/accel    (jerk cost
 │                                hard bounds)   soft penalty)
 │No
 ▼
 멀티모달 환경(좁은 통로)? ──Yes──► SVMPC 또는 SVG-MPPI
 │                                 (SVGD 기반 다중 분포)
 │No
 ▼
 Swerve 로봇? ──Yes──► MPPI-H (Hybrid Swerve)
 │                     (상황별 Low-D↔4D 전환)
 │No
 ▼
 기본 Vanilla MPPI (가장 빠름, 가장 단순)
```

**플러그인 비교표**

| 플러그인 | 이론 | 강점 | 약점 | 오버헤드 | 추천 시나리오 |
|---------|------|------|------|---------|-------------|
| **Vanilla** | Williams 2017 | 빠름, 단순 | 비용 landscape 의존 | 기준 | 단순 환경, 빠른 제어 |
| **Log** | Log-space softmax | 수치 안정성 | 효과 미미 | < 1% | 큰 K에서 overflow 방지 |
| **Tsallis** | q-exponential | 탐색/활용 조절 | q 튜닝 필요 | < 1% | 탐색 강화 필요 시 |
| **CVaR** | Tail-risk | 보수적 경로 | 샘플 효율 낮음 | < 1% | 위험 회피 필수 환경 |
| **Smooth** | Kim 2021 | 부드러운 제어 | 수렴 느림 | ~3% | 기계 마모 최소화 |
| **Spline** | ICRA 2024 | 매끄러운 궤적 | knot 수 튜닝 | ~5% | 고속 주행 |
| **Biased** | RA-L 2024 | 보조 컨트롤러 | 추가 컨트롤러 필요 | ~8% | 좁은 통로 + 빠른 회피 |
| **DIAL** | ICRA 2025 | 어닐링 탐색 | 좁은 공간 불안정 | ~15% | 넓은 환경, 다수 장애물 |
| **Shield** | CBF 투영 | 안전 보장 | 보수적 | ~5% | 안전 필수 환경 |
| **AdaptiveShield** | 적응형 α | 상황 적응 | 파라미터 4개 | ~5% | 동적 장애물 |
| **CLF-CBF** | Ames 2019 | 수렴+안전 동시 | QP 비용 | ~8% | 수렴 보장 필요 |
| **PredictiveSafety** | N-step CBF | 재귀적 안전성 | 계산 비용 | ~12% | 최고 안전성 요구 |
| **iLQR** | IT-MPC 2018 | 빠른 수렴 | 선형화 오차 | ~2% | 빠른 warm-start |
| **CS** | CoRL 2023 | 적응적 노이즈 | 선형화 의존 | ~5% | 감도 기반 탐색 |
| **π-MPPI** | RA-L 2025 | 하드 제약 보장 | ADMM 수렴 | ~8% | rate/accel 제한 |
| **MPPI-H** | IROS 2024 | 동적 차원 전환 | Swerve 전용 | ~3% | Swerve 최적화 |
| **SVMPC** | SVGD | 멀티모달 | 느림 | ~20% | 다중 경로 |
| **SVG-MPPI** | ICRA 2024 | 효율적 다중분포 | G 튜닝 | ~15% | 복잡한 환경 |

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
│  │              │ │  홀로노믹     │ │ (v,omega,d_dot)│ │  theta_dot=v*tan(d)/L │
│  └──────────────┘ └──────────────┘ └────────────────┘ └────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  ResidualDynamicsModel (Decorator)                                   │  │
│  │  f_total = f_nominal + alpha * MLP([x, u])  <-- Sim-to-Real 보정   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  EnsembleDynamicsModel (Decorator)                                   │  │
│  │  f_total = f_nominal + alpha * mean(MLP_1..M([x, u]))               │  │
│  │  PredictionResult: mean + variance --> UncertaintyAwareCost         │  │
│  │  updateEnsemble(): mutex-protected hot-swap (ModelReloader)         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  YAML: motion_model: "diff_drive" | "swerve"                               │
│                     | "non_coaxial_swerve" | "ackermann"                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 안전 스택 (5단계)

```
┌──────────────────────────────────────────────────────────────────┐
│                        Safety Stack                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Level 1: Soft Cost (MPPI 비용 함수)                              │
│  ├── CBFCost: DCBF 이산 조건 위반 제곱 페널티                     │
│  ├── BarrierRateCost (BR-MPPI): dh/dt 접근율 페널티              │
│  ├── C3BFCost: Collision Cone CBF (속도+방향 인식)               │
│  ├── CLFCost: Lyapunov 감소 조건 위반 페널티                     │
│  └── UncertaintyAwareCost: 앙상블 분산 기반 불확실 영역 회피      │
│                                                                    │
│  Level 2: Per-Step Projection (ShieldMPPI)                        │
│  └── 최적 u* 시퀀스에 CBF 투영 (stride 기반)                     │
│                                                                    │
│  Level 3: Adaptive Projection (AdaptiveShieldMPPI)                │
│  └── alpha(d,v) = alpha_min + (alpha_max-alpha_min)               │
│      * exp(-k_d*d) * (1 + k_v*v)                                 │
│                                                                    │
│  Level 4: CLF-CBF-QP (CLFCBFMPPIControllerPlugin)                │
│  ├── CLF 수렴 + CBF 안전 통합 QP                                 │
│  ├── Slack delta로 안전 우선 (CBF hard, CLF relaxed)              │
│  └── 다중 CBF 합성 (smooth-min, log-sum-exp, product)            │
│                                                                    │
│  Level 5: N-Step Predictive (PredictiveSafetyMPPI)               │
│  ├── 전체 horizon forward rollout + CBF 투영                     │
│  ├── 보정 전파: step t --> step t+1..N                            │
│  └── horizon_decay: gamma 시간 감쇠                               │
│                                                                    │
│  부가 기능:                                                        │
│  ├── ConformalPredictor (ACP): 동적 안전 마진 조정                │
│  ├── OnlineDataCollector: 런타임 데이터 수집                      │
│  └── ModelReloader: 앙상블 MLP 핫 리로드                         │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### 빠른 시작 (Gazebo Harmonic + ros2_control + nav2)

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

# CLF-CBF-MPPI (통합 안전 필터)
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=clf_cbf

# Predictive Safety MPPI (N-step CBF)
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=predictive_safety
```

**사용 가능한 controller 옵션:**

```
custom, nav2, log, tsallis, risk_aware, svmpc, smooth, spline, svg,
biased, dial, shield, adaptive_shield, clf_cbf, predictive_safety,
ilqr_mppi, cs_mppi, pi_mppi, stress_test,
swerve, non_coaxial, non_coaxial_60deg, ackermann
```

### 시작 순서 (OnProcessExit 이벤트 체인)

```
┌─────────────────────────────────────────────────────────────────┐
│  0s   │ Gazebo Harmonic + Robot State Publisher + ros_gz_bridge │
│  5s   │ Robot Spawn (gz service -s /world/*/create)            │
│       │     --> OnProcessExit                                   │
│  ~8s  │ joint_state_broadcaster (unload --> spawn 패턴)         │
│       │     --> OnProcessExit                                   │
│  ~11s │ diff_drive_controller (unload --> spawn 패턴)           │
│ 10s   │ Localization (map_server, amcl)                         │
│ 15s   │ Navigation (controller_server, planner, bt_navigator)   │
│ 20s   │ RVIZ                                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 목표 지점 설정

**RVIZ에서:**
1. **2D Pose Estimate** 클릭 -> 초기 위치 설정
2. **Nav2 Goal** 클릭 -> 목표 지점 설정

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
                             v
    ┌──────────────────────────────────────────────────────┐
    │  /diff_drive_controller/cmd_vel_unstamped            │
    └────────────────────────┬─────────────────────────────┘
                             │
                             v
                    ┌──────────────────┐
                    │  diff_drive_     │
                    │  controller      │
                    │  (ros2_control)  │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              v                              v
       /odom (nav_msgs/Odometry)     TF: odom --> base_link
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

### 아키텍처

```
┌────────────────────────────────────────────────────────────────────┐
│                     MPPIControllerPlugin                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │  MotionModel   │  │  CostFunctions   │  │    Sampler        │  │
│  │  (4종 + 2종   │  │  (9종 비용 함수) │  │  (Gaussian /      │  │
│  │   Decorator)   │  │  + CompositeCost │  │   ColoredNoise)   │  │
│  └───────┬────────┘  └───────┬──────────┘  └───────┬───────────┘  │
│          │                   │                     │              │
│  ┌───────┴────────┐         │    ┌─────────────────┴───────────┐  │
│  │ BatchDynamics  │         │    │  WeightComputation          │  │
│  │ (RK4 벡터화)   │         │    │  (Vanilla/Log/Tsallis/CVaR) │  │
│  └───────┬────────┘         │    └─────────────┬───────────────┘  │
│          │                  │                  │                  │
│          └──────────────────┼──────────────────┘                  │
│                             v                                     │
│                 ┌──────────────────────────┐                      │
│                 │     MPPI Algorithm       │                      │
│                 │  1. 제어열 shift          │                      │
│                 │  2. K개 샘플 생성         │                      │
│                 │  3. 배치 rollout          │                      │
│                 │  4. 비용 계산             │                      │
│                 │  5. 가중치 (Strategy)     │                      │
│                 │  6. 가중 평균 업데이트    │                      │
│                 └──────────┬───────────────┘                      │
│                            v                                      │
│                 ┌──────────────────────────┐                      │
│                 │     Safety Stack         │                      │
│                 │  L1: Soft Cost (CBF/CLF) │                      │
│                 │  L2: Shield (per-step)   │                      │
│                 │  L3: Adaptive Shield     │                      │
│                 │  L4: CLF-CBF-QP          │                      │
│                 │  L5: Predictive Safety   │                      │
│                 │  +  Conformal ACP        │                      │
│                 │  +  CBF 합성 (4종)       │                      │
│                 └──────────────────────────┘                      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 파라미터 튜닝 가이드

### 기본 MPPI 파라미터

```yaml
FollowPath:
  plugin: "mpc_controller_ros2::MPPIControllerPlugin"

  # 예측 호라이즌
  N: 30              # 예측 스텝 수 (길수록 먼 미래, 느림)
  dt: 0.1            # 시간 간격 [s] (작을수록 정밀, 느림)

  # 샘플링
  K: 512             # 샘플 수 (많을수록 좋은 해, 느림)
  lambda: 10.0       # 온도 (낮을수록 탐욕적, 높을수록 탐색적)

  # 노이즈
  noise_sigma_v: 0.3     # 선속도 노이즈 (클수록 넓은 탐색)
  noise_sigma_omega: 0.4 # 각속도 노이즈

  # 제어 제한
  v_max: 0.5         # 최대 선속도 [m/s]
  v_min: -0.2        # 최소 선속도 (음수=후진 허용)
  omega_max: 1.0     # 최대 각속도 [rad/s]
  omega_min: -1.0

  # 비용 가중치
  Q_x: 15.0          # x 추적 (클수록 위치 정밀)
  Q_y: 15.0          # y 추적
  Q_theta: 2.0       # 방향 추적 (작게 유지, 너무 크면 진동)
  Qf_x: 30.0         # 터미널 x (Q의 2배 권장)
  Qf_y: 30.0
  Qf_theta: 4.0
  R_v: 0.2           # 제어 비용 (클수록 에너지 절약)
  R_omega: 0.2
  R_rate_v: 1.5      # 변화율 비용 (클수록 부드러운 제어)
  R_rate_omega: 1.5
```

### 튜닝 시나리오별 권장값

| 시나리오 | K | N | lambda | Q_x/y | R_rate | 플러그인 |
|---------|---|---|--------|-------|--------|---------|
| 빠른 반응 (넓은 공간) | 256 | 20 | 5.0 | 20 | 0.5 | Vanilla |
| 좁은 통로 통과 | 512 | 30 | 15.0 | 10 | 2.0 | Biased/Shield |
| 고속 주행 (>1 m/s) | 1024 | 40 | 10.0 | 15 | 3.0 | Smooth/Spline |
| 동적 장애물 회피 | 512 | 30 | 10.0 | 15 | 1.5 | AdaptiveShield |
| 안전 최우선 | 512 | 30 | 10.0 | 10 | 1.0 | PredictiveSafety |
| Swerve 홀로노믹 | 512 | 30 | 10.0 | 10 | 2.0 | MPPI-H |

### Safety 파라미터

```yaml
# CBF 기본
cbf_enabled: true
cbf_gamma: 0.5              # CBF 감쇠율 (0.1~1.0, 클수록 보수적)
cbf_safety_margin: 0.3      # 안전 마진 [m]
cbf_robot_radius: 0.2       # 로봇 반경 [m]
cbf_activation_distance: 2.0 # CBF 활성화 거리 [m]

# Shield-MPPI
shield_cbf_stride: 3        # CBF 투영 간격 (1=매 스텝, 5=5번째마다)
shield_max_iterations: 5    # 투영 최대 반복

# Adaptive Shield
adaptive_shield_enabled: true
adaptive_shield_alpha_min: 0.1  # 최소 alpha (원거리)
adaptive_shield_alpha_max: 1.0  # 최대 alpha (근거리)

# CLF-CBF-QP
clf_cbf_enabled: true
clf_decay_rate: 1.0         # CLF 감쇠 c
clf_slack_penalty: 100.0    # 클수록 CLF 강제 (안전과 충돌 시 완화)
clf_P_scale: 1.0            # P = scale * Q

# CBF 합성 (다중 장애물)
cbf_composition_enabled: true
cbf_composition_method: 1   # 0=MIN, 1=SMOOTH_MIN, 2=LOG_SUM_EXP, 3=PRODUCT
cbf_composition_alpha: 10.0 # smooth-min 파라미터

# Predictive Safety
predictive_safety_enabled: true
predictive_safety_horizon: 0        # 0 = 전체 N
predictive_safety_decay: 0.95       # gamma 시간 감쇠
predictive_safety_max_iterations: 8
```

### 동적 파라미터 조정

```bash
# 런타임 중 파라미터 변경
ros2 param set /controller_server FollowPath.lambda 15.0
ros2 param set /controller_server FollowPath.noise_sigma_v 0.5
ros2 param set /controller_server FollowPath.Q_x 20.0
```

---

## 성능 벤치마크

### MPPI Pipeline (Release -O2 -march=native)

```
┌──────────────────────────────────────────────────────────────────┐
│  MPPI Pipeline Benchmark (DiffDrive, N=30)                       │
├──────────┬───────────┬───────────┬──────────┬───────────────────┤
│ K        │ Pipeline  │ Frequency │ Rollout  │ Cost              │
├──────────┼───────────┼───────────┼──────────┼───────────────────┤
│ 256      │ 0.92ms    │ 1,091 Hz  │ 567us    │ 194us             │
│ 512      │ 1.88ms    │ 532 Hz    │ 1.15ms   │ 407us             │
│ 1024     │ 3.88ms    │ 258 Hz    │ 2.43ms   │ 805us             │
├──────────┼───────────┼───────────┼──────────┼───────────────────┤
│ Swerve   │ 1.95ms    │ 512 Hz    │ (K=512)  │                   │
│ NonCoax  │ 2.80ms    │ 357 Hz    │ (K=512)  │                   │
│ Ackermann│ ~1.9ms    │ ~530 Hz   │ (K=512)  │ nu=2, nx=4        │
└──────────┴───────────┴───────────┴──────────┴───────────────────┘

최적화 기법:
  - True Batch Rollout: K*N*4 --> N*4 propagateBatch (512배 감소)
  - InPlace: heap alloc 1536/call --> 0/call
  - 대각 Q/R: cwiseAbs2().dot() 벡터 연산
  - -march=native: Eigen AVX2 SIMD 활성화
```

```bash
# 벤치마크 실행
colcon build --packages-select mpc_controller_ros2 --cmake-args -DCMAKE_BUILD_TYPE=Release
./build/mpc_controller_ros2/bench_mppi_pipeline --K 512 --N 30
./build/mpc_controller_ros2/bench_mppi_pipeline --scaling
```

### Swerve 벤치마크 (Narrow Passage, Normal vs Stress)

```
┌──────────────────┬─────────────────────────────┬─────────────────────────────┐
│ Controller       │ Normal (planner 20Hz)       │ Stress (planner 1Hz)        │
├──────────────────┼─────────────────────────────┼─────────────────────────────┤
│ swerve (vanilla) │ OK  55s |  28 near | 0 coll│ NG 120s | 448 near|87 coll │
│ non_coaxial      │ OK  45s |   0 near | 0 coll│ NG 120s | 119 near| 0 coll │
│ biased_swerve    │ OK  64s | 102 near |14 coll│ OK  20s | 117 near|21 coll │
│ pi_swerve        │ OK  15s |  12 near | 0 coll│ OK  83s | 400 near|51 coll │
│ smooth_swerve    │ OK  24s |  35 near | 6 coll│ NG  66s |  46 near| 2 coll │
│ cs_swerve        │ OK  19s |  49 near |18 coll│ NG  60s | 270 near|61 coll │
│ dial_swerve      │ OK  80s |  50 near | 0 coll│ NG 120s | 415 near|39 coll │
│ log_swerve       │ NG  15s |   1 near | 0 coll│ OK  88s | 235 near|64 coll │
└──────────────────┴─────────────────────────────┴─────────────────────────────┘

자체 회피력 순위: 1.biased > 2.pi > 3.log > 4.non_coaxial > 5.smooth
```

---

## 디버깅

```bash
# cmd_vel 체인 확인
ros2 topic echo /diff_drive_controller/cmd_vel_unstamped

# TF 확인
ros2 run tf2_tools view_frames
ros2 run tf2_ros tf2_echo map odom

# Controller 상태
ros2 control list_controllers

# nav2 라이프사이클
ros2 lifecycle get /controller_server
```

### 트러블슈팅

| 문제 | 원인 | 해결 |
|------|------|------|
| "Control loop missed rate" | cmd_vel 토픽 연결 | `ros2 topic info -v` 확인 |
| TF map->odom 없음 | AMCL 라이다 미수신 | `/scan` 토픽 확인 |
| 로봇 미동작 | Controller 비활성 | `ros2 control list_controllers` |
| Twist/TwistStamped 충돌 | 메시지 타입 불일치 | `use_stamped_vel: false` |
| Gazebo 센서 미동작 | 관성 텐서 오류 | Error Code 19 확인, ixx/iyy/izz 검증 |

---

## MPC Controller (Python Node)

CasADi/IPOPT 기반 MPC 컨트롤러 (Standalone ROS2 노드).

```bash
ros2 launch mpc_controller_ros2 mpc_controller.launch.py
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `mpc.N` | int | 20 | 예측 구간 |
| `mpc.dt` | float | 0.1 | 시간 간격 [s] |
| `mpc.Q` | list | [10, 10, 1] | 상태 가중치 [x, y, theta] |
| `mpc.R` | list | [0.1, 0.1] | 제어 가중치 [v, omega] |
| `robot.max_velocity` | float | 1.0 | 최대 선속도 [m/s] |
| `robot.max_omega` | float | 1.5 | 최대 각속도 [rad/s] |

---

## 테스트

### C++ 단위 테스트 (494개, 26 스위트)

```bash
# 전체 테스트
colcon test --packages-select mpc_controller_ros2
colcon test-result --verbose

# 개별 테스트
./build/mpc_controller_ros2/test_mppi_algorithm
./build/mpc_controller_ros2/test_cbf_composition
```

| 카테고리 | 테스트 | 수 |
|---------|--------|-----|
| **핵심 MPPI** | batch_dynamics, cost_functions, sampling, mppi_algorithm | 61 |
| **M2 고도화** | adaptive_temperature, tube_mppi, trajectory_stability | 47 |
| **SOTA 변형** | weight_computation, svmpc, m35_plugins, biased, dial | 93 |
| **모션 모델** | motion_model (DiffDrive/Swerve/NonCoaxial/Ackermann) | 68 |
| **안전성** | cbf, safety_enhancement, advanced_cbf, clf_cbf_qp, cbf_composition, predictive_safety | 114 |
| **고급 최적화** | ilqr_solver, ilqr_mppi, cs_mppi, pi_mppi, hybrid_swerve | 70 |
| **학습 기반** | residual_dynamics, ensemble_dynamics, online_learning | 41 |
| **총계** | **26 스위트** | **494** |

---

## 개발 이력

| 마일스톤 | 내용 | PR |
|----------|------|----|
| M1 | Vanilla MPPI (Python) | - |
| M2 | Colored Noise, Adaptive Temp, Tube-MPPI | - |
| M3 | Log, Tsallis, CVaR, SVMPC | - |
| M3.5 | Smooth, Spline, SVG-MPPI | - |
| M4 | ROS2 nav2 통합 + Gazebo | - |
| M5 | C++ 포팅 (SOTA + M2 + M3.5) | - |
| MPPI-CBF | Control Barrier Function (Python + C++) | #98 |
| 성능 최적화 | True Batch + InPlace + SIMD (K=512->532Hz) | #132 |
| Ackermann | Bicycle model MotionModel (nx=4, nu=2) | #138 |
| Safety Enhancement | EigenMLP + ResidualDynamics + Shield-MPPI + BR-MPPI + ACP | #140 |
| iLQR-MPPI | iLQR warm-start + MPPI 파이프라인 | #142 |
| CS-MPPI | Covariance Steering (CoVO-MPC, CoRL 2023) | #150 |
| pi-MPPI | ADMM QP 투영 필터 (RA-L 2025) | #152 |
| MPPI-H | Hybrid Swerve Low-D<->4D (IROS 2024) | #153 |
| Biased-MPPI | Ancillary biased sampling (RA-L 2024) | #123 |
| DIAL-MPPI | Diffusion annealing (ICRA 2025) | #125 |
| Learning MPPI | Ensemble Dynamics + C3BF + Adaptive Shield | #159 |
| CLF-CBF-QP | CLF-CBF 통합 안전 필터 (Ames 2019) | #161 |
| CBF 합성 + Predictive Safety | 다중 CBF + Online Learning + N-step CBF | #165 |

---

## 참고 자료

- [nav2 Controller Plugin Tutorial](https://docs.nav2.org/plugin_tutorials/docs/writing_new_nav2controller_plugin.html)
- [MPPI Paper (Williams 2017)](https://ieeexplore.ieee.org/document/7487277)
- [CLF-CBF-QP (Ames 2019)](https://ieeexplore.ieee.org/document/8796030)
- [CoVO-MPC (CoRL 2023)](https://openreview.net/forum?id=xGfSoLtlvc)
- [pi-MPPI (RA-L 2025)](https://arxiv.org/abs/2409.08648)
- [CasADi Documentation](https://web.casadi.org/)
- [Eigen Quick Reference](https://eigen.tuxfamily.org/dox/group__QuickRefPage.html)

## 라이센스

MIT License
