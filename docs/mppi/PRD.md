# MPPI (Model Predictive Path Integral) Control - PRD

## 1. 프로젝트 배경

### 기존 MPC (CasADi/IPOPT) 대비 MPPI 장점

| 항목 | MPC (CasADi/IPOPT) | MPPI |
|------|---------------------|------|
| 최적화 방식 | Gradient-based NLP | Sampling-based (derivative-free) |
| 비용 함수 | 미분 가능해야 함 | 임의 비용 함수 가능 |
| 제약 조건 | Hard/Soft constraint | 비용 함수에 포함 |
| 병렬성 | 직렬 (IPOPT) | K개 샘플 병렬 처리 |
| 실시간성 | 수렴까지 가변 시간 | 고정 계산 시간 (샘플 수 의존) |
| 비선형 동역학 | CasADi symbolic 필요 | NumPy forward simulation |
| 로컬 최적 | 빠짐 가능 | 샘플링으로 탐색 범위 넓음 |

### MPPI 기본 원리 (Williams et al., 2016)

```
MPPI 알고리즘 흐름:
                                     ┌─────────────┐
                                     │ 초기 제어열  │
                                     │  U (N, nu)  │
                                     └──────┬──────┘
                                            │
                            ┌───────────────┼───────────────┐
                            │               │               │
                    ┌───────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
                    │ 노이즈 샘플  │ │ 노이즈 샘플 │ │ 노이즈 샘플 │
                    │  ε_1 ~ N(0,Σ)│ │  ε_2 ~ N(0,Σ)│ │  ε_K ~ N(0,Σ)│
                    └───────┬──────┘ └──────┬──────┘ └──────┬──────┘
                            │               │               │
                    ┌───────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
                    │ U + ε_1      │ │ U + ε_2     │ │ U + ε_K     │
                    │ rollout_1    │ │ rollout_2   │ │ rollout_K   │
                    └───────┬──────┘ └──────┬──────┘ └──────┬──────┘
                            │               │               │
                    ┌───────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
                    │ cost_1       │ │ cost_2      │ │ cost_K      │
                    │ = S(τ_1)     │ │ = S(τ_2)    │ │ = S(τ_K)    │
                    └───────┬──────┘ └──────┬──────┘ └──────┬──────┘
                            │               │               │
                            └───────────────┼───────────────┘
                                            │
                                    ┌───────▼──────┐
                                    │ Softmax 가중  │
                                    │ w_k ∝ exp    │
                                    │ (-S_k / λ)  │
                                    └───────┬──────┘
                                            │
                                    ┌───────▼──────┐
                                    │ 가중 평균     │
                                    │ U* = Σ w_k   │
                                    │     × ε_k    │
                                    └───────┬──────┘
                                            │
                                    ┌───────▼──────┐
                                    │ 첫 제어 적용  │
                                    │ u* = U*[0]   │
                                    └──────────────┘
```

## 2. 기능 요구사항

### 2.1 핵심 기능 (Milestone 1: Vanilla MPPI) ✅

- **FR-1**: MPPIParams 데이터클래스 (N, dt, K, lambda, sigma, Q, R, Qf)
- **FR-2**: BatchDynamicsWrapper - DifferentialDriveModel을 K개 샘플 벡터화
- **FR-3**: 비용 함수 모듈 (StateTracking, ControlEffort, Terminal, Obstacle)
- **FR-4**: Gaussian 노이즈 샘플링 모듈
- **FR-5**: Vanilla MPPI 컨트롤러 (`compute_control` 인터페이스 호환)
- **FR-6**: RVIZ 시각화 (샘플 궤적, 가중 궤적, 비용 히트맵)
- **FR-7**: 원형 궤적 추적 데모

### 2.2 고도화 기능 (Milestone 2: M2 MPPI) ✅

- **FR-8**: ControlRateCost — 제어 변화율(Δu) 비용으로 부드러운 제어 유도
- **FR-9**: AdaptiveTemperature — ESS 기반 λ 자동 튜닝 (목표 ESS 비율 유지)
- **FR-10**: ColoredNoiseSampler — OU 프로세스 기반 시간 상관 노이즈 생성
- **FR-11**: Tube-MPPI — AncillaryController(body frame 피드백) + TubeMPPIController(명목 상태 전파)
- **FR-12**: TubeAwareCost — 장애물 safety_margin + tube_margin 확장
- **FR-13**: Vanilla vs M2 / Vanilla vs Tube 비교 데모 (--live, --noise 지원)

### 2.3 SOTA 변형 (Milestone 3) ✅

- **FR-14~15**: Log-MPPI — log-space softmax 가중치
- **FR-16~20**: Tsallis-MPPI — q-exponential 가중치 (heavy/light-tail 제어)
- **FR-21~22**: Risk-Aware MPPI (CVaR) — alpha 기반 가중치 절단
- **FR-23~25**: SVMPC — SVGD 커널 기반 샘플 다양성 유도

### 2.4 SOTA 변형 확장 (Milestone 3.5) ✅

- **FR-26~28**: Smooth MPPI — Δu space 최적화 + cumsum 복원 (input-lifting)
- **FR-29~31**: Spline-MPPI — P개 knot 노이즈 → B-spline basis(N,P) 보간
- **FR-32~34**: SVG-MPPI — G개 guide particle SVGD + (K-G)개 follower resampling

### 2.5 ROS2 C++ nav2 플러그인 (Milestone 4/5) ✅

- **FR-39~44**: M4 ROS2 nav2 통합 (Vanilla MPPI, Gazebo, costmap, 파라미터)
- **FR-45~54**: M5a C++ SOTA (WeightComputation Strategy, Log/Tsallis/CVaR/SVMPC)
- **FR-55~57**: M5b C++ M2 (Colored Noise, Adaptive Temp, Tube-MPPI)
- **FR-58~65**: M3.5 C++ (Smooth/Spline/SVG-MPPI)
- **FR-66~72**: Swerve 모션 품질 (theta smoothing, vy_max, VelocityTrackingCost)

### 2.6 벤치마크 도구 ✅

- **FR-35~38**: 9종 변형 벤치마크 + Python vs C++ 벤치마크

---

### 2.7 [신규] C++ 컨트롤러 고도화 로드맵 (M7~)

```
┌──────────────────────────────────────────────────────────────┐
│  ROS2 C++ 컨트롤러 고도화 방향                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  M7: Biased-MPPI C++ nav2 플러그인 (#121)                   │
│  ├── ancillary 컨트롤러 편향 샘플링                         │
│  ├── J deterministic + (K-J) Gaussian 하이브리드             │
│  └── 기존 WeightComputation Strategy 재사용                  │
│                                                              │
│  M8: Ackermann MotionModel C++                               │
│  ├── nx=4 (x, y, θ, δ), nu=2 (v, δ_dot)                   │
│  ├── MotionModelFactory 확장                                 │
│  ├── Ackermann 전용 nav2 파라미터 YAML                      │
│  └── 전체 플러그인 8종 자동 호환                             │
│                                                              │
│  M9: C++ MPPI 성능 최적화                                    │
│  ├── perf/vtune hotspot 프로파일링                           │
│  ├── Eigen SIMD 최적화 (Map + aligned alloc)                │
│  ├── 캐시 지역성 개선 (SoA layout)                          │
│  ├── 멀티스레드 rollout (OpenMP 심화)                        │
│  └── 벤치마크: 제어 주파수 20Hz → 50Hz 목표                 │
│                                                              │
│  M10: 최신 MPPI 변형 C++                                     │
│  ├── Covariance Steering MPPI (공분산 제어)                  │
│  ├── π-MPPI (policy gradient + MPPI 융합)                   │
│  ├── BR-MPPI (Barrier-Regularized MPPI)                     │
│  └── SOPPI (Second-Order Path Integral)                      │
│                                                              │
│  M11: 안전성 고도화 C++                                      │
│  ├── CLF-CBF-QP 통합 컨트롤러                               │
│  ├── 다중 CBF 합성 (교집합/합집합)                           │
│  └── 적응형 CBF 감마 자동 조정                               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

#### M7: Biased-MPPI C++ nav2 플러그인 (Issue #121)
- **FR-73**: BiasedMPPIControllerPlugin — MPPIControllerPlugin 상속
- **FR-74**: AncillaryController 샘플 주입 (J deterministic + (K-J) Gaussian)
- **FR-75**: bias_ratio 파라미터 (J/K 비율, 기본 0.3)
- **FR-76**: ancillary_type 파라미터 ("tube", "pid", "lqr")
- **FR-77**: 기존 WeightComputation 전략과 조합 가능 (Biased + Log, Biased + CVaR 등)
- **FR-78**: nav2_params_biased_mppi.yaml 설정 파일
- **FR-79**: 단위 테스트 (bias injection, weight 호환, Vanilla 동등성)

#### M8: Ackermann MotionModel C++
- **FR-80**: AckermannModel 클래스 — MotionModel 인터페이스 구현
- **FR-81**: nx=4 (x, y, θ, δ), nu=2 (velocity, steering_rate)
- **FR-82**: 최소 회전 반경 제약 (δ_max)
- **FR-83**: MotionModelFactory에 "ackermann" 등록
- **FR-84**: nav2_params_ackermann_mppi.yaml 설정 파일
- **FR-85**: launch `controller:=ackermann` 분기
- **FR-86**: Ackermann URDF + Gazebo 모델

#### M9: C++ MPPI 성능 최적화
- **FR-87**: hotspot 프로파일 + 분석 레포트 생성 스크립트
- **FR-88**: Eigen Map 기반 zero-copy 데이터 전달
- **FR-89**: OpenMP parallel_for rollout 최적화
- **FR-90**: SoA (Structure of Arrays) 메모리 레이아웃
- **FR-91**: 성능 벤치마크 (제어 주파수, 레이턴시 p50/p99)

### 2.8 비기능 요구사항

- **NFR-1**: 순수 NumPy 구현 (CasADi 의존성 없음) — Python
- **NFR-2**: K=1024 샘플, N=30 호라이즌에서 실행 가능
- **NFR-3**: 기존 `compute_control(state, ref) -> (control, info)` 시그니처 준수
- **NFR-4**: for-loop 금지, NumPy broadcasting으로 배치 처리 (GPU: JAX lax.scan + vmap)
- **NFR-5**: 원형 궤적 추적 Position RMSE < 0.2m
- **NFR-6**: Tube-MPPI tube_enabled=False 시 Vanilla와 100% 동일 동작 (하위 호환)
- **NFR-7**: [신규] C++ 신규 플러그인은 기존 nav2 파라미터 체계와 100% 호환
- **NFR-8**: [신규] MotionModel 추가 시 기존 8종 플러그인 코드 변경 불필요

## 3. 아키텍처

### C++ nav2 플러그인 아키텍처 (핵심)

```
nav2_core::Controller
       │
  ┌────┴─────────────────────────────────────────────────────┐
  │                                                           │
MPPIControllerPlugin (base)                                   │
  │ protected: computeControl() (virtual)                     │
  │ protected: params_, control_sequence_, dynamics_, ...     │
  │ uses WeightComputation strategy                           │
  │ uses MotionModel (DiffDrive/Swerve/NonCoaxial/Ackermann) │
  │                                                           │
  ├── LogMPPIControllerPlugin (상속 + LogMPPIWeights)         │
  ├── TsallisMPPIControllerPlugin (상속 + TsallisMPPIWeights) │
  ├── RiskAwareMPPIControllerPlugin (상속 + RiskAwareMPPIWeights)
  ├── BiasedMPPIControllerPlugin (M7: ancillary bias inject)  │  ← 신규
  ├── SmoothMPPIControllerPlugin (Δu space + jerk cost)      │
  ├── SplineMPPIControllerPlugin (B-spline basis 보간)       │
  └── SVMPCControllerPlugin (상속 + computeControl override)  │
        ├─ SVGD Loop: RBF kernel, attractive+repulsive force │
        └── SVGMPPIControllerPlugin (Guide SVGD + follower)  │
                                                              │
  WeightComputation (Strategy 인터페이스)                     │
  ├── VanillaMPPIWeights (softmax, max-shift)                 │
  ├── LogMPPIWeights (log-space 정규화)                       │
  ├── TsallisMPPIWeights (q-exponential)                      │
  └── RiskAwareMPPIWeights (CVaR 절단)                        │
                                                              │
  MotionModel (다형성 인터페이스)                              │
  ├── DiffDriveModel (nx=3, nu=2)                             │
  ├── SwerveDriveModel (nx=3, nu=3)                           │
  ├── NonCoaxialSwerveModel (nx=4, nu=3)                      │
  └── AckermannModel (M8: nx=4, nu=2)                         │  ← 신규

C++ 파일 구조:
ros2_ws/src/mpc_controller_ros2/
├── include/mpc_controller_ros2/
│   ├── mppi_controller_plugin.hpp         # Vanilla MPPI (base, virtual computeControl)
│   ├── log_mppi_controller_plugin.hpp     # Log-MPPI 플러그인
│   ├── tsallis_mppi_controller_plugin.hpp # Tsallis-MPPI 플러그인
│   ├── risk_aware_mppi_controller_plugin.hpp # Risk-Aware MPPI 플러그인
│   ├── svmpc_controller_plugin.hpp        # SVMPC 플러그인 (SVGD override)
│   ├── smooth_mppi_controller_plugin.hpp  # Smooth-MPPI 플러그인 (Δu space)
│   ├── spline_mppi_controller_plugin.hpp  # Spline-MPPI 플러그인 (B-spline)
│   ├── svg_mppi_controller_plugin.hpp     # SVG-MPPI 플러그인 (Guide SVGD)
│   ├── weight_computation.hpp             # Strategy 인터페이스 + 4종 구현
│   ├── mppi_params.hpp                    # 파라미터 (M2+SOTA+SVGD+CBF+Stability)
│   ├── batch_dynamics_wrapper.hpp         # 배치 동역학 (Eigen)
│   ├── cost_functions.hpp                 # 비용 함수 모듈 (VelocityTrackingCost 포함)
│   ├── sampling.hpp                       # 노이즈 샘플러
│   ├── adaptive_temperature.hpp           # ESS 기반 λ 자동 조정
│   ├── tube_mppi.hpp                      # Tube-MPPI
│   ├── motion_model.hpp                   # MotionModel 인터페이스 + Factory
│   ├── diff_drive_model.hpp               # DiffDrive 모델
│   ├── swerve_drive_model.hpp             # Swerve 모델
│   ├── non_coaxial_swerve_model.hpp       # NonCoaxial Swerve 모델
│   ├── barrier_function.hpp               # CBF 장벽 함수
│   ├── cbf_safety_filter.hpp              # CBF 안전 필터
│   ├── savitzky_golay_filter.hpp          # SG 필터
│   └── utils.hpp                          # logSumExp, softmaxWeights 등
├── src/
│   ├── mppi_controller_plugin.cpp
│   ├── log_mppi_controller_plugin.cpp
│   ├── tsallis_mppi_controller_plugin.cpp
│   ├── risk_aware_mppi_controller_plugin.cpp
│   ├── svmpc_controller_plugin.cpp
│   ├── smooth_mppi_controller_plugin.cpp
│   ├── spline_mppi_controller_plugin.cpp
│   ├── svg_mppi_controller_plugin.cpp
│   ├── weight_computation.cpp
│   ├── batch_dynamics_wrapper.cpp
│   ├── cost_functions.cpp
│   ├── sampling.cpp
│   ├── adaptive_temperature.cpp
│   ├── ancillary_controller.cpp
│   ├── tube_mppi.cpp
│   ├── motion_model.cpp / diff_drive_model.cpp / ...
│   ├── barrier_function.cpp / cbf_safety_filter.cpp
│   └── savitzky_golay_filter.cpp
├── plugins/
│   └── mppi_controller_plugin.xml         # 8종 플러그인 등록
├── config/
│   ├── nav2_params.yaml                   # DiffDrive Vanilla
│   ├── nav2_params_swerve_mppi.yaml       # Swerve MPPI
│   ├── nav2_params_non_coaxial_mppi.yaml  # NonCoaxial MPPI
│   ├── nav2_params_log_mppi.yaml          # Log-MPPI
│   ├── nav2_params_tsallis_mppi.yaml      # Tsallis-MPPI
│   ├── nav2_params_risk_aware_mppi.yaml   # Risk-Aware MPPI
│   ├── nav2_params_svmpc.yaml             # SVMPC
│   ├── nav2_params_smooth_mppi.yaml       # Smooth-MPPI
│   ├── nav2_params_spline_mppi.yaml       # Spline-MPPI
│   └── nav2_params_svg_mppi.yaml          # SVG-MPPI
└── test/unit/
    ├── test_batch_dynamics.cpp            # 8개
    ├── test_cost_functions.cpp            # 38개
    ├── test_sampling.cpp                  # 8개
    ├── test_mppi_algorithm.cpp            # 7개
    ├── test_adaptive_temperature.cpp      # 9개
    ├── test_tube_mppi.cpp                 # 13개
    ├── test_weight_computation.cpp        # 30개
    ├── test_svmpc.cpp                     # 13개
    ├── test_m35_plugins.cpp               # 18개
    ├── test_motion_model.cpp              # 48개
    ├── test_cbf.cpp                       # 20개
    └── test_trajectory_stability.cpp      # 25개
    (총 237 gtest)
```

## 4. 마일스톤 로드맵

```
═══════════════════════════════════════════════════════
  완료된 마일스톤 (M1~M5 + 부가)
═══════════════════════════════════════════════════════

M1: Vanilla MPPI ✅
M2: 고도화 (Colored Noise, Adaptive Temp, Tube-MPPI) ✅
M3: SOTA 변형 (Log, Tsallis, CVaR, SVMPC) ✅
M3.5: 확장 (Smooth, Spline, SVG-MPPI) ✅
M4: ROS2 nav2 통합 (C++ Vanilla MPPI) ✅
M5a: C++ SOTA (Log/Tsallis/CVaR/SVMPC) ✅
M5b: C++ M2 고도화 ✅
M3.5 C++: Smooth/Spline/SVG-MPPI ✅
MotionModel 추상화: DiffDrive/Swerve/NonCoaxial ✅
MPPI-CBF: Python + C++ ✅
궤적 안정화: SG Filter + IT 정규화 ✅
Swerve 모션 품질: theta smoothing + vy_max ✅
GPU 가속: JAX JIT (8종 확장) ✅
pybind11 바인딩 + 벤치마크 ✅

═══════════════════════════════════════════════════════
  진행 예정 마일스톤 — ROS2 C++ 중심
═══════════════════════════════════════════════════════

M7: Biased-MPPI C++ (Issue #121)          ◀ 다음
├── BiasedMPPIControllerPlugin
├── Ancillary 편향 샘플 주입
├── J/K 비율 파라미터 (bias_ratio)
├── 기존 WeightComputation 조합 가능
└── 단위 테스트 + 벤치마크

M8: Ackermann MotionModel C++
├── AckermannModel (nx=4, nu=2)
├── MotionModelFactory 확장
├── 최소 회전 반경 제약
└── URDF + launch 분기

M9: C++ MPPI 성능 최적화
├── hotspot 프로파일링
├── Eigen SIMD/Map 최적화
├── OpenMP 병렬 rollout 심화
└── 제어 주파수 50Hz 목표

M10: 최신 MPPI 변형 C++
├── Covariance Steering MPPI
├── π-MPPI
├── BR-MPPI
└── SOPPI

M11: 안전성 고도화 C++
├── CLF-CBF-QP 통합
├── 다중 CBF 합성
└── 적응형 CBF
```

## 5. 참조

### 논문
- Williams et al. (2016) - "Aggressive Driving with MPPI" (원본 MPPI)
- Williams et al. (2018) - "Robust Sampling Based MPPI" (Tube-MPPI)
- Yin et al. (2021) - "Trajectory Distribution Control via Tsallis Entropy" (Tsallis MPPI)
- Yin et al. (2023) - "Risk-Aware MPPI" (RA-MPPI)
- Lambert et al. (2020) - "Stein Variational Model Predictive Control" (SVMPC)
- Kim et al. (2021) - "Smooth MPPI" (SMPPI — input-lifting)
- Bhardwaj et al. (2024) - "Spline-MPPI" (ICRA 2024, B-spline interpolation)
- Kondo et al. (2024) - "SVG-MPPI" (ICRA 2024, Guide particle SVGD)
- Sacks et al. (2024) - "Biased-MPPI" (RA-L 2024, ancillary controller bias)

### 참조 구현
- `pytorch_mppi` - PyTorch GPU 가속 MPPI
- `mppic` - ROS2 nav2 MPPI Controller 플러그인 (C++)
- `PythonLinearNonlinearControl` - Python 제어 알고리즘 모음
