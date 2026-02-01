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

### 2.1 핵심 기능 (Milestone 1: Vanilla MPPI)

- **FR-1**: MPPIParams 데이터클래스 (N, dt, K, lambda, sigma, Q, R, Qf)
- **FR-2**: BatchDynamicsWrapper - DifferentialDriveModel을 K개 샘플 벡터화
- **FR-3**: 비용 함수 모듈 (StateTracking, ControlEffort, Terminal, Obstacle)
- **FR-4**: Gaussian 노이즈 샘플링 모듈
- **FR-5**: Vanilla MPPI 컨트롤러 (`compute_control` 인터페이스 호환)
- **FR-6**: RVIZ 시각화 (샘플 궤적, 가중 궤적, 비용 히트맵)
- **FR-7**: 원형 궤적 추적 데모

### 2.2 고도화 기능 (Milestone 2: M2 MPPI) ✅ 완료

- **FR-8**: ControlRateCost — 제어 변화율(Δu) 비용으로 부드러운 제어 유도
- **FR-9**: AdaptiveTemperature — ESS 기반 λ 자동 튜닝 (목표 ESS 비율 유지)
- **FR-10**: ColoredNoiseSampler — OU 프로세스 기반 시간 상관 노이즈 생성
- **FR-11**: Tube-MPPI — AncillaryController(body frame 피드백) + TubeMPPIController(명목 상태 전파)
- **FR-12**: TubeAwareCost — 장애물 safety_margin + tube_margin 확장
- **FR-13**: Vanilla vs M2 / Vanilla vs Tube 비교 데모 (--live, --noise 지원)

### 2.3 SOTA 변형 (Milestone 3: M3 MPPI)

#### M3a: Log-MPPI ✅ 완료
- **FR-14**: LogMPPIController — log-space softmax 가중치 (참조 구현)
- **FR-15**: Vanilla와 수학적 동등성 확인 (기존 max-shift trick과 동일)

#### M3b: Tsallis-MPPI ✅ 완료
- **FR-16**: TsallisMPPIController — q-exponential 가중치 (heavy/light-tail 제어)
- **FR-17**: q_exponential(), q_logarithm() 유틸리티 함수
- **FR-18**: tsallis_q 파라미터 (기본 1.0 = Vanilla 하위 호환)
- **FR-19**: 비용 min-centering (q-exp translation-invariance 보정)
- **FR-20**: q값 비교 데모 (q=0.5, 1.0, 1.2, 1.5)

#### M3c: Risk-Aware MPPI (CVaR) ✅ 완료
- **FR-21**: RiskAwareMPPIController — CVaR 가중치 절단 (최저 비용 ceil(alpha*K)개만 softmax)
- **FR-22**: cvar_alpha 파라미터 (1.0=risk-neutral/Vanilla, <1=risk-averse, 실용 범위 [0.1, 1.0])

#### M3d: Stein Variational MPPI (SVMPC) ✅ 완료
- **FR-23**: SteinVariationalMPPIController — SVGD 기반 샘플 다양성 유도
- **FR-24**: rbf_kernel, rbf_kernel_grad, median_bandwidth 유틸리티
- **FR-25**: svgd_num_iterations=0 → Vanilla 하위 호환

### 2.4 비기능 요구사항

- **NFR-1**: 순수 NumPy 구현 (CasADi 의존성 없음)
- **NFR-2**: K=1024 샘플, N=30 호라이즌에서 실행 가능
- **NFR-3**: 기존 `compute_control(state, ref) -> (control, info)` 시그니처 준수
- **NFR-4**: for-loop 금지, NumPy broadcasting으로 배치 처리
- **NFR-5**: 원형 궤적 추적 Position RMSE < 0.2m
- **NFR-6**: Tube-MPPI tube_enabled=False 시 Vanilla와 100% 동일 동작 (하위 호환)

## 3. 아키텍처

```
시스템 아키텍처:

┌──────────────────────────────────────────────────────────────────┐
│                     MPPIController (base_mppi.py)                │
│                                                                  │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────────┐   │
│  │ MPPIParams    │  │ GaussianSampler│  │ CompositeMPPICost  │   │
│  │ (mppi_params) │  │ (sampling.py)  │  │ (cost_functions.py)│   │
│  └──────┬───────┘  └───────┬───────┘  └────────┬───────────┘   │
│         │                  │                    │               │
│         │     ┌────────────▼────────────┐       │               │
│         └────►│ BatchDynamicsWrapper    │◄──────┘               │
│               │ (dynamics_wrapper.py)   │                       │
│               └────────────┬────────────┘                       │
│                            │                                    │
│                  ┌─────────▼─────────┐                          │
│                  │ DifferentialDrive  │ (기존 모델 재사용)       │
│                  │ Model             │                           │
│                  └───────────────────┘                           │
│                                                                  │
│  compute_control(state, ref) -> (control, info)                 │
│    info: {sample_trajectories, sample_weights, best_trajectory} │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
┌───────────▼───────────┐  ┌──────▼──────────────┐  ┌────▼────────────────────┐
│ TubeMPPIController    │  │ AdaptiveTemperature │  │ MPPIRVizVisualizer      │
│ (tube_mppi.py) [M2]   │  │ (adaptive_temp.py)  │  │ (mppi_rviz_visualizer)  │
│ - MPPIController 상속 │  │ - ESS 기반 λ 튜닝  │  │ - 샘플 궤적 (투명도)   │
│ - 명목 상태 전파      │  │ - 목표 ESS 비율    │  │ - 가중 평균 궤적 (시안)│
│ - AncillaryController │  └─────────────────────┘  │ - 비용 히트맵           │
│   (body frame 피드백) │                           │ - 온도/ESS 텍스트       │
│ - TubeAwareCost       │                           └─────────────────────────┘
│ - tube 경계 시각화    │
└───────────────────────┘
┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────────┐
│ LogMPPIController     │  │ TsallisMPPIController │  │ RiskAwareMPPIController    │
│ (log_mppi.py) [M3a]   │  │ (tsallis_mppi.py)[M3b]│  │ (risk_aware_mppi.py) [M3c] │
│ - log-space softmax   │  │ - q-exponential 가중치│  │ - CVaR 가중치 절단         │
│ - 참조 구현           │  │ - min-centering       │  │ - alpha=1→Vanilla          │
│   (Vanilla와 동등)    │  │ - q>1=탐색, q<1=집중 │  │ - alpha<1→risk-averse      │
└───────────────────────┘  └───────────────────────┘  └───────────────────────────┘
```

### 파일 구조

```
mpc_controller/controllers/mppi/
  __init__.py                   # MPPIController, TubeMPPIController 등 export
  mppi_params.py                # MPPIParams 데이터클래스 (tube 필드 포함)
  dynamics_wrapper.py           # BatchDynamicsWrapper (RK4 벡터화)
  cost_functions.py             # 비용 함수 모듈 (ControlRateCost, TubeAwareCost 포함)
  sampling.py                   # GaussianSampler, ColoredNoiseSampler
  base_mppi.py                  # Vanilla MPPI 핵심 알고리즘 (_compute_weights 오버라이드 포인트)
  adaptive_temperature.py       # AdaptiveTemperature (ESS 기반 λ 자동 튜닝) [M2]
  ancillary_controller.py       # AncillaryController (body frame 피드백 보정) [M2]
  tube_mppi.py                  # TubeMPPIController (명목 상태 전파 + 피드백) [M2]
  log_mppi.py                   # LogMPPIController (log-space softmax 참조 구현) [M3a]
  tsallis_mppi.py               # TsallisMPPIController (q-exponential 가중치) [M3b]
  risk_aware_mppi.py            # RiskAwareMPPIController (CVaR 가중치 절단) [M3c]
  stein_variational_mppi.py     # SteinVariationalMPPIController (SVGD 샘플 다양성) [M3d]
  utils.py                      # normalize_angle, log_sum_exp, q_exponential, rbf_kernel 등

mpc_controller/ros2/
  mppi_rviz_visualizer.py       # RVIZ 시각화

tests/
  test_mppi.py                  # Vanilla MPPI 유닛 + 통합 테스트
  test_mppi_cost_functions.py   # 비용 함수 테스트
  test_mppi_sampling.py         # 샘플링 테스트
  test_mppi_live_demo.py        # 라이브 데모 테스트
  test_ancillary_controller.py  # AncillaryController 테스트 (14개) [M2]
  test_tube_mppi.py             # TubeMPPIController 테스트 (13개) [M2]
  test_log_mppi.py              # LogMPPIController 테스트 (15개) [M3a]
  test_tsallis_mppi.py          # TsallisMPPIController 테스트 (24개) [M3b]
  test_risk_aware_mppi.py       # RiskAwareMPPIController 테스트 (22개) [M3c]
  test_stein_variational_mppi.py # SteinVariationalMPPIController 테스트 (23개) [M3d]

examples/
  mppi_basic_demo.py            # Vanilla MPPI 데모
  mppi_vanilla_vs_m2_demo.py    # Vanilla vs M2 비교 데모 [M2]
  mppi_vanilla_vs_tube_demo.py  # Vanilla vs Tube 비교 데모 [M2]
  log_mppi_demo.py              # Vanilla vs Log-MPPI 비교 데모 [M3a]
  tsallis_mppi_demo.py          # Tsallis q 파라미터 비교 데모 [M3b]
  risk_aware_mppi_demo.py       # Risk-Aware alpha 비교 데모 (장애물 회피) [M3c]
  stein_variational_mppi_demo.py # SVMPC SVGD iteration 비교 데모 [M3d]

configs/
  mppi_params.yaml              # 기본 설정
```

## 4. 마일스톤 로드맵

```
M1: Vanilla MPPI ✅ 완료
├── MPPIParams 데이터클래스
├── BatchDynamicsWrapper
├── StateTracking / Obstacle 비용 함수
├── Gaussian 샘플링
├── Vanilla MPPI 컨트롤러
├── RVIZ 시각화
└── 원형 궤적 추적 테스트 (RMSE=0.1534m < 0.2m)

M2: 고도화 ✅ 완료 (GPU 가속 잔여)
├── ✅ Colored Noise 샘플링 (OU 프로세스 기반)
├── ✅ Tube-MPPI (AncillaryController + TubeMPPIController)
├── ✅ Adaptive temperature (ESS 기반 λ auto-tuning)
├── ✅ ControlRateCost (제어 변화율 비용)
├── ✅ TubeAwareCost (tube margin 확장)
└── ⬜ GPU 가속 (PyTorch/Numba) — 잔여

M3: SOTA 변형
├── ✅ M3a: Log-MPPI (log-space softmax, 참조 구현)
├── ✅ M3b: Tsallis-MPPI (q-exponential, min-centering)
├── ✅ M3c: Risk-Aware MPPI (CVaR 가중치 절단)
└── ✅ M3d: Stein Variational MPPI (SVMPC — SVGD 커널 기반)

M4: ROS2 통합 마무리 (예정)
├── nav2 플러그인
├── 실제 로봇 인터페이스
├── 파라미터 서버 통합
└── 성능 벤치마크
```

## 5. 참조

### 논문
- Williams et al. (2016) - "Aggressive Driving with MPPI" (원본 MPPI)
- Williams et al. (2018) - "Robust Sampling Based MPPI" (Tube-MPPI)
- Yin et al. (2021) - "Trajectory Distribution Control via Tsallis Entropy" (Tsallis MPPI)
- Yin et al. (2023) - "Risk-Aware MPPI" (RA-MPPI)
- Lambert et al. (2020) - "Stein Variational Model Predictive Control" (SVMPC)

### 참조 구현
- `pytorch_mppi` - PyTorch GPU 가속 MPPI
- `mppic` - ROS2 nav2 MPPI Controller 플러그인 (C++)
- `PythonLinearNonlinearControl` - Python 제어 알고리즘 모음
