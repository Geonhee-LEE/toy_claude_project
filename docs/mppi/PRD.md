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

### 2.2 비기능 요구사항

- **NFR-1**: 순수 NumPy 구현 (CasADi 의존성 없음)
- **NFR-2**: K=1024 샘플, N=30 호라이즌에서 실행 가능
- **NFR-3**: 기존 `compute_control(state, ref) -> (control, info)` 시그니처 준수
- **NFR-4**: for-loop 금지, NumPy broadcasting으로 배치 처리
- **NFR-5**: 원형 궤적 추적 Position RMSE < 0.2m

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
                    ┌──────────────▼──────────────┐
                    │  MPPIRVizVisualizer          │
                    │  (mppi_rviz_visualizer.py)   │
                    │  - 샘플 궤적 (투명도=가중치) │
                    │  - 가중 평균 궤적 (시안)     │
                    │  - 최적 샘플 (마젠타)        │
                    │  - 비용 히트맵 (초록→빨강)   │
                    │  - 온도/ESS 텍스트           │
                    └─────────────────────────────┘
```

### 파일 구조

```
mpc_controller/controllers/mppi/
  __init__.py                   # MPPIController, MPPIParams export
  mppi_params.py                # MPPIParams 데이터클래스
  dynamics_wrapper.py           # BatchDynamicsWrapper
  cost_functions.py             # 비용 함수 모듈
  sampling.py                   # GaussianSampler
  base_mppi.py                  # Vanilla MPPI 핵심 알고리즘
  utils.py                      # normalize_angle_batch, log-sum-exp

mpc_controller/ros2/
  mppi_rviz_visualizer.py       # RVIZ 시각화

tests/
  test_mppi.py                  # 유닛 + 통합 테스트
  test_mppi_cost_functions.py   # 비용 함수 테스트
  test_mppi_sampling.py         # 샘플링 테스트

examples/
  mppi_basic_demo.py            # Vanilla MPPI 데모

configs/
  mppi_params.yaml              # 기본 설정
```

## 4. 마일스톤 로드맵

```
M1: Vanilla MPPI (이번 범위)
├── MPPIParams 데이터클래스
├── BatchDynamicsWrapper
├── StateTracking / Obstacle 비용 함수
├── Gaussian 샘플링
├── Vanilla MPPI 컨트롤러
├── RVIZ 시각화
└── 원형 궤적 추적 테스트

M2: 고도화 (추후)
├── Colored Noise 샘플링 (beta-distribution)
├── Tube-MPPI (ancillary controller)
├── Adaptive temperature (λ auto-tuning)
└── GPU 가속 (PyTorch/Numba)

M3: SOTA 변형 (추후)
├── Tsallis MPPI (일반화 엔트로피)
├── Risk-Aware MPPI (CVaR)
├── Log-MPPI (log-space update)
└── Stein Variational MPPI (SVMPC)

M4: ROS2 통합 마무리 (추후)
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

### 참조 구현
- `pytorch_mppi` - PyTorch GPU 가속 MPPI
- `mppic` - ROS2 nav2 MPPI Controller 플러그인 (C++)
- `PythonLinearNonlinearControl` - Python 제어 알고리즘 모음
