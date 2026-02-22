# Toy Claude Project

Mobile Robot MPC/MPPI Controller with Claude-Driven Development Workflow

## Overview

This project demonstrates:
1. **MPC-based mobile robot control** - CasADi/IPOPT 기반 경로 추종 MPC
2. **MPPI sampling-based control** - 9종 MPPI 변형 + GPU 가속 (JAX)
3. **ROS2 nav2 integration** - 8종 C++ 플러그인 + 3종 모션 모델
4. **Claude-driven development** - GitHub 이슈 자동 처리 워크플로우

## Features

### MPPI 컨트롤러 (9종)

| 변형 | 설명 | 핵심 논문/아이디어 |
|------|------|-------------------|
| **Vanilla MPPI** | 기본 샘플링 기반 MPC | Williams et al. (2017) |
| **Tube-MPPI** | 외란 강건성 (Ancillary 피드백) | 명목/실제 상태 분리 |
| **Log-MPPI** | log-space softmax 수치 안정성 | NaN/Inf 방지 |
| **Tsallis-MPPI** | q-exponential 일반화 엔트로피 | q>1 탐색, q<1 집중 |
| **Risk-Aware (CVaR)** | 가중치 절단 risk-averse | alpha 기반 worst-case |
| **SVMPC** | SVGD 커널 샘플 다양성 유도 | Stein Variational MPC |
| **Smooth-MPPI** | Δu input-lifting 부드러움 | Kim et al. (2021) |
| **Spline-MPPI** | B-spline 보간 smooth sampling | ICRA 2024 |
| **SVG-MPPI** | Guide SVGD + follower 리샘플링 | Kondo et al. (ICRA 2024) |

### 추가 기능

- **GPU 가속** (JAX JIT) — 9종 전체 GPU 지원, K=4096 샘플 실시간 처리
- **MPPI-CBF 통합** — Control Barrier Function 안전성 보장 (Python + C++)
- **궤적 안정화** — SG Filter + IT 정규화 + Exploitation/Exploration
- **3종 모션 모델** — Differential Drive, Swerve, Non-coaxial Swerve
- **ROS2 nav2 플러그인** — 8종 C++ 컨트롤러 플러그인
- **자동화 CI/CD** — GitHub Actions + Claude 이슈 자동 처리

## Quick Start

```bash
# 의존성 설치
pip install -e .

# MPC 데모
python examples/path_tracking_demo.py

# MPPI 데모 (Vanilla)
python examples/mppi_basic_demo.py --trajectory circle --live

# MPPI 비교 데모
python examples/mppi_vanilla_vs_m2_demo.py --live
python examples/mppi_vanilla_vs_tube_demo.py --live --noise 1.0

# 9종 전체 벤치마크
python examples/mppi_all_variants_benchmark.py --trajectory circle --live

# GPU 벤치마크
python examples/gpu_benchmark.py --K 512,1024,2048,4096
```

### 변형별 데모

```bash
# Log-MPPI vs Vanilla
python examples/log_mppi_demo.py --live

# Tsallis q 파라미터 비교
python examples/tsallis_mppi_demo.py --trajectory circle --live --q 0.5 1.0 1.2 1.5

# Risk-Aware (CVaR) alpha 비교
python examples/risk_aware_mppi_demo.py --live

# Smooth-MPPI jerk weight 비교
python examples/smooth_mppi_demo.py --live

# Spline-MPPI P=4 vs P=8
python examples/spline_mppi_demo.py --live

# SVMPC SVGD iteration별 비교
python examples/stein_variational_mppi_demo.py --live

# SVG-MPPI vs SVMPC vs Vanilla
python examples/svg_mppi_demo.py --live

# MPPI vs CBF-MPPI 안전성 비교
python examples/mppi_vs_cbf_mppi_demo.py --live

# CBF-MPPI 전 시나리오 벤치마크
python examples/mppi_vs_cbf_mppi_demo.py --benchmark
```

## MPPI 컨트롤러 계층 구조

```
MPPIController (base_mppi.py) — Vanilla MPPI + GPU/CPU 분기
│
│  가중치(weights) 교체만으로 동작하는 변형:
├── LogMPPIController          ── log-space softmax (수치 안정성)
├── TsallisMPPIController      ── q-exponential (q=1→Vanilla, q>1→탐색, q<1→집중)
├── RiskAwareMPPIController    ── CVaR 가중치 절단 (alpha=1→Vanilla, <1→risk-averse)
│
│  제어 공간(control space) 변환 변형:
├── SmoothMPPIController       ── Δu space 최적화 + jerk cost
├── SplineMPPIController       ── B-spline knot space (P << N 차원 축소)
│
│  샘플 분포(distribution) 개선 변형:
├── SteinVariationalMPPIController ── SVGD 커널 (매력력 + 반발력)
│   └── SVGMPPIController         ── G guide SVGD + follower (G << K)
│
│  강건성(robustness) 변형:
├── TubeMPPIController         ── 명목/실제 분리 + Ancillary 피드백
│
│  안전성(safety) 확장:
└── CBFMPPIController          ── Control Barrier Function 통합
```

### 핵심 인터페이스

모든 컨트롤러는 동일한 인터페이스를 준수합니다:

```python
u, info = controller.compute_control(current_state, reference_trajectory)
# u:    (nu,)  최적 제어 입력 [v, omega] 또는 [vx, vy, omega]
# info: dict   예측 궤적, 샘플 가중치, 비용, 온도, ESS 등
```

**info dict 주요 키:**
- `predicted_trajectory` — 가중 평균 예측 궤적 (N+1, nx)
- `sample_trajectories` — 전체 샘플 궤적 (K, N+1, nx)
- `sample_weights` — softmax 가중치 (K,)
- `best_trajectory` — 최저 비용 궤적
- `temperature` / `ess` — 현재 온도 λ / Effective Sample Size
- `solve_time` — 연산 시간 (초)
- `backend` — "gpu" 또는 "cpu"

## GPU 가속 (JAX)

### 아키텍처

`use_gpu=True` 설정으로 JAX 기반 GPU 가속을 활성화합니다.
**9종 MPPI 변형 전체**가 GPU를 지원합니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GPU 가속 아키텍처                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Python Controller (CPU)                                            │
│  ├── base_mppi.py ──── _compute_control_gpu() ──┐                  │
│  ├── smooth_mppi.py ── _compute_control_gpu() ──┤                  │
│  ├── spline_mppi.py ── _compute_control_gpu() ──┤  GPU 경로        │
│  ├── svmpc.py ──────── _compute_control_gpu() ──┤                  │
│  └── svg_mppi.py ───── _compute_control_gpu() ──┘                  │
│                              │                                      │
│  GPU Kernel Layer (JAX JIT)  ▼                                      │
│  ├── gpu_mppi_kernel.py ── mppi_step()                             │
│  │                         smooth_mppi_step()                       │
│  │                         spline_mppi_step()                       │
│  ├── gpu_weights.py ────── vanilla/log/tsallis/cvar weights        │
│  ├── gpu_dynamics.py ───── lax.scan + vmap rollout                 │
│  ├── gpu_costs.py ──────── 비용 함수 fusion + jerk cost            │
│  ├── gpu_sampling.py ───── JAX PRNG 샘플러                         │
│  └── gpu_svgd.py ───────── SVGD 커널 (SVMPC/SVG-MPPI 공유)        │
│                                                                     │
│  Backend                                                            │
│  └── gpu_backend.py ─── JAX ↔ NumPy 변환, 디바이스 감지            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### GPU 지원 변형 분류

```
┌──────────────────┬────────────────────────────────────────────────┐
│ GPU 전략         │ 해당 변형                                      │
├──────────────────┼────────────────────────────────────────────────┤
│ Weight-fn 교체   │ Log-MPPI, Tsallis-MPPI, Risk-Aware(CVaR)      │
│ 부모 상속        │ Tube-MPPI (super().compute_control() 사용)     │
│ 전용 GPU step    │ Smooth-MPPI (smooth_mppi_step)                 │
│ 전용 GPU step    │ Spline-MPPI (spline_mppi_step)                 │
│ SVGD JIT 커널    │ SVMPC (svgd_step_jit loop)                    │
│ SVGD JIT 커널    │ SVG-MPPI (guide SVGD + follower resampling)   │
└──────────────────┴────────────────────────────────────────────────┘
```

### 성능 비교

```
현재 (CPU NumPy)                        GPU 가속 후 (JAX)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 rollout_batch()     ~65%  ──────►  lax.scan + vmap  (~1-2ms)
 cost.compute()      ~25%  ──────►  단일 JIT fusion  (~0.5ms)
 sampler + softmax   ~10%  ──────►  jax.random + jnp (~0.4ms)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 합계: ~20ms (K=1024)                  합계: ~2-4ms (K=4096)
```

### 사용법

```python
from mpc_controller.controllers.mppi.mppi_params import MPPIParams
from mpc_controller.controllers.mppi.base_mppi import MPPIController
from mpc_controller.controllers.mppi.smooth_mppi import SmoothMPPIController
from mpc_controller.controllers.mppi.tsallis_mppi import TsallisMPPIController

# 1. Vanilla MPPI GPU
params = MPPIParams(K=4096, N=30, use_gpu=True)
ctrl = MPPIController(robot_params, params)
u, info = ctrl.compute_control(state, ref)
# info["backend"] == "gpu"

# 2. Tsallis-MPPI GPU (가중치 전략만 교체, GPU 자동 적용)
params = MPPIParams(K=4096, N=30, use_gpu=True, tsallis_q=1.5)
ctrl = TsallisMPPIController(robot_params, params)
u, info = ctrl.compute_control(state, ref)

# 3. Smooth-MPPI GPU (Δu space 전용 GPU step)
params = MPPIParams(K=4096, N=30, use_gpu=True)
ctrl = SmoothMPPIController(robot_params, params)
u, info = ctrl.compute_control(state, ref)
# info["delta_u_norm"] — Δu 크기 (smooth 효과 측정)

# JAX 미설치 시 자동 CPU fallback (코드 변경 불필요)
```

### JAX 설치

```bash
# CPU only
pip install jax

# GPU (CUDA 12)
pip install "jax[cuda12]>=0.4.20"

# 벤치마크
python examples/gpu_benchmark.py --K 512,1024,2048,4096
```

## Project Structure

```
mpc_controller/
├── models/                          # 로봇 동역학 모델
│   ├── differential_drive/          #   차동 구동 (nx=3, nu=2: v, omega)
│   ├── swerve_drive/                #   스워브 구동 (nx=3, nu=3: vx, vy, omega)
│   └── non_coaxial_swerve/          #   비동축 스워브 (nx=4, nu=3: vx, vy, omega)
├── controllers/
│   ├── mpc/                         # CasADi/IPOPT 기반 MPC
│   ├── mppi/                        # MPPI 샘플링 기반 제어 (9종 + GPU)
│   │   ├── base_mppi.py             #   Vanilla MPPI + GPU/CPU 분기
│   │   ├── tube_mppi.py             #   Tube-MPPI (외란 강건성)
│   │   ├── log_mppi.py              #   Log-MPPI (수치 안정성)
│   │   ├── tsallis_mppi.py          #   Tsallis-MPPI (q-exponential)
│   │   ├── risk_aware_mppi.py       #   Risk-Aware CVaR (가중치 절단)
│   │   ├── stein_variational_mppi.py #  SVMPC (SVGD 커널)
│   │   ├── smooth_mppi.py           #   Smooth-MPPI (Δu input-lifting)
│   │   ├── spline_mppi.py           #   Spline-MPPI (B-spline 보간)
│   │   ├── svg_mppi.py              #   SVG-MPPI (Guide SVGD)
│   │   ├── cbf_mppi.py              #   MPPI-CBF 통합
│   │   ├── ancillary_controller.py  #   Body frame 피드백 (Tube-MPPI)
│   │   ├── adaptive_temperature.py  #   ESS 기반 λ 자동 튜닝
│   │   ├── cost_functions.py        #   비용 함수 모듈
│   │   ├── sampling.py              #   Gaussian + Colored Noise 샘플러
│   │   ├── dynamics_wrapper.py      #   배치 동역학 (RK4 벡터화)
│   │   ├── mppi_params.py           #   파라미터 데이터클래스
│   │   ├── utils.py                 #   유틸리티 (q_exponential 등)
│   │   ├── gpu_backend.py           #   JAX/NumPy 백엔드 추상화
│   │   ├── gpu_dynamics.py          #   JIT rollout (lax.scan + vmap)
│   │   ├── gpu_costs.py             #   JIT 비용 함수 fusion + jerk cost
│   │   ├── gpu_sampling.py          #   JAX PRNG 샘플러
│   │   ├── gpu_mppi_kernel.py       #   GPU MPPI 커널 (vanilla/smooth/spline step)
│   │   ├── gpu_weights.py           #   GPU 가중치 전략 (4종 JIT weight fn)
│   │   └── gpu_svgd.py              #   SVGD JIT 커널 (SVMPC/SVG-MPPI 공유)
│   ├── swerve_mpc/                  # 스워브 MPC
│   └── non_coaxial_swerve_mpc/      # 비동축 스워브 MPC
├── ros2/                            # ROS2 노드 및 RVIZ 시각화
├── simulation/                      # 시뮬레이터
└── utils/                           # 유틸리티 (logger, trajectory 등)

docs/mppi/
├── PRD.md                           # MPPI 제품 요구사항 문서
└── MPPI_GUIDE.md                    # MPPI 기술 가이드 (알고리즘 상세)

tests/
├── test_mppi.py                     # Vanilla MPPI 유닛 + 통합 테스트
├── test_mppi_cost_functions.py      # 비용 함수 테스트
├── test_mppi_sampling.py            # 샘플링 테스트
├── test_ancillary_controller.py     # AncillaryController 테스트
├── test_tube_mppi.py                # Tube-MPPI 테스트
├── test_log_mppi.py                 # Log-MPPI 테스트
├── test_tsallis_mppi.py             # Tsallis-MPPI 테스트
├── test_risk_aware_mppi.py          # Risk-Aware CVaR 테스트
├── test_stein_variational_mppi.py   # SVMPC 테스트
├── test_smooth_mppi.py              # Smooth-MPPI 테스트
├── test_spline_mppi.py              # Spline-MPPI 테스트
├── test_svg_mppi.py                 # SVG-MPPI 테스트
├── test_cbf_barrier.py              # CBF Barrier Function 테스트
├── test_cbf_safety_filter.py        # CBF Safety Filter 테스트
├── test_cbf_mppi.py                 # CBF-MPPI 통합 테스트
├── test_gpu_mppi.py                 # GPU Vanilla MPPI 테스트 (22개)
├── test_gpu_weights.py              # GPU 가중치 전략 테스트 (20개)
├── test_gpu_svgd.py                 # GPU SVGD 커널 테스트 (13개)
└── test_gpu_variants.py             # GPU 8종 변형 통합 테스트 (26개)

examples/
├── mppi_basic_demo.py               # Vanilla MPPI 데모
├── mppi_vanilla_vs_m2_demo.py       # Vanilla vs M2 비교
├── mppi_vanilla_vs_tube_demo.py     # Vanilla vs Tube 비교
├── log_mppi_demo.py                 # Log-MPPI 비교 데모
├── tsallis_mppi_demo.py             # Tsallis q 파라미터 비교
├── risk_aware_mppi_demo.py          # CVaR alpha 비교
├── smooth_mppi_demo.py              # Smooth-MPPI jerk 비교
├── spline_mppi_demo.py              # Spline-MPPI P 비교
├── stein_variational_mppi_demo.py   # SVMPC SVGD 비교
├── svg_mppi_demo.py                 # SVG-MPPI 비교
├── mppi_all_variants_benchmark.py   # 9종 전체 벤치마크
├── mppi_vs_cbf_mppi_demo.py         # MPPI vs CBF-MPPI
├── gpu_benchmark.py                 # GPU 성능 벤치마크
├── path_tracking_demo.py            # MPC 경로 추종 데모
├── mpc_vs_mppi_demo.py              # MPC vs MPPI 비교
└── ...                              # 기타 데모
```

## MPPI 변형별 상세 설명

### Vanilla MPPI (M1)

기본 MPPI 알고리즘. K개 샘플을 병렬 rollout하여 비용 기반 softmax 가중 평균으로 최적 제어를 계산합니다.

```python
from mpc_controller.controllers.mppi.base_mppi import MPPIController
params = MPPIParams(K=1024, N=30, lambda_=1.0)
ctrl = MPPIController(robot_params, params)
```

### Tube-MPPI (M2)

명목(nominal) 상태와 실제(actual) 상태를 분리하여, Ancillary 피드백 컨트롤러로 외란을 보정합니다.

```python
from mpc_controller.controllers.mppi.tube_mppi import TubeMPPIController
params = MPPIParams(K=1024, N=30, tube_enabled=True)
ctrl = TubeMPPIController(robot_params, params)
```

### Log-MPPI (M3a)

log-space에서 softmax를 계산하여 극단적 비용(1e-15~1e15)에서도 NaN/Inf 없이 수치 안정성을 보장합니다.

```python
from mpc_controller.controllers.mppi.log_mppi import LogMPPIController
ctrl = LogMPPIController(robot_params, params)
```

### Tsallis-MPPI (M3b)

q-exponential 가중치로 탐색/집중 정도를 조절합니다.
- `q=1.0` → Vanilla 동일
- `q>1` → heavy-tail (넓은 탐색)
- `q<1` → light-tail (최적 집중)

```python
from mpc_controller.controllers.mppi.tsallis_mppi import TsallisMPPIController
params = MPPIParams(K=1024, N=30, tsallis_q=1.5)
ctrl = TsallisMPPIController(robot_params, params)
```

### Risk-Aware CVaR (M3c)

최저 비용 상위 `ceil(alpha*K)`개 샘플만으로 softmax를 계산하여 worst-case에 강건합니다.
- `alpha=1.0` → Vanilla 동일
- `alpha<1.0` → risk-averse (보수적)

```python
from mpc_controller.controllers.mppi.risk_aware_mppi import RiskAwareMPPIController
params = MPPIParams(K=1024, N=30, cvar_alpha=0.5)
ctrl = RiskAwareMPPIController(robot_params, params)
```

### SVMPC (M3d)

Stein Variational Gradient Descent 커널로 샘플 분포의 다양성을 유도합니다. 매력력(저비용 방향)과 반발력(분산 유지)을 균형합니다.

```python
from mpc_controller.controllers.mppi.stein_variational_mppi import SteinVariationalMPPIController
params = MPPIParams(K=512, N=30, svgd_num_iterations=3)
ctrl = SteinVariationalMPPIController(robot_params, params)
# K≤2048 권장 (K×K pairwise 커널 메모리)
```

### Smooth-MPPI (M3.5a)

Δu(제어 변화량) 공간에서 최적화하여 구조적으로 부드러운 제어를 생성합니다.

```
Vanilla:  optimize u[0..N-1]     → jerky 가능
Smooth:   optimize Δu[0..N-1]   → cumsum → u (구조적 smooth)
          cost += R_jerk · ‖ΔΔu‖²  (jerk 페널티)
```

```python
from mpc_controller.controllers.mppi.smooth_mppi import SmoothMPPIController
ctrl = SmoothMPPIController(robot_params, params)
```

### Spline-MPPI (M3.5b)

P개 제어점(knot)에 노이즈를 주고 B-spline basis로 N개 timestep을 보간합니다. P << N으로 노이즈 차원이 축소되어 구조적으로 smooth합니다.

```python
from mpc_controller.controllers.mppi.spline_mppi import SplineMPPIController
params = MPPIParams(K=1024, N=30, spline_num_knots=12)
ctrl = SplineMPPIController(robot_params, params)
```

### SVG-MPPI (M3.5c)

G개 guide 입자만 SVGD를 적용하고, 나머지 K-G follower는 guide 주변에서 리샘플링합니다. SVMPC 대비 `O(G^2D) << O(K^2D)`로 효율적입니다.

```python
from mpc_controller.controllers.mppi.svg_mppi import SVGMPPIController
params = MPPIParams(K=1024, N=30, svg_num_guides=16, svgd_num_iterations=3)
ctrl = SVGMPPIController(robot_params, params)
```

### CBF-MPPI

Control Barrier Function을 MPPI에 통합하여 안전성을 보장합니다. Hybrid 접근: MPPI 비용에 CBFCost(soft)를 추가하고, 최종 제어에 QP Safety Filter(hard)를 적용합니다.

```
  MPPI + CBFCost → u_mppi (soft 유도)
       ↓
  CBF Safety Filter (QP) → u_safe (hard 보장)
  min ‖u - u_mppi‖²
  s.t. ḣ(x,u) + γ·h(x) ≥ 0
```

```python
from mpc_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mpc_controller.controllers.mppi.mppi_params import CBFParams

cbf_params = CBFParams(
    enabled=True, gamma=1.0, safety_margin=0.3,
    robot_radius=0.2, cost_weight=500.0, use_safety_filter=True,
)
ctrl = CBFMPPIController(robot_params, params, obstacles=obstacles, cbf_params=cbf_params)
u, info = ctrl.compute_control(state, ref)
# info["barrier_values"] — 현재 barrier 함수값
# info["cbf_filter_info"] — QP 필터 상세 정보
```

**벤치마크 결과** (4 시나리오 종합):
| 메트릭 | Vanilla | CBF-MPPI |
|--------|---------|----------|
| 충돌 횟수 | 34 | **0** |
| Safety Violations | 974 | **0** |
| RMSE overhead | — | +51.8% |
| Solve time overhead | — | +7.3% |

## ROS2 nav2 통합

### C++ 플러그인 계층 구조

```
MPPIControllerPlugin (base, virtual computeControl)
├── LogMPPIControllerPlugin       ── WeightComputation 교체
├── TsallisMPPIControllerPlugin   ── WeightComputation 교체
├── RiskAwareMPPIControllerPlugin ── WeightComputation 교체
├── SmoothMPPIControllerPlugin    ── Δu space + jerk cost
├── SplineMPPIControllerPlugin    ── B-spline basis 보간
└── SVMPCControllerPlugin         ── SVGD loop
    └── SVGMPPIControllerPlugin   ── Guide SVGD + follower
```

### ROS2 실행

```bash
# ROS2 빌드
cd ros2_ws && source /opt/ros/jazzy/setup.bash
colcon build --packages-select mpc_controller_ros2

# 컨트롤러별 launch
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=custom    # Vanilla
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=log       # Log-MPPI
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=tsallis   # Tsallis
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=risk_aware # CVaR
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=smooth    # Smooth
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=spline    # Spline
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=svmpc     # SVMPC
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=svg       # SVG-MPPI

# 모션 모델 분기
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=swerve        # Swerve
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=non_coaxial   # Non-coaxial
```

### C++ 테스트 (211개)

```bash
cd ros2_ws && colcon test --packages-select mpc_controller_ros2 --event-handlers console_cohesion+
```

## Development Workflow

### Via GitHub Issues (Mobile-friendly)

```
┌─────────────────────────────────────────────────────────────┐
│                      동작 플로우                            │
├─────────────────────────────────────────────────────────────┤
│  1. 핸드폰에서 이슈 등록 + 'claude' 라벨                   │
│         ↓                                                   │
│  2. 랩탑이 이슈 감지 (30초 폴링)                            │
│         ↓                                                   │
│  3. 로컬 Claude Code가 구현                                 │
│         ↓                                                   │
│  4. 자동 커밋 & PR 생성                                     │
│         ↓                                                   │
│  5. 핸드폰으로 알림 (이슈 댓글)                             │
└─────────────────────────────────────────────────────────────┘
```

### Claude Issue Watcher 설치

```bash
# 필수: GitHub CLI (gh) + Claude Code 설치 확인
gh auth status
claude --version

# Issue Watcher 설치
cd .claude/scripts && ./install-watcher.sh

# 서비스 제어
systemctl --user start claude-watcher     # 시작
systemctl --user status claude-watcher    # 상태 확인
systemctl --user stop claude-watcher      # 중지
journalctl --user -u claude-watcher -f    # 로그
```

### Claude TODO Worker

GitHub 이슈 대신 `TODO.md` 파일 기반으로 Claude가 순차 처리하는 시스템입니다.

```bash
claude-todo-worker          # 다음 작업 하나 처리
claude-todo-task "#101"     # 특정 작업 처리
claude-todo-all             # 전체 작업 연속 처리
```

| 방식 | Issue Watcher | TODO Worker |
|------|---------------|-------------|
| **온라인 필요** | 필수 | 선택 |
| **설정** | systemd | 스크립트 |
| **진행 상황** | GitHub 이슈 | TODO.md |
| **우선순위** | 라벨 | P0/P1/P2 |
| **속도** | 30초 폴링 | 즉시 실행 |

## Dependencies

- Python >= 3.10
- NumPy >= 1.24
- Matplotlib >= 3.7
- CasADi >= 3.6 (MPC 컨트롤러용)
- JAX >= 0.4.20 (optional, GPU 가속용)

MPPI 컨트롤러는 순수 NumPy로 구현되어 CasADi/JAX 없이도 동작합니다.

## Testing

```bash
# 전체 Python 테스트
pytest tests/ -v --override-ini="addopts="

# MPPI CPU 테스트만
pytest tests/test_mppi*.py tests/test_log_mppi.py tests/test_tsallis_mppi.py \
  tests/test_risk_aware_mppi.py tests/test_tube_mppi.py tests/test_ancillary_controller.py \
  tests/test_stein_variational_mppi.py tests/test_smooth_mppi.py tests/test_spline_mppi.py \
  tests/test_svg_mppi.py -v --override-ini="addopts="

# GPU 테스트 (JAX 필요)
pytest tests/test_gpu_mppi.py tests/test_gpu_weights.py \
  tests/test_gpu_svgd.py tests/test_gpu_variants.py -v --override-ini="addopts="

# CBF 테스트
pytest tests/test_cbf_*.py -v --override-ini="addopts="

# 특정 테스트
pytest tests/test_tsallis_mppi.py -v -k "circle_tracking" --override-ini="addopts="

# C++ 테스트 (ROS2)
cd ros2_ws && colcon test --packages-select mpc_controller_ros2 --event-handlers console_cohesion+
```

## Milestones

| 마일스톤 | 상태 | 설명 |
|----------|------|------|
| M1 Vanilla MPPI | **완료** | 기본 MPPI 구현 (Python) |
| M2 고도화 | **완료** | Colored Noise, Adaptive Temp, Tube-MPPI |
| M3 SOTA 변형 | **완료** | Log, Tsallis, Risk-Aware, SVMPC |
| M3.5 확장 | **완료** | Smooth, Spline, SVG-MPPI |
| M4 ROS2 nav2 | **완료** | 8종 C++ 플러그인 + Swerve |
| M5 C++ 포팅 | **완료** | SOTA + M2 고도화 + M3.5 |
| GPU 가속 (Vanilla) | **완료** | JAX JIT + lax.scan + vmap (PR #103) |
| GPU 가속 (8종 확장) | **완료** | 전 변형 GPU 지원 + SVGD JIT (PR #105) |
| MPPI-CBF 통합 | **완료** | Safety Filter + Barrier Cost (Python + C++) |
| 궤적 안정화 | **완료** | SG Filter + IT 정규화 |

## License

MIT
