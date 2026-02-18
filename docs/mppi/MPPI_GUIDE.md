# MPPI (Model Predictive Path Integral) 기술 가이드

본 문서는 프로젝트에 구현된 MPPI 알고리즘 변형들의 개념, 수학적 배경, 사용법을 정리합니다.

---

## 1. MPPI 기본 개념

### 1.1 MPC vs MPPI

```
┌─────────────────────────────────────────────────────────────────┐
│                     제어 최적화 패러다임                          │
├────────────────────────────┬────────────────────────────────────┤
│   Gradient-based MPC       │   Sampling-based MPPI              │
│                            │                                    │
│   min J(u)                 │   u* = Σ w_k · ε_k                │
│   s.t. g(x,u) ≤ 0         │   w_k = softmax(-S_k / λ)         │
│                            │                                    │
│   CasADi + IPOPT           │   NumPy + 병렬 rollout             │
│   미분 가능 비용 필수       │   임의 비용 함수 가능              │
│   수렴 시간 가변            │   고정 계산 시간                   │
│   로컬 최적에 빠질 수 있음  │   샘플링으로 넓은 탐색             │
└────────────────────────────┴────────────────────────────────────┘
```

### 1.2 MPPI 알고리즘 흐름

```
 현재 상태 x₀, 이전 제어열 U
         │
         ▼
 ┌───────────────────────────────────────────┐
 │  1. Shift  ─  이전 제어열 한 스텝 이동     │
 │     U[:-1] = U[1:]                        │
 │     U[-1]  = 0                            │
 ├───────────────────────────────────────────┤
 │  2. Sample ─  K개 노이즈 ε_k ~ P(ε)      │
 │     Gaussian / Colored (OU process)       │
 ├───────────────────────────────────────────┤
 │  3. Rollout ─  K개 궤적 시뮬레이션        │
 │     τ_k = f(x₀, U + ε_k)    (배치 RK4)   │
 ├───────────────────────────────────────────┤
 │  4. Cost   ─  각 궤적의 비용 계산         │
 │     S_k = Σ[ Q·(x-ref)² + R·u² + ... ]   │
 ├───────────────────────────────────────────┤
 │  5. Weight ─  비용 → 가중치 변환          │
 │     w_k = WeightFunction(S_k, λ)          │
 │     ┌─────────────────────────────────┐   │
 │     │ Vanilla : softmax(-S/λ)         │   │
 │     │ Log     : exp(log_softmax(-S/λ))│   │
 │     │ Tsallis : q_exp(-ΔS/λ, q)      │   │
 │     └─────────────────────────────────┘   │
 ├───────────────────────────────────────────┤
 │  6. Update ─  가중 평균으로 제어열 갱신   │
 │     U += Σ w_k · ε_k                     │
 ├───────────────────────────────────────────┤
 │  7. Apply  ─  첫 번째 제어 적용           │
 │     u* = U[0]                             │
 └───────────────────────────────────────────┘
```

### 1.3 핵심 파라미터

| 파라미터 | 기호 | 역할 | 기본값 |
|---------|------|------|--------|
| 호라이즌 | N | 예측 스텝 수 | 30 |
| 샘플 수 | K | 병렬 롤아웃 수 | 1024 |
| 온도 | λ | 가중치 집중도 (작을수록 최적 집중) | 10.0 |
| 노이즈 | σ | 탐색 범위 [v, ω] | [0.3, 0.3] |
| 상태 가중 | Q | [x, y, θ] 추적 가중치 | diag(10, 10, 1) |
| 제어 가중 | R | [v, ω] 제어 비용 | diag(0.01, 0.01) |

---

## 2. 마일스톤별 구현

### 2.1 M1: Vanilla MPPI

```
기본 가중치 함수:

            exp(-S_k / λ)
  w_k = ─────────────────────
          Σ_j exp(-S_j / λ)

Shannon 엔트로피 기반 softmax.
λ 작음 → 최적 샘플에 집중 (exploitation)
λ 큼   → 균등 분포에 가까움 (exploration)
```

**구현**: `base_mppi.py` → `MPPIController`

**핵심 인터페이스**:
```python
controller = MPPIController(mppi_params=MPPIParams(N=20, K=512), seed=42)
control, info = controller.compute_control(state, reference_trajectory)
# control: [v, omega]
# info: {predicted_trajectory, sample_weights, ess, temperature, ...}
```

### 2.2 M2: 고도화

#### Colored Noise (OU Process)

```
Gaussian (iid):     ε₁ ε₂ ε₃ ε₄ ε₅    ← 시간축 독립, 진동 발생
                    ↕  ↕  ↕  ↕  ↕

Colored (OU):       ε₁→ε₂→ε₃→ε₄→ε₅    ← 시간 상관, 부드러운 제어
                    ε[t+1] = decay·ε[t] + diffusion·w[t]
                    decay = exp(-β·dt)

β 큼  → 빠르게 decorrelate → 백색에 가까움
β 작음 → 강한 상관 → 부드러운 샘플
```

#### Adaptive Temperature (ESS 기반 λ 자동 튜닝)

```
ESS(w) = 1 / Σ w_k²     ← Effective Sample Size

ESS 낮음 (소수 샘플 지배)  → λ 증가 → 탐색 강화
ESS 높음 (가중치 균등)     → λ 감소 → 최적 집중

업데이트: log(λ) += rate × (target_ratio - ESS/K)
```

#### ControlRateCost (제어 변화율 비용)

```
추가 비용: Σ (u_{t+1} - u_t)^T R_rate (u_{t+1} - u_t)

R_rate 없음 → 제어 입력 진동 가능
R_rate 있음 → 부드러운 제어 전환
```

#### Tube-MPPI (외란 강건성)

```
┌──────────────────────────────────────────────────────┐
│  Tube-MPPI 개념도                                     │
│                                                      │
│  명목 상태(x_nom)에서 MPPI 계획                       │
│  실제 상태(x_act)와의 편차를 ancillary가 보정         │
│                                                      │
│  u_applied = u_nominal + K_fb · (x_nom - x_act)     │
│                                                      │
│         x_nom ──────────────●───────── 명목 궤적     │
│        ╱                   ╱                         │
│       ╱    tube_width     ╱   ← 보장 가능 영역       │
│      ╱                   ╱                           │
│     x_act ─ ─ ─ ─ ─ ─ ●─ ─ ─ ─  실제 궤적          │
│                                                      │
│  TubeAwareCost: safety_margin + tube_margin          │
│  → 명목 궤적이 보수적으로 계획됨                      │
└──────────────────────────────────────────────────────┘
```

### 2.3 M3a: Log-MPPI (수치 안정성)

```
가중치 계산을 log-space에서 수행:

  log w_k = -S_k / λ
  log w_k -= log_sum_exp(log w)    ← log-space 정규화
  w_k     = exp(log w_k)

수학적으로 Vanilla softmax와 동일 결과.
(현재 Vanilla도 max-shift trick을 사용하므로 성능 차이 없음)
후속 확장(importance sampling 보정 등)의 기반 클래스로 활용.
```

**구현**: `log_mppi.py` → `LogMPPIController(MPPIController)`

### 2.4 M3b: Tsallis-MPPI (일반화 엔트로피)

**논문**: Yin et al. (2021) "Variational Inference MPC using Tsallis Divergence"

```
Tsallis q-exponential 가중치:

  exp_q(x) = [1 + (1-q)·x]_+^{1/(1-q)}

  q → 1.0 : 표준 exp (Vanilla MPPI = Shannon 엔트로피)
  q > 1.0 : heavy-tail (탐색 범위 확대, 다양한 해 고려)
  q < 1.0 : light-tail (최적 해 주변 집중)

  ┌─────────────────────────────────────────────────┐
  │  가중치 분포 형태 (개념)                          │
  │                                                 │
  │  w │  q<1       q=1       q>1                   │
  │    │  ╱╲        ╱╲       ╱────╲                 │
  │    │ ╱  ╲      ╱  ╲     ╱      ╲                │
  │    │╱    ╲    ╱    ╲   ╱        ╲               │
  │    └──────── ──────── ──────────── cost          │
  │    집중↑    표준      탐색↑                       │
  └─────────────────────────────────────────────────┘

주의: q-exponential은 translation-invariant가 아님.
→ 비용 min-centering 필수: ΔS_k = S_k - min(S)
→ λ와 q는 함께 튜닝 (q>1이면 λ를 줄여야 함)
→ adaptive temperature 병용 권장
```

**구현**: `tsallis_mppi.py` → `TsallisMPPIController(MPPIController)`

**실용적 q 범위**: 0.5 ~ 1.5 (q≥2는 polynomial decay가 너무 완만)

### 2.5 M3c: Risk-Aware MPPI (CVaR)

**논문**: Yin et al. (2023) "Risk-Aware MPPI"

```
Conditional Value at Risk (CVaR):

  1. K개 비용 정렬 → 하위 ceil(α·K)개만 선택
  2. 선택된 샘플만 softmax → 가중치

  α = 1.0 → risk-neutral (Vanilla 동등)
  α = 0.3 → 상위 30%만 사용 (보수적 경로)
  α = 0.1 → 최저 비용 10%만 (매우 보수적)

  ┌───────────────────────────────────────────┐
  │  비용 분포에서 CVaR 절단                    │
  │                                           │
  │  ╱╲                                       │
  │ ╱  ╲    ←── 이 영역(α 비율)만 사용         │
  │╱    ╲──────────────── 나머지 무시          │
  │      ┊                                    │
  │      α·K                                  │
  └───────────────────────────────────────────┘
```

### 2.6 M3d: Stein Variational MPPI (SVMPC)

**논문**: Lambert et al. (2020) "Stein Variational Model Predictive Control"

```
SVGD로 샘플 간 상호작용 유도:

  1. Vanilla 동일: shift → sample → rollout → cost
  2. SVGD Loop (L회):
     ├── flatten: (K, N, nu) → (K, D)
     ├── RBF kernel: k(x_i, x_j) = exp(-||x_i-x_j||²/2h²)
     ├── attractive force: w_j · k(x_j,x_i) · (x_j-x_i)
     ├── repulsive force: k(x_j,x_i) · (x_j-x_i) / h²
     └── particles += step_size × (attractive + repulsive)
  3. 가중 평균 업데이트

  svgd_iterations=0 → Vanilla 동등 (backward compatible)
```

### 2.7 M3.5a: Smooth-MPPI (구조적 부드러움)

**논문**: Kim et al. (2021) "Smooth MPPI"

```
Δu space 최적화로 구조적 부드러움:

  Vanilla:  u[0] u[1] u[2] ... u[N-1]     ← 각 스텝 독립
  Smooth:   Δu[0] Δu[1] Δu[2] ... Δu[N-1]  ← 변화량 공간

  u[t] = u_prev + Σ_{i=0}^{t} Δu[i]     (cumsum 복원)

  추가 비용: jerk cost = R_jerk · ‖ΔΔu‖²  (2차 변화율)

  ┌──────────────────────────────────────────┐
  │  Vanilla vs Smooth 제어 비교              │
  │                                          │
  │  Vanilla: ╱╲╱╲╱╲  ← 진동 가능            │
  │  Smooth:  ╱─────╲  ← 구조적으로 부드러움   │
  │           (cumsum이 저역 통과 필터 역할)   │
  └──────────────────────────────────────────┘
```

**핵심 파라미터**:
| 파라미터 | 역할 | 기본값 |
|---------|------|--------|
| smooth_R_jerk_v | v방향 jerk 가중치 | 0.1 |
| smooth_R_jerk_omega | omega방향 jerk 가중치 | 0.1 |
| smooth_action_cost_weight | jerk cost 전체 스케일 | 1.0 |

### 2.8 M3.5b: Spline-MPPI (B-spline 보간)

**논문**: Bhardwaj et al. (2024) "Spline-MPPI" (ICRA 2024)

```
P개 knot에만 노이즈 → B-spline basis로 N개 보간:

  noise: (K, P, nu)  ← P ≈ 8 << N = 30
  basis: (N, P)      ← de Boor 재귀, clamped uniform
  controls = basis @ knots → (K, N, nu)  ← 구조적 smooth

  ┌──────────────────────────────────────────┐
  │  B-spline 보간 개념                       │
  │                                          │
  │  knots:    *     *     *     *     *     │
  │            P=5 개 제어점                  │
  │                                          │
  │  보간:  ─*─────*─────*─────*─────*──     │
  │         N=30 개 부드러운 제어 시점         │
  │                                          │
  │  장점: 노이즈 차원 P/N 배 축소            │
  │        → 구조적 smooth + 탐색 효율        │
  └──────────────────────────────────────────┘
```

**핵심 파라미터**:
| 파라미터 | 역할 | 기본값 |
|---------|------|--------|
| spline_num_knots | B-spline 제어점 수 (P) | 8 |
| spline_degree | B-spline 차수 | 3 (cubic) |

### 2.9 M3.5c: SVG-MPPI (Guide Particle)

**논문**: Kondo et al. (2024) "SVG-MPPI" (ICRA 2024)

```
G개 guide particle만 SVGD → 나머지 K-G개는 follower:

  Phase 1: K개 전체 rollout → cost
  Phase 2: 비용 최저 G개 → guide 선택
  Phase 3: SVGD(G×G) L회 반복 (G << K)
  Phase 4: 각 guide 주변 (K-G)/G개 follower 리샘플링
  Phase 5: 전체 K개 rollout → weight → U 업데이트

  ┌──────────────────────────────────────────┐
  │  SVMPC vs SVG-MPPI 비교                   │
  │                                          │
  │  SVMPC:    K×K SVGD  → O(K²D)  (느림)    │
  │  SVG-MPPI: G×G SVGD  → O(G²D)  (빠름)    │
  │            + follower resample            │
  │                                          │
  │  G=10, K=512 → 계산량 2600배 감소         │
  │  다중 모드 탐색 능력은 유지               │
  └──────────────────────────────────────────┘
```

**핵심 파라미터**:
| 파라미터 | 역할 | 기본값 |
|---------|------|--------|
| svg_num_guide_particles | guide 수 (G) | 10 |
| svg_guide_iterations | SVGD 반복 횟수 | 3 |
| svg_guide_step_size | SVGD step size | 0.1 |
| svg_resample_std | follower 노이즈 표준편차 | 0.3 |

---

## 3. 클래스 계층 구조

```
Python 클래스 계층:

MPPIController (base_mppi.py) ── M1 Vanilla
│  오버라이드: _compute_weights(costs) → weights
│
├── TubeMPPIController ── M2
├── LogMPPIController ── M3a (log-space softmax)
├── TsallisMPPIController ── M3b (q-exponential)
├── RiskAwareMPPIController ── M3c (CVaR 절단)
├── SteinVariationalMPPIController ── M3d (SVGD)
│   └── SVGMPPIController ── M3.5c (Guide SVGD)
├── SmoothMPPIController ── M3.5a (Δu space)
└── SplineMPPIController ── M3.5b (B-spline basis)

C++ nav2 플러그인 계층:

MPPIControllerPlugin (base, virtual computeControl)
├── LogMPPIControllerPlugin (WeightComputation 교체)
├── TsallisMPPIControllerPlugin (WeightComputation 교체)
├── RiskAwareMPPIControllerPlugin (WeightComputation 교체)
├── SmoothMPPIControllerPlugin (computeControl: Δu space)
├── SplineMPPIControllerPlugin (computeControl: B-spline)
└── SVMPCControllerPlugin (computeControl: SVGD loop)
    └── SVGMPPIControllerPlugin (computeControl: Guide SVGD)
```

---

## 4. 비용 함수 구성

```
CompositeMPPICost (합산)
├── StateTrackingCost   ─  Σ (x_t - ref_t)^T Q (x_t - ref_t)
├── TerminalCost        ─  (x_N - ref_N)^T Qf (x_N - ref_N)
├── ControlEffortCost   ─  Σ u_t^T R u_t
├── ControlRateCost     ─  Σ Δu_t^T R_rate Δu_t         [M2]
├── ObstacleCost        ─  Σ max(0, d_safe - dist)²
└── TubeAwareCost       ─  ObstacleCost + tube_margin    [M2]
```

---

## 5. 파라미터 튜닝 가이드

### 5.1 기본 튜닝 순서

```
1. K (샘플 수)     ─  128(빠름) → 512(균형) → 2048(정밀)
2. N (호라이즌)     ─  너무 짧으면 근시안, 길면 계산 비용↑
3. λ (온도)        ─  작으면 최적 집중, 크면 탐색 넓음
4. σ (노이즈)      ─  제어 범위의 20~50% 정도
5. Q/R (가중치)     ─  추적 정밀도 vs 에너지 절약 균형
```

### 5.2 Tsallis-MPPI q 파라미터

```
┌──────────┬────────────────────────────────────────────┐
│  q 값    │  사용 시나리오                              │
├──────────┼────────────────────────────────────────────┤
│  0.5     │  정밀 추적, 최적 해 주변 집중               │
│  1.0     │  표준 (Vanilla MPPI와 동일)                │
│  1.2     │  약간의 탐색 확대, 로컬 최적 회피           │
│  1.5     │  넓은 탐색, 좁은 복도/장애물 밀집 환경      │
├──────────┼────────────────────────────────────────────┤
│  주의    │  q>1이면 λ를 줄여야 함                     │
│          │  adaptive_temperature 병용 권장             │
│          │  q≥2는 실용적이지 않음 (polynomial decay)   │
└──────────┴────────────────────────────────────────────┘
```

### 5.3 Adaptive Temperature 설정

| 파라미터 | 설명 | 권장값 |
|---------|------|--------|
| target_ess_ratio | 목표 ESS/K 비율 | 0.3~0.7 |
| adaptation_rate | 조정 속도 | 0.1(안정) ~ 1.0(공격적) |
| lambda_min | λ 하한 | 0.001~1.0 |
| lambda_max | λ 상한 | 50~100 |

---

## 6. 데모 실행법

### 6.1 기본 MPPI 데모

```bash
# Vanilla MPPI 기본 데모
python examples/mppi_basic_demo.py
python examples/mppi_basic_demo.py --live    # 실시간 시각화
```

### 6.2 비교 데모

```bash
# Vanilla vs M2 (ControlRateCost + AdaptiveTemp + ColoredNoise)
python examples/mppi_vanilla_vs_m2_demo.py --trajectory circle --live

# Vanilla vs Tube-MPPI (외란 강건성)
python examples/mppi_vanilla_vs_tube_demo.py --live --noise

# Vanilla vs Log-MPPI (수치 안정성 실험)
python examples/log_mppi_demo.py --trajectory figure8 --live

# Tsallis-MPPI q 파라미터 비교
python examples/tsallis_mppi_demo.py --trajectory circle --live
python examples/tsallis_mppi_demo.py --q 0.5 1.0 1.5  # 커스텀 q값
```

### 6.3 C++ nav2 플러그인 (ROS2)

```bash
# 빌드
cd ros2_ws && source /opt/ros/jazzy/setup.bash
colcon build --packages-select mpc_controller_ros2

# Vanilla MPPI
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=custom

# Log-MPPI
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=log

# Tsallis-MPPI
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=tsallis

# Risk-Aware MPPI (CVaR)
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=risk_aware

# SVMPC (Stein Variational MPC)
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=svmpc

# Smooth-MPPI (Δu space 최적화)
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=smooth

# Spline-MPPI (B-spline 보간)
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=spline

# SVG-MPPI (Guide particle SVGD)
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=svg

# nav2 기본 MPPI
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=nav2
```

### 6.4 MPC 기본 데모

```bash
python examples/path_tracking_demo.py          # MPC 경로 추종
python examples/obstacle_avoidance_demo.py     # 장애물 회피
python examples/pid_vs_mpc_demo.py             # PID vs MPC 비교
python examples/mpc_benchmark.py               # 성능 벤치마크
```

---

## 7. 참고 논문

| 마일스톤 | 논문 | 핵심 기여 |
|---------|------|---------|
| M1 | Williams et al. (2016) "Aggressive Driving with MPPI" | 원본 MPPI 알고리즘 |
| M2 | Williams et al. (2018) "Robust Sampling Based MPPI" | Tube-MPPI, 외란 강건성 |
| M3b | Yin et al. (2021) "Trajectory Distribution Control via Tsallis Entropy" | Tsallis q-exponential |
| M3c | Yin et al. (2023) "Risk-Aware MPPI" | CVaR 기반 위험 인지 |
| M3c | Yin et al. (2023) "Risk-Aware MPPI" | CVaR 기반 위험 인지 |
| M3d | Lambert et al. (2020) "Stein Variational MPC" | 커널 기반 샘플 다양성 |
| M3.5a | Kim et al. (2021) "Smooth MPPI" | Δu input-lifting, jerk cost |
| M3.5b | Bhardwaj et al. (2024) "Spline-MPPI" (ICRA 2024) | B-spline basis 보간 |
| M3.5c | Kondo et al. (2024) "SVG-MPPI" (ICRA 2024) | Guide particle SVGD |
