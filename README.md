# Toy Claude Project

Mobile Robot MPC/MPPI Controller with Claude-Driven Development Workflow

## Overview

This project demonstrates:
1. **MPC-based mobile robot control** - CasADi/IPOPT 기반 경로 추종 MPC
2. **MPPI sampling-based control** - 22종 C++ MPPI 플러그인 + 9종 Python MPPI + GPU 가속 (JAX)
3. **ROS2 nav2 integration** - 22종 C++ 플러그인 + 4종 모션 모델 + 5단계 Safety Stack
4. **Paper-Ready Benchmarking** - 다중 시행 통계 분석 + LaTeX 테이블 + 파레토 분석
5. **Claude-driven development** - GitHub 이슈 자동 처리 워크플로우

## C++ MPPI 플러그인 (22종)

### 플러그인 계층 구조

```
MPPIControllerPlugin (base, virtual computeControl)
│
│  가중치(weights) 교체 변형:
├── LogMPPIControllerPlugin         ── log-space softmax (수치 안정성)
├── TsallisMPPIControllerPlugin     ── q-exponential (Tsallis 엔트로피)
├── RiskAwareMPPIControllerPlugin   ── CVaR 가중치 절단 (risk-averse)
│
│  제어 공간(control space) 변환 변형:
├── SmoothMPPIControllerPlugin      ── Δu space + jerk cost
├── SplineMPPIControllerPlugin      ── B-spline basis 보간 (P << N)
├── LPMPPIControllerPlugin          ── 1차 IIR Low-Pass 필터 (주파수 도메인)
│
│  샘플링(sampling) 개선 변형:
├── BiasedMPPIControllerPlugin      ── Ancillary biased sampling (RA-L 2024)
├── DialMPPIControllerPlugin        ── Diffusion Annealing (ICRA 2025)
│
│  고급 최적화 변형:
├── IlqrMPPIControllerPlugin        ── iLQR warm-start + MPPI 파이프라인
├── CSMPPIControllerPlugin          ── Covariance Steering (CoVO-MPC, CoRL 2023)
├── PiMPPIControllerPlugin          ── ADMM QP 투영 필터 (π-MPPI, RA-L 2025)
│
│  분포(distribution) 개선 변형:
├── SVMPCControllerPlugin           ── SVGD 커널 (Stein Variational MPC)
│   └── SVGMPPIControllerPlugin     ── Guide SVGD + follower (G << K)
│
│  강건성(robustness) 변형:
├── TubeMPPIControllerPlugin        ── Nominal state MPPI + body frame 피드백
│
│  안전성(safety) 스택 (5단계):
├── ShieldMPPIControllerPlugin      ── per-step CBF 투영
│   ├── AdaptiveShieldMPPIControllerPlugin ── 거리/속도 적응 α
│   ├── CLFCBFMPPIControllerPlugin         ── CLF-CBF-QP 통합 안전 필터 (Ames 2019)
│   └── PredictiveSafetyMPPIControllerPlugin ── N-step CBF 투영
│
│  하이브리드 변형:
└── MPPIHControllerPlugin           ── Hybrid Swerve Low-D↔4D 전환 (IROS 2024)
```

### 플러그인 상세

| # | 플러그인 | 핵심 논문/아이디어 | 모션 모델 |
|---|---------|-------------------|----------|
| 1 | Vanilla MPPI | Williams et al. (2017) | All |
| 2 | Log-MPPI | log-space softmax 수치 안정성 | All |
| 3 | Tsallis-MPPI | q-exponential 일반화 엔트로피 | All |
| 4 | Risk-Aware (CVaR) | alpha 기반 worst-case | All |
| 5 | Smooth-MPPI | Kim et al. (2021), Δu space | All |
| 6 | Spline-MPPI | B-spline 보간 (ICRA 2024) | All |
| 7 | LP-MPPI | 1차 IIR Low-Pass 필터 (2025) | All |
| 8 | Biased-MPPI | Ancillary biased sampling (RA-L 2024) | All |
| 9 | DIAL-MPPI | Diffusion Annealing (ICRA 2025) | All |
| 10 | iLQR-MPPI | iLQR warm-start + MPPI | DiffDrive, Ackermann |
| 11 | CS-MPPI | Covariance Steering (CoVO-MPC, CoRL 2023) | All |
| 12 | π-MPPI | ADMM QP 투영 (RA-L 2025) | All |
| 13 | SVMPC | Stein Variational MPC | All |
| 14 | SVG-MPPI | Guide SVGD + follower (ICRA 2024) | All |
| 15 | Tube-MPPI | Nominal/actual 분리 + 피드백 | DiffDrive |
| 16 | Shield-MPPI | per-step CBF 투영 | All |
| 17 | Adaptive Shield | 거리/속도 적응 α | All |
| 18 | CLF-CBF-QP | Ames (2019) 통합 안전 필터 | All |
| 19 | Predictive Safety | N-step CBF 투영 | All |
| 20 | MPPI-H | Hybrid Swerve (IROS 2024) | Swerve |
| 21 | Ensemble + C3BF | Collision Cone CBF + 불확실성 | All |
| 22 | BR-MPPI | Barrier Rate Cost + ACP | All |

### 4종 모션 모델

```
MotionModelFactory::create(string, params)
├── "diff_drive"           ── nx=3 [x,y,θ],     nu=2 [v,ω]
├── "swerve"               ── nx=3 [x,y,θ],     nu=3 [vx,vy,ω]
├── "non_coaxial_swerve"   ── nx=4 [x,y,θ,φ],   nu=3 [vx,vy,ω]
└── "ackermann"            ── nx=4 [x,y,θ,δ],   nu=2 [v,δ̇]  (Bicycle model)
```

### Safety Stack (5단계)

```
┌─────────────────────────────────────────────────────────┐
│ Level 5: Predictive Safety (N-step CBF 투영)            │
├─────────────────────────────────────────────────────────┤
│ Level 4: CLF-CBF-QP (Lyapunov + Barrier 통합)          │
├─────────────────────────────────────────────────────────┤
│ Level 3: Adaptive Shield (d,v 적응 α)                  │
├─────────────────────────────────────────────────────────┤
│ Level 2: Shield-MPPI (per-step CBF 투영)               │
├─────────────────────────────────────────────────────────┤
│ Level 1: CBF Cost (soft penalty in MPPI cost)          │
├─────────────────────────────────────────────────────────┤
│ Level 0: Costmap lethal cell → obstacle extraction     │
└─────────────────────────────────────────────────────────┘
+ Ensemble Dynamics (불확실성) + C3BF (Collision Cone)
+ Dynamic Obstacle Tracker (clustering + EMA velocity)
```

## Quick Start

### C++ (ROS2 nav2)

```bash
# 빌드
cd ros2_ws && source /opt/ros/jazzy/setup.bash
colcon build --packages-select mpc_controller_ros2

# Gazebo + nav2 실행 (21종 컨트롤러 전환)
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=custom
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=smooth
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=dial
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=lp
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=shield
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=tube_mppi
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=pi_mppi

# 모션 모델 분기
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=swerve
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=non_coaxial
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py controller:=ackermann

# 환경 분기 (6종 World)
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py \
    world:=maze_world.world map:=maze_map.yaml
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py \
    world:=narrow_passage_world.world map:=narrow_passage_map.yaml
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py \
    world:=random_forest_world.world map:=random_forest_map.yaml

# Headless 모드 (CI/벤치마크용)
ros2 launch mpc_controller_ros2 mppi_ros2_control_nav2.launch.py headless:=true
```

### Python

```bash
pip install -e .

# MPPI 데모
python examples/mppi_basic_demo.py --trajectory circle --live
python examples/mppi_all_variants_benchmark.py --trajectory circle --live

# GPU 벤치마크
python examples/gpu_benchmark.py --K 512,1024,2048,4096
```

## Benchmarking

### Paper-Ready 벤치마크 스위트

```bash
# 논문용 벤치마크 (10종 컨트롤러, maze, 3회 반복)
python3 scripts/paper_benchmark.py --scenario maze_nav --group paper_core

# 분석 + 시각화 + LaTeX 테이블
python3 scripts/paper_benchmark_analysis.py \
    --input ~/paper_benchmark_results/bench_maze_nav_*/aggregated_stats.json
```

**출력물:**
- ASCII 요약 테이블 (mean +/- 95% CI)
- Kruskal-Wallis 비모수 통계 검정
- 파레토 최적 분석 (Time vs Safety)
- LaTeX 테이블 (최적값 자동 `\textbf` 볼드)
- matplotlib 4종 차트 (에러바, 박스플롯, 레이더, 파레토)

### 컨트롤러 벤치마크 (단일 시행)

```bash
# 22종 자동 비교
python3 scripts/controller_benchmark.py --group all

# C++ 파이프라인 마이크로벤치마크
./build/mpc_controller_ros2/bench_mppi_pipeline --K 512 --N 30

# 스트레스 테스트 (동적 장애물)
ros2 run mpc_controller_ros2 stress_test.py --speed high --frequency 20
```

### 성능 (K=512, N=30, DiffDrive)

```
Pipeline: 1.88ms mean (532 Hz)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Sampling      :   42μs ( 2.2%)
 Rollout       :  882μs (46.9%)
 Cost          :  510μs (27.1%)
 Weight+Update :  446μs (23.7%)
```

## Testing

### C++ (542+ gtest, 31 스위트)

```bash
cd ros2_ws && colcon test --packages-select mpc_controller_ros2 \
    --event-handlers console_cohesion+
```

| 테스트 스위트 | 수 | 영역 |
|---|---|---|
| test_mppi_algorithm | 7 | 핵심 MPPI 루프 |
| test_batch_dynamics | 8 | 배치 RK4 |
| test_cost_functions | 38 | 비용 함수 |
| test_sampling | 8 | Gaussian + Colored Noise |
| test_motion_model | 68 | 4종 모션 모델 |
| test_weight_computation | 30 | 가중치 전략 |
| test_adaptive_temperature | 9 | ESS λ 자동 튜닝 |
| test_tube_mppi | 13 | Tube-MPPI (레거시) |
| test_tube_mppi_plugin | 15 | Tube-MPPI Plugin |
| test_dynamic_obstacle_tracker | 15 | 동적 장애물 추적 |
| test_svmpc | 13 | SVMPC/SVG-MPPI |
| test_m35_plugins | 18 | Smooth/Spline/SVG |
| test_biased_mppi | 15 | Biased-MPPI |
| test_dial_mppi | 17 | DIAL-MPPI |
| test_lp_mppi | 15 | LP-MPPI |
| test_ilqr_solver + ilqr_mppi | 20 | iLQR-MPPI |
| test_cs_mppi | 16 | CS-MPPI |
| test_pi_mppi | 16 | π-MPPI |
| test_hybrid_swerve_mppi | 18 | MPPI-H |
| test_cbf | 22 | CBF 기본 |
| test_advanced_cbf | 16 | C3BF + Ensemble |
| test_clf_cbf_qp | 21 | CLF-CBF-QP |
| test_cbf_composition | 15 | 다중 CBF 합성 |
| test_predictive_safety | 16 | N-step CBF |
| test_residual_dynamics | 15 | EigenMLP Residual |
| test_safety_enhancement | 24 | Shield/BR-MPPI/ACP |
| test_ensemble_dynamics | 14 | Ensemble 불확실성 |
| test_online_learning | 12 | Online Learning |
| test_trajectory_stability | 25 | SG Filter + IT |
| test_tube_tracker_integration | 3 | Tube+Tracker 통합 |

### Python

```bash
# CPU 테스트
pytest tests/ -v --override-ini="addopts="

# GPU 테스트 (JAX 필요)
pytest tests/test_gpu_*.py -v --override-ini="addopts="

# pybind11 바인딩 테스트
python3 -m pytest test/python/ -v --override-ini="addopts="
```

## Python MPPI (9종 + GPU)

```
MPPIController (base_mppi.py) — Vanilla MPPI + GPU/CPU 분기
├── LogMPPIController          ── log-space softmax
├── TsallisMPPIController      ── q-exponential
├── RiskAwareMPPIController    ── CVaR 가중치 절단
├── SmoothMPPIController       ── Δu space + jerk cost
├── SplineMPPIController       ── B-spline knot space
├── SteinVariationalMPPIController ── SVGD 커널
│   └── SVGMPPIController         ── Guide SVGD + follower
├── TubeMPPIController         ── 명목/실제 분리 + 피드백
└── CBFMPPIController          ── Control Barrier Function 통합
```

GPU 가속: `use_gpu=True`로 JAX JIT 활성화 (K=4096, ~2-4ms)

## Project Structure

```
ros2_ws/src/mpc_controller_ros2/     # ROS2 C++ 패키지
├── include/mpc_controller_ros2/     # C++ 헤더
│   ├── mppi_controller_plugin.hpp   #   Base MPPI Plugin
│   ├── mppi_params.hpp              #   파라미터 데이터클래스
│   ├── motion_model.hpp             #   4종 모션 모델
│   ├── tube_mppi_controller_plugin.hpp
│   ├── dynamic_obstacle_tracker.hpp
│   └── ...                          #   21종 플러그인 헤더
├── src/                             # C++ 구현
├── config/                          # 39개 nav2 파라미터 YAML
├── worlds/                          # 7개 Gazebo World
├── maps/                            # 맵 파일
├── launch/                          # Launch 파일
├── scripts/                         # 벤치마크 + E2E 테스트
│   ├── controller_benchmark.py      #   21종 자동 비교
│   ├── benchmark_report.py          #   리포트 생성
│   ├── paper_benchmark.py           #   논문용 다중 시행
│   ├── paper_benchmark_analysis.py  #   통계 분석 + LaTeX
│   ├── stress_test.py               #   동적 장애물 스트레스
│   └── nav2_e2e_test.py             #   E2E 네비게이션 테스트
├── test/                            # 527+ gtest + Python 테스트
└── plugins/                         # Plugin XML 등록

mpc_controller/                      # Python 패키지
├── models/                          # 로봇 동역학 (DiffDrive, Swerve)
├── controllers/
│   ├── mpc/                         # CasADi/IPOPT MPC
│   └── mppi/                        # 9종 MPPI + GPU 가속
├── ros2/                            # ROS2 노드 + RVIZ
├── simulation/                      # 시뮬레이터
└── utils/                           # 유틸리티
```

## Milestones

| 마일스톤 | 상태 | 설명 | PR |
|----------|------|------|----|
| M1 Vanilla MPPI | 완료 | 기본 MPPI 구현 (Python) | — |
| M2 고도화 | 완료 | Colored Noise, Adaptive Temp, Tube-MPPI | — |
| M3 SOTA 변형 | 완료 | Log, Tsallis, Risk-Aware, SVMPC | — |
| M3.5 확장 | 완료 | Smooth, Spline, SVG-MPPI | — |
| M4 ROS2 nav2 | 완료 | C++ 플러그인 + nav2 통합 | — |
| M5 C++ 포팅 | 완료 | M2+M3+M3.5 전체 C++ | — |
| GPU 가속 | 완료 | JAX JIT + 9종 전체 GPU | #103, #105 |
| MPPI-CBF 통합 | 완료 | Safety Filter + Barrier Cost | #98 |
| pybind11 바인딩 | 완료 | C++ MPPI → Python | #115 |
| Biased-MPPI | 완료 | RA-L 2024 | #123 |
| DIAL-MPPI | 완료 | ICRA 2025 + 최적화 | #125, #129 |
| 성능 최적화 | 완료 | K=512→1.88ms/532Hz | #132 |
| Ackermann | 완료 | Bicycle model nx=4 nu=2 | #138 |
| Residual + Safety | 완료 | EigenMLP + Shield + BR-MPPI + ACP | #140 |
| iLQR-MPPI | 완료 | iLQR warm-start + MPPI | #142 |
| CS-MPPI | 완료 | Covariance Steering (CoRL 2023) | #150 |
| π-MPPI | 완료 | ADMM QP 투영 (RA-L 2025) | #152 |
| MPPI-H | 완료 | Hybrid Swerve (IROS 2024) | #153 |
| Learning + C3BF | 완료 | Ensemble + Collision Cone CBF | #159 |
| CLF-CBF-QP | 완료 | Ames 2019 통합 필터 | #161 |
| 다중 CBF + Predictive | 완료 | CBF 합성 + N-step 투영 | #165 |
| 벤치마크 대시보드 | 완료 | 21종 자동 비교 + 리포트 | #171 |
| Tube-MPPI Plugin | 완료 | Nominal state MPPI + DynObs Tracker | #173 |
| LP-MPPI | 완료 | Low-Pass filtering (IIR, 2025) | #181 |
| 시뮬레이션 인프라 | 완료 | World physics 통일 + E2E 테스트 | #175 |
| Paper 벤치마크 | 완료 | 다중 시행 + 통계 + LaTeX | #177 |

## Dependencies

- **C++ (ROS2)**: Eigen3, nav2_core, pluginlib, tf2_ros, ros:jazzy
- **Python**: NumPy >= 1.24, Matplotlib >= 3.7
- **Optional**: CasADi >= 3.6 (MPC), JAX >= 0.4.20 (GPU), scipy (통계 검정)

## Development Workflow

```
┌──────────────────────────────────────────────────────┐
│  GitHub Issue → feature branch → 구현 → PR → merge  │
│                                                      │
│  해결 이슈: #63~#178 (28개)                          │
│  CI: .github/workflows/ros2-ci.yml                   │
│      (ros:jazzy Docker, colcon build+test)            │
└──────────────────────────────────────────────────────┘
```

## License

MIT
