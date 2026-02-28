# MPC Robot Navigation - TODO

프로젝트의 개발 작업 목록입니다. Claude가 순차적으로 처리합니다.

---

## 🔴 High Priority (P0)

- [ ] 실제 로봇 인터페이스 — 하드웨어 연동 테스트

## 🟠 Medium Priority (P1)

- [x] MPPI GPU 가속 — JAX JIT + lax.scan + vmap (PR #103, Issue #63)
- [x] MPPI 8종 변형 GPU 가속 — 가중치 Strategy + SVGD JIT + Smooth/Spline GPU step (PR #105)
- [x] MPPI-CBF 통합 — Control Barrier Function 안전성 보장 (PR #98, Issue #97)
- [x] MPPI 궤적 안정화 — SG Filter + IT 정규화 + Exploitation/Exploration (PR #98)
- [x] MPPI vs MPPI-CBF 비교 데모 — 안전성 및 성능 벤치마크 (PR #107, Issue #106)
- [ ] MPC vs MPPI 비교 데모 파라미터 공정화 — 호라이즌 통일 (MPC 2.0s vs MPPI 1.0s)
- [ ] `--live` 리플레이에 MPPI 샘플 궤적 시각화 추가
- [ ] Ackermann 조향 모델 추가 — 자동차형 로봇 지원
- [ ] 속도 제약 고려 MPC — 가속도/저크 제한
- [x] CI/CD 파이프라인 — GitHub Actions ROS2 빌드 + 테스트 자동화 (PR #102, Issue #101)
- [x] Swerve E2E 시뮬레이션 검증 스크립트 — jerk/smoothness 정량 분석 (PR #100, Issue #99)
- [ ] pybind11 Python 바인딩 — C++ ↔ Python 연동

## 🟢 Low Priority (P2)

- [ ] CLF-CBF-QP 컨트롤러 — Lyapunov + Barrier 통합 제어
- [ ] 다중 CBF 합성 — 복잡한 제약 조건 처리 (교집합/합집합)
- [ ] CBF GPU 가속 — JAX 기반 CBF 제약 병렬 계산
- [ ] Omnidirectional 로봇 모델 — 전방향 이동 로봇 (Mecanum/Omni wheel)
- [ ] 성능 프로파일링 및 최적화 — 실시간 성능 개선
- [ ] 웹 기반 시각화 대시보드 — 실시간 모니터링
- [ ] Docker 컨테이너화 — 배포 및 재현성 개선
- [ ] Multi-robot MPC — 다중 로봇 협조 제어
- [ ] NMPC (Nonlinear MPC) 구현 — 비선형 최적화
- [ ] Covariance Steering MPPI — 공분산 제어 기반 정밀 분포 조정
- [ ] Biased-MPPI (RA-L 2024) — 편향 샘플링 기반 효율 개선
- [ ] π-MPPI / BR-MPPI / SOPPI (2025) — 최신 MPPI 변형

## 📚 Documentation

- [ ] API 문서 자동 생성 — Sphinx/MkDocs
- [ ] 튜토리얼 작성 — 사용법 상세 가이드
- [ ] 아키텍처 문서 작성 — 시스템 설계 문서
- [x] MPPI 기술 가이드 업데이트 — M3.5 변형 설명 추가

## 🐛 Bug Fixes

- [ ] 각도 정규화 엣지 케이스 수정
- [ ] 고속 주행 시 경로 추적 오버슈트 개선
---

## ✅ Completed

### 2026-02-22 (MPPI vs CBF-MPPI 벤치마크)
- [x] MPPI vs CBF-MPPI 비교 데모 + 벤치마크 (PR #107, Issue #106)
  * 4 시나리오: head_on, narrow, multi, dense
  * 안전성 메트릭: 충돌 횟수, safety violation, 최소 표면 거리, barrier 위반 비율
  * 성능 메트릭: RMSE, control rate, jerk, solve time, path length
  * --live 실시간 리플레이 (Vanilla vs CBF 동시 시각화)
  * --benchmark 전 시나리오 일괄 비교 + 종합 ASCII 리포트
  * 3x3 비교 그래프 (궤적, 오차, barrier, 거리, 제어, jerk, solve time)
  * CBF 효과: 충돌 34→0, violation 974→0, solve +7.3%

### 2026-02-22 (GPU 8종 변형 확장)
- [x] MPPI 8종 변형 GPU 가속 확장 (PR #105, Issue #104)
  * gpu_weights.py: 4종 JIT 가중치 전략 (vanilla/log/tsallis/cvar) + registry
  * gpu_svgd.py: SVGD JIT 커널 (svgd_step, median_bandwidth, diversity, rbf_kernel)
  * gpu_mppi_kernel.py: weight_fn 주입 + smooth_mppi_step + spline_mppi_step
  * gpu_costs.py: jerk_cost_jit 추가 (Smooth-MPPI용)
  * base_mppi.py: _get_gpu_weight_fn() 가상 메서드
  * Log/Tsallis/CVaR: _get_gpu_weight_fn() 오버라이드 (가중치만 교체)
  * Tube-MPPI: 부모 GPU 경로 자동 상속 (코드 변경 없음)
  * Smooth/Spline: 전용 _compute_control_gpu() + 지연 GPU 초기화
  * SVMPC/SVG-MPPI: gpu_svgd.py SVGD JIT 커널 활용
  * 신규 테스트 59개 (weights 20 + svgd 13 + variants 26)
  * 기존 322개 MPPI 테스트 회귀 없음

### 2026-02-22 (GPU 가속)
- [x] #63 MPPI GPU 가속 — JAX JIT + lax.scan + vmap (PR #103)
  * JAX 기반 GPU 가속: rollout (~65%) + cost (~25%) 핵심 병목 해결
  * lax.scan: N=30 순차 rollout → XLA fused kernel (1회 커널 실행)
  * vmap: K 샘플 차원 자동 벡터화 (수동 broadcasting 불필요)
  * 비용 함수 fusion: 8종 비용 함수를 단일 JIT kernel로 통합
  * 장애물 벡터화: (K,N+1,1,2)-(1,1,M,2) → Python for-loop 제거
  * GPU↔CPU 전송 최소화: 호출당 2회 (입력 전송 + 결과 반환)
  * 신규 파일: gpu_backend.py, gpu_dynamics.py, gpu_costs.py, gpu_sampling.py, gpu_mppi_kernel.py
  * mppi_params.py: use_gpu, gpu_warmup, gpu_float32 파라미터 추가
  * base_mppi.py: GPU/CPU 분기 (_init_gpu, _compute_control_gpu)
  * use_gpu=False 기본값 → 기존 CPU 코드 100% 보존, 회귀 없음
  * diff_drive / swerve / non_coaxial_swerve 3종 모델 지원
  * 테스트 22개 통과 (동역학 5, 비용 6, 샘플링 2, 커널 5, 백엔드 4)
  * 벤치마크: examples/gpu_benchmark.py (K별 CPU/GPU 비교 테이블)

### 2026-02-22
- [x] Swerve MPPI 오실레이션 진단 + 옵티마이저 수렴 수정 (Phase C)
  * cmd_vel 녹화/분석 파이프라인으로 오실레이션 정량 진단
  * vx≡0 근본원인: Spline-MPPI 비대칭 clipping bias + auto lookahead 도달불가 + ESS 균등화
  * v_min: 0.0→-0.5 (비대칭 clipping 제거), lookahead_dist: 0→1.5 (고정)
  * target_ess_ratio: 0.5→0.2 (가중치 집중도 증가)
  * noise_sigma_vy: 0.5→0.2, R_vy: 0.3→1.0, control_smoothing_alpha: 0.8→0.5
  * prefer_forward_velocity_incentive: 0→2.0 (전진 인센티브)
  * Spline-MPPI RCLCPP_DEBUG 진단 로그 추가

### 2026-02-21
- [x] #95 MotionModel 추상화 — DiffDrive/Swerve/NonCoaxialSwerve 다모델 지원 (PR #96)
  * `MotionModel` 인터페이스 → DiffDrive(nx=3,nu=2), Swerve(nx=3,nu=3), NonCoaxialSwerve(nx=4,nu=3)
  * `MotionModelFactory::create(string, params)` — YAML `motion_model` 파라미터로 전환
  * VectorXd/MatrixXd 동적 차원 리팩토링 (Q/R/noise_sigma 런타임 크기)
  * swerve_robot.urdf: 4륜 스티어링 + 4륜 휠 URDF
  * launch: `controller:=swerve`, `controller:=non_coaxial` 분기
  * 단위 테스트 36개 통과 (test_motion_model 신규)
- [x] Goal approach 수렴 + 장애물 회피 보수화 튜닝
  * `min_lookahead` 파라미터 구현 (goal 근처 수렴 보장)
  * Q_theta: 1.0→3.0, Qf_theta: 2.0→6.0 (yaw 수렴 가속)
  * R_omega: 0.3→0.8, R_rate_omega: 0.5→1.0 (heading 유지 강화)
  * goal_slowdown_dist: 1.0→0.5 (감속 거리 축소)
  * costmap_lethal_cost: 500→5000, costmap_critical_cost: 50→500 (장애물 비용 10x)
  * obstacle_weight: 100→300, inflation_radius: 1.0→1.5 (회피 영역 확장)
  * `setLethalCost`/`setCriticalCost` 런타임 setter 추가
  * onSetParametersCallback에 min_lookahead, goal_slowdown_dist, costmap costs 추가

### 2026-02-18 (Issue #64)
- [x] Spline-MPPI figure8 궤적 추적 RMSE 개선 (2.17m → <0.5m)
  * Auto knot sigma: basis 감쇠 자동 보정 (amp_factor)
  * LS warm-start: pseudo-inverse 재투영으로 시간 정렬
  * 기본 knots 8→12, Python/C++ 동기화

### 2026-02-18
- [x] #87 M3.5 Smooth/Spline/SVG-MPPI C++ nav2 플러그인 구현 (PR #88)
  * SmoothMPPIControllerPlugin: Δu space 최적화 + jerk cost (Kim et al. 2021)
  * SplineMPPIControllerPlugin: B-spline basis 보간 P knots → N timesteps (ICRA 2024)
  * SVGMPPIControllerPlugin: G guide SVGD + follower resampling (Kondo et al., ICRA 2024)
  * 12개 파라미터 추가 (mppi_params.hpp)
  * svmpc private→protected (SVG-MPPI SVGD 메서드 재사용)
  * plugin XML + CMakeLists + YAML 설정 3개 + launch 분기 3개
  * 단위 테스트 16개 통과, 기존 8개 회귀 없음
  * launch: `controller:=smooth/spline/svg`
  * **M3.5 C++ 포팅 완료** (M5a + M5b + M3.5 C++ 전체 완료)
- [x] #85 SVMPC (Stein Variational MPC) C++ nav2 플러그인 구현 (PR #86)
  * SVMPCControllerPlugin: SVGD 커널 기반 샘플 다양성 유도
  * computeControl() virtual화 + private→protected 리팩터링
  * computeSVGDForce(): attractive + repulsive force
  * medianBandwidth(): median heuristic, computeDiversity(): pairwise L2
  * nav2_params_svmpc.yaml, launch `controller:=svmpc` 분기
  * 단위 테스트 13개 통과 (SVGD Force, Diversity, MedianBandwidth, RBF Kernel)
  * .gitignore 정리: build artifacts, Graphviz 출력 제외
  * **M5a C++ SOTA 변형 완료** (Log-MPPI PR #82 + Tsallis/CVaR PR #84 + SVMPC PR #86)
- [x] MPPI M5b: C++ M2 고도화 머지 완료 (PR #74)
  * Colored Noise Sampler, Adaptive Temperature, Tube-MPPI C++ 구현
- [x] #83 Tsallis-MPPI + Risk-Aware(CVaR) C++ nav2 플러그인 구현 (PR #84)
  * TsallisMPPIWeights: q-exponential 가중치 (heavy/light-tail 조절)
  * RiskAwareMPPIWeights: CVaR 가중치 절단 (risk-averse)
  * TsallisMPPIControllerPlugin, RiskAwareMPPIControllerPlugin
  * qExponential() 유틸리티 함수
  * nav2_params_tsallis_mppi.yaml, nav2_params_risk_aware_mppi.yaml
  * launch에 `controller:=tsallis/risk_aware` 옵션 추가
  * 단위 테스트 30개 통과 (기존 12 + 신규 18)
- [x] launch 파일 정리 — 구버전 5개 삭제 (689줄 제거)
  * 삭제: mppi_nav2_gazebo, gazebo_mppi_test, mppi_navigation, gazebo_harmonic_test, nav2_mppi
  * 잔여: mppi_ros2_control_nav2 (주력), gazebo_ros2_control, mpc_controller, test_urdf

### 2026-02-09
- [x] #81 Log-MPPI C++ nav2 플러그인 구현 (PR #82)
  * WeightComputation Strategy 인터페이스 (Vanilla/Log 분리)
  * LogMPPIControllerPlugin (상속 + 전략 교체)
  * logSumExp 유틸리티 함수
  * nav2_params_log_mppi.yaml 설정 파일
  * launch에 `controller:=log` 옵션 추가
  * 단위 테스트 12개 통과 (Vanilla/Log 동등성, 극단 비용 안정성, greedy fallback)

### 2026-02-08
- [x] #79 PreferForwardCost 추가로 후진 편향 해소 (PR #80)
- [x] #77 controller_server local_costmap 파라미터 누락 수정 (PR #78)
- [x] #75 커스텀 MPPI vs nav2 기본 MPPI 비교 전환 환경 (PR #76)

### 2026-02-07
- [x] MPPI M4: ROS2 nav2 통합 완료 (PR #72)
  * C++ Vanilla MPPI nav2 플러그인
  * Gazebo Harmonic + ros2_control + nav2 통합 launch
  * local_costmap 장애물 추출
  * 동적 파라미터 재설정

### 2026-02-07
- [x] #104 실시간 경로 재계획 기능 — 환경 변화 대응
  * RealtimeReplanner 클래스 (realtime_replanner.py)
  * 재계획 트리거: 충돌 위험, 경로 이탈, 신규 장애물, 목표 변경
  * 환경 변화 자동 감지 및 실시간 재계획
  * 장애물 회피 웨이포인트 생성
  * 부드러운 궤적 전환 (블렌딩)
  * 단위 테스트 8개 통과 (test_realtime_replanner.py)
  * 통합 데모 (realtime_replanning_demo.py) — 동적 장애물 시나리오

### 2026-02-01 (Benchmark)
- [x] MPPI 전체 9종 변형 벤치마크 도구
  * `examples/mppi_all_variants_benchmark.py` — 9종 동시 비교
  * Vanilla, Tube, Log, Tsallis, CVaR, SVMPC, Smooth, Spline, SVG
  * `--live` 실시간 시뮬레이션 모드 지원
  * `--trajectory {circle,figure8,sine}` 궤적 선택
  * ASCII 요약 테이블 + 6패널 정적 비교 차트
  * RMSE, Smoothness, Speed 3개 카테고리 랭킹

### 2026-02-01 (M3.5)
- [x] MPPI M3.5a: Smooth MPPI (SMPPI) — Δu input-lifting 구조적 부드러움 (#56)
  * SmoothMPPIController (Δu space 최적화 + cumsum 복원)
  * Jerk cost (ΔΔu 페널티)로 액추에이터 보호
  * Vanilla 대비 제어 변화율 감소 검증
  * 단위 테스트 17개 통과
  * Vanilla vs SMPPI jerk weight 비교 데모
- [x] MPPI M3.5b: Spline-MPPI — B-spline 보간 기반 smooth sampling (#57)
  * SplineMPPIController (P개 knot에 노이즈 → B-spline basis 보간)
  * 순수 NumPy B-spline basis (de Boor 재귀, scipy 미사용)
  * P << N으로 노이즈 차원 축소 → 구조적 smooth 제어
  * 단위 테스트 23개 통과
  * Vanilla vs Spline P=4/P=8 비교 데모
- [x] MPPI M3.5c: SVG-MPPI — Guide particle 다중 모드 탐색 (#58)
  * SVGMPPIController (G개 guide SVGD + follower resampling)
  * G << K로 SVGD 계산량 O(G²D) << O(K²D)
  * SVMPC 대비 속도 향상 + 다중 모드 유지
  * 단위 테스트 21개 통과
  * Vanilla vs SVMPC vs SVG-MPPI 장애물 환경 비교 데모

### 2026-02-01 (M3)
- [x] MPPI M3d: Stein Variational MPPI (SVMPC) — SVGD 커널 기반 샘플 다양성
  * SteinVariationalMPPIController (SVGD 기반 gradient-free 샘플 분포 개선)
  * rbf_kernel, rbf_kernel_grad, median_bandwidth 유틸리티
  * svgd_num_iterations=0 → Vanilla 완전 동등성 검증
  * compute_control 전체 오버라이드 (SVGD 루프: 매력력+반발력)
  * 단위 테스트 23개 통과
  * SVGD iteration수별 비교 데모
- [x] MPPI M3c: Risk-Aware MPPI (CVaR) — alpha 기반 가중치 절단
  * RiskAwareMPPIController (CVaR 가중치 절단, 최저 비용 ceil(alpha*K)개만 softmax)
  * cvar_alpha 파라미터 (1.0=risk-neutral/Vanilla, <1=risk-averse)
  * alpha=1.0 → Vanilla 완전 동등성 검증
  * 장애물 회피 시 risk-averse가 더 보수적 경로 선택
  * 단위 테스트 22개 통과
  * alpha별 장애물 회피 비교 데모
- [x] MPPI M3a: Log-MPPI — log-space softmax 수치 안정성 (#51)
  * LogMPPIController (log-space 가중치 계산)
  * 극단적 cost(1e-15~1e15)에서 NaN/Inf 방지
  * Vanilla와 일반 범위에서 동일 결과 (차이 < 1e-6)
  * 단위 테스트 15개 통과
  * Vanilla vs Log-MPPI 비교 데모
- [x] MPPI M3b: Tsallis-MPPI — q-exponential 일반화 엔트로피 (#52)
  * TsallisMPPIController (q-exponential 가중치 + min-centering)
  * q_exponential, q_logarithm 유틸리티
  * q=1.0 → Vanilla 하위 호환 (차이 < 1e-8)
  * q>1 heavy-tail(탐색↑), q<1 light-tail(집중↑) 검증
  * min-centering 적용 (q-exp translation-invariance 보정)
  * 단위 테스트 24개 통과
  * q값 비교 데모 (q=0.5, 1.0, 1.2, 1.5)

### 2026-01-31
- [x] MPPI M2: Tube-MPPI — Ancillary 피드백 컨트롤러 (#49)
  * AncillaryController (body frame 오차 변환 + 피드백 보정)
  * TubeMPPIController (MPPIController 상속, 명목 상태 전파)
  * TubeAwareCost (장애물 safety_margin + tube_margin)
  * MPPIParams 확장 (tube_enabled, tube_K_fb 등)
  * 단위 테스트 27개 통과 (ancillary 14 + tube_mppi 13)
  * Vanilla vs Tube 비교 데모 (--live/--noise 지원)
- [x] MPPI M2: 핵심 기능 — ControlRateCost, Adaptive Temp, Colored Noise (#47)
  * ControlRateCost (제어 변화율 비용 함수)
  * AdaptiveTemperature (ESS 기반 λ 자동 튜닝)
  * ColoredNoiseSampler (OU 프로세스 기반 시간 상관 노이즈)
  * Vanilla vs M2 비교 데모 (`examples/mppi_vanilla_vs_m2_demo.py`)
- [x] MPC vs MPPI 비교 데모 (#45, #46)
  * 비교 데모 스크립트 (`examples/mpc_vs_mppi_demo.py`)
  * `--live` 실시간 리플레이 모드
- [x] MPPI M1: Vanilla MPPI 구현 (#31~#36)
  * PRD 문서 작성 (docs/mppi/PRD.md)
  * MPPIParams 데이터클래스 & BatchDynamicsWrapper (RK4 벡터화)
  * 비용 함수 모듈 (StateTracking, Terminal, ControlEffort, Obstacle)
  * GaussianSampler 노이즈 샘플링
  * Vanilla MPPI 컨트롤러 (compute_control 인터페이스 호환)
  * RVIZ 시각화 (샘플 궤적, 가중 궤적, 비용 히트맵)
  * 원형 궤적 추적 RMSE = 0.1534m (< 0.2m 기준 통과)

### 2026-01-25
- [x] #103 동적 장애물 회피 기능 - 움직이는 장애물 대응
- [x] #102 RVIZ 시각화 마커 구현 - 예측 궤적, 제약조건, 장애물 표시
- [x] #101 ROS2 노드 기본 구조 구현 - MPC Controller ROS2 wrapper
- [x] #010 Claude Code 상세 로깅 개선 - watcher 실시간 출력
- [x] #009 GitHub Issue Watcher 자동화 - 이슈 자동 처리

### 2026-01-24
- [x] #008 Logger 유틸리티 구현 - utils/logger.py
- [x] #007 MPC 파라미터 튜닝 가이드 - examples/mpc_tuning_guide.py
- [x] #006 성능 벤치마크 스크립트 - examples/mpc_benchmark.py
- [x] #005 정적 장애물 회피 기능 - examples/obstacle_avoidance_demo.py

### 2026-01-22
- [x] #004 소프트 제약조건 추가 - ObstacleSoftConstraint

### 2026-01-21
- [x] #003 경로 추종 시뮬레이션 기본 루프
- [x] #002 Differential drive 로봇 모델 구현
- [x] #001 MPC 컨트롤러 기본 구현 - CasADi 기반

### 2026-01-20
- [x] #000 프로젝트 초기 구조 설정

### v0.1.0 (초기 구현)
- [x] Swerve Drive 모델 구현
- [x] Non-coaxial Swerve Drive 모델 구현
- [x] 시뮬레이션 환경 구축

---

## 💡 Ideas / Backlog

- 강화학습 기반 MPC 튜닝
- ~~ROS2 nav2 플러그인 통합~~ → M4 완료, ~~M5a/M5b 완료~~, ~~M3.5 C++ 완료~~, ~~MotionModel 추상화 완료~~
- 실제 로봇 테스트 환경 구축
- 슬립 모델 적용
- 적응형 MPC 가중치 튜닝
- pybind11 Python 바인딩 (C++ ↔ Python 연동)

---

## 사용 방법

### 다음 작업 하나 처리
```bash
claude-todo-worker
```

### 특정 작업 처리
```bash
claude-todo-task "#101"
```

### 모든 작업 연속 처리
```bash
claude-todo-all
```

---

## 우선순위 기준

- **P0 (High)**: 핵심 기능, 즉시 필요
- **P1 (Medium)**: 중요하지만 급하지 않음
- **P2 (Low)**: 추가 개선사항, 여유 있을 때

## 작업 규칙

1. 각 작업은 독립적인 기능 단위
2. 작업 완료 시 테스트 필수
3. PR 생성 및 리뷰 후 머지
4. TODO.md 업데이트는 자동으로 처리됨
