# MPC Robot Navigation - TODO

프로젝트의 개발 작업 목록입니다. Claude가 순차적으로 처리합니다.

---

## 🔴 High Priority (P0)

- [x] MPPI M3d: Stein Variational MPPI (SVMPC) ✅
- [ ] MPPI M5a: C++ MPPI 코어 변환 — Python → C++ 포팅 (실시간 성능)
- [ ] MPPI M5b: ROS2 nav2 Controller 플러그인 — C++ MPPI nav2 Server 플러그인
- [ ] MPPI M2: 고도화 - GPU 가속 (잔여) — CuPy 기반 NumPy drop-in 또는 JAX jit
- [ ] MPPI M3d-GPU: SVMPC CUDA 가속 — pairwise kernel (K²) + rollout 병렬화
- [ ] MPPI M4: ROS2 통합 마무리 - nav2 플러그인, 실제 로봇, 파라미터 서버

## 🟠 Medium Priority (P1)

- [ ] MPC vs MPPI 비교 데모 파라미터 공정화 — 호라이즌 통일 (MPC 2.0s vs MPPI 1.0s)
- [ ] `--live` 리플레이에 MPPI 샘플 궤적 시각화 추가
- [ ] #104 실시간 경로 재계획 기능 - 환경 변화 대응
- [ ] #105 Ackermann 조향 모델 추가 - 자동차형 로봇 지원
- [ ] #106 속도 제약 고려 MPC - 가속도/저크 제한
- [ ] #107 CI/CD 파이프라인 개선 - 자동 테스트 및 배포
- [ ] #108 시뮬레이션 환경 고도화 - 다양한 맵, 시나리오

## 🟢 Low Priority (P2)

- [ ] #109 Omnidirectional 로봇 모델 - 전방향 이동 로봇 (Mecanum/Omni wheel)
- [ ] #110 성능 프로파일링 및 최적화 - 실시간 성능 개선
- [ ] #111 웹 기반 시각화 대시보드 - 실시간 모니터링
- [ ] #112 Docker 컨테이너화 - 배포 및 재현성 개선
- [ ] #113 Multi-robot MPC - 다중 로봇 협조 제어
- [ ] #114 NMPC (Nonlinear MPC) 구현 - 비선형 최적화

## 📚 Documentation

- [ ] #115 API 문서 자동 생성 - Sphinx/MkDocs
- [ ] #116 튜토리얼 작성 - 사용법 상세 가이드
- [ ] #117 아키텍처 문서 작성 - 시스템 설계 문서

## 🐛 Bug Fixes

- [ ] #118 각도 정규화 엣지 케이스 수정
- [ ] #119 고속 주행 시 경로 추적 오버슈트 개선

---

## ✅ Completed

### 2026-02-01
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
  * DynamicObstaclePredictor 예측 알고리즘 구현
  * 충돌 시간 계산 및 위험 평가
  * 동적 장애물 회피 데모 예제
  * 단위 테스트 (5개 케이스 통과)
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
- ROS2 nav2 플러그인 통합
- 실제 로봇 테스트 환경 구축
- 슬립 모델 적용
- 적응형 MPC 가중치 튜닝

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
