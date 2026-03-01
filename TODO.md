# MPC Robot Navigation - TODO

ROS2 C++ 기반 MPPI 컨트롤러 고도화를 최우선으로 진행합니다.

---

## 🔴 High Priority (P0) — ROS2 C++ 컨트롤러 고도화

- [ ] #121 Biased-MPPI C++ nav2 플러그인 — ancillary 컨트롤러 편향 샘플링 (RA-L 2024)
- [ ] Ackermann MotionModel C++ — 자동차형 로봇 모델 (nx=4, nu=2: 속도+스티어링)
- [ ] C++ MPPI 성능 프로파일링 — hotspot 분석 + Eigen 최적화 (SIMD, 캐시 지역성)
- [ ] Covariance Steering MPPI C++ — 공분산 제어 기반 정밀 분포 조정

## 🟠 Medium Priority (P1) — C++ 확장 및 알고리즘 개선

- [ ] π-MPPI / BR-MPPI / SOPPI C++ 플러그인 — 최신 MPPI 변형 (2025)
- [ ] CLF-CBF-QP C++ 컨트롤러 — Lyapunov + Barrier 통합 제어
- [ ] 다중 CBF 합성 C++ — 복잡한 제약 조건 처리 (교집합/합집합)
- [ ] Omnidirectional MotionModel C++ — Mecanum/Omni wheel 모델
- [ ] MPC vs MPPI 비교 데모 파라미터 공정화 — 호라이즌 통일 (MPC 2.0s vs MPPI 1.0s)
- [ ] `--live` 리플레이에 MPPI 샘플 궤적 시각화 추가
- [ ] 각도 정규화 엣지 케이스 수정
- [ ] 고속 주행 시 경로 추적 오버슈트 개선

## 🟢 Low Priority (P2) — Python / 인프라 / 기타

- [ ] CBF GPU 가속 — JAX 기반 CBF 제약 병렬 계산
- [ ] 속도 제약 고려 MPC — 가속도/저크 제한
- [ ] Multi-robot MPC — 다중 로봇 협조 제어
- [ ] NMPC (Nonlinear MPC) 구현 — 비선형 최적화
- [ ] 웹 기반 시각화 대시보드 — 실시간 모니터링
- [ ] Docker 컨테이너화 — 배포 및 재현성 개선

## 📚 Documentation

- [ ] API 문서 자동 생성 — Sphinx/MkDocs
- [ ] C++ 플러그인 개발 가이드 — 신규 MPPI 변형 추가 방법
- [ ] 아키텍처 문서 작성 — 시스템 설계 문서
- [x] MPPI 기술 가이드 업데이트 — M3.5 변형 설명 추가

---

## ✅ Completed

### ROS2 C++ 플러그인 (8종 + 다모델 + CBF + 안정화)
- [x] M4: ROS2 nav2 통합 — C++ Vanilla MPPI nav2 플러그인 (PR #72)
- [x] M5a: C++ SOTA — Log/Tsallis/CVaR/SVMPC 플러그인 (PR #82, #84, #86)
- [x] M5b: C++ M2 — Colored Noise, Adaptive Temp, Tube-MPPI (PR #74)
- [x] M3.5 C++: Smooth/Spline/SVG-MPPI 플러그인 (PR #88)
- [x] MotionModel 추상화 — DiffDrive/Swerve/NonCoaxialSwerve (PR #96)
- [x] MPPI-CBF 통합 C++ — BarrierFunction + CBFSafetyFilter (PR #98)
- [x] 궤적 안정화 — SG Filter + IT 정규화 + Exploitation/Exploration
- [x] Swerve 모션 품질 — theta smoothing + vy_max + VelocityTrackingCost (PR #113)
- [x] 60° 스티어링 제한 — NonCoaxialSwerve 데모 + 벤치마크 (PR #118)
- [x] nav2 Bond-Free Lifecycle 관리 — nav2_lifecycle_bringup.py

### pybind11 바인딩 + 벤치마크
- [x] pybind11 Python 바인딩 — C++ MPPI 코어 Python 노출 (PR #115)
- [x] Python vs C++ MPPI 벤치마크 스위트 (PR #117)
- [x] LookaheadInterpolator 전체 예제 통합 (PR #117)

### Python MPPI (M1~M3.5 + GPU)
- [x] M1: Vanilla MPPI (Python)
- [x] M2: Tube-MPPI, Colored Noise, Adaptive Temp, ControlRateCost
- [x] M3: Log, Tsallis, CVaR, SVMPC
- [x] M3.5: Smooth, Spline, SVG-MPPI
- [x] GPU 가속 — JAX JIT + lax.scan + vmap (PR #103)
- [x] GPU 8종 변형 확장 — 가중치 Strategy + SVGD JIT (PR #105)
- [x] MPPI-CBF 통합 Python (PR #98)
- [x] MPPI vs CBF-MPPI 벤치마크 (PR #107)

### 인프라 + 시뮬레이션
- [x] CI/CD 파이프라인 — GitHub Actions ROS2 빌드 + 테스트 (PR #102)
- [x] Swerve E2E 시뮬레이션 검증 스크립트 (PR #100)
- [x] Goal 수렴 + 장애물 회피 튜닝
- [x] Swerve MPPI 오실레이션 진단 + 수렴 수정

### MPC 기반 (초기 구현)
- [x] CasADi 기반 MPC 컨트롤러
- [x] DiffDrive / Swerve / NonCoaxial Swerve 모델
- [x] 시뮬레이션 환경 구축
- [x] ROS2 노드 기본 구조 + RVIZ 시각화

---

## 💡 Ideas / Backlog

- 강화학습 기반 MPC 튜닝
- 슬립 모델 적용
- 적응형 MPC 가중치 튜닝
- 실제 로봇 테스트 환경 구축

---

## 우선순위 기준

```
┌──────────────────────────────────────────────────────────┐
│  P0 (High)  │ ROS2 C++ 컨트롤러 신규 알고리즘 + 모델   │
│             │ → Biased-MPPI, Ackermann, 성능 최적화     │
├─────────────┼────────────────────────────────────────────┤
│  P1 (Medium)│ C++ 확장 + 알고리즘 연구                  │
│             │ → 최신 MPPI 변형, CLF-CBF, 새 MotionModel │
├─────────────┼────────────────────────────────────────────┤
│  P2 (Low)   │ Python / MPC / 인프라                     │
│             │ → GPU CBF, Multi-robot, 시각화             │
└──────────────────────────────────────────────────────────┘
```

## 작업 규칙

1. 각 작업은 독립적인 기능 단위
2. 작업 완료 시 테스트 필수
3. PR 생성 및 리뷰 후 머지
4. TODO.md 업데이트는 자동으로 처리됨
