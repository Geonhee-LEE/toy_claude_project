# MPC Robot Navigation - TODO

프로젝트의 개발 작업 목록입니다. Claude가 순차적으로 처리합니다.

---

## 🔴 High Priority (P0)

- [ ] #102 RVIZ 시각화 마커 구현 - 예측 궤적, 제약조건, 장애물 표시
- [ ] #103 동적 장애물 회피 기능 - 움직이는 장애물 대응

## 🟠 Medium Priority (P1)

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

### 2026-01-25
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
