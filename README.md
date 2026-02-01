# Toy Claude Project

Mobile Robot MPC Controller with Claude-Driven Development Workflow

## Overview

This project demonstrates:
1. **MPC-based mobile robot control** - Path tracking with Model Predictive Control
2. **MPPI sampling-based control** - Derivative-free parallel sampling controller
3. **Claude-driven development** - Automated development workflow via GitHub Issues

## Features

- Differential drive robot model (Swerve, Non-coaxial Swerve 포함)
- CasADi-based MPC controller
- MPPI (Model Predictive Path Integral) 샘플링 기반 제어
  - Vanilla MPPI (M1)
  - Tube-MPPI, Adaptive Temperature, Colored Noise, ControlRateCost (M2)
  - Log-MPPI, Tsallis-MPPI (M3)
- 2D simulation with visualization
- Automated CI/CD with Claude integration

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

# Log-MPPI vs Vanilla 비교
python examples/log_mppi_demo.py --live

# Tsallis-MPPI q 파라미터 비교
python examples/tsallis_mppi_demo.py --trajectory circle --live
python examples/tsallis_mppi_demo.py --trajectory circle --live --q 0.5 1.0 1.2 1.5
```

## Project Structure

```
mpc_controller/
├── models/                       # 로봇 동역학 모델
│   ├── differential_drive/       # 차동 구동 (v, omega)
│   ├── swerve_drive/             # 스워브 구동
│   └── non_coaxial_swerve/       # 비동축 스워브 구동
├── controllers/
│   ├── mpc/                      # CasADi/IPOPT 기반 MPC
│   ├── mppi/                     # MPPI 샘플링 기반 제어
│   │   ├── base_mppi.py          #   Vanilla MPPI (M1)
│   │   ├── tube_mppi.py          #   Tube-MPPI (M2)
│   │   ├── ancillary_controller.py #  Body frame 피드백 보정 (M2)
│   │   ├── adaptive_temperature.py #  ESS 기반 λ 자동 튜닝 (M2)
│   │   ├── log_mppi.py           #   Log-MPPI (M3a)
│   │   ├── tsallis_mppi.py       #   Tsallis-MPPI (M3b)
│   │   ├── cost_functions.py     #   비용 함수 모듈
│   │   ├── sampling.py           #   Gaussian + Colored Noise 샘플러
│   │   ├── dynamics_wrapper.py   #   배치 동역학 (RK4 벡터화)
│   │   ├── mppi_params.py        #   파라미터 데이터클래스
│   │   └── utils.py              #   유틸리티 (q_exponential 등)
│   ├── swerve_mpc/               # 스워브 MPC
│   └── non_coaxial_swerve_mpc/   # 비동축 스워브 MPC
├── ros2/                         # ROS2 노드 및 RVIZ 시각화
├── simulation/                   # 시뮬레이터
└── utils/                        # 유틸리티 (logger, trajectory 등)

docs/mppi/
├── PRD.md                        # MPPI 제품 요구사항 문서
└── MPPI_GUIDE.md                 # MPPI 기술 가이드 (알고리즘 상세 설명)

tests/
├── test_mppi.py                  # Vanilla MPPI 유닛 + 통합 테스트
├── test_mppi_cost_functions.py   # 비용 함수 테스트
├── test_mppi_sampling.py         # 샘플링 테스트
├── test_ancillary_controller.py  # AncillaryController 테스트 (M2)
├── test_tube_mppi.py             # TubeMPPIController 테스트 (M2)
├── test_log_mppi.py              # LogMPPIController 테스트 (M3a)
└── test_tsallis_mppi.py          # TsallisMPPIController 테스트 (M3b)

examples/
├── mppi_basic_demo.py            # Vanilla MPPI 데모
├── mppi_vanilla_vs_m2_demo.py    # Vanilla vs M2 비교
├── mppi_vanilla_vs_tube_demo.py  # Vanilla vs Tube 비교
├── log_mppi_demo.py              # Log-MPPI 비교 데모 (M3a)
├── tsallis_mppi_demo.py          # Tsallis q 파라미터 비교 (M3b)
├── path_tracking_demo.py         # MPC 경로 추종 데모
└── ...                           # 기타 데모
```

## MPPI 컨트롤러 계층 구조

```
MPPIController (base_mppi.py) — Vanilla MPPI
├── _compute_weights()         ← 서브클래스 오버라이드 포인트
│
├── TubeMPPIController         ── 외란 강건성 (M2)
│   └── AncillaryController    ── body frame 피드백
│
├── LogMPPIController          ── log-space softmax (M3a)
│   └── 참조 구현 (Vanilla와 수학적 동등)
│
└── TsallisMPPIController      ── q-exponential 가중치 (M3b)
    └── q=1.0→Vanilla, q>1→탐색↑, q<1→집중↑
```

자세한 알고리즘 설명은 [docs/mppi/MPPI_GUIDE.md](docs/mppi/MPPI_GUIDE.md) 참조.

## Development Workflow

### Via GitHub Issues (Mobile-friendly)

1. Create an issue with label `claude-task`
2. Describe what you want in the issue body
3. Claude automatically creates a PR with the implementation
4. Review and merge

### Issue Template Example

```markdown
Title: Add obstacle avoidance to MPC

## Task
Implement obstacle avoidance constraints in the MPC controller.

## Requirements
- Support circular obstacles
- Soft constraints with slack variables
- Visualization of obstacle regions
```

## Claude Issue Watcher

로컬 머신에서 GitHub 이슈를 자동으로 감지하고 Claude Code로 구현하는 자동화 도구입니다.

### 동작 플로우

```
┌─────────────────────────────────────────────────────────────┐
│                      동작 플로우                            │
├─────────────────────────────────────────────────────────────┤
│  📱 핸드폰에서 이슈 등록 + 'claude' 라벨                    │
│         ↓                                                   │
│  💻 랩탑이 이슈 감지 (30초 폴링)                            │
│         ↓                                                   │
│  🤖 로컬 Claude Code가 구현                                 │
│         ↓                                                   │
│  📤 자동 커밋 & PR 생성                                     │
│         ↓                                                   │
│  📱 핸드폰으로 알림 (이슈 댓글)                             │
└─────────────────────────────────────────────────────────────┘
```

### 설치 방법

#### 1. 필수 요구사항 확인

- GitHub CLI (`gh`) 설치 및 인증
- Claude Code 설치
- systemd (Linux)

```bash
# GitHub CLI 설치 확인
gh auth status

# Claude Code 설치 확인
claude --version
```

#### 2. Issue Watcher 설치

```bash
# 설치 스크립트 실행
cd .claude/scripts
./install-watcher.sh
```

설치 스크립트는 다음을 수행합니다:
- systemd user 서비스 파일 복사
- 서비스 활성화
- 필요한 디렉토리 생성

### 사용 방법

#### systemd 서비스 제어

```bash
# 서비스 시작
systemctl --user start claude-watcher

# 서비스 상태 확인
systemctl --user status claude-watcher

# 서비스 중지
systemctl --user stop claude-watcher

# 로그 실시간 보기
journalctl --user -u claude-watcher -f

# 로그 파일 확인
tail -f ~/.claude/issue-watcher.log
```

#### 수동 실행 (테스트용)

```bash
# 직접 실행하여 동작 테스트
.claude/scripts/issue-watcher.sh
```

### 이슈에 'claude' 라벨 붙이는 방법

#### GitHub 웹에서

1. 이슈 페이지 열기
2. 오른쪽 사이드바에서 "Labels" 클릭
3. `claude` 라벨 선택

#### GitHub CLI로

```bash
# 라벨 추가
gh issue edit <issue-number> --add-label claude

# 예시: 이슈 #15에 claude 라벨 추가
gh issue edit 15 --add-label claude
```

#### 모바일 GitHub 앱에서

1. 이슈 상세 페이지 열기
2. 상단 메뉴 (⋯) 클릭
3. "Edit" 선택
4. "Labels" 섹션에서 `claude` 선택
5. 저장

### 주요 기능

- **자동 이슈 감지**: 30초마다 `claude` 라벨이 붙은 이슈 확인
- **브랜치 자동 생성**: `feature/issue-{번호}` 형식으로 생성
- **Claude Code 실행**: 비대화형 모드로 자동 구현
- **PR 자동 생성**: 구현 완료 후 자동으로 Pull Request 생성
- **실시간 알림**: 이슈 댓글로 진행 상황 업데이트
- **중복 처리 방지**: 처리된 이슈는 `~/.claude/processed_issues.txt`에 기록

### 설정 파일 위치

- 서비스 파일: `~/.config/systemd/user/claude-watcher.service`
- 스크립트: `.claude/scripts/issue-watcher.sh`
- 처리 기록: `~/.claude/processed_issues.txt`
- 로그 파일: `~/.claude/issue-watcher.log`

### 문제 해결

#### 서비스가 시작되지 않는 경우

```bash
# 서비스 로그 확인
journalctl --user -u claude-watcher -n 50

# 권한 확인
chmod +x .claude/scripts/issue-watcher.sh

# systemd 재로드
systemctl --user daemon-reload
```

#### GitHub 인증 문제

```bash
# GitHub CLI 재인증
gh auth login

# 인증 상태 확인
gh auth status
```

---

## Claude TODO Worker

GitHub 이슈 대신 프로젝트 내 `TODO.md` 파일을 기반으로 Claude가 순차적으로 개발하는 시스템입니다.

### 특징

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   TODO.md       │────▶│  Claude Code     │────▶│  자동 커밋/PR   │
│   작업 목록      │     │  순차 처리        │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

- ✅ 단순함: 파일 하나로 작업 관리
- ✅ 추적 용이: Git 히스토리로 진행 상황 확인
- ✅ 유연성: 로컬/오프라인 작업 가능
- ✅ 우선순위: P0/P1/P2 라벨로 작업 구분

### 사용 방법

#### 1. 다음 작업 하나 처리
```bash
claude-todo-worker
```

첫 번째 미완료 작업을 자동으로 찾아서 처리합니다.

#### 2. 특정 작업 처리
```bash
claude-todo-task "#101"
```

작업 ID를 지정해서 해당 작업만 처리합니다.

#### 3. 모든 작업 연속 처리
```bash
claude-todo-all
```

TODO.md의 모든 미완료 작업을 순차적으로 처리합니다. (30초 간격으로 자동 처리)

### TODO.md 구조

```markdown
# MPC Robot Navigation - TODO

## 🔴 High Priority (P0)
- [ ] #101 ROS2 노드 기본 구조 구현
- [ ] #102 RVIZ 시각화 마커 구현

## 🟠 Medium Priority (P1)
- [ ] #104 실시간 경로 재계획 기능

## 🟢 Low Priority (P2)
- [ ] #109 Omnidirectional 로봇 모델

## ✅ Completed
- [x] #001 MPC 컨트롤러 기본 구현
```

### 워크플로우

1. **작업 추가**: `TODO.md`에 새 작업 추가
2. **자동 처리**: `claude-todo-worker` 실행
3. **확인**: Claude가 코드 구현, 테스트, 커밋 자동 수행
4. **PR 리뷰**: 생성된 PR 확인 및 머지
5. **TODO 업데이트**: 자동으로 완료 표시

### 장점

| 방식 | Issue Watcher | TODO Worker |
|------|---------------|-------------|
| **온라인 필요** | ✅ 필수 | ❌ 선택 |
| **설정** | 복잡 (systemd) | 간단 (스크립트) |
| **진행 상황** | GitHub 이슈 | TODO.md 파일 |
| **우선순위** | 라벨 | P0/P1/P2 구분 |
| **속도** | 30초 폴링 | 즉시 실행 |

### 설치

스크립트는 `~/.local/bin/` 에 자동 설치되어 있습니다:
- `claude-todo-worker`: 단일 작업 처리
- `claude-todo-task`: 특정 작업 처리
- `claude-todo-all`: 전체 작업 처리

PATH 설정 확인:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## Dependencies

- Python >= 3.10
- NumPy >= 1.24
- Matplotlib >= 3.7
- CasADi >= 3.6 (MPC 컨트롤러용)

MPPI 컨트롤러는 순수 NumPy로 구현되어 CasADi 없이도 동작합니다.

## Testing

```bash
# 전체 테스트 실행
pytest tests/ -v

# MPPI 테스트만 실행
pytest tests/test_mppi*.py tests/test_log_mppi.py tests/test_tsallis_mppi.py tests/test_tube_mppi.py tests/test_ancillary_controller.py -v

# 특정 테스트
pytest tests/test_tsallis_mppi.py -v -k "circle_tracking"
```

## License

MIT
