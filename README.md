# Toy Claude Project

Mobile Robot MPC Controller with Claude-Driven Development Workflow

## Overview

This project demonstrates:
1. **MPC-based mobile robot control** - Path tracking with Model Predictive Control
2. **Claude-driven development** - Automated development workflow via GitHub Issues

## Features

- Differential drive robot model
- CasADi-based MPC controller
- 2D simulation with visualization
- Automated CI/CD with Claude integration

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run demo
python examples/path_tracking_demo.py
```

## Project Structure

```
├── mpc_controller/       # Core MPC implementation
│   ├── models/           # Robot kinematic models
│   ├── controllers/      # MPC controller
│   └── utils/            # Trajectory utilities
├── simulation/           # 2D simulator & visualizer
├── tests/                # Unit tests
├── examples/             # Demo scripts
└── .github/workflows/    # CI/CD & Claude automation
```

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

## Dependencies

- Python >= 3.10
- CasADi >= 3.6
- NumPy
- Matplotlib

## License

MIT
