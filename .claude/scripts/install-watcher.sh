#!/bin/bash
#
# Claude Issue Watcher 설치 스크립트
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="$SCRIPT_DIR/claude-watcher.service"
USER_SERVICE_DIR="$HOME/.config/systemd/user"

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║          Claude Issue Watcher 설치                                ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# 디렉토리 생성
mkdir -p "$USER_SERVICE_DIR"
mkdir -p "$HOME/.claude"

# 서비스 파일 복사
echo "1. systemd 서비스 파일 설치 중..."
cp "$SERVICE_FILE" "$USER_SERVICE_DIR/"

# 서비스 리로드
echo "2. systemd 리로드 중..."
systemctl --user daemon-reload

# 서비스 활성화
echo "3. 서비스 활성화 중..."
systemctl --user enable claude-watcher.service

echo ""
echo "✅ 설치 완료!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "사용 방법:"
echo ""
echo "  # 서비스 시작"
echo "  systemctl --user start claude-watcher"
echo ""
echo "  # 상태 확인"
echo "  systemctl --user status claude-watcher"
echo ""
echo "  # 로그 보기"
echo "  journalctl --user -u claude-watcher -f"
echo ""
echo "  # 서비스 중지"
echo "  systemctl --user stop claude-watcher"
echo ""
echo "  # 수동 실행 (테스트)"
echo "  $SCRIPT_DIR/issue-watcher.sh"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "이슈에 'claude' 라벨을 붙이면 자동으로 처리됩니다!"
echo ""
