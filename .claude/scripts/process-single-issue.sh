#!/bin/bash
#
# 단일 이슈 처리 스크립트 (테스트용)
#

set -e

ISSUE_NUM="${1:-26}"
REPO="Geonhee-LEE/toy_claude_project"
PROJECT_DIR="$HOME/toy_claude_project"

# 색상
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}======================================"
echo -e "이슈 #$ISSUE_NUM 직접 처리 (수정된 로깅 테스트)"
echo -e "======================================${NC}"
echo ""

# 이슈 정보 가져오기
echo -e "${BLUE}📋 이슈 정보 가져오는 중...${NC}"
ISSUE_DATA=$(gh issue view "$ISSUE_NUM" --repo "$REPO" --json number,title,body)
ISSUE_TITLE=$(echo "$ISSUE_DATA" | jq -r '.title')
ISSUE_BODY=$(echo "$ISSUE_DATA" | jq -r '.body')

echo -e "${GREEN}✓${NC} 제목: $ISSUE_TITLE"
echo ""

# 프로젝트 디렉토리로 이동
cd "$PROJECT_DIR"

# 최신 코드 가져오기
echo -e "${BLUE}📥 최신 코드 가져오는 중...${NC}"
git checkout main 2>/dev/null || git checkout master
git pull origin main 2>/dev/null || git pull origin master

# 새 브랜치 생성
BRANCH_NAME="feature/issue-${ISSUE_NUM}"
echo -e "${BLUE}🌿 브랜치 생성: $BRANCH_NAME${NC}"
git checkout -b "$BRANCH_NAME" 2>/dev/null || {
    git checkout "$BRANCH_NAME"
    git pull origin "$BRANCH_NAME" 2>/dev/null || true
}

# 이슈에 시작 댓글
gh issue comment "$ISSUE_NUM" --repo "$REPO" \
    --body "🤖 **Claude가 작업을 시작합니다...**

⏱️ 시작 시간: $(date '+%Y-%m-%d %H:%M:%S')
💻 실행 환경: 로컬 머신 ($(hostname))
📁 프로젝트: $PROJECT_DIR

진행 상황은 이 이슈에 업데이트됩니다."

echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}🤖 Claude Code 실행 시작${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

PROMPT="GitHub 이슈 #$ISSUE_NUM 를 구현해주세요.

## 이슈 제목
$ISSUE_TITLE

## 이슈 내용
$ISSUE_BODY

## 지침
1. 필요한 코드를 구현하세요
2. RVIZ 마커 시각화를 고려해주세요 (해당되는 경우)
3. 테스트를 작성하고 실행하세요
4. 모든 변경사항을 커밋하세요
5. 커밋 메시지 형식: 'feat: 기능설명 (closes #$ISSUE_NUM)'
6. 한국어로 주석과 문서를 작성해주세요"

echo -e "${BLUE}📝 프롬프트 전달 중...${NC}"
echo -e "${CYAN}┌─────────────────────────────────────────┐${NC}"
echo -e "${CYAN}│  Claude가 다음 작업을 수행합니다:       │${NC}"
echo -e "${CYAN}│  • 코드 분석 및 이해                    │${NC}"
echo -e "${CYAN}│  • 요구사항 구현                        │${NC}"
echo -e "${CYAN}│  • 테스트 작성 및 실행                  │${NC}"
echo -e "${CYAN}│  • 커밋 생성                            │${NC}"
echo -e "${CYAN}└─────────────────────────────────────────┘${NC}"
echo ""

# Claude Code 실행 로그
CLAUDE_LOG="$HOME/.claude/claude-run-${ISSUE_NUM}.log"

echo -e "${BLUE}🚀 Claude Code 실행 중... (로그: $CLAUDE_LOG)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━ Claude 출력 시작 ━━━━━━━━━━━━━${NC}"
echo ""

# Claude Code 실행
if claude --dangerously-skip-permissions -p "$PROMPT" 2>&1 | tee "$CLAUDE_LOG"; then
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━ Claude 출력 종료 ━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${GREEN}✅ Claude Code 실행 완료${NC}"
    echo -e "${BLUE}📊 실행 로그가 저장되었습니다: $CLAUDE_LOG${NC}"
else
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━ Claude 출력 종료 ━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${RED}❌ Claude Code 실행 실패${NC}"
    exit 1
fi

echo ""
echo -e "${CYAN}======================================"
echo -e "처리 완료"
echo -e "======================================${NC}"
