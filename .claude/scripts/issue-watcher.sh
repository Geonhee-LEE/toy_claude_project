#!/bin/bash
#
# Claude Issue Watcher - 로컬에서 GitHub 이슈 자동 처리
#
# ┌─────────────────────────────────────────────────────────────┐
# │                      동작 플로우                            │
# ├─────────────────────────────────────────────────────────────┤
# │  📱 핸드폰에서 이슈 등록 + 'claude' 라벨                    │
# │         ↓                                                   │
# │  💻 랩탑이 이슈 감지 (30초 폴링)                            │
# │         ↓                                                   │
# │  🤖 로컬 Claude Code가 구현                                 │
# │         ↓                                                   │
# │  📤 자동 커밋 & PR 생성                                     │
# │         ↓                                                   │
# │  📱 핸드폰으로 알림 (이슈 댓글)                             │
# └─────────────────────────────────────────────────────────────┘

set -e

# 설정
REPO="Geonhee-LEE/toy_claude_project"
PROJECT_DIR="$HOME/toy_claude_project"
PROCESSED_FILE="$HOME/.claude/processed_issues.txt"
LOG_FILE="$HOME/.claude/issue-watcher.log"
POLL_INTERVAL=30  # 초
LABEL="claude"    # 감지할 라벨

# 색상
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 디렉토리 생성
mkdir -p "$(dirname "$PROCESSED_FILE")"
mkdir -p "$(dirname "$LOG_FILE")"
touch "$PROCESSED_FILE"

# 로그 함수
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "$msg" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    log "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

# 시작 배너
show_banner() {
    echo -e "${CYAN}"
    cat << 'EOF'
╔═══════════════════════════════════════════════════════════════════╗
║                  🤖 Claude Issue Watcher                          ║
╠═══════════════════════════════════════════════════════════════════╣
║  GitHub 이슈를 자동으로 감지하고 Claude Code로 구현합니다         ║
╚═══════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    log_info "프로젝트: $PROJECT_DIR"
    log_info "저장소: $REPO"
    log_info "라벨: $LABEL"
    log_info "폴링 간격: ${POLL_INTERVAL}초"
    log_info "로그 파일: $LOG_FILE"
    echo ""
}

# GitHub CLI 확인
check_prerequisites() {
    if ! command -v gh &> /dev/null; then
        log_error "GitHub CLI (gh)가 설치되어 있지 않습니다."
        log_error "설치: https://cli.github.com/"
        exit 1
    fi

    if ! gh auth status &> /dev/null; then
        log_error "GitHub CLI 인증이 필요합니다."
        log_error "실행: gh auth login"
        exit 1
    fi

    if ! command -v claude &> /dev/null; then
        log_error "Claude Code가 설치되어 있지 않습니다."
        exit 1
    fi

    if [ ! -d "$PROJECT_DIR" ]; then
        log_error "프로젝트 디렉토리가 존재하지 않습니다: $PROJECT_DIR"
        exit 1
    fi

    log_success "환경 확인 완료"
}

# 이슈 처리
process_issue() {
    local issue_num="$1"
    local issue_title="$2"
    local issue_body="$3"

    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "🆕 새 이슈 처리 시작: #$issue_num"
    log_info "제목: $issue_title"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 이슈에 시작 댓글
    gh issue comment "$issue_num" --repo "$REPO" \
        --body "🤖 **Claude가 작업을 시작합니다...**

⏱️ 시작 시간: $(date '+%Y-%m-%d %H:%M:%S')
💻 실행 환경: 로컬 머신 ($(hostname))
📁 프로젝트: $PROJECT_DIR

진행 상황은 이 이슈에 업데이트됩니다."

    # 프로젝트 디렉토리로 이동
    cd "$PROJECT_DIR" || {
        log_error "프로젝트 디렉토리 접근 실패"
        return 1
    }

    # 최신 코드 가져오기
    log_info "최신 코드 가져오는 중..."
    git checkout main 2>/dev/null || git checkout master
    git pull origin main 2>/dev/null || git pull origin master

    # 새 브랜치 생성
    local branch_name="feature/issue-${issue_num}"
    log_info "브랜치 생성: $branch_name"
    git checkout -b "$branch_name" 2>/dev/null || {
        git checkout "$branch_name"
        git pull origin "$branch_name" 2>/dev/null || true
    }

    # Claude Code 실행
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "🤖 Claude Code 실행 시작"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    local prompt="GitHub 이슈 #$issue_num 를 구현해주세요.

## 이슈 제목
$issue_title

## 이슈 내용
$issue_body

## 지침
1. 필요한 코드를 구현하세요
2. RVIZ 마커 시각화를 고려해주세요 (해당되는 경우)
3. 테스트를 작성하고 실행하세요
4. 모든 변경사항을 커밋하세요
5. 커밋 메시지 형식: 'feat: 기능설명 (closes #$issue_num)'
6. 한국어로 주석과 문서를 작성해주세요"

    log_info "📝 프롬프트 전달 중..."
    echo -e "${CYAN}┌─────────────────────────────────────────┐${NC}"
    echo -e "${CYAN}│  Claude가 다음 작업을 수행합니다:       │${NC}"
    echo -e "${CYAN}│  • 코드 분석 및 이해                    │${NC}"
    echo -e "${CYAN}│  • 요구사항 구현                        │${NC}"
    echo -e "${CYAN}│  • 테스트 작성 및 실행                  │${NC}"
    echo -e "${CYAN}│  • 커밋 생성                            │${NC}"
    echo -e "${CYAN}└─────────────────────────────────────────┘${NC}"
    echo ""

    # Claude Code 실행 로그 파일
    local claude_log="$HOME/.claude/claude-run-${issue_num}.log"

    log_info "🚀 Claude Code 실행 중... (로그: $claude_log)"
    echo -e "${YELLOW}━━━━━━━━━━━━━ Claude 출력 시작 ━━━━━━━━━━━━━${NC}"
    echo ""

    # Claude Code 실행 (비대화형 모드) - 출력을 터미널과 로그 파일에 모두 표시
    if claude --dangerously-skip-permissions -p "$prompt" 2>&1 | tee "$claude_log"; then
        echo ""
        echo -e "${YELLOW}━━━━━━━━━━━━━ Claude 출력 종료 ━━━━━━━━━━━━━${NC}"
        echo ""
        log_success "✅ Claude Code 실행 완료"
        log_info "📊 실행 로그가 저장되었습니다: $claude_log"
    else
        echo ""
        echo -e "${YELLOW}━━━━━━━━━━━━━ Claude 출력 종료 ━━━━━━━━━━━━━${NC}"
        echo ""
        log_error "❌ Claude Code 실행 실패"
        log_error "📋 상세 로그: $claude_log"

        gh issue comment "$issue_num" --repo "$REPO" \
            --body "❌ **Claude 실행 중 오류 발생**

로그를 확인해주세요:
- 전체 로그: $LOG_FILE
- Claude 실행 로그: $claude_log"
        return 1
    fi

    echo ""
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 변경사항 확인
    if [ -z "$(git status --porcelain)" ]; then
        log_warn "변경사항 없음"
        gh issue comment "$issue_num" --repo "$REPO" \
            --body "⚠️ **변경사항 없음**

Claude가 분석했지만 코드 변경이 필요하지 않았습니다."
        git checkout main
        return 0
    fi

    # 커밋되지 않은 변경사항 커밋
    if [ -n "$(git status --porcelain)" ]; then
        log_info "변경사항 커밋 중..."
        git add -A
        git commit -m "feat: $issue_title (closes #$issue_num)

Co-Authored-By: Claude <noreply@anthropic.com>" || true
    fi

    # 푸시
    log_info "변경사항 푸시 중..."
    git push -u origin "$branch_name"

    # PR 생성
    log_info "PR 생성 중..."
    local pr_url
    pr_url=$(gh pr create --repo "$REPO" \
        --title "feat: $issue_title" \
        --body "## Summary
Closes #$issue_num

## 구현 내용
Claude가 자동으로 구현했습니다.

## 변경사항
$(git log main.."$branch_name" --oneline | head -10)

---
🤖 Generated with Claude Code (Local)" \
        --base main \
        --head "$branch_name" 2>/dev/null) || {
        log_warn "PR이 이미 존재하거나 생성 실패"
        pr_url=$(gh pr view "$branch_name" --repo "$REPO" --json url -q '.url' 2>/dev/null || echo "PR 링크 없음")
    }

    # 이슈에 완료 댓글
    gh issue comment "$issue_num" --repo "$REPO" \
        --body "✅ **구현 완료!**

🔗 PR: $pr_url
⏱️ 완료 시간: $(date '+%Y-%m-%d %H:%M:%S')

### 변경된 파일
\`\`\`
$(git diff main.."$branch_name" --stat | tail -20)
\`\`\`"

    # 처리 완료 기록
    echo "$issue_num" >> "$PROCESSED_FILE"

    # main으로 돌아가기
    git checkout main

    log_success "이슈 #$issue_num 처리 완료!"
    return 0
}

# 메인 루프
main_loop() {
    log_info "이슈 감시 시작... (Ctrl+C로 종료)"
    echo ""

    while true; do
        # 'claude' 라벨이 있는 열린 이슈 확인
        local issues
        issues=$(gh issue list --repo "$REPO" --state open --label "$LABEL" --json number,title,body 2>/dev/null) || {
            log_warn "이슈 목록 가져오기 실패, 재시도..."
            sleep "$POLL_INTERVAL"
            continue
        }

        # 이슈가 있는지 확인
        local count
        count=$(echo "$issues" | jq 'length')

        if [ "$count" -gt 0 ]; then
            echo "$issues" | jq -c '.[]' | while read -r issue; do
                local issue_num issue_title issue_body

                issue_num=$(echo "$issue" | jq -r '.number')
                issue_title=$(echo "$issue" | jq -r '.title')
                issue_body=$(echo "$issue" | jq -r '.body')

                # 이미 처리된 이슈인지 확인
                if grep -q "^${issue_num}$" "$PROCESSED_FILE" 2>/dev/null; then
                    continue
                fi

                # 이슈 처리
                process_issue "$issue_num" "$issue_title" "$issue_body" || {
                    log_error "이슈 #$issue_num 처리 실패"
                }
            done
        fi

        # 대기
        sleep "$POLL_INTERVAL"
    done
}

# 메인 실행
show_banner
check_prerequisites
main_loop
