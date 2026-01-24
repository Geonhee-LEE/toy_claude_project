#!/usr/bin/env python3
"""
Claude Development Agent for GitHub Automation.

This script handles:
1. Issue to PR conversion (--action issue-to-pr)
2. Code review automation (--action code-review)
3. @claude mention responses (--action respond-mention)
"""

import argparse
import os
import json
import subprocess
from pathlib import Path
from typing import Optional

import anthropic
from github import Github


# Configuration
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 8192


def get_project_context() -> str:
    """Get project structure and key files for context."""
    context_parts = []

    # Project structure
    result = subprocess.run(
        ["find", ".", "-type", "f", "-name", "*.py", "-not", "-path", "./.venv/*"],
        capture_output=True,
        text=True,
    )
    context_parts.append(f"## Project Python Files\n```\n{result.stdout}\n```")

    # Read CLAUDE.md for project context
    claude_md = Path("CLAUDE.md")
    if claude_md.exists():
        content = claude_md.read_text()
        context_parts.append(f"## Project Instructions (CLAUDE.md)\n```\n{content}\n```")

    # Read key files
    key_files = [
        "mpc_controller/models/differential_drive.py",
        "mpc_controller/controllers/mpc.py",
        "mpc_controller/utils/trajectory.py",
        "simulation/simulator.py",
    ]

    for file_path in key_files:
        if Path(file_path).exists():
            content = Path(file_path).read_text()
            context_parts.append(f"## {file_path}\n```python\n{content}\n```")

    return "\n\n".join(context_parts)


def create_system_prompt(action: str) -> str:
    """Create system prompt for Claude based on action type."""
    base_prompt = """You are an expert robotics software engineer specializing in:
- Model Predictive Control (MPC)
- Mobile robot motion planning and control
- ROS2 navigation development
- CasADi optimization
- Python software architecture

"""

    if action == "issue-to-pr":
        return base_prompt + """Your task is to implement features requested in GitHub issues.

When implementing a feature, respond with a JSON object containing:
{
    "summary": "Brief description of changes",
    "files": [
        {
            "path": "path/to/file.py",
            "action": "create" | "modify" | "delete",
            "content": "full file content for create/modify"
        }
    ],
    "test_files": [
        {
            "path": "tests/test_*.py",
            "content": "test file content"
        }
    ],
    "branch_name": "feature/short-description",
    "commit_message": "feat: description of the change"
}

Guidelines:
1. Write clean, modular, well-documented code
2. Follow existing code patterns and style
3. Include type hints
4. Add appropriate tests
5. Keep backward compatibility

Only respond with valid JSON. No markdown code blocks."""

    elif action == "code-review":
        return base_prompt + """Your task is to review code changes in a Pull Request.

Provide a comprehensive code review in Korean with the following sections:

## 코드 리뷰 요약
[Overall assessment]

## 좋은 점 ✅
- [Positive aspects]

## 개선 제안 💡
- [Suggestions for improvement]

## 잠재적 이슈 ⚠️
- [Potential bugs or issues]

## 보안 검토 🔒
- [Security considerations]

Focus on:
- Code quality and best practices
- Performance considerations
- Error handling
- Test coverage
- Documentation"""

    elif action == "respond-mention":
        return base_prompt + """You are responding to a @claude mention in a GitHub issue or PR.

Respond helpfully in Korean to the user's question or request.
Be concise but thorough. If code changes are needed, explain them clearly.
Use markdown formatting for readability."""

    return base_prompt


def call_claude(prompt: str, system_prompt: str) -> str:
    """Call Claude API with the given prompt."""
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text


def apply_changes(changes: dict) -> None:
    """Apply file changes from Claude's response."""
    for file_info in changes.get("files", []):
        path = Path(file_info["path"])
        action = file_info["action"]

        if action == "delete":
            if path.exists():
                path.unlink()
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file_info["content"])
        print(f"{'Created' if action == 'create' else 'Modified'}: {path}")

    for test_info in changes.get("test_files", []):
        path = Path(test_info["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(test_info["content"])
        print(f"Test file: {path}")


def create_pr(changes: dict, issue_number: int) -> str:
    """Create a branch, commit changes, and open a PR."""
    branch_name = changes.get("branch_name", f"claude/issue-{issue_number}")
    commit_message = changes.get("commit_message", f"feat: implement issue #{issue_number}")

    subprocess.run(["git", "config", "user.name", "Claude Bot"], check=True)
    subprocess.run(["git", "config", "user.email", "claude@anthropic.com"], check=True)
    subprocess.run(["git", "checkout", "-b", branch_name], check=True)
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", commit_message], check=True)
    subprocess.run(
        ["git", "push", "-u", "origin", branch_name],
        check=True,
        env={
            **os.environ,
            "GIT_ASKPASS": "echo",
            "GIT_USERNAME": "x-access-token",
            "GIT_PASSWORD": os.environ["GITHUB_TOKEN"],
        },
    )

    gh = Github(os.environ["GITHUB_TOKEN"])
    repo = gh.get_repo(os.environ["GITHUB_REPOSITORY"])

    pr_body = f"""## Summary
{changes.get('summary', 'Automated implementation by Claude')}

## Related Issue
Closes #{issue_number}

## Changes
This PR was automatically generated by Claude in response to the issue.

---
🤖 *Generated by Claude Development Agent*
"""

    pr = repo.create_pull(
        title=commit_message,
        body=pr_body,
        head=branch_name,
        base="main",
    )

    return pr.html_url


def handle_issue_to_pr(args) -> None:
    """Handle issue to PR conversion."""
    issue_number = int(args.issue_number)
    issue_title = args.issue_title
    issue_body = args.issue_body or ""

    print(f"Processing issue #{issue_number}: {issue_title}")

    project_context = get_project_context()

    prompt = f"""# GitHub Issue

## Title
{issue_title}

## Description
{issue_body}

# Current Project Context
{project_context}

Please implement this feature. Respond with the JSON object containing the implementation."""

    print("Calling Claude API...")
    response_text = call_claude(prompt, create_system_prompt("issue-to-pr"))

    # Parse JSON response
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]

    changes = json.loads(response_text)

    print("Applying changes...")
    apply_changes(changes)

    print("Creating PR...")
    pr_url = create_pr(changes, issue_number)

    response_md = f"""## 구현 완료 ✅

**요약:** {changes.get('summary', 'PR을 확인해주세요')}

**Pull Request:** {pr_url}

### 변경된 파일
"""
    for f in changes.get("files", []):
        response_md += f"- `{f['path']}` ({f['action']})\n"

    Path("claude_response.md").write_text(response_md)
    print(f"Done! PR created: {pr_url}")


def handle_code_review(args) -> None:
    """Handle PR code review."""
    pr_number = args.pr_number
    changed_files = args.files.split() if args.files else []

    print(f"Reviewing PR #{pr_number}")
    print(f"Changed files: {changed_files}")

    # Read changed files content
    file_contents = []
    for file_path in changed_files:
        if Path(file_path).exists():
            content = Path(file_path).read_text()
            file_contents.append(f"## {file_path}\n```python\n{content}\n```")

    # Get git diff
    diff_result = subprocess.run(
        ["git", "diff", "origin/main...HEAD", "--"],
        capture_output=True,
        text=True,
    )

    prompt = f"""# Pull Request #{pr_number} Code Review

## Changed Files
{chr(10).join(file_contents)}

## Git Diff
```diff
{diff_result.stdout[:10000]}
```

Please review the code changes and provide feedback in Korean."""

    print("Calling Claude API for code review...")
    review = call_claude(prompt, create_system_prompt("code-review"))

    Path("review_output.md").write_text(review)
    print("Code review completed!")


def handle_mention(args) -> None:
    """Handle @claude mention response."""
    comment = args.comment
    issue_number = args.issue_number

    print(f"Responding to mention in issue #{issue_number}")

    # Remove @claude from the comment to get the actual question
    question = comment.replace("@claude", "").strip()

    project_context = get_project_context()

    prompt = f"""# User Question/Request

{question}

# Project Context
{project_context}

Please respond to this question or request in Korean."""

    print("Calling Claude API...")
    response = call_claude(prompt, create_system_prompt("respond-mention"))

    Path("claude_response.md").write_text(response)
    print("Response generated!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Claude GitHub Automation Agent")
    parser.add_argument(
        "--action",
        choices=["issue-to-pr", "code-review", "respond-mention"],
        required=True,
        help="Action to perform",
    )
    parser.add_argument("--issue-number", help="GitHub issue number")
    parser.add_argument("--issue-title", help="Issue title")
    parser.add_argument("--issue-body", help="Issue body")
    parser.add_argument("--pr-number", help="Pull request number")
    parser.add_argument("--files", help="Changed files (space-separated)")
    parser.add_argument("--comment", help="Comment body for mention response")

    args = parser.parse_args()

    if args.action == "issue-to-pr":
        handle_issue_to_pr(args)
    elif args.action == "code-review":
        handle_code_review(args)
    elif args.action == "respond-mention":
        handle_mention(args)


if __name__ == "__main__":
    main()
