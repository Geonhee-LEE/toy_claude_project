#!/usr/bin/env python3
"""
Claude Development Agent for GitHub Issues.

This script:
1. Reads a GitHub issue with 'claude-task' label
2. Sends the task to Claude API
3. Creates a branch and commits the changes
4. Opens a Pull Request
"""

import os
import json
import subprocess
from pathlib import Path

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


def create_system_prompt() -> str:
    """Create system prompt for Claude."""
    return """You are an expert robotics software engineer specializing in:
- Model Predictive Control (MPC)
- Mobile robot motion planning and control
- CasADi optimization
- Python software architecture

You are working on a mobile robot MPC project. Your task is to implement features
requested in GitHub issues. Follow these guidelines:

1. Write clean, modular, well-documented code
2. Follow existing code patterns and style
3. Include type hints
4. Add appropriate tests
5. Keep backward compatibility

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

Only respond with valid JSON. No markdown code blocks."""


def call_claude(issue_title: str, issue_body: str, project_context: str) -> dict:
    """Call Claude API with the issue details."""
    client = anthropic.Anthropic()
    
    user_message = f"""# GitHub Issue

## Title
{issue_title}

## Description
{issue_body}

# Current Project Context
{project_context}

Please implement this feature. Respond with the JSON object containing the implementation."""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=create_system_prompt(),
        messages=[{"role": "user", "content": user_message}],
    )
    
    # Parse JSON response
    response_text = response.content[0].text
    
    # Try to extract JSON if wrapped in code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]
    
    return json.loads(response_text)


def apply_changes(changes: dict) -> None:
    """Apply file changes from Claude's response."""
    # Create/modify files
    for file_info in changes.get("files", []):
        path = Path(file_info["path"])
        action = file_info["action"]
        
        if action == "delete":
            if path.exists():
                path.unlink()
            continue
        
        # Create parent directories
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content
        path.write_text(file_info["content"])
        print(f"{'Created' if action == 'create' else 'Modified'}: {path}")
    
    # Create/modify test files
    for test_info in changes.get("test_files", []):
        path = Path(test_info["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(test_info["content"])
        print(f"Test file: {path}")


def create_pr(changes: dict, issue_number: int) -> str:
    """Create a branch, commit changes, and open a PR."""
    branch_name = changes.get("branch_name", f"claude/issue-{issue_number}")
    commit_message = changes.get("commit_message", f"feat: implement issue #{issue_number}")
    
    # Configure git
    subprocess.run(["git", "config", "user.name", "Claude Bot"], check=True)
    subprocess.run(["git", "config", "user.email", "claude@anthropic.com"], check=True)
    
    # Create and checkout branch
    subprocess.run(["git", "checkout", "-b", branch_name], check=True)
    
    # Add and commit changes
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", commit_message], check=True)
    
    # Push branch
    subprocess.run(
        ["git", "push", "-u", "origin", branch_name],
        check=True,
        env={**os.environ, "GIT_ASKPASS": "echo", "GIT_USERNAME": "x-access-token", "GIT_PASSWORD": os.environ["GITHUB_TOKEN"]},
    )
    
    # Create PR using GitHub API
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


def main():
    """Main entry point."""
    issue_number = int(os.environ["ISSUE_NUMBER"])
    issue_title = os.environ["ISSUE_TITLE"]
    issue_body = os.environ.get("ISSUE_BODY", "")
    
    print(f"Processing issue #{issue_number}: {issue_title}")
    
    # Get project context
    project_context = get_project_context()
    
    # Call Claude
    print("Calling Claude API...")
    changes = call_claude(issue_title, issue_body, project_context)
    
    # Apply changes
    print("Applying changes...")
    apply_changes(changes)
    
    # Create PR
    print("Creating PR...")
    pr_url = create_pr(changes, issue_number)
    
    # Write response for GitHub Action to comment
    response_md = f"""## Implementation Complete

**Summary:** {changes.get('summary', 'See PR for details')}

**Pull Request:** {pr_url}

### Files Changed
"""
    for f in changes.get("files", []):
        response_md += f"- `{f['path']}` ({f['action']})\n"
    
    Path("claude_response.md").write_text(response_md)
    print(f"Done! PR created: {pr_url}")


if __name__ == "__main__":
    main()
