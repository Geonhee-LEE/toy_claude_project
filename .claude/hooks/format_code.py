#!/usr/bin/env python3
"""
Custom code formatting hook for Claude Code.

This hook runs automatically after code changes to ensure
consistent formatting across the project.
"""

import subprocess
import sys
from pathlib import Path


def run_formatter(file_path: str) -> bool:
    """Run black formatter on the file."""
    try:
        result = subprocess.run(
            ["black", "--quiet", file_path],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        print("Warning: black formatter not installed")
        return True


def run_linter(file_path: str) -> list[str]:
    """Run ruff linter and return issues."""
    try:
        result = subprocess.run(
            ["ruff", "check", file_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return result.stdout.strip().split("\n")
        return []
    except FileNotFoundError:
        print("Warning: ruff linter not installed")
        return []


def run_type_check(file_path: str) -> list[str]:
    """Run mypy type checker and return issues."""
    try:
        result = subprocess.run(
            ["mypy", "--ignore-missing-imports", file_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return result.stdout.strip().split("\n")
        return []
    except FileNotFoundError:
        print("Warning: mypy not installed")
        return []


def main():
    """Main hook entry point."""
    if len(sys.argv) < 2:
        print("Usage: format_code.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not file_path.endswith(".py"):
        print(f"Skipping non-Python file: {file_path}")
        sys.exit(0)

    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    print(f"Processing: {file_path}")

    # Format
    if run_formatter(file_path):
        print("  ✓ Formatted")
    else:
        print("  ✗ Format failed")

    # Lint
    lint_issues = run_linter(file_path)
    if lint_issues:
        print("  ⚠ Lint issues:")
        for issue in lint_issues[:5]:
            print(f"    {issue}")
    else:
        print("  ✓ No lint issues")

    # Type check (optional, can be slow)
    # type_issues = run_type_check(file_path)
    # if type_issues:
    #     print("  ⚠ Type issues:")
    #     for issue in type_issues[:5]:
    #         print(f"    {issue}")


if __name__ == "__main__":
    main()
