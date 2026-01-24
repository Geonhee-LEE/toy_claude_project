#!/bin/bash

# Read file path from stdin JSON
FILE_PATH=$(jq -r '.tool_input.file_path')

# Only process Python files
if [[ "$FILE_PATH" == *.py ]]; then
    # Format with Black
    black --quiet "$FILE_PATH" 2>/dev/null

    # Sort imports
    isort --quiet "$FILE_PATH" 2>/dev/null

    # Run quick lint (non-blocking)
    ruff check "$FILE_PATH" --fix --quiet 2>/dev/null || true

    echo "Formatted: $FILE_PATH"
fi

exit 0
