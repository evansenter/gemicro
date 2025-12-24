#!/bin/bash

# Session initialization for gemicro project
# Checks environment setup and project health

echo "=== gemicro Project Session Init ==="
echo ""

# Check for GEMINI_API_KEY
if [ -n "$GEMINI_API_KEY" ]; then
  echo "✓ GEMINI_API_KEY is configured"
  if [ -n "$CLAUDE_ENV_FILE" ]; then
    echo 'export GEMINI_API_KEY="'$GEMINI_API_KEY'"' >> "$CLAUDE_ENV_FILE"
  fi
else
  echo "⚠ GEMINI_API_KEY not set - integration tests will be skipped"
  echo "  Set it with: export GEMINI_API_KEY=your_key"
fi

echo ""

# Quick build check
cd "$CLAUDE_PROJECT_DIR"
echo "Checking if project builds..."
if cargo check --quiet 2>/dev/null; then
  echo "✓ Project builds successfully"
else
  echo "⚠ Project has build issues - run 'cargo build' for details"
fi

echo ""
echo "=== Session Ready ==="
echo "Remember: Use 'cargo test -- --include-ignored' for full testing"
echo ""

exit 0
