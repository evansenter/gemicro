#!/bin/bash
# gemicro REPL Demo Script
#
# This script demonstrates common REPL usage patterns.
# Run with: ./examples/repl_demo.sh
#
# Prerequisites:
#   - GEMINI_API_KEY environment variable set
#   - cargo build completed

set -e

echo "=== gemicro REPL Demo ==="
echo ""
echo "This demo shows common REPL interactions."
echo "Press Ctrl+C to exit at any time."
echo ""

# Check for API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY environment variable not set"
    echo "Set it with: export GEMINI_API_KEY='your-api-key'"
    exit 1
fi

# Build if needed
echo "Building gemicro..."
cargo build -p gemicro-cli --quiet

GEMICRO="cargo run -p gemicro-cli --quiet --"

echo ""
echo "=== Demo 1: List Available Agents ==="
echo "Command: /agent"
echo ""
echo "/agent
/quit" | $GEMICRO --interactive

echo ""
echo "=== Demo 2: Show Help ==="
echo "Command: /help (via unknown command)"
echo ""
echo "/help
/quit" | $GEMICRO --interactive

echo ""
echo "=== Demo 3: Interactive Research Query ==="
echo "Starting interactive session..."
echo "Try commands like:"
echo "  - Type a question to research it"
echo "  - /agent to list agents"
echo "  - /history to see past queries"
echo "  - /clear to clear history"
echo "  - /quit to exit"
echo ""

# Start interactive session
$GEMICRO --interactive
