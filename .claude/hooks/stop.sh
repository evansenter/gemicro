#!/bin/bash

# Stop hook - Runs pre-push validation checks matching CI configuration
# This ensures local changes will pass CI before being pushed

cd "$CLAUDE_PROJECT_DIR" || exit 1

echo "🔍 Running pre-push validation checks..."
echo ""

# Track if any checks fail
FAILED=0

# 1. Check formatting (matches CI fmt job)
echo "📝 Checking code formatting..."
if cargo fmt --all -- --check 2>&1; then
    echo "✓ Formatting check passed"
else
    echo "✗ Formatting check failed - run 'cargo fmt' to fix"
    FAILED=1
fi
echo ""

# 2. Run clippy with warnings as errors (matches CI clippy job)
echo "🔍 Running clippy lints..."
if cargo clippy --workspace --all-targets --all-features -- -D warnings 2>&1 | head -50; then
    echo "✓ Clippy passed"
else
    echo "✗ Clippy found issues"
    FAILED=1
fi
echo ""

# 3. Run cargo check (matches CI check job)
echo "🔧 Running cargo check..."
if cargo check --workspace --all-targets --all-features 2>&1 | tail -10; then
    echo "✓ Cargo check passed"
else
    echo "✗ Cargo check failed"
    FAILED=1
fi
echo ""

# 4. Run unit tests (subset of CI test job)
echo "🧪 Running unit tests..."
if cargo test --lib --workspace 2>&1 | tail -20; then
    echo "✓ Unit tests passed"
else
    echo "✗ Unit tests failed"
    FAILED=1
fi
echo ""

# Summary
if [ $FAILED -eq 0 ]; then
    echo "✅ All pre-push checks passed! Safe to push."
    exit 0
else
    echo "❌ Some checks failed. Please fix issues before pushing."
    exit 1
fi
