.PHONY: check fmt clippy test test-all docs clean

# Run all quality gates (format check, clippy, tests)
check: fmt clippy test

# Check formatting
fmt:
	cargo fmt --all -- --check

# Run clippy with warnings as errors (matches CI: --all-targets --all-features)
clippy:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

# Unit tests only (doctests run in CI - excluded locally for speed)
test:
	cargo nextest run --workspace

# Full test suite including integration tests (requires GEMINI_API_KEY)
# Doctests excluded locally - they add compile overhead and CI catches them
test-all:
	cargo nextest run --workspace --run-ignored all

# Build documentation
docs:
	cargo doc --workspace --no-deps

# Clean build artifacts
clean:
	cargo clean
