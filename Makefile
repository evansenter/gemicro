.PHONY: check fmt clippy test test-all docs clean

# Run all quality gates (format check, clippy, tests)
check: fmt clippy test

# Check formatting
fmt:
	cargo fmt --all -- --check

# Run clippy with warnings as errors (matches CI: --all-targets --all-features)
clippy:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

# Run unit and doc tests (matches CI)
test:
	cargo test --workspace

# Run all tests including LLM integration tests (requires GEMINI_API_KEY)
test-all:
	cargo test --workspace -- --include-ignored

# Build documentation
docs:
	cargo doc --workspace --no-deps

# Clean build artifacts
clean:
	cargo clean
