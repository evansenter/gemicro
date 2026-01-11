.PHONY: check fmt clippy test test-all docs clean

# Run all quality gates (format check, clippy, tests)
check: fmt clippy test

# Check formatting
fmt:
	cargo fmt --all -- --check

# Run clippy with warnings as errors (matches CI: --all-targets --all-features)
clippy:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

# Run unit and doc tests with nextest (parallel execution)
# Falls back to cargo test if nextest not installed
test:
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		cargo nextest run --workspace; \
	else \
		echo "cargo-nextest not found, using cargo test (install: cargo install cargo-nextest)"; \
		cargo test --workspace; \
	fi

# Run all tests including LLM integration tests (requires GEMINI_API_KEY)
test-all:
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		cargo nextest run --workspace --run-ignored all; \
	else \
		cargo test --workspace -- --include-ignored; \
	fi

# Build documentation
docs:
	cargo doc --workspace --no-deps

# Clean build artifacts
clean:
	cargo clean
