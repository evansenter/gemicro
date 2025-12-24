//! Integration tests for the gemicro CLI.
//!
//! These tests require GEMINI_API_KEY to be set and are marked #[ignore].
//! Run with: cargo test -p gemicro-cli -- --include-ignored

use std::process::Command;
use std::time::Duration;

/// Helper to run the CLI binary with arguments.
fn run_cli(args: &[&str]) -> std::process::Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_gemicro"));
    cmd.args(args);
    cmd.output().expect("Failed to execute gemicro CLI")
}

/// Helper to check if GEMINI_API_KEY is set.
fn has_api_key() -> bool {
    std::env::var("GEMINI_API_KEY").is_ok()
}

#[test]
fn test_cli_help() {
    let output = run_cli(&["--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("gemicro"));
    assert!(stdout.contains("query"));
    assert!(stdout.contains("--api-key"));
}

#[test]
fn test_cli_version() {
    let output = run_cli(&["--version"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("gemicro"));
    assert!(stdout.contains("0.1.0"));
}

#[test]
fn test_cli_missing_query() {
    // Running without a query should fail (unless GEMINI_API_KEY is not set,
    // in which case it fails on missing API key first)
    let output = run_cli(&[]);
    assert!(!output.status.success());
}

#[test]
fn test_cli_invalid_temperature() {
    let output = run_cli(&[
        "test query",
        "--api-key",
        "fake-key",
        "--temperature",
        "2.0",
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("temperature"));
}

#[test]
fn test_cli_invalid_min_max_queries() {
    let output = run_cli(&[
        "test query",
        "--api-key",
        "fake-key",
        "--min-sub-queries",
        "10",
        "--max-sub-queries",
        "5",
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("min-sub-queries"));
}

#[test]
fn test_cli_zero_timeout() {
    let output = run_cli(&["test query", "--api-key", "fake-key", "--timeout", "0"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("timeout"));
}

#[test]
fn test_cli_llm_timeout_exceeds_total() {
    let output = run_cli(&[
        "test query",
        "--api-key",
        "fake-key",
        "--timeout",
        "60",
        "--llm-timeout",
        "120",
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("llm-timeout"));
}

#[test]
#[ignore] // Requires GEMINI_API_KEY
fn test_cli_simple_query() {
    if !has_api_key() {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    }

    let output = run_cli(&[
        "What is 2 + 2?",
        "--min-sub-queries",
        "2",
        "--max-sub-queries",
        "3",
        "--timeout",
        "120",
    ]);

    assert!(output.status.success(), "CLI failed: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Check for expected output sections
    assert!(stdout.contains("gemicro Deep Research"), "Missing header");
    assert!(
        stdout.contains("SYNTHESIZED ANSWER"),
        "Missing answer section"
    );
    assert!(stdout.contains("Performance:"), "Missing performance stats");
    assert!(stdout.contains("Sub-queries:"), "Missing sub-query stats");
}

#[test]
#[ignore] // Requires GEMINI_API_KEY
fn test_cli_verbose_mode() {
    if !has_api_key() {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    }

    let output = run_cli(&[
        "What is Rust?",
        "--verbose",
        "--min-sub-queries",
        "2",
        "--max-sub-queries",
        "2",
        "--timeout",
        "120",
    ]);

    assert!(output.status.success(), "CLI failed: {:?}", output);

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Verbose mode should produce debug output
    assert!(
        stderr.contains("DEBUG") || stderr.contains("Creating interaction"),
        "Verbose mode should produce debug output"
    );
}

#[test]
#[ignore] // Requires GEMINI_API_KEY
fn test_cli_token_counts_displayed() {
    if !has_api_key() {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    }

    let output = run_cli(&[
        "Explain recursion briefly",
        "--min-sub-queries",
        "2",
        "--max-sub-queries",
        "2",
        "--timeout",
        "120",
    ]);

    assert!(output.status.success(), "CLI failed: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Token counts should be displayed (now that rust-genai is fixed)
    assert!(
        stdout.contains("tokens)") || stdout.contains("Tokens used:"),
        "Token counts should be displayed in output"
    );
}

#[test]
#[ignore] // Requires GEMINI_API_KEY
fn test_cli_parallel_speedup_displayed() {
    if !has_api_key() {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    }

    let output = run_cli(&[
        "Compare cats and dogs",
        "--min-sub-queries",
        "3",
        "--max-sub-queries",
        "4",
        "--timeout",
        "180",
    ]);

    assert!(output.status.success(), "CLI failed: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parallel speedup should be displayed
    assert!(
        stdout.contains("Parallel speedup:") || stdout.contains("saved"),
        "Parallel speedup should be displayed"
    );
}
