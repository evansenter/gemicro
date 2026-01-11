//! Integration tests for the gemicro CLI.
//!
//! These tests require GEMINI_API_KEY to be set and are marked #[ignore].
//! Run with: cargo test -p gemicro-cli -- --include-ignored

use std::process::Command;

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
        "--agent",
        "echo",
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
        "--agent",
        "echo",
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
    let output = run_cli(&[
        "test query",
        "--agent",
        "echo",
        "--api-key",
        "fake-key",
        "--timeout",
        "0",
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("timeout"));
}

#[test]
fn test_cli_llm_timeout_exceeds_total() {
    let output = run_cli(&[
        "test query",
        "--agent",
        "echo",
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
        "--agent",
        "deep_research",
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
    assert!(stdout.contains("Performance:"), "Missing performance stats");
    assert!(stdout.contains("Steps:"), "Missing step stats");
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
        "--agent",
        "deep_research",
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
        "--agent",
        "deep_research",
        "--min-sub-queries",
        "2",
        "--max-sub-queries",
        "2",
        "--timeout",
        "120",
    ]);

    assert!(output.status.success(), "CLI failed: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Token counts should be displayed (now that genai-rs is fixed)
    assert!(
        stdout.contains("tokens)") || stdout.contains("Tokens used:"),
        "Token counts should be displayed in output"
    );
}

// NOTE: test_cli_parallel_speedup_displayed was removed in #124.
// The simplified CLI no longer tracks step-level timing for parallel speedup
// calculation. This is an intentional breaking change when migrating to the
// ExecutionTracking pattern (which is agent-agnostic and doesn't provide
// step-level timing data).

// ============================================================================
// Sandbox Path Tests
// ============================================================================

#[test]
fn test_cli_sandbox_path_help() {
    let output = run_cli(&["--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--sandbox-path"),
        "Help should mention --sandbox-path flag"
    );
    assert!(
        stdout.contains("Restrict file operations"),
        "Help should describe sandbox restriction"
    );
}

#[test]
fn test_cli_sandbox_path_nonexistent() {
    let output = run_cli(&[
        "test query",
        "--agent",
        "echo",
        "--api-key",
        "fake-key",
        "--sandbox-path",
        "/nonexistent/path/that/should/not/exist",
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("does not exist"),
        "Should report path does not exist"
    );
}

#[test]
fn test_cli_sandbox_path_not_directory() {
    // Use the test binary itself as a file that exists but isn't a directory
    let binary_path = env!("CARGO_BIN_EXE_gemicro");
    let output = run_cli(&[
        "test query",
        "--agent",
        "echo",
        "--api-key",
        "fake-key",
        "--sandbox-path",
        binary_path,
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("must be a directory"),
        "Should report path must be directory"
    );
}

#[test]
fn test_cli_sandbox_path_valid() {
    // /tmp should exist on most systems
    if !std::path::Path::new("/tmp").exists() {
        return;
    }

    // With a valid sandbox path, should fail on missing agent or other reasons,
    // but NOT on sandbox path validation
    let output = run_cli(&[
        "test query",
        "--agent",
        "echo",
        "--api-key",
        "fake-key",
        "--sandbox-path",
        "/tmp",
    ]);

    // Check that it didn't fail on sandbox validation
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("sandbox path does not exist"),
        "Valid sandbox path should not trigger existence error"
    );
    assert!(
        !stderr.contains("must be a directory"),
        "Valid sandbox path should not trigger directory error"
    );
}

#[test]
fn test_cli_multiple_sandbox_paths() {
    // Test that multiple --sandbox-path flags are accepted
    if !std::path::Path::new("/tmp").exists() {
        return;
    }

    let output = run_cli(&[
        "test query",
        "--agent",
        "echo",
        "--api-key",
        "fake-key",
        "--sandbox-path",
        "/tmp",
        "--sandbox-path",
        "/tmp", // Duplicate is fine, just testing multiple args work
    ]);

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("sandbox path"),
        "Multiple valid sandbox paths should not trigger validation error"
    );
}

// ============================================================================
// Interactive REPL Tests
// ============================================================================

#[test]
fn test_cli_interactive_help() {
    let output = run_cli(&["--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--interactive"),
        "Help should mention --interactive flag"
    );
    assert!(stdout.contains("-i"), "Help should mention -i short flag");
}

#[test]
fn test_cli_interactive_no_query_required() {
    // Interactive mode should not require a query argument
    // This will fail on missing API key, but that's expected
    let output = run_cli(&["--interactive", "--agent", "echo", "--api-key", "fake-key"]);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should NOT fail with "Query is required"
    assert!(
        !stderr.contains("Query is required"),
        "Interactive mode should not require query"
    );
}

/// Helper to run CLI with stdin input
fn run_cli_with_stdin(args: &[&str], stdin: &str) -> std::process::Output {
    use std::io::Write;
    use std::process::Stdio;

    let mut child = Command::new(env!("CARGO_BIN_EXE_gemicro"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn gemicro CLI");

    if let Some(mut child_stdin) = child.stdin.take() {
        child_stdin.write_all(stdin.as_bytes()).ok();
    }

    child
        .wait_with_output()
        .expect("Failed to wait on gemicro CLI")
}

#[test]
#[ignore] // Requires GEMINI_API_KEY
fn test_cli_interactive_quit() {
    if !has_api_key() {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    }

    // Send /quit to exit the REPL immediately
    let output = run_cli_with_stdin(&["--interactive", "--agent", "deep_research"], "/quit\n");

    assert!(output.status.success(), "REPL should exit cleanly on /quit");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Goodbye"), "Should show goodbye message");
}

#[test]
#[ignore] // Requires GEMINI_API_KEY
fn test_cli_interactive_list_agents() {
    if !has_api_key() {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    }

    // List agents then quit
    let output = run_cli_with_stdin(
        &["--interactive", "--agent", "deep_research"],
        "/agent\n/quit\n",
    );

    assert!(output.status.success(), "REPL should exit cleanly");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("deep_research"),
        "Should list deep_research agent"
    );
    assert!(
        stdout.contains("Available agents"),
        "Should show agent list header"
    );
}

#[test]
#[ignore] // Requires GEMINI_API_KEY
fn test_cli_interactive_unknown_command() {
    if !has_api_key() {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    }

    // Try unknown command then quit
    let output = run_cli_with_stdin(
        &["--interactive", "--agent", "deep_research"],
        "/foobar\n/quit\n",
    );

    assert!(
        output.status.success(),
        "REPL should handle unknown commands gracefully"
    );

    // Unknown command message goes to stderr
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Unknown command"),
        "Should show unknown command message on stderr"
    );
}

#[test]
#[ignore] // Requires GEMINI_API_KEY
fn test_cli_interactive_simple_query() {
    if !has_api_key() {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    }

    // Run a simple query then quit
    // Note: Rustyline may not work perfectly with piped stdin in all environments
    let output = run_cli_with_stdin(
        &[
            "--interactive",
            "--agent",
            "deep_research",
            "--timeout",
            "120",
            "--min-sub-queries",
            "2",
            "--max-sub-queries",
            "2",
        ],
        "What is 1+1?\n/quit\n",
    );

    // Check that the REPL at least started (shows header or prompt)
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}{}", stdout, stderr);

    assert!(
        combined.contains("gemicro REPL")
            || combined.contains("deep_research")
            || combined.contains("Decomposing"),
        "Should show REPL activity. stdout: {}, stderr: {}",
        stdout,
        stderr
    );
}
