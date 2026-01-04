//! Example: Using PathSandbox to restrict tool file access.
//!
//! This example demonstrates how PathSandbox intercepts tool calls and
//! makes allow/deny decisions based on the sandbox configuration.
//!
//! # Running
//!
//! ```bash
//! cargo run -p gemicro-path-sandbox --example sandboxed_tools
//! ```

use gemicro_core::interceptor::{InterceptDecision, Interceptor, InterceptorChain, ToolCall};
use gemicro_core::tool::{AutoApprove, GemicroToolService, ToolRegistry, ToolResult};
use gemicro_path_sandbox::PathSandbox;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // Set up logging to see sandbox decisions
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("=== PathSandbox Example ===\n");

    // 1. Create a sandbox that only allows access to /tmp
    let sandbox = PathSandbox::new(vec![PathBuf::from("/tmp")]);
    println!("Created sandbox allowing: /tmp\n");

    // 2. Demonstrate actual interception with ToolCall objects
    println!("--- Testing Interception Decisions ---\n");

    let test_cases = vec![
        (
            "file_read",
            json!({"path": "/tmp/allowed.txt"}),
            "in sandbox",
        ),
        (
            "file_read",
            json!({"path": "/etc/passwd"}),
            "outside sandbox",
        ),
        (
            "file_write",
            json!({"path": "/tmp/new.txt"}),
            "new file in sandbox",
        ),
        (
            "file_read",
            json!({"path": "/tmp/../etc/passwd"}),
            "path traversal attempt",
        ),
        (
            "grep",
            json!({"path": "/var/log/syslog", "pattern": "error"}),
            "grep outside sandbox",
        ),
        (
            "glob",
            json!({"pattern": "*.txt", "base_dir": "/tmp"}),
            "glob in sandbox",
        ),
        (
            "bash",
            json!({"command": "ls /"}),
            "non-file tool (passthrough)",
        ),
    ];

    for (tool_name, arguments, description) in test_cases {
        let call = ToolCall::new(tool_name, arguments.clone());
        let decision = sandbox.intercept(&call).await.unwrap();

        let status = match &decision {
            InterceptDecision::Allow => "ALLOWED",
            InterceptDecision::Deny { .. } => "DENIED ",
            InterceptDecision::Transform { .. } => "TRANSFORMED",
            InterceptDecision::Confirm { .. } => "CONFIRM",
        };

        println!(
            "  {}: {}({}) -> {}",
            description, tool_name, arguments, status
        );

        if let InterceptDecision::Deny { reason } = decision {
            println!("    Reason: {}", reason);
        }
        println!();
    }

    // 3. Show how to wire into a tool service
    println!("--- Wiring into Tool Service ---\n");

    let interceptors: InterceptorChain<ToolCall, ToolResult> =
        InterceptorChain::new().with(sandbox);

    let registry = ToolRegistry::new();
    let _service = GemicroToolService::new(Arc::new(registry))
        .with_interceptors(Arc::new(interceptors))
        .with_confirmation_handler(Arc::new(AutoApprove));

    println!("Tool service configured with sandbox protection.");
    println!("All file operations will be intercepted before execution.\n");

    // 4. Defense-in-depth recommendation
    println!("--- Defense-in-Depth ---\n");
    println!("Tip: Combine PathSandbox (whitelist) with FileSecurity (blacklist):");
    println!("  - PathSandbox: Only /workspace allowed");
    println!("  - FileSecurity: Block /workspace/.git, /workspace/.env");
    println!("  - Result: Agent accesses only /workspace, excluding sensitive files");
}
