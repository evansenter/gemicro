//! Example: Using PathSandbox to restrict tool file access.
//!
//! This example demonstrates how to wire PathSandbox into a tool execution
//! pipeline to restrict file operations to specific directories.
//!
//! # Running
//!
//! ```bash
//! cargo run -p gemicro-path-sandbox --example sandboxed_tools
//! ```

use gemicro_core::interceptor::{InterceptorChain, ToolCall};
use gemicro_core::tool::{AutoApprove, GemicroToolService, ToolRegistry, ToolResult};
use gemicro_path_sandbox::PathSandbox;
use std::path::PathBuf;
use std::sync::Arc;

fn main() {
    // Set up logging to see sandbox decisions
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("=== PathSandbox Example ===\n");

    // 1. Create a sandbox that only allows access to /tmp
    let sandbox = PathSandbox::new(vec![PathBuf::from("/tmp")]);
    println!("Created sandbox allowing: /tmp");
    println!();

    // 2. Build an interceptor chain with the sandbox
    let interceptors: InterceptorChain<ToolCall, ToolResult> =
        InterceptorChain::new().with(sandbox);
    println!("Interceptor chain configured");
    println!();

    // 3. Create a tool service with the interceptors
    //
    // In a real application, you'd register your tools here.
    // The sandbox will intercept file_read, file_write, file_edit,
    // grep, and glob operations.
    let registry = ToolRegistry::new();
    let _service = GemicroToolService::new(Arc::new(registry))
        .with_interceptors(Arc::new(interceptors))
        .with_confirmation_handler(Arc::new(AutoApprove));

    println!("Tool service ready with sandbox protection");
    println!();

    // 4. Demonstrate what would happen with different paths
    //
    // When agents try to use file tools, the sandbox intercepts:
    //
    // file_read("/tmp/allowed.txt")      -> ALLOWED (in sandbox)
    // file_read("/etc/passwd")           -> DENIED  (outside sandbox)
    // file_write("/tmp/new.txt")         -> ALLOWED (in sandbox)
    // file_write("/home/user/.ssh/key")  -> DENIED  (outside sandbox)
    // file_read("/tmp/../etc/passwd")    -> DENIED  (traversal detected)

    println!("Example decisions:");
    println!("  /tmp/allowed.txt        -> ALLOWED (in sandbox)");
    println!("  /etc/passwd             -> DENIED  (outside sandbox)");
    println!("  /tmp/../etc/passwd      -> DENIED  (traversal detected)");
    println!();

    // 5. For production use, combine with FileSecurity for defense-in-depth
    //
    // PathSandbox: Whitelist - only /workspace allowed
    // FileSecurity: Blacklist - block /workspace/.git, /workspace/.env
    //
    // This gives you:
    // - Agent can only access /workspace (sandbox)
    // - Within /workspace, sensitive files are still protected (security)

    println!("Tip: Combine PathSandbox (whitelist) with FileSecurity (blacklist)");
    println!("     for defense-in-depth protection.");
}
