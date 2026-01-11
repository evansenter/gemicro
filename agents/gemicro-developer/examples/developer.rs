//! Developer Agent Example
//!
//! Run with:
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-developer --example developer
//!
//! Or with a custom query:
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-developer --example developer -- "List files in src/"
//!
//! For full wire-level debugging (API traffic + tool args/results):
//!   LOUD_WIRE=1 GEMINI_API_KEY=your_key cargo run -p gemicro-developer --example developer
//!
//! Press Ctrl+C to cancel gracefully.

use futures_util::StreamExt;
use gemicro_bash::Bash;
use gemicro_core::tool::{AutoApprove, ToolRegistry};
use gemicro_core::{Agent, AgentContext, AgentError, LlmClient, LlmConfig, MODEL};
use gemicro_developer::{DeveloperAgent, DeveloperConfig};
use gemicro_file_read::FileRead;
use gemicro_glob::Glob;
use gemicro_grep::Grep;
use serde_json::Value;
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// Truncate a string for display, adding ellipsis if needed.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = env::var("GEMINI_API_KEY").expect(
        "GEMINI_API_KEY environment variable not set.\n\
         Set it with: export GEMINI_API_KEY=your_key",
    );

    // Get query from args or use default
    let query = env::args().nth(1).unwrap_or_else(|| {
        "Read the CLAUDE.md file and summarize what this project is about".into()
    });

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                   Developer Agent Demo                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Show configuration
    println!("┌─ Configuration ─────────────────────────────────────────────┐");
    println!("│ Model: {:<54} │", MODEL);
    println!(
        "│ Working dir: {:<48} │",
        truncate(&env::current_dir()?.display().to_string(), 48)
    );
    println!("│ Max iterations: {:<45} │", 50);
    println!("│ LLM timeout: {:<48} │", "60s");
    println!("│ Temperature: {:<48} │", 0.7);
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    // Set up cancellation token for graceful shutdown
    let cancellation_token = CancellationToken::new();
    let token_clone = cancellation_token.clone();

    // Spawn signal handler for Ctrl+C
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            eprintln!("\n⚠ Interrupt received - cancelling gracefully...");
            token_clone.cancel();

            // Wait for second Ctrl+C to force exit
            if tokio::signal::ctrl_c().await.is_ok() {
                eprintln!("\n✗ Force exit");
                std::process::exit(130);
            }
        }
    });

    // Create tool registry with tools for this demo
    // Note: Bash requires confirmation (handled by AutoApprove for demos)
    let mut tools = ToolRegistry::new();
    tools.register(FileRead);
    tools.register(Glob);
    tools.register(Grep);
    tools.register(Bash);

    // Show registered tools
    println!("┌─ Registered Tools ──────────────────────────────────────────┐");
    for tool_name in tools.list() {
        println!("│ • {:<58} │", tool_name);
    }
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    // Create LLM client
    let genai_client = genai_rs::Client::builder(api_key)
        .build()
        .map_err(|e| AgentError::Other(e.to_string()))?;
    let llm_config = LlmConfig::default()
        .with_timeout(Duration::from_secs(60))
        .with_max_tokens(4096)
        .with_temperature(0.7);
    let llm = LlmClient::new(genai_client, llm_config);

    // Create context with tools and auto-approval (safe for read-only tools)
    let context = AgentContext::new_with_cancellation(llm, cancellation_token)
        .with_tools(tools)
        .with_confirmation_handler(Arc::new(AutoApprove));

    // Create developer agent
    let config = DeveloperConfig::default().with_max_iterations(50);
    let agent = DeveloperAgent::new(config)?;

    println!("┌─ Query ──────────────────────────────────────────────────────┐");
    // Word-wrap the query for display
    let wrapped = textwrap::wrap(&query, 60);
    for line in &wrapped {
        println!("│ {:<60} │", line);
    }
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    // Execute and stream updates
    println!("─── Execution ───");
    println!();
    let start = Instant::now();
    let stream = agent.execute(&query, context);
    futures_util::pin_mut!(stream);

    let mut tool_call_count = 0;

    while let Some(result) = stream.next().await {
        let update = result?;

        match update.event_type.as_str() {
            "developer_started" => {
                println!("▶ Agent started");
            }
            "tool_call_started" => {
                tool_call_count += 1;
                let tool_name = update.data["tool_name"].as_str().unwrap_or("unknown");
                let args = update.data.get("arguments").unwrap_or(&Value::Null);

                // Format args preview - show full paths
                let args_preview = if let Some(path) = args["path"].as_str() {
                    path.to_string()
                } else if let Some(pattern) = args["pattern"].as_str() {
                    pattern.to_string()
                } else {
                    truncate(&args.to_string(), 60)
                };

                println!();
                println!("🔧 [{:>2}] {} {}", tool_call_count, tool_name, args_preview);
                // Full args/results available via LOUD_WIRE=1
            }
            "tool_result" => {
                let tool_name = update.data["tool_name"].as_str().unwrap_or("unknown");
                let success = update.data["success"].as_bool().unwrap_or(false);
                let duration_ms = update.data["duration_ms"].as_u64().unwrap_or(0);
                let result = update.data.get("result").unwrap_or(&Value::Null);

                let status_icon = if success { "✓" } else { "✗" };
                let status_text = if success { "OK" } else { "FAILED" };

                println!(
                    "   {} {} {} ({:.2}s)",
                    status_icon,
                    status_text,
                    tool_name,
                    duration_ms as f64 / 1000.0
                );

                // Show errors inline (full results available via LOUD_WIRE=1)
                if !success {
                    let error_str = if let Some(err) = result.get("error") {
                        format!("Error: {}", err.as_str().unwrap_or("unknown"))
                    } else if let Some(content) = result.get("content") {
                        truncate(content.as_str().unwrap_or(&content.to_string()), 80)
                    } else {
                        truncate(&result.to_string(), 80)
                    };
                    println!("     {}", error_str);
                }
            }
            "final_result" => {
                println!();
                println!();

                println!("╔══════════════════════════════════════════════════════════════╗");
                println!("║                          RESULT                              ║");
                println!("╚══════════════════════════════════════════════════════════════╝");
                println!();

                // Extract the answer
                if let Some(result) = update.as_final_result() {
                    if let Some(answer) = result.result.as_str() {
                        println!("{}", answer);
                    } else {
                        println!("{}", result.result);
                    }
                    println!();

                    // Show metadata
                    println!("┌─ Execution Summary ─────────────────────────────────────────┐");
                    println!(
                        "│ Total time: {:<49} │",
                        format!("{:.2}s", start.elapsed().as_secs_f64())
                    );
                    println!("│ Tool calls: {:<49} │", tool_call_count);

                    // Extract iterations from metadata extra
                    let extra = &result.metadata.extra;
                    if let Some(iterations) = extra.get("iterations") {
                        println!(
                            "│ LLM iterations: {:<45} │",
                            iterations.as_u64().unwrap_or(0)
                        );
                    }
                    if let Some(incomplete) = extra.get("incomplete") {
                        if incomplete.as_bool().unwrap_or(false) {
                            let reason = extra
                                .get("reason")
                                .and_then(|r| r.as_str())
                                .unwrap_or("unknown");
                            println!("│ ⚠ Incomplete: {:<47} │", reason);
                        }
                    }

                    if result.metadata.total_tokens > 0 {
                        println!("│ Tokens used: {:<48} │", result.metadata.total_tokens);
                    }
                    println!(
                        "│ Duration (agent): {:<43} │",
                        format!("{}ms", result.metadata.duration_ms)
                    );
                    println!("└──────────────────────────────────────────────────────────────┘");
                }
            }
            _ => {
                // Unknown events are silently ignored
                // (full event stream available via LOUD_WIRE=1)
            }
        }
    }

    Ok(())
}
