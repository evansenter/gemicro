//! Developer Agent Example
//!
//! Run with:
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-developer --example developer
//!
//! Or with a custom query:
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-developer --example developer -- "List files in src/"
//!
//! Press Ctrl+C to cancel gracefully.

use futures_util::StreamExt;
use gemicro_core::tool::{AutoApprove, ToolRegistry};
use gemicro_core::{Agent, AgentContext, LlmClient, LlmConfig};
use gemicro_developer::{DeveloperAgent, DeveloperConfig};
use gemicro_file_read::FileRead;
use gemicro_glob::Glob;
use serde_json::Value;
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

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

    println!("==============================================================");
    println!("                  Developer Agent Demo                       ");
    println!("==============================================================");
    println!();
    println!("Query: {}", query);
    println!();

    // Set up cancellation token for graceful shutdown
    let cancellation_token = CancellationToken::new();
    let token_clone = cancellation_token.clone();

    // Spawn signal handler for Ctrl+C
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            eprintln!("\nInterrupt received - cancelling...");
            token_clone.cancel();

            // Wait for second Ctrl+C to force exit
            if tokio::signal::ctrl_c().await.is_ok() {
                eprintln!("\nForce exit");
                std::process::exit(130);
            }
        }
    });

    // Create tool registry with read-only tools for this demo
    let mut tools = ToolRegistry::new();
    tools.register(FileRead);
    tools.register(Glob);

    // Create LLM client
    let genai_client = rust_genai::Client::builder(api_key).build();
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
    let config = DeveloperConfig::default().with_max_iterations(10); // Limit for demo
    let agent = DeveloperAgent::new(config)?;

    // Execute and stream updates
    let start = Instant::now();
    let stream = agent.execute(&query, context);
    futures_util::pin_mut!(stream);

    let mut tool_call_count = 0;

    while let Some(result) = stream.next().await {
        let update = result?;

        match update.event_type.as_str() {
            "developer_started" => {
                println!("Starting developer agent...");
                println!();
            }
            "tool_call_started" => {
                let tool_name = update.data["tool_name"].as_str().unwrap_or("unknown");
                let args = update.data.get("arguments").unwrap_or(&Value::Null);

                // Format args preview
                let args_preview = if let Some(path) = args["path"].as_str() {
                    path.to_string()
                } else if let Some(pattern) = args["pattern"].as_str() {
                    pattern.to_string()
                } else {
                    format!("{}", args)
                };

                println!("  {} {} ...", tool_name, args_preview);
                tool_call_count += 1;
            }
            "tool_result" => {
                let tool_name = update.data["tool_name"].as_str().unwrap_or("unknown");
                let success = update.data["success"].as_bool().unwrap_or(false);
                let duration_ms = update.data["duration_ms"].as_u64().unwrap_or(0);

                let status = if success { "OK" } else { "FAILED" };
                println!(
                    "    -> {} {} ({:.1}s)",
                    status,
                    tool_name,
                    duration_ms as f64 / 1000.0
                );
            }
            "final_result" => {
                println!();
                println!("==============================================================");
                println!("                         RESULT                              ");
                println!("==============================================================");
                println!();

                // Extract the answer
                if let Some(result) = update.as_final_result() {
                    if let Some(answer) = result.result.as_str() {
                        println!("{}", answer);
                    }
                    println!();
                    println!("--------------------------------------------------------------");
                    println!("Performance:");
                    println!("  Total time: {:.1}s", start.elapsed().as_secs_f64());
                    println!("  Tool calls: {}", tool_call_count);
                    if result.metadata.total_tokens > 0 {
                        println!("  Tokens used: {}", result.metadata.total_tokens);
                    }
                }
            }
            _ => {
                // Log unknown events at debug level
                log::debug!("Unknown event: {}", update.event_type);
            }
        }
    }

    Ok(())
}
