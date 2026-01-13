//! Subagent Delegation Example
//!
//! Demonstrates DeveloperAgent delegating to subagents via the Task tool.
//!
//! Run with:
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-developer-agent --example subagent_delegation
//!
//! Or with a custom query:
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-developer-agent --example subagent_delegation -- \
//!     "Research the tradeoffs between tokio and async-std"
//!
//! Self-critique example (validates work against conventions):
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-developer-agent --example subagent_delegation -- \
//!     "Review the changes in src/lib.rs and validate against project conventions"
//!
//! For full wire-level debugging:
//!   LOUD_WIRE=1 GEMINI_API_KEY=your_key cargo run -p gemicro-developer-agent --example subagent_delegation

use futures_util::StreamExt;
use gemicro_bash::Bash;
use gemicro_core::tool::{AutoApprove, ToolRegistry};
use gemicro_core::{Agent, AgentContext, AgentError, LlmClient, LlmConfig};
use gemicro_critique_agent::CritiqueAgent;
use gemicro_deep_research_agent::{DeepResearchAgent, DeepResearchAgentConfig};
use gemicro_developer_agent::{DeveloperAgent, DeveloperAgentConfig};
use gemicro_file_read::FileRead;
use gemicro_glob::Glob;
use gemicro_grep::Grep;
use gemicro_prompt_agent::{PromptAgent, PromptAgentConfig};
use gemicro_runner::AgentRegistry;
use gemicro_task::Task;
use serde_json::Value;
use std::env;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

fn truncate(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        format!(
            "{}...",
            s.chars()
                .take(max_len.saturating_sub(3))
                .collect::<String>()
        )
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = env::var("GEMINI_API_KEY").expect(
        "GEMINI_API_KEY environment variable not set.\n\
         Set it with: export GEMINI_API_KEY=your_key",
    );

    // Query that encourages subagent delegation
    let query = env::args().nth(1).unwrap_or_else(|| {
        "Research the key differences between sync and async Rust for CLI applications. \
         Synthesize your findings into a recommendation."
            .into()
    });

    println!("========================================");
    println!("   Subagent Delegation Demo");
    println!("========================================");
    println!();
    println!("Model: gemini-3-flash-preview (default)");
    println!();

    let cancellation_token = CancellationToken::new();
    let token_clone = cancellation_token.clone();

    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            eprintln!("\nInterrupt received - cancelling...");
            token_clone.cancel();
            if tokio::signal::ctrl_c().await.is_ok() {
                std::process::exit(130);
            }
        }
    });

    // Create shared agent registry
    let registry = Arc::new(RwLock::new(AgentRegistry::new()));

    // Create LLM config (shared across all agent factories)
    let llm_config = LlmConfig::default()
        .with_timeout(Duration::from_secs(120))
        .with_max_tokens(4096)
        .with_temperature(0.7);

    // Create LLM client for Task tool (needs Arc for sharing)
    let genai_client_for_task = genai_rs::Client::builder(api_key.clone())
        .build()
        .map_err(|e| AgentError::Other(e.to_string()))?;
    let llm_for_task = Arc::new(LlmClient::new(genai_client_for_task, llm_config.clone()));

    // Register subagents using factory closures
    {
        let mut reg = registry.write().unwrap();

        // Deep Research agent for complex research queries
        let research_config = DeepResearchAgentConfig::default()
            .with_min_sub_queries(2)
            .with_max_sub_queries(4);
        reg.register("deep_research", move || {
            Box::new(DeepResearchAgent::new(research_config.clone()).unwrap()) as Box<dyn Agent>
        });

        // Prompt agent for simpler prompt-based tasks
        let prompt_config = PromptAgentConfig::default();
        reg.register("prompt_agent", move || {
            Box::new(PromptAgent::new(prompt_config.clone()).unwrap()) as Box<dyn Agent>
        });

        // Critique agent for self-validation against project conventions
        reg.register("critique", || {
            Box::new(CritiqueAgent::default_agent()) as Box<dyn Agent>
        });

        println!("Registered agents:");
        for name in reg.list() {
            println!("  - {}", name);
        }
    }
    println!();

    // Create tool registry with Task tool for subagent spawning
    let mut tools = ToolRegistry::new();
    tools.register(FileRead);
    tools.register(Glob);
    tools.register(Grep);
    tools.register(Bash);
    let (task, _task_context) = Task::new(Arc::clone(&registry), llm_for_task);
    tools.register(task);

    println!("Available tools:");
    for tool_name in tools.list() {
        println!("  - {}", tool_name);
    }
    println!();

    // Create a separate LLM client for the agent context
    let genai_client = genai_rs::Client::builder(api_key)
        .build()
        .map_err(|e| AgentError::Other(e.to_string()))?;
    let llm = LlmClient::new(genai_client, llm_config);

    // Create context
    let context = AgentContext::new_with_cancellation(llm, cancellation_token)
        .with_tools(tools)
        .with_confirmation_handler(Arc::new(AutoApprove));

    // Create developer agent
    let config = DeveloperAgentConfig::default().with_max_iterations(30);
    let agent = DeveloperAgent::new(config)?;

    println!("Query: {}", truncate(&query, 70));
    println!();
    println!("--- Execution ---");
    println!();

    let start = Instant::now();
    let stream = agent.execute(&query, context);
    futures_util::pin_mut!(stream);

    let mut tool_call_count = 0;
    let mut subagent_depth = 0;

    while let Some(result) = stream.next().await {
        let update = result?;
        let indent = "    ".repeat(subagent_depth);

        match update.event_type.as_str() {
            "developer_started" => {
                println!("{}> Developer agent started", indent);
            }
            "tool_call_started" => {
                tool_call_count += 1;
                let tool_name = update.data["tool_name"].as_str().unwrap_or("unknown");
                let args = update.data.get("arguments").unwrap_or(&Value::Null);

                let args_preview = if tool_name == "task" {
                    let agent = args["agent"].as_str().unwrap_or("?");
                    let query = args["query"]
                        .as_str()
                        .map(|s| truncate(s, 40))
                        .unwrap_or_default();
                    format!("{}: {}", agent, query)
                } else if let Some(path) = args["path"].as_str() {
                    path.to_string()
                } else {
                    truncate(&args.to_string(), 50)
                };

                println!(
                    "{}[{}] {} {}",
                    indent, tool_call_count, tool_name, args_preview
                );
            }
            "tool_result" => {
                let success = update.data["success"].as_bool().unwrap_or(false);
                let duration_ms = update.data["duration_ms"].as_u64().unwrap_or(0);
                let icon = if success { "+" } else { "x" };
                println!("{}  {} ({:.1}s)", indent, icon, duration_ms as f64 / 1000.0);
            }
            "subagent_started" => {
                let agent_name = update.data["agent"].as_str().unwrap_or("unknown");
                let query_preview = update.data["query_preview"].as_str().unwrap_or("");
                println!();
                println!("{}>>> Delegating to {}", indent, agent_name);
                if !query_preview.is_empty() {
                    println!("{}    Query: {}", indent, query_preview);
                }
                subagent_depth += 1;
            }
            "subagent_completed" => {
                subagent_depth = subagent_depth.saturating_sub(1);
                let agent_name = update.data["agent"].as_str().unwrap_or("unknown");
                let success = update.data["success"].as_bool().unwrap_or(false);
                let duration_ms = update.data["duration_ms"].as_u64().unwrap_or(0);
                let icon = if success { "+++" } else { "xxx" };
                println!(
                    "{}<<< {} {} ({:.1}s)",
                    indent,
                    agent_name,
                    icon,
                    duration_ms as f64 / 1000.0
                );
                println!();
            }
            "final_result" => {
                println!();
                println!("========================================");
                println!("   RESULT");
                println!("========================================");
                println!();

                if let Some(result) = update.as_final_result() {
                    if let Some(answer) = result.result.as_str() {
                        println!("{}", answer);
                    } else {
                        println!("{}", result.result);
                    }
                    println!();

                    println!("--- Summary ---");
                    println!("Time: {:.1}s", start.elapsed().as_secs_f64());
                    println!("Tool calls: {}", tool_call_count);
                    if result.metadata.total_tokens > 0 {
                        println!("Tokens: {}", result.metadata.total_tokens);
                    }
                }
            }
            _ => {
                // Other events (sub_query_started, etc.) from subagents
                // can be logged at debug level if needed
            }
        }
    }

    Ok(())
}
