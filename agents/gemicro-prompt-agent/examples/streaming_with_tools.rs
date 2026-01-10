//! Streaming Example with Tools
//!
//! Demonstrates PromptAgent with function calling and real-time streaming.
//!
//! Run with:
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-prompt-agent --example streaming_with_tools
//!
//! This example shows:
//! 1. Creating a PromptAgent with a custom system prompt
//! 2. Registering tools (Calculator, CurrentDatetime)
//! 3. Executing with function calling
//! 4. Streaming events in real-time

use futures_util::StreamExt;
use gemicro_core::{Agent, AgentContext, LlmClient, LlmConfig, ToolRegistry};
use gemicro_prompt_agent::tools::{Calculator, CurrentDatetime};
use gemicro_prompt_agent::{PromptAgent, PromptAgentConfig};
use std::env;
use std::sync::Arc;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = env::var("GEMINI_API_KEY").expect(
        "GEMINI_API_KEY environment variable not set.\n\
         Set it with: export GEMINI_API_KEY=your_key",
    );

    println!("========================================");
    println!("  PromptAgent Streaming with Tools");
    println!("========================================\n");

    // Create the agent with a math-focused system prompt
    let config = PromptAgentConfig::default()
        .with_system_prompt(
            "You are a helpful assistant with access to tools. \
             Use the calculator for math and datetime for time queries. \
             Always show your work.",
        )
        .with_timeout(Duration::from_secs(60));

    let agent = PromptAgent::new(config)?;

    // Create LLM client
    let genai_client = genai_rs::Client::builder(api_key).build()?;
    let llm = LlmClient::new(
        genai_client,
        LlmConfig::default()
            .with_timeout(Duration::from_secs(60))
            .with_max_tokens(1024),
    );

    // Register tools
    let mut registry = ToolRegistry::new();
    registry.register(Calculator);
    registry.register(CurrentDatetime);

    println!("Registered tools:");
    for name in registry.list() {
        println!("  - {}", name);
    }
    println!();

    // Create context WITH tools
    let context = AgentContext::new(llm).with_tools_arc(Arc::new(registry));

    // Execute with a query that requires tool use
    let query = "What is 47 * 89, and what time is it right now?";
    println!("Query: {}\n", query);
    println!("--- Streaming Events ---\n");

    let stream = agent.execute(query, context);
    futures_util::pin_mut!(stream);

    // Stream events in real-time
    while let Some(update) = stream.next().await {
        match update {
            Ok(event) => {
                match event.event_type.as_str() {
                    "prompt_agent_started" => {
                        println!("[started] {}", event.message);
                    }
                    "prompt_agent_result" => {
                        println!("\n[result]");
                        println!("{}", event.message);
                    }
                    "final_result" => {
                        // Extract metadata
                        if let Some(tokens) = event.data.get("tokens_used") {
                            println!("\n--- Metadata ---");
                            println!("Tokens used: {}", tokens);
                        }
                    }
                    _ => {
                        // Log unknown events (demonstrates soft-typed handling)
                        println!("[{}] {}", event.event_type, event.message);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        }
    }

    println!("\n========================================");
    println!("  Demo Complete!");
    println!("========================================");

    Ok(())
}
