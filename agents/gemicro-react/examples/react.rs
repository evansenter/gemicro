//! ReAct Agent Example
//!
//! Demonstrates the Reasoning + Acting pattern where the agent iteratively
//! thinks, acts with tools (calculator, web search), and observes results.
//!
//! Run with:
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-react --example react
//!
//! Or with a custom query:
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-react --example react -- "Your question here"
//!
//! Available tools:
//!   - calculator: Evaluates mathematical expressions
//!   - web_search: Searches the web using Google grounding (requires API support)
//!
//! Press Ctrl+C to cancel gracefully.

use futures_util::StreamExt;
use gemicro_core::{AgentContext, AgentError, LlmClient, LlmConfig};
use gemicro_react::{ReactAgent, ReactConfig};
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// Truncate text to a maximum length, adding ellipsis if needed
fn truncate(s: &str, max_chars: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars - 3).collect();
        format!("{}...", truncated.trim_end())
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
    let query = env::args()
        .nth(1)
        .unwrap_or_else(|| "What is 25 * 4 + 100 / 5?".into());

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                   ReAct Agent Demo                           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Pattern: Think → Act → Observe → Repeat                      ║");
    println!("║ Tools: calculator, web_search                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Query: {}", query);
    println!();

    // Set up cancellation token for graceful shutdown
    let cancellation_token = CancellationToken::new();
    let token_clone = cancellation_token.clone();
    let interrupted = Arc::new(AtomicBool::new(false));
    let interrupted_clone = interrupted.clone();

    // Spawn signal handler for Ctrl+C
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            eprintln!("\n⚠️  Interrupt received - cancelling...");
            interrupted_clone.store(true, Ordering::SeqCst);
            token_clone.cancel();

            // Wait for second Ctrl+C to force exit
            if tokio::signal::ctrl_c().await.is_ok() {
                eprintln!("\n❌ Force exit");
                std::process::exit(130);
            }
        }
    });

    // Create LLM client with cancellation support
    let genai_client = rust_genai::Client::builder(api_key).build();
    let llm_config = LlmConfig::default()
        .with_timeout(Duration::from_secs(60))
        .with_max_tokens(1024)
        .with_temperature(0.0) // Lower temperature for more consistent tool use
        .with_max_retries(2)
        .with_retry_base_delay_ms(1000);
    let llm = LlmClient::new(genai_client, llm_config);
    let context = AgentContext::new_with_cancellation(llm, cancellation_token);

    // Create agent with config
    let react_config = ReactConfig::default()
        .with_max_iterations(10)
        .with_available_tools(vec!["calculator".to_string(), "web_search".to_string()])
        .with_use_google_search(true) // Enable for web_search tool
        .with_total_timeout(Duration::from_secs(120));
    let agent = ReactAgent::new(react_config)?;

    // Execute and stream results
    let stream = agent.execute(&query, context);
    futures_util::pin_mut!(stream);

    let overall_start = Instant::now();
    let mut iteration_count = 0;

    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => match update.event_type.as_str() {
                "react_started" => {
                    if let Some(max) = update.data.get("max_iterations").and_then(|v| v.as_u64()) {
                        println!("🔄 Starting ReAct loop (max {} iterations)", max);
                        println!();
                    }
                }
                "react_thought" => {
                    if let Some(iteration) = update.data.get("iteration").and_then(|v| v.as_u64()) {
                        let thought = update
                            .data
                            .get("thought")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");

                        iteration_count = iteration as usize;
                        println!("┌─────────────────────────────────────────────────────────────┐");
                        println!(
                            "│ 💭 Iteration {}                                              │",
                            iteration
                        );
                        println!("├─────────────────────────────────────────────────────────────┤");
                        println!("│ Thought: {}│", format_box_line(&truncate(thought, 48)));
                    }
                }
                "react_action" => {
                    let tool = update
                        .data
                        .get("tool")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let input = update
                        .data
                        .get("input")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    let icon = match tool {
                        "calculator" => "🔢",
                        "web_search" => "🌐",
                        "final_answer" => "✅",
                        _ => "🔧",
                    };

                    println!(
                        "│ Action:  {} {}[{}]│",
                        icon,
                        tool,
                        format_input(&truncate(input, 40))
                    );
                }
                "react_observation" => {
                    let result_text = update
                        .data
                        .get("result")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let is_error = update
                        .data
                        .get("is_error")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);

                    if is_error {
                        println!(
                            "│ Observe: ❌ {}│",
                            format_box_line(&truncate(result_text, 46))
                        );
                    } else {
                        println!(
                            "│ Observe: ✓ {}│",
                            format_box_line(&truncate(result_text, 47))
                        );
                    }
                    println!("└─────────────────────────────────────────────────────────────┘");
                    println!();
                }
                "react_complete" => {
                    let iterations = update
                        .data
                        .get("iterations_used")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(iteration_count as u64);
                    let answer = update
                        .data
                        .get("final_answer")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    let duration = overall_start.elapsed();

                    println!("╔══════════════════════════════════════════════════════════════╗");
                    println!("║                       FINAL ANSWER                           ║");
                    println!("╚══════════════════════════════════════════════════════════════╝");
                    println!();
                    println!("{}", answer);
                    println!();
                    println!("══════════════════════════════════════════════════════════════");
                    println!("📊 Performance:");
                    println!("   Iterations: {}", iterations);
                    println!("   Total time: {:.1}s", duration.as_secs_f64());
                }
                "react_max_iterations" => {
                    let max = update
                        .data
                        .get("max_iterations")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let last_thought = update
                        .data
                        .get("last_thought")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    println!();
                    println!("⚠️  Max iterations ({}) reached without final answer", max);
                    if !last_thought.is_empty() {
                        println!("   Last thought: {}", truncate(last_thought, 60));
                    }
                }
                _ => {}
            },
            Err(AgentError::Cancelled) => {
                println!();
                println!("══════════════════════════════════════════════════════════════");
                println!("⚠️  ReAct loop cancelled at iteration {}", iteration_count);
                println!("══════════════════════════════════════════════════════════════");
                break;
            }
            Err(e) => {
                eprintln!("❌ Error: {:?}", e);
                return Err(e.into());
            }
        }
    }

    // Exit with SIGINT code if interrupted
    if interrupted.load(Ordering::SeqCst) {
        std::process::exit(130);
    }

    Ok(())
}

/// Format text to fit in a 59-char box line with right padding
fn format_box_line(s: &str) -> String {
    let width = 48;
    let len = s.chars().count();
    if len >= width {
        s.to_string()
    } else {
        format!("{}{}", s, " ".repeat(width - len))
    }
}

/// Format input text with appropriate padding
fn format_input(s: &str) -> String {
    let width = 40;
    let len = s.chars().count();
    if len >= width {
        s.to_string()
    } else {
        format!("{}{}", s, " ".repeat(width - len))
    }
}
