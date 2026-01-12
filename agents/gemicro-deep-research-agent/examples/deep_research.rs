//! Deep Research Agent Example
//!
//! Run with:
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-deep-research-agent --example deep_research
//!
//! Or with a custom query:
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-deep-research-agent --example deep_research -- "Your question here"
//!
//! Press Ctrl+C to cancel gracefully - partial results will be shown.
//!
//! Note: For real-time web data, use the CLI with `--google-search`:
//!   gemicro "What happened in tech news today?" --google-search

use futures_util::StreamExt;
use gemicro_core::{first_sentence, truncate, AgentContext, AgentError, LlmClient, LlmConfig};
use gemicro_deep_research_agent::{
    DeepResearchAgent, DeepResearchAgentConfig, DeepResearchEventExt,
};
use std::collections::HashMap;
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
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
    let query = env::args()
        .nth(1)
        .unwrap_or_else(|| "What are the main benefits of the Rust programming language?".into());

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              Deep Research Agent Demo                        ║");
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
            eprintln!("   (Press Ctrl+C again to exit immediately)\n");
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
    let genai_client = genai_rs::Client::builder(api_key)
        .build()
        .map_err(|e| AgentError::Other(e.to_string()))?;
    let llm_config = LlmConfig::default()
        .with_timeout(Duration::from_secs(60))
        .with_max_tokens(1024)
        .with_temperature(0.7)
        .with_max_retries(2)
        .with_retry_base_delay_ms(1000);
    let llm = LlmClient::new(genai_client, llm_config);
    let context = AgentContext::new_with_cancellation(llm, cancellation_token);

    // Create agent with config
    let research_config = DeepResearchAgentConfig::default()
        .with_min_sub_queries(3)
        .with_max_sub_queries(5)
        .with_continue_on_partial_failure(true)
        .with_total_timeout(Duration::from_secs(180));
    let agent = DeepResearchAgent::new(research_config)?;

    // Execute and stream results
    let stream = agent.execute(&query, context);
    futures_util::pin_mut!(stream);

    // Track state for enhanced output
    let mut sub_queries: Vec<String> = Vec::new();
    let mut sub_query_start_times: HashMap<usize, Instant> = HashMap::new();
    let mut findings: Vec<(usize, String, Duration)> = Vec::new(); // (id, result, duration)
    let overall_start = Instant::now();

    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => match update.event_type.as_str() {
                "decomposition_started" => {
                    println!("🔍 Analyzing query and generating research plan...");
                }
                "decomposition_complete" => {
                    if let Some(queries) = update.as_decomposition_complete() {
                        sub_queries = queries.clone();
                        println!();
                        println!("┌─────────────────────────────────────────────────────────────┐");
                        println!(
                            "│ 📋 RESEARCH PLAN ({} sub-queries)                            │",
                            sub_queries.len()
                        );
                        println!("├─────────────────────────────────────────────────────────────┤");
                        for (i, q) in sub_queries.iter().enumerate() {
                            // Wrap long queries
                            let display = truncate(q, 55);
                            println!(
                                "│ {}. {}{}│",
                                i + 1,
                                display,
                                " ".repeat(56 - display.chars().count())
                            );
                        }
                        println!("└─────────────────────────────────────────────────────────────┘");
                        println!();
                        println!("⚡ Executing {} queries in parallel...", sub_queries.len());
                        println!();
                    }
                }
                "sub_query_started" => {
                    if let Some(id) = update.data.get("id").and_then(|v| v.as_u64()) {
                        sub_query_start_times.insert(id as usize, Instant::now());
                        let query_text = sub_queries
                            .get(id as usize)
                            .map(|q| truncate(q, 50))
                            .unwrap_or_else(|| "...".to_string());
                        println!("   ⏳ [{}] {}", id + 1, query_text);
                    }
                }
                "sub_query_completed" => {
                    if let Some(result) = update.as_sub_query_completed() {
                        let duration = sub_query_start_times
                            .get(&result.id)
                            .map(|start| start.elapsed())
                            .unwrap_or_default();

                        let duration_str = format!("{:.1}s", duration.as_secs_f64());
                        let preview = first_sentence(&result.result);

                        println!(
                            "   ✅ [{}] {} → \"{}\"",
                            result.id + 1,
                            duration_str,
                            preview
                        );

                        findings.push((result.id, result.result.clone(), duration));
                    }
                }
                "sub_query_failed" => {
                    if let Some(id) = update.data.get("id").and_then(|v| v.as_u64()) {
                        let duration = sub_query_start_times
                            .get(&(id as usize))
                            .map(|start| start.elapsed())
                            .unwrap_or_default();

                        let err = update
                            .data
                            .get("error")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Unknown error");

                        println!(
                            "   ❌ [{}] {:.1}s → Failed: {}",
                            id + 1,
                            duration.as_secs_f64(),
                            truncate(err, 50)
                        );
                    }
                }
                "synthesis_started" => {
                    println!();

                    // Show findings summary before synthesis
                    if !findings.is_empty() {
                        println!("┌─────────────────────────────────────────────────────────────┐");
                        println!("│ 📚 FINDINGS SUMMARY                                         │");
                        println!("├─────────────────────────────────────────────────────────────┤");

                        // Sort findings by ID for consistent display
                        findings.sort_by_key(|(id, _, _)| *id);

                        for (id, result, duration) in &findings {
                            let query_text = sub_queries
                                .get(*id)
                                .map(|q| truncate(q, 40))
                                .unwrap_or_else(|| "Query".to_string());

                            // Show first meaningful line of the result
                            let first_line = result
                                .lines()
                                .find(|line| !line.trim().is_empty() && line.len() > 10)
                                .unwrap_or("(result available)");
                            let preview = truncate(first_line, 55);

                            println!(
                                "│                                                             │"
                            );
                            println!(
                                "│ {}. {} ({:.1}s){}│",
                                id + 1,
                                query_text,
                                duration.as_secs_f64(),
                                " ".repeat(55 - query_text.chars().count() - 7)
                            );
                            println!(
                                "│    → {}{}│",
                                preview,
                                " ".repeat(55 - preview.chars().count())
                            );
                        }
                        println!("│                                                             │");
                        println!("└─────────────────────────────────────────────────────────────┘");
                        println!();
                    }

                    println!(
                        "🧠 Synthesizing {} findings into comprehensive answer...",
                        findings.len()
                    );
                }
                "final_result" => {
                    if let Some(result) = update.as_final_result() {
                        let total_duration = overall_start.elapsed();
                        let answer = result.result.as_str().unwrap_or("");

                        println!();
                        println!(
                            "╔══════════════════════════════════════════════════════════════╗"
                        );
                        println!(
                            "║                     SYNTHESIZED ANSWER                       ║"
                        );
                        println!(
                            "╚══════════════════════════════════════════════════════════════╝"
                        );
                        println!();
                        println!("{}", answer);
                        println!();
                        println!("══════════════════════════════════════════════════════════════");

                        let steps_succeeded = result.metadata.extra["steps_succeeded"]
                            .as_u64()
                            .unwrap_or(0);
                        let steps_failed =
                            result.metadata.extra["steps_failed"].as_u64().unwrap_or(0);
                        let total_queries = steps_succeeded + steps_failed;

                        // Calculate parallel efficiency
                        let sequential_time: f64 =
                            findings.iter().map(|(_, _, d)| d.as_secs_f64()).sum();
                        let parallel_time = total_duration.as_secs_f64();
                        let speedup = if parallel_time > 0.0 {
                            sequential_time / parallel_time
                        } else {
                            1.0
                        };

                        println!("📊 Performance:");
                        println!("   Total time: {:.1}s", parallel_time);
                        println!(
                            "   Sub-queries: {}/{} succeeded",
                            steps_succeeded, total_queries
                        );
                        if speedup > 1.1 {
                            println!(
                                "   Parallel speedup: {:.1}x (saved {:.1}s)",
                                speedup,
                                sequential_time - parallel_time
                            );
                        }

                        // Show token info only if available
                        if result.metadata.tokens_unavailable_count == 0
                            && result.metadata.total_tokens > 0
                        {
                            println!("   Tokens used: {}", result.metadata.total_tokens);
                        }
                    }
                }
                _ => {}
            },
            Err(AgentError::Cancelled) => {
                // Graceful cancellation - show what we have
                println!();
                println!("══════════════════════════════════════════════════════════════");
                println!("⚠️  Research cancelled");
                if !findings.is_empty() {
                    println!(
                        "   Collected {} partial results before cancellation",
                        findings.len()
                    );
                }
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
