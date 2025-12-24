//! Deep Research Agent Example
//!
//! Run with:
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-core --example deep_research
//!
//! Or with a custom query:
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-core --example deep_research -- "Your question here"

use futures_util::StreamExt;
use gemicro_core::{
    AgentContext, DeepResearchAgent, LlmClient, LlmConfig, ResearchConfig,
    EVENT_DECOMPOSITION_COMPLETE, EVENT_DECOMPOSITION_STARTED, EVENT_FINAL_RESULT,
    EVENT_SUB_QUERY_COMPLETED, EVENT_SUB_QUERY_FAILED, EVENT_SUB_QUERY_STARTED,
    EVENT_SYNTHESIS_STARTED,
};
use std::collections::HashMap;
use std::env;
use std::time::{Duration, Instant};

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

/// Extract the first sentence or line from text
fn first_sentence(s: &str) -> String {
    let s = s.trim();
    // Try to find end of first sentence
    if let Some(pos) = s.find(['.', '\n']) {
        let sentence = s[..=pos].trim();
        if sentence.len() > 10 {
            return truncate(sentence, 100);
        }
    }
    truncate(s, 100)
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
        .unwrap_or_else(|| "What are the main benefits of the Rust programming language?".into());

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              Deep Research Agent Demo                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Query: {}", query);
    println!();

    // Create LLM client
    let genai_client = rust_genai::Client::builder(api_key).build();
    let llm_config = LlmConfig {
        timeout: Duration::from_secs(60),
        max_tokens: 1024,
        temperature: 0.7,
        max_retries: 2,
        retry_base_delay_ms: 1000,
    };
    let llm = LlmClient::new(genai_client, llm_config);
    let context = AgentContext::new(llm);

    // Create agent with config
    let research_config = ResearchConfig {
        min_sub_queries: 3,
        max_sub_queries: 5,
        continue_on_partial_failure: true,
        total_timeout: Duration::from_secs(180),
        ..Default::default()
    };
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
                EVENT_DECOMPOSITION_STARTED => {
                    println!("🔍 Analyzing query and generating research plan...");
                }
                EVENT_DECOMPOSITION_COMPLETE => {
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
                EVENT_SUB_QUERY_STARTED => {
                    if let Some(id) = update.data.get("id").and_then(|v| v.as_u64()) {
                        sub_query_start_times.insert(id as usize, Instant::now());
                        let query_text = sub_queries
                            .get(id as usize)
                            .map(|q| truncate(q, 50))
                            .unwrap_or_else(|| "...".to_string());
                        println!("   ⏳ [{}] {}", id + 1, query_text);
                    }
                }
                EVENT_SUB_QUERY_COMPLETED => {
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
                EVENT_SUB_QUERY_FAILED => {
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
                EVENT_SYNTHESIS_STARTED => {
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
                EVENT_FINAL_RESULT => {
                    if let Some(result) = update.as_final_result() {
                        let total_duration = overall_start.elapsed();

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
                        println!("{}", result.answer);
                        println!();
                        println!("══════════════════════════════════════════════════════════════");

                        let total_queries = result.metadata.sub_queries_succeeded
                            + result.metadata.sub_queries_failed;

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
                            result.metadata.sub_queries_succeeded, total_queries
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
            Err(e) => {
                eprintln!("❌ Error: {:?}", e);
                return Err(e.into());
            }
        }
    }

    Ok(())
}
