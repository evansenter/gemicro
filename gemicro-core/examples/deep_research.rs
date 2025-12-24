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
use std::env;
use std::time::Duration;

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
    };
    let agent = DeepResearchAgent::new(research_config)?;

    // Execute and stream results
    let stream = agent.execute(&query, context);
    futures_util::pin_mut!(stream);

    let mut sub_query_count = 0;

    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => match update.event_type.as_str() {
                EVENT_DECOMPOSITION_STARTED => {
                    println!("🔍 Decomposing query into sub-questions...");
                }
                EVENT_DECOMPOSITION_COMPLETE => {
                    if let Some(sub_queries) = update.as_decomposition_complete() {
                        sub_query_count = sub_queries.len();
                        println!("📋 Generated {} sub-queries:", sub_query_count);
                        for (i, q) in sub_queries.iter().enumerate() {
                            println!("   {}. {}", i + 1, q);
                        }
                        println!();
                    }
                }
                EVENT_SUB_QUERY_STARTED => {
                    if let Some(id) = update.data.get("id").and_then(|v| v.as_u64()) {
                        if let Some(q) = update.data.get("query").and_then(|v| v.as_str()) {
                            println!("⏳ [{}/{}] Researching: {}...", id + 1, sub_query_count, q);
                        }
                    }
                }
                EVENT_SUB_QUERY_COMPLETED => {
                    if let Some(result) = update.as_sub_query_completed() {
                        if result.tokens_used > 0 {
                            println!(
                                "✅ [{}/{}] Complete ({} tokens)",
                                result.id + 1,
                                sub_query_count,
                                result.tokens_used
                            );
                        } else {
                            println!("✅ [{}/{}] Complete", result.id + 1, sub_query_count);
                        }
                    }
                }
                EVENT_SUB_QUERY_FAILED => {
                    if let Some(id) = update.data.get("id").and_then(|v| v.as_u64()) {
                        if let Some(err) = update.data.get("error").and_then(|v| v.as_str()) {
                            println!("❌ [{}/{}] Failed: {}", id + 1, sub_query_count, err);
                        }
                    }
                }
                EVENT_SYNTHESIS_STARTED => {
                    println!();
                    println!("🧠 Synthesizing findings...");
                }
                EVENT_FINAL_RESULT => {
                    if let Some(result) = update.as_final_result() {
                        println!();
                        println!("╔══════════════════════════════════════════════════════════════╗");
                        println!("║                        ANSWER                                ║");
                        println!("╚══════════════════════════════════════════════════════════════╝");
                        println!();
                        println!("{}", result.answer);
                        println!();
                        println!("──────────────────────────────────────────────────────────────");

                        let total_queries = result.metadata.sub_queries_succeeded
                            + result.metadata.sub_queries_failed;

                        // Format duration nicely
                        let duration = if result.metadata.duration_ms >= 1000 {
                            format!("{:.1}s", result.metadata.duration_ms as f64 / 1000.0)
                        } else {
                            format!("{}ms", result.metadata.duration_ms)
                        };

                        // Show token info only if available
                        if result.metadata.tokens_unavailable_count == 0
                            && result.metadata.total_tokens > 0
                        {
                            println!(
                                "📊 Stats: {} tokens | {} | {}/{} sub-queries succeeded",
                                result.metadata.total_tokens,
                                duration,
                                result.metadata.sub_queries_succeeded,
                                total_queries
                            );
                        } else {
                            println!(
                                "📊 Stats: {} | {}/{} sub-queries succeeded",
                                duration,
                                result.metadata.sub_queries_succeeded,
                                total_queries
                            );
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
