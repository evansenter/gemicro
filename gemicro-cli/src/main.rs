//! gemicro CLI - Deep research agent powered by Gemini.

mod cli;
mod display;
mod format;

use anyhow::{Context, Result};
use clap::Parser;
use display::{DisplayState, IndicatifRenderer, Phase, Renderer};
use futures_util::StreamExt;
use gemicro_core::{AgentContext, AgentError, DeepResearchAgent, LlmClient};

#[tokio::main]
async fn main() -> Result<()> {
    let args = cli::Args::parse();

    // Initialize logging
    if args.verbose {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    }

    // Print header
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    gemicro Deep Research                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Query: {}", args.query);
    println!();

    // Run the research
    run_research(&args).await
}

async fn run_research(args: &cli::Args) -> Result<()> {
    // Create LLM client and context
    let genai_client = rust_genai::Client::builder(args.api_key.clone()).build();
    let llm = LlmClient::new(genai_client, args.llm_config());
    let context = AgentContext::new(llm);

    // Create agent
    let agent = DeepResearchAgent::new(args.research_config())
        .context("Failed to create research agent")?;

    // Initialize state and renderer
    let mut state = DisplayState::new();
    let mut renderer = IndicatifRenderer::new();

    // Get stream
    let stream = agent.execute(&args.query, context);
    futures_util::pin_mut!(stream);

    // Consume stream
    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => {
                let prev_phase = state.phase();
                let updated_id = state.update(&update);

                // Notify renderer of phase changes
                if state.phase() != prev_phase {
                    renderer
                        .on_phase_change(&state)
                        .context("Renderer phase change failed")?;
                }

                // Notify renderer of sub-query updates
                if let Some(id) = updated_id {
                    renderer
                        .on_sub_query_update(&state, id)
                        .context("Renderer sub-query update failed")?;
                }
            }
            Err(e) => {
                renderer.finish().ok();
                return Err(format_agent_error(e));
            }
        }
    }

    // Render final result
    if state.phase() == Phase::Complete {
        renderer
            .on_final_result(&state)
            .context("Renderer final result failed")?;
    }

    renderer.finish().context("Renderer cleanup failed")?;

    Ok(())
}

/// Format an AgentError with helpful suggestions.
fn format_agent_error(e: AgentError) -> anyhow::Error {
    let suggestion = match &e {
        AgentError::Timeout { phase, .. } => Some(format!(
            "💡 Timeout during {}. Try increasing --timeout or --llm-timeout",
            phase
        )),
        AgentError::AllSubQueriesFailed => {
            Some("💡 All sub-queries failed. Check your API key and network connection".to_string())
        }
        AgentError::InvalidConfig(msg) => Some(format!("💡 Configuration error: {}", msg)),
        AgentError::Llm(llm_err) => Some(format!("💡 LLM error: {}", llm_err)),
        _ => None,
    };

    let err = anyhow::anyhow!("Research failed: {}", e);
    if let Some(hint) = suggestion {
        eprintln!("\n{}", hint);
    }
    err
}
