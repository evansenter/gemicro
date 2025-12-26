//! gemicro CLI - Deep research agent powered by Gemini.

mod cli;
mod display;
mod error;
mod format;
mod repl;

use anyhow::{bail, Context, Result};
use clap::Parser;
use display::{ExecutionState, IndicatifRenderer, Phase, Renderer};
use futures_util::StreamExt;
use gemicro_core::{AgentContext, AgentError, DeepResearchAgent, LlmClient};
use repl::Session;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> Result<()> {
    let args = cli::Args::parse();

    // Validate arguments
    if let Err(e) = args.validate() {
        bail!("{}", e);
    }

    // Initialize logging
    if args.verbose {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    }

    // Log if Google Search grounding is enabled
    if args.google_search {
        log::info!(
            "Google Search grounding enabled - sub-queries will search the web for real-time data"
        );
    }

    if args.interactive {
        run_interactive(&args).await
    } else {
        // Print header for single query mode
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║                    gemicro Deep Research                     ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();
        // Safe to unwrap - validation ensures query exists when not interactive
        let query = args.query.as_ref().unwrap();
        println!("Query: {}", query);
        println!();

        run_research(&args, query).await
    }
}

/// Run the interactive REPL
async fn run_interactive(args: &cli::Args) -> Result<()> {
    let genai_client = rust_genai::Client::builder(args.api_key.clone()).build();
    let llm = LlmClient::new(genai_client, args.llm_config());

    let mut session = Session::new(llm, args.plain);

    // Register available agents
    let research_config = args.research_config();
    session.registry.register("deep_research", move || {
        Box::new(DeepResearchAgent::new(research_config.clone()).expect("Invalid research config"))
    });

    // Set the initial agent
    session
        .set_current_agent("deep_research")
        .expect("deep_research agent should be registered");

    session.run().await
}

async fn run_research(args: &cli::Args, query: &str) -> Result<()> {
    // Create cancellation token for cooperative shutdown
    let cancellation_token = CancellationToken::new();

    // Create LLM client and context with cancellation support
    let genai_client = rust_genai::Client::builder(args.api_key.clone()).build();
    let llm = LlmClient::new(genai_client, args.llm_config());
    let context = AgentContext::new_with_cancellation(llm, cancellation_token.clone());

    // Create agent
    let agent = DeepResearchAgent::new(args.research_config())
        .context("Failed to create research agent")?;

    // Initialize state and renderer
    let mut state = ExecutionState::new();
    let mut renderer = IndicatifRenderer::new(args.plain);

    // Set up interrupt handling with AtomicU8 for cross-thread visibility.
    // Signal handler writes, main loop reads. Uses SeqCst for simplicity.
    // 0 = no interrupt, 1 = first interrupt (graceful), 2+ = force exit
    let interrupt_count = Arc::new(AtomicU8::new(0));
    let interrupt_count_clone = interrupt_count.clone();
    let cancellation_token_clone = cancellation_token.clone();

    // Spawn signal handler task
    let signal_task = tokio::spawn(async move {
        loop {
            if tokio::signal::ctrl_c().await.is_err() {
                log::error!("Failed to listen for Ctrl+C signal");
                return;
            }

            let count = interrupt_count_clone.fetch_add(1, Ordering::SeqCst) + 1;
            if count == 1 {
                eprintln!("\n⚠️  Interrupt received - cancelling in-flight requests...");
                eprintln!("   (Press Ctrl+C again to exit immediately)\n");
                // Cancel all in-flight tasks cooperatively
                cancellation_token_clone.cancel();
            } else {
                eprintln!("\n❌ Force exit requested");
                std::process::exit(130); // Standard exit code for SIGINT
            }
        }
    });

    // Helper to check for interrupts
    let is_interrupted = || interrupt_count.load(Ordering::SeqCst) > 0;

    // Get stream
    let stream = agent.execute(query, context);
    futures_util::pin_mut!(stream);

    // Track if we were interrupted
    let mut interrupted = false;

    // Consume stream
    while let Some(result) = stream.next().await {
        // Check if interrupted at start of each iteration
        if is_interrupted() {
            interrupted = true;
            break;
        }

        match result {
            Ok(update) => {
                // Check again before processing (reduces latency to respond to interrupt)
                if is_interrupted() {
                    interrupted = true;
                    break;
                }

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
            Err(AgentError::Cancelled) => {
                // Cancellation is not an error - treat as interrupt
                interrupted = true;
                break;
            }
            Err(e) => {
                signal_task.abort();
                renderer.finish().ok();
                return Err(format_agent_error(e));
            }
        }
    }

    // Render final result or partial results
    if state.phase() == Phase::Complete {
        renderer
            .on_final_result(&state)
            .context("Renderer final result failed")?;
    } else if interrupted {
        // Show partial results if we were interrupted
        renderer
            .on_interrupted(&state)
            .context("Renderer interrupted state failed")?;
    }

    // Cleanup
    signal_task.abort();
    renderer.finish().context("Renderer cleanup failed")?;

    // Exit with SIGINT code if interrupted (convention: 128 + signal number)
    if interrupted {
        std::process::exit(130);
    }

    Ok(())
}

/// Format an AgentError with helpful suggestions (with emoji hints).
fn format_agent_error(e: AgentError) -> anyhow::Error {
    error::ErrorFormatter::with_emoji().format(e)
}
