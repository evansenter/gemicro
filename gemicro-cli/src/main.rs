//! gemicro CLI - Deep research agent powered by Gemini.

mod cli;
pub mod config;
pub mod confirmation;
mod display;
mod error;
mod format;
mod repl;

use anyhow::{bail, Context, Result};
use clap::Parser;
use display::{IndicatifRenderer, Renderer};
use futures_util::StreamExt;
use gemicro_core::{
    enforce_final_result_contract, Agent, AgentContext, AgentError, Coordination, HubCoordination,
    LlmClient,
};
use gemicro_deep_research_agent::DeepResearchAgent;
use gemicro_prompt_agent::{tools, PromptAgent, PromptAgentConfig};
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
    // Filter out noisy third-party crates, show only gemicro-related debug output
    if args.verbose {
        env_logger::Builder::from_env(
            env_logger::Env::default()
                .default_filter_or("debug,rustyline=warn,hyper=warn,reqwest=warn,h2=warn"),
        )
        .write_style(env_logger::WriteStyle::Always) // Force colors even when piped
        .format(|buf, record| {
            use std::io::Write;
            // Color-code by module: genai_rs = dim cyan, gemicro = bright
            let style = buf.default_level_style(record.level());
            let module = record.module_path().unwrap_or("unknown");
            let is_wire = module.starts_with("genai_rs");
            if is_wire {
                // Dim style for wire-level debug (less important)
                writeln!(buf, "\x1b[2m[{}] {}\x1b[0m", module, record.args())
            } else {
                // Normal style with level coloring for app logs
                writeln!(
                    buf,
                    "{style}[{}]{style:#} {}",
                    record.level(),
                    record.args()
                )
            }
        })
        .init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    }

    // Log if Google Search grounding is enabled
    if args.google_search {
        log::info!(
            "Google Search grounding enabled - sub-queries will search the web for real-time data"
        );
    }

    // Interactive mode: explicit -i flag OR no query provided (default)
    let interactive_mode = args.interactive || args.query.is_none();

    if interactive_mode {
        run_interactive(&args).await
    } else {
        // Safe to unwrap - validation ensures query exists when not interactive
        let query = args.query.as_ref().unwrap();
        log::debug!("Starting single-query execution with agent: {}", args.agent);

        // Print header for single query mode
        // Note: Only hardcoded agents (deep_research, prompt_agent) are supported in single-query mode.
        // Runtime-loaded markdown agents (like codebase-explorer) require interactive mode.
        let agent_title = match args.agent.as_str() {
            "deep_research" => "gemicro Deep Research",
            "prompt_agent" => "gemicro Prompt Agent",
            other => {
                anyhow::bail!(
                    "Single-query mode supports: deep_research, prompt_agent. \
                     Use --interactive for runtime-loaded agents like '{}'.",
                    other
                );
            }
        };

        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║{:^62}║", agent_title);
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();
        println!("Query: {}", query);
        println!();

        run_single_query(&args, query).await
    }
}

/// Run the interactive REPL
async fn run_interactive(args: &cli::Args) -> Result<()> {
    let genai_client = genai_rs::Client::builder(args.api_key.clone())
        .build()
        .context("Failed to create Gemini client")?;
    let llm = LlmClient::new(genai_client, args.llm_config());

    let mut session = Session::new(llm, args.plain);

    // Set CLI overrides (these take precedence over file config)
    session.set_cli_overrides(repl::CliOverrides {
        research_config: Some(args.research_config()),
    });

    // Set sandbox paths for file access restriction (if configured)
    if !args.sandbox_paths.is_empty() {
        log::info!(
            "File sandbox enabled, restricted to: {}",
            args.sandbox_paths
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        session.set_sandbox_paths(args.sandbox_paths.clone());
    }

    // Enable auto-approve mode for tool confirmations (if configured)
    // This is useful for piped/scripted input where stdin is not a terminal
    if args.auto_approve {
        session.set_auto_approve(true);
    }

    // Load config files and register agents (always succeeds, logs warnings on errors)
    let loaded_files = session.load_config_and_register_agents();
    if !loaded_files.is_empty() {
        log::info!(
            "Loaded config from: {}",
            loaded_files
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // Set the initial agent from CLI flag
    session.set_current_agent(&args.agent).unwrap_or_else(|_| {
        eprintln!("Error: Unknown agent '{}'. Available agents:", args.agent);
        eprintln!("  deep_research, developer, prompt_agent, react, critique");
        std::process::exit(1);
    });

    session.run().await
}

async fn run_single_query(args: &cli::Args, query: &str) -> Result<()> {
    // Create cancellation token for cooperative shutdown
    let cancellation_token = CancellationToken::new();

    // Create LLM client
    let genai_client = genai_rs::Client::builder(args.api_key.clone())
        .build()
        .context("Failed to create Gemini client")?;
    let llm = LlmClient::new(genai_client, args.llm_config());

    // Create agent and context based on --agent flag
    // Context is agent-specific: prompt_agent gets tools, others don't
    let (agent, context): (Box<dyn Agent>, AgentContext) = match args.agent.as_str() {
        "deep_research" => {
            let agent = Box::new(
                DeepResearchAgent::new(args.research_config())
                    .context("Failed to create research agent")?,
            );
            let context = AgentContext::new_with_cancellation(llm, cancellation_token.clone());
            (agent, context)
        }
        "prompt_agent" => {
            let agent = Box::new(
                PromptAgent::new(PromptAgentConfig::default())
                    .context("Failed to create prompt agent")?,
            );
            // Add bundled tools (Calculator, CurrentDatetime)
            let context = AgentContext::new_with_cancellation(llm, cancellation_token.clone())
                .with_tools(tools::default_registry());
            (agent, context)
        }
        other => anyhow::bail!("Unsupported agent for single-query mode: {}", other),
    };

    let mut tracker = agent.create_tracker();
    // Disable spinner in verbose mode to avoid interference with log output
    let mut renderer = IndicatifRenderer::new(args.plain || args.verbose);

    // Connect to event bus if URL provided
    let mut coordination = if let Some(ref url) = args.event_bus_url {
        match HubCoordination::connect(url, "gemicro-cli").await {
            Ok(coord) => {
                log::info!("Connected to event bus at {}", url);
                Some(coord)
            }
            Err(e) => {
                log::warn!(
                    "Failed to connect to event bus: {} - continuing without coordination",
                    e
                );
                None
            }
        }
    } else {
        None
    };

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

    // Get stream with contract enforcement
    let stream = agent.execute(query, context);
    let stream = enforce_final_result_contract(Box::pin(stream));
    futures_util::pin_mut!(stream);

    // Track if we were interrupted
    let mut interrupted = false;

    // Consume stream, optionally with coordination
    if let Some(ref mut coord) = coordination {
        // With coordination: use select! to interleave external events
        // Track if coordination channel is still active to avoid busy-loop
        let mut coord_active = true;

        loop {
            if is_interrupted() {
                interrupted = true;
                break;
            }

            // If coordination is closed, fall back to simple stream consumption
            if !coord_active {
                match stream.next().await {
                    Some(Ok(update)) => {
                        if is_interrupted() {
                            interrupted = true;
                            break;
                        }
                        // Handle event-specific rendering first
                        renderer
                            .on_event(&update)
                            .context("Renderer event handling failed")?;
                        tracker.handle_event(&update);
                        renderer
                            .on_status(tracker.as_ref())
                            .context("Renderer status update failed")?;
                        if tracker.is_complete() {
                            renderer
                                .on_complete(tracker.as_ref())
                                .context("Renderer completion failed")?;
                            break;
                        }
                    }
                    Some(Err(AgentError::Cancelled)) => {
                        interrupted = true;
                        break;
                    }
                    Some(Err(e)) => {
                        signal_task.abort();
                        if let Err(finish_err) = renderer.finish() {
                            log::warn!("Failed to clean up renderer during error: {}", finish_err);
                        }
                        return Err(format_agent_error(e));
                    }
                    None => break,
                }
                continue;
            }

            tokio::select! {
                // Agent stream has priority (biased) to avoid starving agent events
                biased;

                result = stream.next() => {
                    match result {
                        Some(Ok(update)) => {
                            if is_interrupted() {
                                interrupted = true;
                                break;
                            }

                            // Handle event-specific rendering first
                            renderer
                                .on_event(&update)
                                .context("Renderer event handling failed")?;

                            tracker.handle_event(&update);
                            renderer
                                .on_status(tracker.as_ref())
                                .context("Renderer status update failed")?;

                            if tracker.is_complete() {
                                renderer
                                    .on_complete(tracker.as_ref())
                                    .context("Renderer completion failed")?;
                                break;
                            }
                        }
                        Some(Err(AgentError::Cancelled)) => {
                            interrupted = true;
                            break;
                        }
                        Some(Err(e)) => {
                            signal_task.abort();
                            if let Err(finish_err) = renderer.finish() {
                                log::warn!("Failed to clean up renderer during error: {}", finish_err);
                            }
                            return Err(format_agent_error(e));
                        }
                        None => break, // Stream exhausted
                    }
                }

                event = coord.recv_event() => {
                    if let Some(external_event) = event {
                        // Display external event inline
                        let source = external_event.source_session.as_deref().unwrap_or("unknown");
                        eprintln!(
                            "\n  [{}] {} (from: {})\n",
                            external_event.event_type,
                            external_event.payload,
                            source
                        );
                    } else {
                        // Coordination channel closed - stop polling to avoid busy-loop
                        log::debug!("Coordination channel closed, continuing with agent stream only");
                        coord_active = false;
                    }
                }
            }
        }
    } else {
        // Without coordination: simple stream consumption
        while let Some(result) = stream.next().await {
            if is_interrupted() {
                interrupted = true;
                break;
            }

            match result {
                Ok(update) => {
                    if is_interrupted() {
                        interrupted = true;
                        break;
                    }

                    // Handle event-specific rendering first
                    renderer
                        .on_event(&update)
                        .context("Renderer event handling failed")?;

                    tracker.handle_event(&update);
                    renderer
                        .on_status(tracker.as_ref())
                        .context("Renderer status update failed")?;

                    if tracker.is_complete() {
                        renderer
                            .on_complete(tracker.as_ref())
                            .context("Renderer completion failed")?;
                        break;
                    }
                }
                Err(AgentError::Cancelled) => {
                    interrupted = true;
                    break;
                }
                Err(e) => {
                    signal_task.abort();
                    if let Err(finish_err) = renderer.finish() {
                        log::warn!("Failed to clean up renderer during error: {}", finish_err);
                    }
                    return Err(format_agent_error(e));
                }
            }
        }
    }

    // Handle interrupted state
    if interrupted {
        renderer
            .on_interrupted(tracker.as_ref())
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
