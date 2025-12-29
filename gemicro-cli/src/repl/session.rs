//! REPL session management
//!
//! Handles the interactive session loop, command processing, and agent execution.

use super::commands::Command;
use crate::confirmation::InteractiveConfirmation;
use crate::display::{IndicatifRenderer, Renderer};
use crate::error::ErrorFormatter;
use crate::format::truncate;
use anyhow::{Context, Result};
use futures_util::StreamExt;
use gemicro_core::{
    enforce_final_result_contract, AgentContext, AgentError, AgentUpdate, ConfirmationHandler,
    ConversationHistory, HistoryEntry, LlmClient,
};
use gemicro_runner::AgentRegistry;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::SystemTime;
use tokio_util::sync::CancellationToken;

/// Maximum characters for answer previews in the `/history` command.
///
/// When users run `/history` in the REPL, this controls how much of each
/// previous answer is shown. Full answers are always available in the
/// conversation context sent to the LLM for follow-up queries.
const HISTORY_PREVIEW_CHARS: usize = 256;

/// REPL session state
pub struct Session {
    /// Agent registry
    pub registry: AgentRegistry,

    /// Currently selected agent name
    current_agent_name: String,

    /// Conversation history
    pub history: ConversationHistory,

    /// LLM client (shared across agents)
    llm: Arc<LlmClient>,

    /// Path to the CLI binary (for mtime checking).
    ///
    /// Used to detect when the binary has been recompiled, showing a `[stale]`
    /// indicator in the prompt. This enables future hot-reload functionality.
    /// See: <https://github.com/evansenter/gemicro/issues/36>
    binary_path: Option<PathBuf>,

    /// Last known mtime of the binary.
    ///
    /// Compared against current mtime to detect staleness.
    /// TODO(#36): Implement actual hot-reload when binary changes.
    binary_mtime: Option<SystemTime>,

    /// Whether to use plain text output (no markdown rendering)
    plain: bool,

    /// Cumulative tokens used in this session
    session_tokens: u64,

    /// Confirmation handler for tools that require user approval
    confirmation_handler: Arc<dyn ConfirmationHandler>,
}

impl Session {
    /// Create a new session.
    ///
    /// If `plain` is true, markdown rendering will be disabled for output.
    pub fn new(llm: LlmClient, plain: bool) -> Self {
        let binary_path = std::env::current_exe().ok();
        let binary_mtime = binary_path
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok())
            .and_then(|m| m.modified().ok());

        Self {
            registry: AgentRegistry::new(),
            current_agent_name: String::new(),
            history: ConversationHistory::new(),
            llm: Arc::new(llm),
            binary_path,
            binary_mtime,
            plain,
            session_tokens: 0,
            confirmation_handler: Arc::new(InteractiveConfirmation::default()),
        }
    }

    /// Set the current agent by name.
    ///
    /// This should be called after registering agents to set the initial selection.
    /// Resets the session token count when switching to a different agent.
    /// Returns `Ok(())` if the agent exists, `Err` with message otherwise.
    pub fn set_current_agent(&mut self, name: &str) -> Result<(), String> {
        if self.registry.contains(name) {
            // Reset token count when switching agents
            if self.current_agent_name != name {
                self.session_tokens = 0;
            }
            self.current_agent_name = name.to_string();
            Ok(())
        } else {
            Err(format!(
                "Unknown agent '{}'. Available: {}",
                name,
                self.registry.list().join(", ")
            ))
        }
    }

    /// Get the current agent name.
    pub fn current_agent_name(&self) -> &str {
        &self.current_agent_name
    }

    /// Check if the binary has been modified since session started
    pub fn is_stale(&self) -> bool {
        if let (Some(path), Some(original_mtime)) = (&self.binary_path, &self.binary_mtime) {
            match std::fs::metadata(path) {
                Ok(metadata) => match metadata.modified() {
                    Ok(current_mtime) => {
                        let stale = current_mtime > *original_mtime;
                        if stale {
                            log::debug!("Binary is stale: {:?}", path);
                        }
                        return stale;
                    }
                    Err(e) => {
                        log::debug!("Failed to get mtime for {:?}: {}", path, e);
                    }
                },
                Err(e) => {
                    log::debug!("Failed to get metadata for {:?}: {}", path, e);
                }
            }
        }
        false
    }

    /// Get the current agent context with cancellation support
    fn agent_context(&self, cancellation_token: CancellationToken) -> AgentContext {
        AgentContext {
            llm: self.llm.clone(),
            cancellation_token,
            tools: None,
            confirmation_handler: Some(Arc::clone(&self.confirmation_handler)),
        }
    }

    /// Build prompt prefix with context
    fn build_prompt_prefix(&self) -> String {
        // Include conversation history as context
        self.history.context_for_prompt(3) // Last 3 exchanges
    }

    /// Run a query through the current agent
    ///
    /// Supports Ctrl+C cancellation - pressing Ctrl+C during execution will
    /// cancel the query gracefully and return to the prompt with partial results.
    pub async fn run_query(&mut self, query: &str) -> Result<()> {
        if self.current_agent_name.is_empty() {
            anyhow::bail!("No agent selected. Register agents and call set_current_agent() first.");
        }
        let agent = self
            .registry
            .get(&self.current_agent_name)
            .context("Selected agent no longer available")?;

        let agent_name = agent.name().to_string();

        // Build query with context
        let full_query = if self.history.is_empty() {
            query.to_string()
        } else {
            format!("{}\n\nCurrent query: {}", self.build_prompt_prefix(), query)
        };

        // Set up cancellation infrastructure for this query
        let cancellation_token = CancellationToken::new();
        let interrupt_count = Arc::new(AtomicU8::new(0));

        // Spawn signal handler task for Ctrl+C
        let signal_task = tokio::spawn({
            let interrupt_count = interrupt_count.clone();
            let cancellation_token = cancellation_token.clone();
            async move {
                loop {
                    if tokio::signal::ctrl_c().await.is_err() {
                        log::error!("Failed to listen for Ctrl+C signal in REPL");
                        return;
                    }
                    let count = interrupt_count.fetch_add(1, Ordering::SeqCst) + 1;
                    if count == 1 {
                        eprintln!("\nCancelling query...");
                        cancellation_token.cancel();
                    } else {
                        eprintln!("\nAlready cancelling... please wait");
                    }
                }
            }
        });

        // Helper to check if interrupted
        let is_interrupted = || interrupt_count.load(Ordering::SeqCst) > 0;

        // Create tracker and renderer
        let mut tracker = agent.create_tracker();
        let mut renderer = IndicatifRenderer::new(self.plain);
        let mut events = Vec::new();
        let mut interrupted = false;

        // Execute and stream with contract enforcement
        let stream = agent.execute(&full_query, self.agent_context(cancellation_token));
        let stream = enforce_final_result_contract(stream);
        futures_util::pin_mut!(stream);

        while let Some(result) = stream.next().await {
            // Check for interrupt before processing
            if is_interrupted() {
                interrupted = true;
                break;
            }

            match result {
                Ok(update) => {
                    // Check again after receiving update
                    if is_interrupted() {
                        interrupted = true;
                        break;
                    }

                    // Update tracker with the event
                    tracker.handle_event(&update);

                    // Store event for history
                    events.push(update);

                    // Update renderer with current status
                    renderer
                        .on_status(tracker.as_ref())
                        .context("Renderer status update failed")?;

                    // Check if complete
                    if tracker.is_complete() {
                        renderer
                            .on_complete(tracker.as_ref())
                            .context("Renderer completion failed")?;
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

        // Clean up signal handler
        signal_task.abort();

        // Handle interruption
        if interrupted {
            renderer
                .on_interrupted(tracker.as_ref())
                .context("Renderer interrupted state failed")?;
            return Ok(()); // Cancellation is not an error
        }

        renderer.finish().context("Renderer cleanup failed")?;

        // Extract tokens before moving events to history
        let tokens_used = extract_tokens_from_events(&events);

        // Store in history
        self.history
            .push(HistoryEntry::new(query.to_string(), agent_name, events));

        // Accumulate session token count
        self.session_tokens += tokens_used;

        Ok(())
    }

    /// Run the interactive REPL loop
    pub async fn run(&mut self) -> Result<()> {
        let mut rl = DefaultEditor::new().context("Failed to initialize readline")?;

        println!("gemicro REPL - Type /help for commands, /quit to exit");
        println!();

        loop {
            // Build prompt with optional token count and stale indicator
            let stale_indicator = if self.is_stale() { " [stale]" } else { "" };
            let token_display = if self.session_tokens > 0 {
                format!(" ({})", format_tokens(self.session_tokens))
            } else {
                String::new()
            };
            let agent_display = if self.current_agent_name.is_empty() {
                "no agent"
            } else {
                &self.current_agent_name
            };
            let prompt = format!("[{}{}{}] > ", agent_display, token_display, stale_indicator);

            match rl.readline(&prompt) {
                Ok(line) => {
                    let _ = rl.add_history_entry(&line);

                    match Command::parse(&line) {
                        Command::Query(query) => {
                            if let Err(e) = self.run_query(&query).await {
                                eprintln!("Error: {:#}", e);
                            }
                        }
                        Command::Agent(name) => {
                            if let Err(e) = self.set_current_agent(&name) {
                                eprintln!("{}", e);
                            } else {
                                println!("Switched to agent: {}", name);
                            }
                        }
                        Command::ListAgents => {
                            println!("Available agents:");
                            for name in self.registry.list() {
                                let marker = if name == self.current_agent_name() {
                                    " *"
                                } else {
                                    ""
                                };
                                if let Some(agent) = self.registry.get(name) {
                                    println!("  {}{} - {}", name, marker, agent.description());
                                }
                            }
                        }
                        Command::Help => {
                            println!("{}", Command::help_text());
                        }
                        Command::History => {
                            if self.history.is_empty() {
                                println!("No conversation history yet.");
                            } else {
                                println!("Conversation history ({} entries):", self.history.len());
                                for (i, entry) in self.history.iter().enumerate() {
                                    println!(
                                        "\n[{}] ({}) Q: {}",
                                        i + 1,
                                        entry.agent_name,
                                        entry.query
                                    );
                                    if let Some(result) = entry.final_result() {
                                        let preview = truncate(result, HISTORY_PREVIEW_CHARS);
                                        println!("    A: {}", preview);
                                    }
                                }
                            }
                        }
                        Command::Clear => {
                            self.history.clear();
                            println!("Conversation history cleared.");
                        }
                        Command::Reload => {
                            println!("Hot-reload not yet implemented. See issue #36.");
                            // TODO: Implement state serialization, cargo build, exec
                        }
                        Command::Quit => {
                            println!("Goodbye!");
                            break;
                        }
                        Command::Empty => {
                            // Just show prompt again
                        }
                        Command::Unknown(cmd) => {
                            eprintln!("Unknown command: /{}", cmd);
                            println!("{}", Command::help_text());
                        }
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                    // Continue - don't exit on Ctrl+C
                }
                Err(ReadlineError::Eof) => {
                    println!("Goodbye!");
                    break;
                }
                Err(e) => {
                    eprintln!("Readline error: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }
}

/// Format an AgentError with helpful suggestions (plain, no emoji).
fn format_agent_error(e: AgentError) -> anyhow::Error {
    ErrorFormatter::plain().format(e)
}

/// Format a token count for display (e.g., "1.2k" for 1200).
fn format_tokens(count: u64) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}k", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

/// Extract total tokens from a list of agent events.
///
/// Looks for final_result events and sums their total_tokens metadata.
fn extract_tokens_from_events(events: &[AgentUpdate]) -> u64 {
    events
        .iter()
        .filter_map(|e| e.as_final_result())
        .map(|r| u64::from(r.metadata.total_tokens))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gemicro_core::{LlmConfig, ResultMetadata};

    // Note: Most session tests require a mock LLM client which we don't have yet.
    // These are basic structural tests.

    #[test]
    fn test_session_stale_detection_fresh() {
        // A fresh session should not be stale
        let genai_client = rust_genai::Client::builder("test-key".to_string()).build();
        let llm = LlmClient::new(genai_client, LlmConfig::default());
        let session = Session::new(llm, false);

        // Fresh session should not be stale
        assert!(!session.is_stale());
    }

    #[test]
    fn test_session_initial_token_count() {
        let genai_client = rust_genai::Client::builder("test-key".to_string()).build();
        let llm = LlmClient::new(genai_client, LlmConfig::default());
        let session = Session::new(llm, false);

        // Initial token count should be zero
        assert_eq!(session.session_tokens, 0);
    }

    // Compile-time validation of constants
    const _: () = assert!(HISTORY_PREVIEW_CHARS >= 100, "too small");
    const _: () = assert!(HISTORY_PREVIEW_CHARS <= 500, "too large");

    #[test]
    fn test_history_truncation() {
        let long_answer = "a".repeat(500);
        let truncated = truncate(&long_answer, HISTORY_PREVIEW_CHARS);

        // Should be truncated to HISTORY_PREVIEW_CHARS (including "...")
        assert!(truncated.chars().count() <= HISTORY_PREVIEW_CHARS);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_short_history_not_truncated() {
        let short_answer = "The answer is 42";
        let result = truncate(short_answer, HISTORY_PREVIEW_CHARS);

        // Short answers should pass through unchanged
        assert_eq!(result, short_answer);
    }

    // Token formatting tests

    #[test]
    fn test_format_tokens_small() {
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(1), "1");
        assert_eq!(format_tokens(999), "999");
    }

    #[test]
    fn test_format_tokens_thousands() {
        assert_eq!(format_tokens(1000), "1.0k");
        assert_eq!(format_tokens(1234), "1.2k");
        assert_eq!(format_tokens(12345), "12.3k");
        assert_eq!(format_tokens(999999), "1000.0k");
    }

    #[test]
    fn test_format_tokens_millions() {
        assert_eq!(format_tokens(1_000_000), "1.0M");
        assert_eq!(format_tokens(1_234_567), "1.2M");
        assert_eq!(format_tokens(12_345_678), "12.3M");
    }

    // Token extraction tests

    #[test]
    fn test_extract_tokens_empty_events() {
        let events: Vec<AgentUpdate> = vec![];
        assert_eq!(extract_tokens_from_events(&events), 0);
    }

    #[test]
    fn test_extract_tokens_no_final_result() {
        use serde_json::json;
        let events = vec![
            AgentUpdate::custom("decomposition_started", "Decomposing query", json!({})),
            AgentUpdate::custom(
                "decomposition_complete",
                "Decomposed into 1 sub-query",
                json!({ "sub_queries": ["Q1"] }),
            ),
        ];
        assert_eq!(extract_tokens_from_events(&events), 0);
    }

    #[test]
    fn test_extract_tokens_with_final_result() {
        use serde_json::json;
        let events = vec![
            AgentUpdate::custom("decomposition_started", "Decomposing query", json!({})),
            AgentUpdate::final_result(
                "The answer".to_string(),
                ResultMetadata::with_extra(
                    1500,
                    0,
                    1000,
                    json!({
                        "steps_succeeded": 2,
                        "steps_failed": 0,
                    }),
                ),
            ),
        ];
        assert_eq!(extract_tokens_from_events(&events), 1500);
    }
}
