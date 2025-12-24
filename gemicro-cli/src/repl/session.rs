//! REPL session management
//!
//! Handles the interactive session loop, command processing, and agent execution.

use super::commands::Command;
use super::registry::AgentRegistry;
use crate::display::{DisplayState, IndicatifRenderer, Phase, Renderer};
use crate::format::truncate;
use anyhow::{Context, Result};
use futures_util::StreamExt;
use gemicro_core::{AgentContext, AgentError, ConversationHistory, HistoryEntry, LlmClient};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

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

    /// Conversation history
    pub history: ConversationHistory,

    /// LLM client (shared across agents)
    llm: Arc<LlmClient>,

    /// Path to the CLI binary (for mtime checking)
    binary_path: Option<PathBuf>,

    /// Last known mtime of the binary
    binary_mtime: Option<SystemTime>,
}

impl Session {
    /// Create a new session.
    pub fn new(llm: LlmClient) -> Self {
        let binary_path = std::env::current_exe().ok();
        let binary_mtime = binary_path
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok())
            .and_then(|m| m.modified().ok());

        Self {
            registry: AgentRegistry::new(),
            history: ConversationHistory::new(),
            llm: Arc::new(llm),
            binary_path,
            binary_mtime,
        }
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

    /// Get the current agent context
    fn agent_context(&self) -> AgentContext {
        AgentContext::from_arc(self.llm.clone())
    }

    /// Build prompt prefix with context
    fn build_prompt_prefix(&self) -> String {
        // Include conversation history as context
        self.history.context_for_prompt(3) // Last 3 exchanges
    }

    /// Run a query through the current agent
    pub async fn run_query(&mut self, query: &str) -> Result<()> {
        let agent = self
            .registry
            .current_agent()
            .context("No agent available")?;

        let agent_name = agent.name().to_string();

        // Build query with context
        let full_query = if self.history.is_empty() {
            query.to_string()
        } else {
            format!("{}\n\nCurrent query: {}", self.build_prompt_prefix(), query)
        };

        // Initialize state and renderer
        let mut state = DisplayState::new();
        let mut renderer = IndicatifRenderer::new();
        let mut events = Vec::new();

        // Execute and stream
        let stream = agent.execute(&full_query, self.agent_context());
        futures_util::pin_mut!(stream);

        while let Some(result) = stream.next().await {
            match result {
                Ok(update) => {
                    let prev_phase = state.phase();
                    let updated_id = state.update(&update);

                    // Store event for history
                    events.push(update);

                    if state.phase() != prev_phase {
                        renderer.on_phase_change(&state)?;
                    }

                    if let Some(id) = updated_id {
                        renderer.on_sub_query_update(&state, id)?;
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
            renderer.on_final_result(&state)?;
        }

        renderer.finish()?;

        // Store in history
        self.history
            .push(HistoryEntry::new(query.to_string(), agent_name, events));

        Ok(())
    }

    /// Run the interactive REPL loop
    pub async fn run(&mut self) -> Result<()> {
        let mut rl = DefaultEditor::new().context("Failed to initialize readline")?;

        println!("gemicro REPL - Type /help for commands, /quit to exit");
        println!();

        loop {
            // Build prompt with stale indicator
            let stale_indicator = if self.is_stale() { " [stale]" } else { "" };
            let prompt = format!("[{}{}] > ", self.registry.current_name(), stale_indicator);

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
                            if let Err(e) = self.registry.switch(&name) {
                                eprintln!("{}", e);
                            } else {
                                println!("Switched to agent: {}", name);
                            }
                        }
                        Command::ListAgents => {
                            println!("Available agents:");
                            for name in self.registry.list() {
                                let marker = if name == self.registry.current_name() {
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

/// Format an AgentError with helpful suggestions
fn format_agent_error(e: AgentError) -> anyhow::Error {
    let suggestion = match &e {
        AgentError::Timeout { phase, .. } => Some(format!(
            "Timeout during {}. Try increasing timeout settings.",
            phase
        )),
        AgentError::AllSubQueriesFailed => {
            Some("All sub-queries failed. Check your API key and network connection.".to_string())
        }
        AgentError::InvalidConfig(msg) => Some(format!("Configuration error: {}", msg)),
        AgentError::Llm(llm_err) => Some(format!("LLM error: {}", llm_err)),
        _ => None,
    };

    let err = anyhow::anyhow!("Agent error: {}", e);
    if let Some(hint) = suggestion {
        eprintln!("{}", hint);
    }
    err
}

#[cfg(test)]
mod tests {
    use super::*;
    use gemicro_core::LlmConfig;

    // Note: Most session tests require a mock LLM client which we don't have yet.
    // These are basic structural tests.

    #[test]
    fn test_session_stale_detection_fresh() {
        // A fresh session should not be stale
        let genai_client = rust_genai::Client::builder("test-key".to_string()).build();
        let llm = LlmClient::new(genai_client, LlmConfig::default());
        let session = Session::new(llm);

        // Fresh session should not be stale
        assert!(!session.is_stale());
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
}
