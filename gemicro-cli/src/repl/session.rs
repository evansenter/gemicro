//! REPL session management
//!
//! Handles the interactive session loop, command processing, and agent execution.

use super::commands::Command;
use crate::config::loader::ConfigChange;
use crate::config::{ConfigLoader, GemicroConfig};
use crate::confirmation::InteractiveConfirmation;
use crate::display::{IndicatifRenderer, Renderer};
use crate::error::ErrorFormatter;
use crate::format::truncate;
use crate::registry::{register_builtin_agents, register_markdown_agents, RegistryOptions};
use anyhow::{Context, Result};
use futures_util::StreamExt;
use gemicro_audit_log::AuditLog;
use gemicro_core::{
    enforce_final_result_contract,
    interceptor::{InterceptorChain, ToolCall},
    tool::{ToolRegistry, ToolResult},
    AgentContext, AgentError, AgentUpdate, AutoApprove, BatchConfirmationHandler,
    ConversationHistory, HistoryEntry, LlmClient,
};
use gemicro_path_sandbox::PathSandbox;
use gemicro_prompt_agent::tools as prompt_tools;
use gemicro_runner::AgentRegistry;
use gemicro_task::{Task, TaskContext};
// Explicit tool imports (per "Explicit Over Implicit" principle)
use gemicro_bash::Bash;
use gemicro_file_edit::FileEdit;
use gemicro_file_read::FileRead;
use gemicro_file_write::FileWrite;
use gemicro_glob::Glob;
use gemicro_grep::Grep;
use gemicro_web_fetch::WebFetch;
use gemicro_web_search::WebSearch;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;
use tokio_util::sync::CancellationToken;

/// Maximum characters for answer previews in the `/history` command.
///
/// When users run `/history` in the REPL, this controls how much of each
/// previous answer is shown. Full answers are always available in the
/// conversation context sent to the LLM for follow-up queries.
const HISTORY_PREVIEW_CHARS: usize = 256;

/// Default directory for runtime-loaded markdown agents.
///
/// This path is relative to the workspace root. Agents in this directory
/// are loaded at startup and registered in the agent registry.
const RUNTIME_AGENTS_DIR: &str = "agents/runtime-agents";

/// REPL session state
pub struct Session {
    /// Agent registry (shared with Task tool for subagent access).
    ///
    /// Wrapped in RwLock to allow mutation on reload while Task tool holds
    /// a read-only reference for agent lookups.
    pub registry: Arc<RwLock<AgentRegistry>>,

    /// Currently selected agent name
    current_agent_name: String,

    /// Conversation history
    pub history: ConversationHistory,

    /// LLM client (shared across agents)
    llm: Arc<LlmClient>,

    /// Config loader for hot-reload support
    config_loader: ConfigLoader,

    /// CLI argument overrides (take precedence over file config)
    cli_overrides: CliOverrides,

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

    /// Confirmation handler for tools that require user approval (with batch support)
    confirmation_handler: Arc<dyn BatchConfirmationHandler>,

    /// Tool registry for agents that need tools (e.g., developer agent)
    tool_registry: Arc<ToolRegistry>,

    /// Optional interceptor chain for tool interception (e.g., PathSandbox)
    ///
    /// When set, tool calls are passed through this chain for validation,
    /// logging, and security controls. Built from CLI --sandbox-path args.
    interceptors: Option<Arc<InterceptorChain<ToolCall, ToolResult>>>,

    /// Shared context for the Task tool.
    ///
    /// This allows us to update the Task tool's context (tools, confirmation
    /// handler, interceptors) after configuration changes, without needing
    /// to recreate the tool registry.
    task_context: Arc<TaskContext>,
}

/// CLI argument overrides that take precedence over file config.
#[derive(Debug, Clone, Default)]
pub struct CliOverrides {
    /// Model override from CLI --model flag or GEMINI_MODEL env var.
    /// Applied to all agents when set.
    pub model: Option<String>,
}

/// Result of a config reload operation.
#[derive(Debug)]
pub struct ReloadResult {
    /// Config files that were loaded
    pub loaded_files: Vec<PathBuf>,
    /// Changes detected between old and new config
    pub changes: Vec<ConfigChange>,
    /// The old config (before reload) - kept for debugging
    #[allow(dead_code)]
    pub old_config: GemicroConfig,
    /// The new config (after reload) - kept for debugging
    #[allow(dead_code)]
    pub new_config: GemicroConfig,
}

impl ReloadResult {
    /// Check if any changes were detected.
    pub fn has_changes(&self) -> bool {
        !self.changes.is_empty()
    }
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

        let llm = Arc::new(llm);

        // Create agent registry (shared with Task tool for subagent access)
        let registry = Arc::new(RwLock::new(AgentRegistry::new()));

        // Create tool registry with Task tool for subagent spawning
        let (tool_registry, task_context) =
            Self::create_tool_registry(Arc::clone(&llm), Arc::clone(&registry));
        let tool_registry = Arc::new(tool_registry);

        // Initialize interceptor chain with AuditLog for LOUD_WIRE support
        let interceptors = {
            let chain: InterceptorChain<ToolCall, ToolResult> =
                InterceptorChain::new().with(AuditLog);
            Some(Arc::new(chain))
        };

        // Initialize TaskContext with the tools, default confirmation handler, and interceptors
        let confirmation_handler: Arc<dyn BatchConfirmationHandler> =
            Arc::new(InteractiveConfirmation::default());
        task_context.set_tools(Some(Arc::clone(&tool_registry)));
        task_context.set_confirmation_handler(Some(Arc::clone(&confirmation_handler)));
        task_context.set_interceptors(interceptors.clone());

        Self {
            registry,
            current_agent_name: String::new(),
            history: ConversationHistory::new(),
            llm,
            config_loader: ConfigLoader::new(),
            cli_overrides: CliOverrides::default(),
            binary_path,
            binary_mtime,
            plain,
            session_tokens: 0,
            confirmation_handler,
            tool_registry,
            interceptors,
            task_context,
        }
    }

    /// Create a tool registry with developer-appropriate tools.
    ///
    /// Includes:
    /// - Read-only tools (file_read, glob, grep)
    /// - Web search tools
    /// - Write tools that require confirmation (file_write, file_edit, bash)
    /// - Task tool for spawning subagents (shares the Session's agent registry,
    ///   enabling recursive delegation to any registered agent)
    ///
    /// Returns both the registry and the TaskContext for updating Task's shared state.
    ///
    /// Per "Explicit Over Implicit" principle, all tools are registered explicitly
    /// rather than using default registries.
    fn create_tool_registry(
        llm: Arc<LlmClient>,
        agent_registry: Arc<RwLock<AgentRegistry>>,
    ) -> (ToolRegistry, Arc<TaskContext>) {
        let mut registry = ToolRegistry::new();

        // Built-in tools from prompt-agent crate
        registry.register(prompt_tools::Calculator);
        registry.register(prompt_tools::CurrentDatetime);

        // Read-only tools (no confirmation needed)
        registry.register(FileRead);
        registry.register(WebFetch::new());
        registry.register(Glob);
        registry.register(Grep);

        // Web search tool (requires LlmClient)
        registry.register(WebSearch::new(Arc::clone(&llm)));

        // Write tools (require confirmation)
        registry.register(FileWrite);
        registry.register(FileEdit);
        registry.register(Bash);

        // Task tool for subagent spawning
        let (task, task_context) = Task::new(agent_registry, llm);
        registry.register(task);

        (registry, task_context)
    }

    /// Set CLI overrides that take precedence over file config.
    pub fn set_cli_overrides(&mut self, overrides: CliOverrides) {
        self.cli_overrides = overrides;
    }

    /// Enable auto-approve mode for tool confirmations.
    ///
    /// When enabled, tools that normally require confirmation (like bash commands)
    /// are automatically approved without user prompting. This is useful for:
    /// - Scripted/piped input where stdin is not a terminal
    /// - Trusted automation contexts
    /// - Testing and development
    ///
    /// SECURITY WARNING: Use with caution - this bypasses human review.
    pub fn set_auto_approve(&mut self, enabled: bool) {
        if enabled {
            log::info!("Auto-approve mode enabled - tool confirmations will be automatic");
            self.confirmation_handler = Arc::new(AutoApprove);
        } else {
            self.confirmation_handler = Arc::new(InteractiveConfirmation::default());
        }
        // Update Task's shared context so subagents see the change
        self.task_context
            .set_confirmation_handler(Some(Arc::clone(&self.confirmation_handler)));
    }

    /// Set sandbox paths for file access restriction.
    ///
    /// When paths are provided, builds a PathSandbox interceptor that restricts
    /// all file operations to the specified paths. This is a security feature
    /// controlled by the CLI --sandbox-path flag.
    ///
    /// The interceptor chain always includes AuditLog for LOUD_WIRE debugging
    /// support. When sandbox paths are provided, PathSandbox is added to enforce
    /// the file access restrictions.
    pub fn set_sandbox_paths(&mut self, paths: Vec<PathBuf>) {
        // Always include AuditLog for LOUD_WIRE support
        let mut chain: InterceptorChain<ToolCall, ToolResult> =
            InterceptorChain::new().with(AuditLog);

        // Add PathSandbox if paths are provided
        if !paths.is_empty() {
            chain = chain.with(PathSandbox::new(paths));
        }

        self.interceptors = Some(Arc::new(chain));
        // Update Task's shared context so subagents see the change
        self.task_context
            .set_interceptors(self.interceptors.clone());
    }

    /// Load config and register agents.
    ///
    /// This is called on startup and on /reload to (re)register agents
    /// with the current config. Always succeeds - config errors are logged
    /// as warnings and agents are registered with defaults.
    pub fn load_config_and_register_agents(&mut self) -> Vec<PathBuf> {
        let (config, loaded_files) = match self.config_loader.load() {
            Ok((config, files)) => (config, files),
            Err(e) => {
                log::warn!("Failed to load config files: {}. Using defaults.", e);
                (GemicroConfig::default(), Vec::new())
            }
        };

        self.register_agents_from_config(&config);

        loaded_files
    }

    /// Register agents based on config.
    ///
    /// Delegates to the shared registry module for builtin agents, then loads
    /// markdown agents from the runtime-agents directory.
    fn register_agents_from_config(&mut self, _config: &GemicroConfig) {
        // Build registry options from CLI overrides
        let options = match &self.cli_overrides.model {
            Some(model) => RegistryOptions::default().with_model(model),
            None => RegistryOptions::default(),
        };

        // Acquire write lock and register all agents
        let mut registry = self.registry.write().expect("agent registry lock poisoned");

        // Register builtin agents (deep_research, prompt_agent, developer, react, echo, critique)
        register_builtin_agents(&mut registry, &options);

        // Register bundled markdown agents from runtime-agents directory
        register_markdown_agents(
            &mut registry,
            &options,
            std::path::Path::new(RUNTIME_AGENTS_DIR),
        );
    }

    /// Reload configuration from files.
    ///
    /// Re-reads config files, shows what changed, and re-registers agents.
    /// Preserves conversation history across reload.
    pub fn reload(&mut self) -> Result<ReloadResult> {
        let old_config = self.config_loader.last_config().clone();

        let (new_config, loaded_files) = self
            .config_loader
            .load()
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        let changes = self.config_loader.diff(&new_config);

        // Re-register agents with new config
        self.register_agents_from_config(&new_config);

        Ok(ReloadResult {
            loaded_files,
            changes,
            old_config,
            new_config,
        })
    }

    /// Check if config files have changed since last load.
    pub fn is_config_stale(&self) -> bool {
        self.config_loader.is_stale()
    }

    /// Set the current agent by name.
    ///
    /// This should be called after registering agents to set the initial selection.
    /// Resets the session token count when switching to a different agent.
    /// Returns `Ok(())` if the agent exists, `Err` with message otherwise.
    pub fn set_current_agent(&mut self, name: &str) -> Result<(), String> {
        let registry = self
            .registry
            .read()
            .map_err(|_| "Agent registry lock poisoned".to_string())?;

        if registry.contains(name) {
            drop(registry); // Release lock before modifying self
                            // Reset token count when switching agents
            if self.current_agent_name != name {
                self.session_tokens = 0;
            }
            self.current_agent_name = name.to_string();
            Ok(())
        } else {
            let available = registry.list().join(", ");
            Err(format!(
                "Unknown agent '{}'. Available: {}",
                name, available
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

    /// Get the current agent context with cancellation support.
    ///
    /// Tools are always provided to agents. Each agent controls which tools it
    /// uses via its `tool_filter` configuration:
    /// - `ToolSet::None` → agent won't use any tools
    /// - `ToolSet::Inherit` → use all available tools
    /// - `ToolSet::Specific([...])` → use only listed tools
    ///
    /// If sandbox paths are configured, interceptors are attached for security.
    fn agent_context(&self, cancellation_token: CancellationToken) -> AgentContext {
        AgentContext {
            llm: self.llm.clone(),
            cancellation_token,
            tools: Some(Arc::clone(&self.tool_registry)),
            confirmation_handler: Some(Arc::clone(&self.confirmation_handler)),
            execution: gemicro_core::ExecutionContext::root(),
            orchestration: None,
            interceptors: self.interceptors.clone(),
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
        let agent = {
            let registry = self
                .registry
                .read()
                .map_err(|_| anyhow::anyhow!("Agent registry lock poisoned"))?;
            registry
                .get(&self.current_agent_name)
                .context("Selected agent no longer available")?
        };

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

                    // Handle event-specific rendering first
                    renderer
                        .on_event(&update)
                        .context("Renderer event handling failed")?;

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
            // Build prompt with optional token count and stale indicators
            let binary_stale = if self.is_stale() { " [stale]" } else { "" };
            let config_stale = if self.is_config_stale() {
                " [config]"
            } else {
                ""
            };
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
            let prompt = format!(
                "[{}{}{}{}] > ",
                agent_display, token_display, binary_stale, config_stale
            );

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
                            if let Ok(registry) = self.registry.read() {
                                for name in registry.list() {
                                    let marker = if name == self.current_agent_name() {
                                        " *"
                                    } else {
                                        ""
                                    };
                                    if let Some(agent) = registry.get(name) {
                                        println!("  {}{} - {}", name, marker, agent.description());
                                    }
                                }
                            } else {
                                eprintln!("Error: Agent registry lock poisoned");
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
                        Command::Reload => match self.reload() {
                            Ok(result) => {
                                if result.loaded_files.is_empty() {
                                    println!("No config files found.");
                                    println!(
                                        "  Create {} or {}",
                                        ConfigLoader::local_config_path().display(),
                                        ConfigLoader::global_config_path().display()
                                    );
                                } else {
                                    println!("Reloaded config from:");
                                    for path in &result.loaded_files {
                                        println!("  {}", path.display());
                                    }

                                    if result.has_changes() {
                                        println!("\nChanges:");
                                        for change in &result.changes {
                                            println!("  {}", change);
                                        }
                                    } else {
                                        println!("\nNo changes detected.");
                                    }
                                }
                                println!("\nAgents re-registered with updated config.");
                            }
                            Err(e) => {
                                eprintln!("Reload failed: {:#}", e);
                            }
                        },
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
    use async_stream::stream;
    use gemicro_core::{
        Agent, AgentStream, DefaultTracker, ExecutionTracking, LlmConfig, ResultMetadata,
    };
    use serde_json::json;

    // ========================================================================
    // Mock Agent for Cancellation Testing
    // ========================================================================

    /// A mock agent that yields events and can be cancelled mid-execution.
    ///
    /// This agent yields a configurable number of events, then checks for
    /// cancellation before emitting the final result. It's designed to test:
    /// - CancellationToken triggering graceful exit
    /// - AgentError::Cancelled handling
    /// - Partial state preservation after cancellation
    struct MockCancellableAgent {
        /// Number of events to emit before checking cancellation
        events_before_cancel_check: usize,
    }

    impl MockCancellableAgent {
        fn new(events_before_cancel_check: usize) -> Self {
            Self {
                events_before_cancel_check,
            }
        }
    }

    impl Agent for MockCancellableAgent {
        fn name(&self) -> &str {
            "mock_cancellable"
        }

        fn description(&self) -> &str {
            "A mock agent for testing cancellation"
        }

        fn execute(&self, query: &str, context: AgentContext) -> AgentStream<'_> {
            let query = query.to_string();
            let events_before_check = self.events_before_cancel_check;

            Box::pin(stream! {
                // Emit start event
                yield Ok(AgentUpdate::custom(
                    "mock_started",
                    format!("Processing: {}", query),
                    json!({ "query": query }),
                ));

                // Emit intermediate events
                for i in 0..events_before_check {
                    yield Ok(AgentUpdate::custom(
                        "mock_progress",
                        format!("Step {} of {}", i + 1, events_before_check),
                        json!({ "step": i + 1 }),
                    ));
                }

                // Check cancellation - this is a non-blocking check
                if context.cancellation_token.is_cancelled() {
                    yield Err(AgentError::Cancelled);
                    return;
                }

                // Emit final result
                yield Ok(AgentUpdate::final_result(
                    json!("Mock answer"),
                    ResultMetadata::new(100, 0, 50),
                ));
            })
        }

        fn create_tracker(&self) -> Box<dyn ExecutionTracking> {
            Box::new(DefaultTracker::default())
        }
    }

    /// A mock agent that waits for cancellation signal before completing.
    ///
    /// This agent uses a notify to properly synchronize with the test:
    /// 1. Emits start event
    /// 2. Emits N progress events
    /// 3. Notifies ready_signal that it's ready
    /// 4. Waits for either cancellation or proceed signal
    /// 5. Returns Cancelled if token was cancelled, or emits final result
    struct SyncMockAgent {
        events_before_check: usize,
        ready_signal: Arc<tokio::sync::Notify>,
        proceed_signal: Arc<tokio::sync::Notify>,
    }

    impl SyncMockAgent {
        fn new(
            events_before_check: usize,
            ready_signal: Arc<tokio::sync::Notify>,
            proceed_signal: Arc<tokio::sync::Notify>,
        ) -> Self {
            Self {
                events_before_check,
                ready_signal,
                proceed_signal,
            }
        }
    }

    impl Agent for SyncMockAgent {
        fn name(&self) -> &str {
            "sync_mock"
        }

        fn description(&self) -> &str {
            "A synchronized mock agent for testing cancellation"
        }

        fn execute(&self, query: &str, context: AgentContext) -> AgentStream<'_> {
            let query = query.to_string();
            let events_before_check = self.events_before_check;
            let ready_signal = self.ready_signal.clone();
            let proceed_signal = self.proceed_signal.clone();

            Box::pin(stream! {
                // Emit start event
                yield Ok(AgentUpdate::custom(
                    "mock_started",
                    format!("Processing: {}", query),
                    json!({ "query": query }),
                ));

                // Emit intermediate events
                for i in 0..events_before_check {
                    yield Ok(AgentUpdate::custom(
                        "mock_progress",
                        format!("Step {} of {}", i + 1, events_before_check),
                        json!({ "step": i + 1 }),
                    ));
                }

                // Signal that we're ready for cancellation
                ready_signal.notify_one();

                // Wait for either cancellation or proceed signal
                let was_cancelled = tokio::select! {
                    biased;
                    _ = context.cancellation_token.cancelled() => true,
                    _ = proceed_signal.notified() => false,
                };

                if was_cancelled {
                    yield Err(AgentError::Cancelled);
                    return;
                }

                // Emit final result
                yield Ok(AgentUpdate::final_result(
                    json!("Mock answer"),
                    ResultMetadata::new(100, 0, 50),
                ));
            })
        }

        fn create_tracker(&self) -> Box<dyn ExecutionTracking> {
            Box::new(DefaultTracker::default())
        }
    }

    // Note: Most session tests require a mock LLM client which we don't have yet.
    // These are basic structural tests.

    #[test]
    fn test_session_stale_detection_fresh() {
        // A fresh session should not be stale
        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let llm = LlmClient::new(genai_client, LlmConfig::default());
        let session = Session::new(llm, false);

        // Fresh session should not be stale
        assert!(!session.is_stale());
    }

    #[test]
    fn test_session_initial_token_count() {
        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
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
                json!("The answer"),
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

    // ========================================================================
    // Cancellation Tests
    // ========================================================================
    //
    // These tests verify the cancellation behavior in run_query:
    // 1. CancellationToken triggers graceful exit
    // 2. AgentError::Cancelled is handled as non-error (returns Ok())
    // 3. Partial events are preserved after cancellation
    // 4. Multiple cancellation attempts (Ctrl+C debouncing) work correctly

    /// Test that cancellation via CancellationToken triggers a graceful exit.
    ///
    /// When the cancellation token is triggered:
    /// - The agent should return AgentError::Cancelled
    /// - run_query should return Ok(()) (not an error)
    /// - The renderer should show interrupted state
    #[tokio::test]
    async fn test_cancellation_token_triggers_graceful_exit() {
        // Create synchronization signals
        let ready_signal = Arc::new(tokio::sync::Notify::new());
        let proceed_signal = Arc::new(tokio::sync::Notify::new());

        let mock_agent = SyncMockAgent::new(3, ready_signal.clone(), proceed_signal.clone());

        // Set up cancellation token that will be triggered during execution
        let cancellation_token = CancellationToken::new();
        let cancel_token_clone = cancellation_token.clone();

        // Create agent context
        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let llm = LlmClient::new(genai_client, LlmConfig::default());
        let context = AgentContext::new_with_cancellation(llm, cancellation_token);

        // Run agent execution in a separate task
        let agent = Arc::new(mock_agent);
        let agent_clone = agent.clone();
        let exec_handle = tokio::spawn(async move {
            let stream = agent_clone.execute("test query", context);
            let stream = enforce_final_result_contract(stream);
            futures_util::pin_mut!(stream);

            let mut events = Vec::new();
            let mut got_cancelled_error = false;

            while let Some(result) = stream.next().await {
                match result {
                    Ok(update) => events.push(update),
                    Err(AgentError::Cancelled) => {
                        got_cancelled_error = true;
                        break;
                    }
                    Err(e) => panic!("Unexpected error: {:?}", e),
                }
            }

            (events, got_cancelled_error)
        });

        // Wait for agent to signal it's ready, then trigger cancellation
        tokio::time::timeout(std::time::Duration::from_secs(5), ready_signal.notified())
            .await
            .expect("Agent should signal readiness within 5 seconds");
        cancel_token_clone.cancel();

        // Wait for execution to complete
        let (events, got_cancelled_error) = exec_handle.await.unwrap();

        // Verify cancellation behavior
        assert!(
            got_cancelled_error,
            "Should have received AgentError::Cancelled"
        );

        // Verify partial events were collected (start + progress events).
        // Cancellation is only checked after all progress events are emitted,
        // so we expect: 1 start + 3 progress = 4 events
        assert_eq!(
            events.len(),
            4,
            "Should have 4 events before cancellation check"
        );
        assert_eq!(
            events[0].event_type, "mock_started",
            "First event should be mock_started"
        );
    }

    /// Test that AgentError::Cancelled is treated as a non-error condition.
    ///
    /// The run_query method should return Ok(()) when cancelled, not propagate
    /// the error, because cancellation is a normal user-initiated flow.
    #[tokio::test]
    async fn test_cancelled_error_treated_as_non_error() {
        // Create agent that yields cancelled error after one event
        struct CancellingAgent;

        impl Agent for CancellingAgent {
            fn name(&self) -> &str {
                "cancelling_agent"
            }

            fn description(&self) -> &str {
                "Agent that returns Cancelled after one event"
            }

            fn execute(&self, _query: &str, _context: AgentContext) -> AgentStream<'_> {
                Box::pin(stream! {
                    // Emit one event before returning cancelled
                    yield Ok(AgentUpdate::custom(
                        "starting",
                        "Starting execution",
                        json!({}),
                    ));
                    // Return cancelled error
                    yield Err(AgentError::Cancelled);
                })
            }

            fn create_tracker(&self) -> Box<dyn ExecutionTracking> {
                Box::new(DefaultTracker::default())
            }
        }

        // Execute the stream
        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let llm = LlmClient::new(genai_client, LlmConfig::default());
        let context = AgentContext::new(llm);

        let agent = CancellingAgent;
        let stream = agent.execute("test", context);
        let stream = enforce_final_result_contract(stream);
        futures_util::pin_mut!(stream);

        // Simulate the run_query cancellation handling logic
        let mut is_cancelled = false;
        let mut event_count = 0;
        while let Some(result) = stream.next().await {
            match result {
                Ok(_update) => event_count += 1,
                Err(AgentError::Cancelled) => {
                    // This is the key assertion: Cancelled should be caught
                    // and treated as a graceful exit, not an error
                    is_cancelled = true;
                    break;
                }
                Err(e) => panic!("Unexpected error type: {:?}", e),
            }
        }

        assert!(
            is_cancelled,
            "Should have detected Cancelled and exited gracefully"
        );
        assert_eq!(
            event_count, 1,
            "Should have received one event before cancellation"
        );
    }

    /// Test that partial events are preserved after cancellation.
    ///
    /// When cancellation occurs mid-execution:
    /// - Events emitted before cancellation should be collected
    /// - The stream should terminate cleanly after Cancelled error
    #[tokio::test]
    async fn test_partial_events_preserved_after_cancellation() {
        let ready_signal = Arc::new(tokio::sync::Notify::new());
        let proceed_signal = Arc::new(tokio::sync::Notify::new());
        let mock_agent = SyncMockAgent::new(5, ready_signal.clone(), proceed_signal.clone());

        let cancellation_token = CancellationToken::new();
        let cancel_token_clone = cancellation_token.clone();

        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let llm = LlmClient::new(genai_client, LlmConfig::default());
        let context = AgentContext::new_with_cancellation(llm, cancellation_token);

        // Run execution
        let agent = Arc::new(mock_agent);
        let agent_clone = agent.clone();
        let exec_handle = tokio::spawn(async move {
            let stream = agent_clone.execute("test query", context);
            futures_util::pin_mut!(stream);

            let mut events = Vec::new();
            while let Some(result) = stream.next().await {
                match result {
                    Ok(update) => events.push(update),
                    Err(AgentError::Cancelled) => break,
                    Err(e) => panic!("Unexpected error: {:?}", e),
                }
            }
            events
        });

        // Wait for agent to be ready, then cancel
        tokio::time::timeout(std::time::Duration::from_secs(5), ready_signal.notified())
            .await
            .expect("Agent should signal readiness within 5 seconds");
        cancel_token_clone.cancel();

        let events = exec_handle.await.unwrap();

        // Should have: 1 start + 5 progress = 6 events before cancellation
        assert_eq!(events.len(), 6, "Should have 6 events before cancellation");

        // Explicit event type checking for each expected event
        assert_eq!(
            events[0].event_type, "mock_started",
            "First event should be start"
        );
        assert_eq!(
            events[1].event_type, "mock_progress",
            "Event 1 should be progress"
        );
        assert_eq!(
            events[2].event_type, "mock_progress",
            "Event 2 should be progress"
        );
        assert_eq!(
            events[3].event_type, "mock_progress",
            "Event 3 should be progress"
        );
        assert_eq!(
            events[4].event_type, "mock_progress",
            "Event 4 should be progress"
        );
        assert_eq!(
            events[5].event_type, "mock_progress",
            "Event 5 should be progress"
        );

        // Verify step data matches expected sequence
        for (i, event) in events.iter().skip(1).enumerate() {
            let step = event.data.get("step").and_then(|v| v.as_u64()).unwrap();
            assert_eq!(
                step,
                (i + 1) as u64,
                "Step {} should have correct data",
                i + 1
            );
        }
    }

    /// Test that interrupt counting works correctly for multiple Ctrl+C presses.
    ///
    /// The interrupt counter should:
    /// - Trigger cancellation on first press
    /// - Not cause issues on subsequent presses
    #[test]
    fn test_interrupt_counter_debouncing() {
        let interrupt_count = Arc::new(AtomicU8::new(0));

        // Simulate first Ctrl+C
        let count1 = interrupt_count.fetch_add(1, Ordering::SeqCst) + 1;
        assert_eq!(count1, 1, "First interrupt should be count 1");

        // Simulate second Ctrl+C
        let count2 = interrupt_count.fetch_add(1, Ordering::SeqCst) + 1;
        assert_eq!(count2, 2, "Second interrupt should be count 2");

        // Verify the check logic
        let is_interrupted = || interrupt_count.load(Ordering::SeqCst) > 0;
        assert!(
            is_interrupted(),
            "Should report interrupted after first Ctrl+C"
        );

        // Verify the debounce logic (count > 1 means already cancelling)
        let is_already_cancelling = interrupt_count.load(Ordering::SeqCst) > 1;
        assert!(
            is_already_cancelling,
            "Should detect already-cancelling state"
        );
    }

    /// Test that pre-cancelled token causes exit at next cancellation check.
    ///
    /// If the cancellation token is already cancelled when execution starts,
    /// the agent emits all events before its cancellation check, then exits
    /// with Cancelled error. This verifies the check happens at the expected point.
    #[tokio::test]
    async fn test_pre_cancelled_token_exits_at_check() {
        let mock_agent = MockCancellableAgent::new(10); // Would emit 10 progress events

        // Pre-cancel the token
        let cancellation_token = CancellationToken::new();
        cancellation_token.cancel();

        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let llm = LlmClient::new(genai_client, LlmConfig::default());
        let context = AgentContext::new_with_cancellation(llm, cancellation_token);

        let stream = mock_agent.execute("test query", context);
        futures_util::pin_mut!(stream);

        let mut event_count = 0;
        let mut got_cancelled = false;

        while let Some(result) = stream.next().await {
            match result {
                Ok(_) => event_count += 1,
                Err(AgentError::Cancelled) => {
                    got_cancelled = true;
                    break;
                }
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        assert!(got_cancelled, "Should get Cancelled error");
        // With pre-cancelled token, the agent checks cancellation after emitting
        // start + progress events: 1 start + 10 progress = 11 events total
        assert_eq!(
            event_count, 11,
            "Should emit exactly 11 events (1 start + 10 progress) before cancellation check"
        );
    }

    /// Test that normal completion (no cancellation) works correctly.
    ///
    /// Without cancellation, the agent should emit all events including final_result.
    #[tokio::test]
    async fn test_normal_completion_without_cancellation() {
        let mock_agent = MockCancellableAgent::new(3);

        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let llm = LlmClient::new(genai_client, LlmConfig::default());
        let context = AgentContext::new(llm);

        let stream = mock_agent.execute("test query", context);
        let stream = enforce_final_result_contract(stream);
        futures_util::pin_mut!(stream);

        let mut events = Vec::new();
        while let Some(result) = stream.next().await {
            events.push(result.expect("Should not error without cancellation"));
        }

        // Should have: 1 start + 3 progress + 1 final_result = 5 events
        assert_eq!(events.len(), 5, "Should have 5 events total");
        assert_eq!(events[0].event_type, "mock_started");
        assert_eq!(events[1].event_type, "mock_progress");
        assert_eq!(events[2].event_type, "mock_progress");
        assert_eq!(events[3].event_type, "mock_progress");
        assert_eq!(events[4].event_type, "final_result");
    }

    /// Test that cancellation takes precedence over timeout.
    ///
    /// The `with_timeout_and_cancellation` helper uses `biased` in `tokio::select!`,
    /// meaning cancellation is checked first. When both are triggered simultaneously,
    /// cancellation should win.
    #[tokio::test]
    async fn test_cancellation_takes_precedence_over_timeout() {
        use gemicro_core::with_timeout_and_cancellation;
        use std::time::Duration;

        let cancellation_token = CancellationToken::new();

        // Pre-cancel the token before calling
        cancellation_token.cancel();

        // Use a generous timeout - but cancellation should win
        let result = with_timeout_and_cancellation(
            async { Ok::<_, AgentError>("should not reach") },
            Duration::from_secs(60),
            &cancellation_token,
            || AgentError::Timeout {
                elapsed_ms: 60_000,
                timeout_ms: 60_000,
                phase: "test".to_string(),
            },
        )
        .await;

        assert!(
            matches!(result, Err(AgentError::Cancelled)),
            "Cancellation should take precedence over timeout, got: {:?}",
            result
        );
    }

    /// Test that timeout triggers when cancellation doesn't happen.
    ///
    /// Without cancellation, the timeout should trigger after the specified duration.
    #[tokio::test]
    async fn test_timeout_triggers_without_cancellation() {
        use gemicro_core::with_timeout_and_cancellation;
        use std::time::Duration;

        let cancellation_token = CancellationToken::new();

        // Use a very short timeout with a long-running future
        let result = with_timeout_and_cancellation(
            async {
                tokio::time::sleep(Duration::from_secs(60)).await;
                Ok::<_, AgentError>("should not reach")
            },
            Duration::from_millis(10),
            &cancellation_token,
            || AgentError::Timeout {
                elapsed_ms: 10,
                timeout_ms: 10,
                phase: "test".to_string(),
            },
        )
        .await;

        assert!(
            matches!(result, Err(AgentError::Timeout { .. })),
            "Should timeout without cancellation, got: {:?}",
            result
        );
    }
}
