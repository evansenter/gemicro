//! Agent infrastructure.
//!
//! This module provides the core [`Agent`] trait for building AI agents,
//! along with the [`AgentContext`] for shared resources across agent executions.
//!
//! ## Design Philosophy
//!
//! Following [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) principles:
//!
//! - **Soft-typed events**: Agents return streams of [`AgentUpdate`] with flexible JSON payloads
//! - **Agent-specific config**: Each agent owns its configuration (passed to constructor)
//! - **Minimal shared context**: [`AgentContext`] contains only cross-agent resources (LLM client)
//! - **Streaming-first**: Real-time observability via async streams
//!
//! ## Subagent Orchestration
//!
//! For agents that spawn subagents:
//! - [`ExecutionContext`]: Tracks parent-child relationships and execution depth
//! - [`SubagentConfig`]: Controls resource isolation for subagents
//! - [`PromptAgentDef`]: Defines ephemeral prompt-based agents inline
//!
//! ## Agent Implementations
//!
//! Agents are in separate crates for hermetic isolation:
//! - `gemicro-deep-research`: Decomposes queries, executes sub-queries in parallel, synthesizes
//! - `gemicro-react`: Reasoning + Acting pattern with iterative tool use
//! - `gemicro-simple-qa`: Minimal single-call agent for reference/demonstration
//! - `gemicro-tool-agent`: Native function calling via rust-genai's `#[tool]` macro
//! - `gemicro-judge`: LLM-based semantic evaluation for scoring
//!
//! ## Example
//!
//! ```no_run
//! use gemicro_core::{
//!     Agent, AgentContext, AgentStream, AgentUpdate, AgentError,
//!     LlmClient, LlmConfig, ExecutionTracking, DefaultTracker,
//! };
//!
//! struct MyAgent;
//!
//! impl Agent for MyAgent {
//!     fn name(&self) -> &str { "my_agent" }
//!     fn description(&self) -> &str { "A custom agent" }
//!     fn execute(&self, query: &str, context: AgentContext) -> AgentStream<'_> {
//!         // Implementation would return a stream of updates
//!         # todo!()
//!     }
//!     fn create_tracker(&self) -> Box<dyn ExecutionTracking> {
//!         Box::new(DefaultTracker::default())
//!     }
//! }
//! ```

mod ephemeral;
mod execution;
mod subagent;

pub use ephemeral::{AgentSpec, PromptAgentDef};
pub use execution::{ExecutionContext, ExecutionId};
pub use subagent::{SubagentConfig, DEFAULT_SUBAGENT_TIMEOUT_SECS};

use crate::error::AgentError;
use crate::llm::LlmClient;
use crate::tool::{ConfirmationHandler, Tool, ToolRegistry};
use crate::tracking::ExecutionTracking;
use crate::update::AgentUpdate;

use futures_util::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// Type alias for the boxed, pinned stream returned by agents.
///
/// This allows agents to return different stream implementations while
/// maintaining a consistent interface for consumers.
pub type AgentStream<'a> = Pin<Box<dyn Stream<Item = Result<AgentUpdate, AgentError>> + Send + 'a>>;

/// Trait for AI agents that process queries and return streaming updates.
///
/// Agents are the core abstraction in Gemicro. Each agent implements a specific
/// pattern (Deep Research, ReAct, Reflexion, etc.) and returns a stream of
/// [`AgentUpdate`] events for real-time observability.
///
/// # Design Notes
///
/// - Config belongs to the agent constructor, not the execute method
/// - [`AgentContext`] provides only shared resources (LLM client, cancellation)
/// - Events are soft-typed via [`AgentUpdate`] for extensibility
///
/// # Example
///
/// ```no_run
/// use gemicro_core::{
///     Agent, AgentContext, AgentStream, AgentUpdate, AgentError,
///     ExecutionTracking, DefaultTracker,
/// };
///
/// struct MyAgent;
///
/// impl Agent for MyAgent {
///     fn name(&self) -> &str { "my_agent" }
///     fn description(&self) -> &str { "A custom agent" }
///     fn execute(&self, query: &str, context: AgentContext) -> AgentStream<'_> {
///         // Implementation would return a stream of updates
///         # todo!()
///     }
///     fn create_tracker(&self) -> Box<dyn ExecutionTracking> {
///         Box::new(DefaultTracker::default())
///     }
/// }
/// ```
pub trait Agent: Send + Sync {
    /// Unique identifier for this agent type.
    fn name(&self) -> &str;

    /// Human-readable description of what this agent does.
    fn description(&self) -> &str;

    /// Execute the agent's logic and return a stream of updates.
    ///
    /// The stream yields [`AgentUpdate`] events as the agent progresses.
    /// Consumers should handle unknown event types gracefully (log and ignore).
    fn execute(&self, query: &str, context: AgentContext) -> AgentStream<'_>;

    /// Create a tracker for this agent's execution.
    ///
    /// Each agent provides its own tracker implementation that understands
    /// its specific event types and execution patterns. This enables the CLI
    /// and runner to remain agent-agnostic while still providing meaningful
    /// progress updates.
    ///
    /// For simple agents without complex tracking needs, use [`DefaultTracker`]:
    ///
    /// ```text
    /// fn create_tracker(&self) -> Box<dyn ExecutionTracking> {
    ///     Box::new(DefaultTracker::default())
    /// }
    /// ```
    ///
    /// [`DefaultTracker`]: crate::DefaultTracker
    fn create_tracker(&self) -> Box<dyn ExecutionTracking>;
}

/// Shared resources available to all agents during execution.
///
/// Contains only cross-agent concerns. Agent-specific configuration
/// belongs in the agent's constructor, not here.
///
/// # Design Note
///
/// Following [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) philosophy,
/// this struct intentionally contains only resources that ALL agents need.
/// Do not add agent-specific config here.
///
/// # Subagent Orchestration
///
/// When spawning subagents, use [`Self::child_context()`] to create a derived context
/// that tracks parent-child relationships:
///
/// ```text
/// let child_ctx = context.child_context("simple_qa");
/// let stream = subagent.execute(query, child_ctx);
/// ```
#[derive(Clone)]
pub struct AgentContext {
    /// Shared LLM client for making API calls.
    pub llm: Arc<LlmClient>,

    /// Cancellation token for cooperative shutdown.
    ///
    /// Agents should check this token periodically and abort gracefully
    /// when cancelled, returning partial results if possible.
    pub cancellation_token: CancellationToken,

    /// Optional tool registry for agents that use tools.
    ///
    /// Not all agents use tools, so this is optional. Agents that need
    /// tools should check for this and either use a default registry
    /// or fail gracefully if tools are required but not provided.
    pub tools: Option<Arc<ToolRegistry>>,

    /// Optional confirmation handler for tools that require user approval.
    ///
    /// When set, tools that return `true` from [`Tool::requires_confirmation`]
    /// will call this handler before execution. If the handler returns `false`,
    /// the tool invocation is denied.
    ///
    /// If not set, tools requiring confirmation will be denied by default.
    pub confirmation_handler: Option<Arc<dyn ConfirmationHandler>>,

    /// Execution context for tracking parent-child agent relationships.
    ///
    /// Used for observability, debugging, and depth limiting. Defaults to
    /// a root context; use [`Self::child_context()`] when spawning subagents.
    pub execution: ExecutionContext,
}

impl AgentContext {
    /// Create a new context with an owned LLM client.
    ///
    /// The client will be wrapped in an Arc for sharing across tasks.
    /// Uses a new cancellation token (never cancelled unless explicitly triggered).
    /// No tools are registered by default. Creates a root execution context.
    pub fn new(llm: LlmClient) -> Self {
        Self {
            llm: Arc::new(llm),
            cancellation_token: CancellationToken::new(),
            tools: None,
            confirmation_handler: None,
            execution: ExecutionContext::root(),
        }
    }

    /// Create a new context with cancellation support.
    ///
    /// Use this when you need to cancel in-flight operations (e.g., on Ctrl+C).
    pub fn new_with_cancellation(llm: LlmClient, cancellation_token: CancellationToken) -> Self {
        Self {
            llm: Arc::new(llm),
            cancellation_token,
            tools: None,
            confirmation_handler: None,
            execution: ExecutionContext::root(),
        }
    }

    /// Create a context from an existing shared LLM client.
    ///
    /// Useful when multiple agents share the same client instance.
    pub fn from_arc(llm: Arc<LlmClient>) -> Self {
        Self {
            llm,
            cancellation_token: CancellationToken::new(),
            tools: None,
            confirmation_handler: None,
            execution: ExecutionContext::root(),
        }
    }

    /// Add a tool registry to this context.
    ///
    /// Returns a new context with the tools field set.
    pub fn with_tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = Some(Arc::new(tools));
        self
    }

    /// Add a shared tool registry to this context.
    ///
    /// Useful when sharing a registry across multiple contexts.
    pub fn with_tools_arc(mut self, tools: Arc<ToolRegistry>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Add a confirmation handler for tools that require user approval.
    ///
    /// Tools that return `true` from [`Tool::requires_confirmation`] will
    /// call this handler before execution.
    pub fn with_confirmation_handler(mut self, handler: Arc<dyn ConfirmationHandler>) -> Self {
        self.confirmation_handler = Some(handler);
        self
    }

    /// Set a specific execution context.
    ///
    /// Use this to create a context with a non-root execution context,
    /// typically when manually constructing contexts for testing or
    /// specialized use cases. For normal subagent spawning, use
    /// [`Self::child_context()`] instead.
    pub fn with_execution(mut self, execution: ExecutionContext) -> Self {
        self.execution = execution;
        self
    }

    /// Create a child context for spawning a subagent.
    ///
    /// The child context:
    /// - Shares the same LLM client
    /// - Inherits the cancellation token (so cancelling parent cancels child)
    /// - Inherits tools and confirmation handler
    /// - Creates a new execution context with this agent as parent
    ///
    /// # Example
    ///
    /// ```text
    /// fn spawn_subagent(&self, agent: &dyn Agent, query: &str, context: &AgentContext) {
    ///     let child_ctx = context.child_context(agent.name());
    ///     let stream = agent.execute(query, child_ctx);
    ///     // Process stream...
    /// }
    /// ```
    pub fn child_context(&self, agent_name: &str) -> Self {
        Self {
            llm: Arc::clone(&self.llm),
            cancellation_token: self.cancellation_token.clone(),
            tools: self.tools.clone(),
            confirmation_handler: self.confirmation_handler.clone(),
            execution: self.execution.child(agent_name),
        }
    }

    /// Get a tool by name from the registry.
    ///
    /// Returns `None` if no registry is set or the tool doesn't exist.
    pub fn get_tool(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.as_ref().and_then(|reg| reg.get(name))
    }
}

// ============================================================================
// Helper functions for agent implementations
// ============================================================================

/// Calculate remaining time from a total timeout budget.
///
/// Returns an error if the timeout has already been exceeded.
///
/// This is a public helper for agent implementations in separate crates.
pub fn remaining_time(
    start: Instant,
    total_timeout: Duration,
    phase: &str,
) -> Result<Duration, AgentError> {
    let elapsed = start.elapsed();
    if elapsed >= total_timeout {
        return Err(AgentError::Timeout {
            elapsed_ms: elapsed.as_millis() as u64,
            timeout_ms: total_timeout.as_millis() as u64,
            phase: phase.to_string(),
        });
    }
    Ok(total_timeout - elapsed)
}

/// Create a timeout error with current elapsed time.
///
/// This is a public helper for agent implementations in separate crates.
pub fn timeout_error(start: Instant, total_timeout: Duration, phase: &str) -> AgentError {
    AgentError::Timeout {
        elapsed_ms: start.elapsed().as_millis() as u64,
        timeout_ms: total_timeout.as_millis() as u64,
        phase: phase.to_string(),
    }
}

/// Execute a future with timeout and cancellation support.
///
/// Returns the future's result, or an error if cancelled/timed out.
///
/// This is a public helper for agent implementations in separate crates.
pub async fn with_timeout_and_cancellation<F, T>(
    future: F,
    timeout: Duration,
    cancellation_token: &CancellationToken,
    timeout_error_fn: impl FnOnce() -> AgentError,
) -> Result<T, AgentError>
where
    F: std::future::Future<Output = Result<T, AgentError>>,
{
    tokio::select! {
        biased;

        _ = cancellation_token.cancelled() => {
            Err(AgentError::Cancelled)
        }

        result = tokio::time::timeout(timeout, future) => {
            match result {
                Ok(inner) => inner,
                Err(_) => Err(timeout_error_fn()),
            }
        }
    }
}

// ============================================================================
// Contract enforcement
// ============================================================================

/// The event type that signals agent completion.
pub const EVENT_FINAL_RESULT: &str = "final_result";

/// Wraps an agent stream to enforce the `final_result` contract.
///
/// The event contract states that `final_result` **MUST** be the last event
/// emitted by any agent. This wrapper detects violations and logs warnings
/// when events are yielded after `final_result`.
///
/// # Why This Matters
///
/// If an agent yields events after `final_result`:
/// - Some consumers might ignore them (already in "done" state)
/// - Others might process them unexpectedly
/// - Metrics could be wrong (duration/tokens already captured)
///
/// # Usage
///
/// ```no_run
/// use gemicro_core::{Agent, AgentContext, enforce_final_result_contract};
///
/// async fn run_agent(agent: &dyn Agent, query: &str, context: AgentContext) {
///     let stream = agent.execute(query, context);
///     let validated_stream = enforce_final_result_contract(stream);
///     // Use validated_stream instead of stream
/// }
/// ```
///
/// # Behavior
///
/// - Events before `final_result` pass through unchanged
/// - The `final_result` event passes through unchanged
/// - Events after `final_result` pass through but trigger a warning log
/// - Errors pass through unchanged (no contract checking on errors)
///
/// # Future Considerations
///
/// Currently logs warnings for violations. Once all agents are verified
/// compliant, this could be promoted to return errors for violations.
pub fn enforce_final_result_contract(stream: AgentStream<'_>) -> AgentStream<'_> {
    let seen_final = Arc::new(AtomicBool::new(false));

    Box::pin(stream.map(move |result| {
        if let Ok(ref update) = result {
            if seen_final.load(Ordering::SeqCst) {
                log::warn!(
                    "Contract violation: event '{}' yielded after final_result (message: '{}')",
                    update.event_type,
                    update.message
                );
            }
            if update.event_type == EVENT_FINAL_RESULT {
                seen_final.store(true, Ordering::SeqCst);
            }
        }
        result
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Timeout enforcement tests

    /// Verifies that remaining_time returns Ok with correct duration when
    /// sufficient time remains in the total timeout budget.
    #[test]
    fn test_remaining_time_not_exceeded() {
        let start = Instant::now();
        let total_timeout = Duration::from_secs(10);

        let result = remaining_time(start, total_timeout, "test_phase");
        assert!(result.is_ok());

        let remaining = result.unwrap();
        // Should have close to 10 seconds remaining (minus small elapsed time)
        assert!(remaining > Duration::from_secs(9));
        assert!(remaining <= Duration::from_secs(10));
    }

    /// Verifies that remaining_time returns a Timeout error when elapsed time
    /// exceeds the total timeout budget.
    #[test]
    fn test_remaining_time_exceeded() {
        let start = Instant::now();
        // Use 10ms timeout with 50ms sleep for reliable CI behavior
        let total_timeout = Duration::from_millis(10);

        std::thread::sleep(Duration::from_millis(50));

        let result = remaining_time(start, total_timeout, "decomposition");
        assert!(result.is_err());

        match result.unwrap_err() {
            AgentError::Timeout {
                elapsed_ms,
                timeout_ms,
                phase,
            } => {
                assert!(elapsed_ms >= 50); // At least 50ms elapsed
                assert_eq!(timeout_ms, 10);
                assert_eq!(phase, "decomposition");
            }
            _ => panic!("Expected Timeout error"),
        }
    }

    /// Verifies that the exact boundary condition (elapsed == timeout) triggers
    /// a timeout error, since we use >= comparison.
    #[test]
    fn test_remaining_time_exact_boundary() {
        let start = Instant::now();
        let total_timeout = Duration::from_millis(20);

        // Sleep slightly past the timeout to ensure we hit the boundary
        std::thread::sleep(Duration::from_millis(25));

        let result = remaining_time(start, total_timeout, "boundary_test");
        // At or past boundary should error
        assert!(result.is_err());

        match result.unwrap_err() {
            AgentError::Timeout { phase, .. } => {
                assert_eq!(phase, "boundary_test");
            }
            _ => panic!("Expected Timeout error"),
        }
    }

    /// Verifies that the phase name is correctly preserved in timeout errors
    /// for all agent execution phases.
    #[test]
    fn test_remaining_time_phase_name_preserved() {
        let start = Instant::now();
        let total_timeout = Duration::from_millis(10);
        std::thread::sleep(Duration::from_millis(50));

        // Test different phase names
        for phase in &["decomposition", "parallel execution", "synthesis"] {
            let result = remaining_time(start, total_timeout, phase);
            assert!(result.is_err());

            match result.unwrap_err() {
                AgentError::Timeout {
                    phase: error_phase, ..
                } => {
                    assert_eq!(&error_phase, phase);
                }
                _ => panic!("Expected Timeout error for phase: {}", phase),
            }
        }
    }

    /// Verifies that timeout_error correctly constructs a Timeout error with
    /// accurate elapsed and timeout millisecond values.
    #[test]
    fn test_timeout_error_creation() {
        let start = Instant::now();
        let total_timeout = Duration::from_secs(5);

        // Sleep a bit so elapsed > 0
        std::thread::sleep(Duration::from_millis(20));

        let error = timeout_error(start, total_timeout, "synthesis");

        match error {
            AgentError::Timeout {
                elapsed_ms,
                timeout_ms,
                phase,
            } => {
                assert!(elapsed_ms >= 20); // At least 20ms elapsed
                assert_eq!(timeout_ms, 5000); // 5 seconds = 5000ms
                assert_eq!(phase, "synthesis");
            }
            _ => panic!("Expected Timeout error"),
        }
    }

    /// Verifies that the Timeout error Display implementation includes all
    /// relevant information: elapsed time, timeout limit, and phase name.
    #[test]
    fn test_timeout_error_display() {
        let error = AgentError::Timeout {
            elapsed_ms: 1500,
            timeout_ms: 1000,
            phase: "decomposition".to_string(),
        };

        let display = error.to_string();
        assert!(display.contains("1500"));
        assert!(display.contains("1000"));
        assert!(display.contains("decomposition"));
    }

    /// Verifies that remaining_time correctly calculates the difference between
    /// total timeout and elapsed time.
    #[test]
    fn test_remaining_time_calculates_difference() {
        let start = Instant::now();
        let total_timeout = Duration::from_millis(500);

        // Sleep for 100ms
        std::thread::sleep(Duration::from_millis(100));

        let result = remaining_time(start, total_timeout, "test");
        assert!(result.is_ok());

        let remaining = result.unwrap();
        // Should have ~400ms remaining (500 - 100)
        // Allow generous tolerance for CI timing variations
        assert!(remaining > Duration::from_millis(300));
        assert!(remaining < Duration::from_millis(450));
    }

    // AgentContext tests

    #[test]
    fn test_agent_context_new() {
        // Just verify it compiles - we can't easily test without a real client
        let _context_fn = |llm: LlmClient| AgentContext::new(llm);
    }

    // Note: AgentError display tests are in error.rs to avoid duplication

    // Contract enforcement tests

    use crate::update::ResultMetadata;
    use async_stream::stream;
    use serde_json::json;

    /// Creates a test stream from a vector of events.
    fn test_stream(events: Vec<Result<AgentUpdate, AgentError>>) -> AgentStream<'static> {
        Box::pin(stream! {
            for event in events {
                yield event;
            }
        })
    }

    /// Verifies that events before final_result pass through unchanged.
    #[tokio::test]
    async fn test_contract_events_before_final_result_pass_through() {
        let events = vec![
            Ok(AgentUpdate::custom("step_1", "Step 1", json!({}))),
            Ok(AgentUpdate::custom("step_2", "Step 2", json!({}))),
            Ok(AgentUpdate::final_result(
                json!("Answer"),
                ResultMetadata::new(100, 0, 1000),
            )),
        ];

        let stream = test_stream(events);
        let wrapped = enforce_final_result_contract(stream);
        futures_util::pin_mut!(wrapped);

        let mut collected = Vec::new();
        while let Some(result) = wrapped.next().await {
            collected.push(result);
        }

        assert_eq!(collected.len(), 3);
        assert!(collected[0].as_ref().unwrap().event_type == "step_1");
        assert!(collected[1].as_ref().unwrap().event_type == "step_2");
        assert!(collected[2].as_ref().unwrap().event_type == "final_result");
    }

    /// Verifies that events after final_result still pass through (with warning logged).
    #[tokio::test]
    async fn test_contract_events_after_final_result_pass_through() {
        let events = vec![
            Ok(AgentUpdate::custom("step_1", "Step 1", json!({}))),
            Ok(AgentUpdate::final_result(
                json!("Answer"),
                ResultMetadata::new(100, 0, 1000),
            )),
            // This violates the contract - events after final_result
            Ok(AgentUpdate::custom("post_final", "Should warn", json!({}))),
        ];

        let stream = test_stream(events);
        let wrapped = enforce_final_result_contract(stream);
        futures_util::pin_mut!(wrapped);

        let mut collected = Vec::new();
        while let Some(result) = wrapped.next().await {
            collected.push(result);
        }

        // All events should still pass through
        assert_eq!(collected.len(), 3);
        assert!(collected[2].as_ref().unwrap().event_type == "post_final");
        // Note: The warning is logged but we can't easily test log output
    }

    /// Verifies that errors pass through without contract checking.
    #[tokio::test]
    async fn test_contract_errors_pass_through() {
        let events = vec![
            Ok(AgentUpdate::custom("step_1", "Step 1", json!({}))),
            Err(AgentError::Llm(crate::error::LlmError::Other(
                "Test error".to_string(),
            ))),
        ];

        let stream = test_stream(events);
        let wrapped = enforce_final_result_contract(stream);
        futures_util::pin_mut!(wrapped);

        let mut collected = Vec::new();
        while let Some(result) = wrapped.next().await {
            collected.push(result);
        }

        assert_eq!(collected.len(), 2);
        assert!(collected[0].is_ok());
        assert!(collected[1].is_err());
    }

    /// Verifies that an empty stream works correctly.
    #[tokio::test]
    async fn test_contract_empty_stream() {
        let events: Vec<Result<AgentUpdate, AgentError>> = vec![];

        let stream = test_stream(events);
        let wrapped = enforce_final_result_contract(stream);
        futures_util::pin_mut!(wrapped);

        let mut count = 0;
        while (wrapped.next().await).is_some() {
            count += 1;
        }

        assert_eq!(count, 0);
    }

    /// Verifies that stream with only final_result works correctly.
    #[tokio::test]
    async fn test_contract_only_final_result() {
        let events = vec![Ok(AgentUpdate::final_result(
            json!("Direct answer"),
            ResultMetadata::new(50, 0, 500),
        ))];

        let stream = test_stream(events);
        let wrapped = enforce_final_result_contract(stream);
        futures_util::pin_mut!(wrapped);

        let mut collected = Vec::new();
        while let Some(result) = wrapped.next().await {
            collected.push(result);
        }

        assert_eq!(collected.len(), 1);
        assert!(collected[0].as_ref().unwrap().event_type == "final_result");
    }

    /// Verifies the EVENT_FINAL_RESULT constant matches the expected value.
    #[test]
    fn test_event_final_result_constant() {
        assert_eq!(EVENT_FINAL_RESULT, "final_result");
    }

    /// Verifies that multiple final_result events trigger warnings for all but the first.
    #[tokio::test]
    async fn test_contract_multiple_final_results_warn() {
        let events = vec![
            Ok(AgentUpdate::custom("step_1", "Step 1", json!({}))),
            Ok(AgentUpdate::final_result(
                json!("First answer"),
                ResultMetadata::new(100, 0, 1000),
            )),
            // Second final_result - violates contract
            Ok(AgentUpdate::final_result(
                json!("Second answer"),
                ResultMetadata::new(50, 0, 500),
            )),
            // Third final_result - also violates contract
            Ok(AgentUpdate::final_result(
                json!("Third answer"),
                ResultMetadata::new(25, 0, 250),
            )),
        ];

        let stream = test_stream(events);
        let wrapped = enforce_final_result_contract(stream);
        futures_util::pin_mut!(wrapped);

        let mut collected = Vec::new();
        while let Some(result) = wrapped.next().await {
            collected.push(result);
        }

        // All events should still pass through (graceful degradation)
        assert_eq!(collected.len(), 4);
        assert!(collected[0].as_ref().unwrap().event_type == "step_1");
        assert!(collected[1].as_ref().unwrap().event_type == "final_result");
        assert!(collected[2].as_ref().unwrap().event_type == "final_result");
        assert!(collected[3].as_ref().unwrap().event_type == "final_result");
        // Note: Warnings are logged for collected[2] and collected[3] but we can't easily verify logs
    }
}
