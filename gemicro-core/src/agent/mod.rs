//! Agent infrastructure and implementations.
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
//! ## Available Agents
//!
//! - [`DeepResearchAgent`]: Decomposes queries, executes sub-queries in parallel, synthesizes results
//!
//! ## Example
//!
//! ```no_run
//! use gemicro_core::{Agent, AgentContext, DeepResearchAgent, ResearchConfig, LlmClient, LlmConfig};
//! use futures_util::StreamExt;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let genai_client = rust_genai::Client::builder("api-key".to_string()).build();
//! let context = AgentContext::new(LlmClient::new(genai_client, LlmConfig::default()));
//! let agent = DeepResearchAgent::new(ResearchConfig::default())?;
//!
//! let stream = agent.execute("What is Rust?", context);
//! futures_util::pin_mut!(stream);
//! while let Some(update) = stream.next().await {
//!     println!("{:?}", update?);
//! }
//! # Ok(())
//! # }
//! ```

mod deep_research;

pub use deep_research::DeepResearchAgent;

use crate::error::AgentError;
use crate::llm::LlmClient;
use crate::update::AgentUpdate;

use futures_util::Stream;
use std::pin::Pin;
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
/// use gemicro_core::{Agent, AgentContext, AgentStream, AgentUpdate, AgentError};
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
#[derive(Clone)]
pub struct AgentContext {
    /// Shared LLM client for making API calls.
    pub llm: Arc<LlmClient>,

    /// Cancellation token for cooperative shutdown.
    ///
    /// Agents should check this token periodically and abort gracefully
    /// when cancelled, returning partial results if possible.
    pub cancellation_token: CancellationToken,
}

impl AgentContext {
    /// Create a new context with an owned LLM client.
    ///
    /// The client will be wrapped in an Arc for sharing across tasks.
    /// Uses a new cancellation token (never cancelled unless explicitly triggered).
    pub fn new(llm: LlmClient) -> Self {
        Self {
            llm: Arc::new(llm),
            cancellation_token: CancellationToken::new(),
        }
    }

    /// Create a new context with cancellation support.
    ///
    /// Use this when you need to cancel in-flight operations (e.g., on Ctrl+C).
    pub fn new_with_cancellation(llm: LlmClient, cancellation_token: CancellationToken) -> Self {
        Self {
            llm: Arc::new(llm),
            cancellation_token,
        }
    }

    /// Create a context from an existing shared LLM client.
    ///
    /// Useful when multiple agents share the same client instance.
    pub fn from_arc(llm: Arc<LlmClient>) -> Self {
        Self {
            llm,
            cancellation_token: CancellationToken::new(),
        }
    }
}

// ============================================================================
// Internal helper functions (used by agents in this module)
// ============================================================================

/// Calculate remaining time from a total timeout budget.
///
/// Returns an error if the timeout has already been exceeded.
pub(crate) fn remaining_time(
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
pub(crate) fn timeout_error(start: Instant, total_timeout: Duration, phase: &str) -> AgentError {
    AgentError::Timeout {
        elapsed_ms: start.elapsed().as_millis() as u64,
        timeout_ms: total_timeout.as_millis() as u64,
        phase: phase.to_string(),
    }
}

/// Execute a future with timeout and cancellation support.
///
/// Returns the future's result, or an error if cancelled/timed out.
pub(crate) async fn with_timeout_and_cancellation<F, T>(
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
}
