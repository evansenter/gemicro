//! Agent execution tracking abstraction.
//!
//! This module provides the `ExecutionTracking` trait which agents implement
//! to track their execution state for display purposes. Each agent provides
//! its own implementation that understands its specific event types and
//! execution patterns.
//!
//! # Design Philosophy
//!
//! Following the [Evergreen spec](https://github.com/google-deepmind/evergreen-spec)
//! philosophy, execution tracking is agent-owned. The CLI and runner don't need
//! to know about agent-specific concepts like "sub-queries" or "ReAct iterations".
//! Instead, they call generic methods like `status_message()` and `is_complete()`.
//!
//! # Usage Example
//!
//! ```text
//! async fn run_with_progress(agent: &dyn Agent, query: &str, context: AgentContext) {
//!     let mut tracker = agent.create_tracker();
//!     let stream = agent.execute(query, context);
//!     futures_util::pin_mut!(stream);
//!
//!     while let Some(result) = stream.next().await {
//!         let update = result?;
//!         tracker.handle_event(&update);
//!         if let Some(msg) = tracker.status_message() {
//!             println!("{}", msg);
//!         }
//!     }
//!
//!     if let Some(result) = tracker.final_result() {
//!         println!("Answer: {}", result.answer);
//!     }
//! }
//! ```

use crate::update::AgentUpdate;

/// Data from a completed agent execution.
///
/// Contains generic fields plus an extensible `extra` for agent-specific data.
/// This mirrors the structure of `ResultMetadata` but is owned by the tracker.
#[derive(Debug, Clone)]
pub struct FinalResultData {
    /// The synthesized answer from the agent.
    pub answer: String,

    /// Total tokens used across all LLM calls.
    pub total_tokens: u32,

    /// Number of LLM calls that didn't report token usage.
    pub tokens_unavailable_count: usize,

    /// Total execution time in milliseconds.
    pub duration_ms: u64,

    /// Agent-specific metadata (steps, iterations, tool calls, etc.).
    ///
    /// Examples:
    /// - DeepResearch: `{"steps_succeeded": 3, "steps_failed": 1}`
    /// - ReAct: `{"iterations": 5, "tools_used": ["calculator", "search"]}`
    /// - SimpleQA: `{}`
    pub extra: serde_json::Value,
}

/// Tracks agent execution state for display purposes.
///
/// Each agent provides its own implementation that understands its specific
/// event types and execution patterns. This allows the CLI and runner to
/// remain agent-agnostic while still providing meaningful progress updates.
///
/// # Implementation Notes
///
/// - `handle_event` is called for each event in the stream
/// - `status_message` returns the current human-readable status
/// - `is_complete` returns true when execution has finished
/// - `final_result` returns the result data once complete
pub trait ExecutionTracking: Send + Sync {
    /// Process an event and update internal state.
    ///
    /// This is called for each `AgentUpdate` received from the agent stream.
    /// The tracker should update its internal state based on the event type
    /// and data.
    fn handle_event(&mut self, event: &AgentUpdate);

    /// Current status message for display.
    ///
    /// Returns a human-readable status message suitable for terminal display,
    /// such as "Researching sub-query 2/5..." or "Synthesizing final answer...".
    ///
    /// Returns `None` when there's nothing meaningful to display.
    fn status_message(&self) -> Option<&str>;

    /// Whether execution is complete.
    ///
    /// Returns `true` once a `final_result` event has been processed.
    fn is_complete(&self) -> bool;

    /// Final result data (available only when `is_complete()` is true).
    ///
    /// Returns `None` if execution hasn't completed yet.
    fn final_result(&self) -> Option<&FinalResultData>;
}

/// A no-op tracker for agents that don't need custom tracking.
///
/// This implementation simply captures the status message from each event
/// and extracts the final result when it arrives. Use this for simple agents
/// that don't have complex multi-step execution patterns.
///
/// # Example
///
/// ```
/// use gemicro_core::{DefaultTracker, ExecutionTracking};
///
/// let tracker = DefaultTracker::default();
/// assert!(!tracker.is_complete());
/// assert!(tracker.status_message().is_none());
/// ```
#[derive(Debug, Default)]
pub struct DefaultTracker {
    status: String,
    result: Option<FinalResultData>,
}

impl ExecutionTracking for DefaultTracker {
    fn handle_event(&mut self, event: &AgentUpdate) {
        // Always update status with the event message
        self.status = event.message.clone();

        // Check for final_result event
        if let Some(result) = event.as_final_result() {
            self.result = Some(FinalResultData {
                answer: result.answer,
                total_tokens: result.metadata.total_tokens,
                tokens_unavailable_count: result.metadata.tokens_unavailable_count,
                duration_ms: result.metadata.duration_ms,
                extra: result.metadata.extra.clone(),
            });
        }
    }

    fn status_message(&self) -> Option<&str> {
        if self.status.is_empty() {
            None
        } else {
            Some(&self.status)
        }
    }

    fn is_complete(&self) -> bool {
        self.result.is_some()
    }

    fn final_result(&self) -> Option<&FinalResultData> {
        self.result.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::update::ResultMetadata;
    use serde_json::json;

    #[test]
    fn test_default_tracker_initial_state() {
        let tracker = DefaultTracker::default();
        assert!(tracker.status_message().is_none());
        assert!(!tracker.is_complete());
        assert!(tracker.final_result().is_none());
    }

    #[test]
    fn test_default_tracker_updates_status() {
        let mut tracker = DefaultTracker::default();

        let event = AgentUpdate::custom("test_event", "Processing...", json!({}));
        tracker.handle_event(&event);

        assert_eq!(tracker.status_message(), Some("Processing..."));
        assert!(!tracker.is_complete());
    }

    #[test]
    fn test_default_tracker_handles_final_result() {
        let mut tracker = DefaultTracker::default();

        let metadata = ResultMetadata::with_extra(
            100,
            0,
            5000,
            json!({
                "steps_succeeded": 3,
                "steps_failed": 1,
            }),
        );
        let event = AgentUpdate::final_result("The answer is 42".to_string(), metadata);
        tracker.handle_event(&event);

        assert!(tracker.is_complete());
        let result = tracker.final_result().unwrap();
        assert_eq!(result.answer, "The answer is 42");
        assert_eq!(result.total_tokens, 100);
        assert_eq!(result.duration_ms, 5000);
        assert_eq!(result.extra["steps_succeeded"], 3);
    }

    #[test]
    fn test_default_tracker_status_updates_with_each_event() {
        let mut tracker = DefaultTracker::default();

        tracker.handle_event(&AgentUpdate::custom("step1", "Step 1", json!({})));
        assert_eq!(tracker.status_message(), Some("Step 1"));

        tracker.handle_event(&AgentUpdate::custom("step2", "Step 2", json!({})));
        assert_eq!(tracker.status_message(), Some("Step 2"));

        tracker.handle_event(&AgentUpdate::custom("step3", "Step 3", json!({})));
        assert_eq!(tracker.status_message(), Some("Step 3"));
    }
}
