//! Terminal-agnostic state tracking for agent execution.
//!
//! This module contains pure state with no terminal dependencies,
//! making it easy to test and enabling renderer swappability.

use crate::utils::first_sentence;
use gemicro_core::{
    AgentUpdate, EVENT_DECOMPOSITION_COMPLETE, EVENT_DECOMPOSITION_STARTED, EVENT_FINAL_RESULT,
    EVENT_SUB_QUERY_COMPLETED, EVENT_SUB_QUERY_FAILED, EVENT_SUB_QUERY_STARTED,
    EVENT_SYNTHESIS_STARTED,
};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Current execution phase of the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Phase {
    /// Initial state before any events
    NotStarted,
    /// Query is being decomposed into sub-queries
    Decomposing,
    /// Sub-queries are being executed in parallel
    Executing,
    /// Results are being synthesized
    Synthesizing,
    /// Execution is complete
    Complete,
}

/// Status of an individual sub-query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SubQueryStatus {
    /// Waiting to start
    Pending,
    /// Currently executing
    InProgress,
    /// Completed successfully
    Completed { result_preview: String, tokens: u32 },
    /// Failed with error
    Failed { error: String },
}

/// State of an individual sub-query.
///
/// # Serialization
///
/// `start_time` is skipped during serialization (`Instant` is not portable).
/// Timing data is preserved via `duration`, which is populated when the
/// sub-query completes. If you serialize mid-execution, in-progress queries
/// will have `start_time: None` after deserialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubQueryState {
    pub id: usize,
    pub query: String,
    pub status: SubQueryStatus,
    /// Skipped during serialization. See struct-level docs.
    #[serde(skip)]
    pub start_time: Option<Instant>,
    pub duration: Option<Duration>,
}

/// Data from the final result event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalResultData {
    pub answer: String,
    pub total_tokens: u32,
    pub tokens_unavailable_count: usize,
    // Note: duration_ms from metadata is not stored here.
    // Use ExecutionState::elapsed() for wall-clock timing instead.
    pub sub_queries_succeeded: usize,
    pub sub_queries_failed: usize,
}

/// Terminal-agnostic state machine for agent execution.
///
/// Tracks the current phase, sub-query statuses, timing information,
/// and final results. Updated via `update()` from `AgentUpdate` events.
#[derive(Clone)]
pub struct ExecutionState {
    phase: Phase,
    sub_queries: Vec<SubQueryState>,
    start_time: Instant,
    final_result: Option<FinalResultData>,
}

impl ExecutionState {
    /// Create a new ExecutionState.
    pub fn new() -> Self {
        Self {
            phase: Phase::NotStarted,
            sub_queries: Vec::new(),
            start_time: Instant::now(),
            final_result: None,
        }
    }

    /// Update state from an AgentUpdate event.
    ///
    /// Returns the ID of the sub-query that was updated, if any.
    ///
    /// # Error Handling
    ///
    /// This method follows the Evergreen philosophy of graceful degradation:
    /// - **Unknown event types** are logged at debug level and ignored
    /// - **Malformed event data** is logged at warn level and ignored
    ///
    /// The state machine continues processing subsequent events even if some
    /// events are malformed. This ensures robustness against protocol evolution
    /// and partial failures, but callers should monitor logs if strict validation
    /// is required.
    pub fn update(&mut self, event: &AgentUpdate) -> Option<usize> {
        match event.event_type.as_str() {
            EVENT_DECOMPOSITION_STARTED => {
                self.phase = Phase::Decomposing;
                None
            }

            EVENT_DECOMPOSITION_COMPLETE => {
                if let Some(queries) = event.as_decomposition_complete() {
                    self.sub_queries = queries
                        .into_iter()
                        .enumerate()
                        .map(|(id, query)| SubQueryState {
                            id,
                            query,
                            status: SubQueryStatus::Pending,
                            start_time: None,
                            duration: None,
                        })
                        .collect();
                    self.phase = Phase::Executing;
                } else {
                    log::warn!(
                        "Received decomposition_complete event with malformed data: {:?}",
                        event.data
                    );
                }
                None
            }

            EVENT_SUB_QUERY_STARTED => {
                if let Some(id) = event.data.get("id").and_then(|v| v.as_u64()) {
                    let id = id as usize;
                    if let Some(sq) = self.sub_queries.get_mut(id) {
                        sq.status = SubQueryStatus::InProgress;
                        sq.start_time = Some(Instant::now());
                        return Some(id);
                    }
                }
                None
            }

            EVENT_SUB_QUERY_COMPLETED => {
                if let Some(result) = event.as_sub_query_completed() {
                    if let Some(sq) = self.sub_queries.get_mut(result.id) {
                        sq.duration = sq.start_time.map(|s| s.elapsed());
                        let preview = first_sentence(&result.result);
                        sq.status = SubQueryStatus::Completed {
                            result_preview: preview,
                            tokens: result.tokens_used,
                        };
                        return Some(result.id);
                    }
                } else {
                    log::warn!(
                        "Received sub_query_completed event with malformed data: {:?}",
                        event.data
                    );
                }
                None
            }

            EVENT_SUB_QUERY_FAILED => {
                if let Some(id) = event.data.get("id").and_then(|v| v.as_u64()) {
                    let id = id as usize;
                    if let Some(sq) = self.sub_queries.get_mut(id) {
                        sq.duration = sq.start_time.map(|s| s.elapsed());
                        let error = event
                            .data
                            .get("error")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Unknown error")
                            .to_string();
                        sq.status = SubQueryStatus::Failed { error };
                        return Some(id);
                    }
                }
                None
            }

            EVENT_SYNTHESIS_STARTED => {
                self.phase = Phase::Synthesizing;
                None
            }

            EVENT_FINAL_RESULT => {
                if let Some(result) = event.as_final_result() {
                    self.final_result = Some(FinalResultData {
                        answer: result.answer,
                        total_tokens: result.metadata.total_tokens,
                        tokens_unavailable_count: result.metadata.tokens_unavailable_count,
                        sub_queries_succeeded: result.metadata.sub_queries_succeeded,
                        sub_queries_failed: result.metadata.sub_queries_failed,
                    });
                    self.phase = Phase::Complete;
                } else {
                    log::warn!(
                        "Received final_result event with malformed data: {:?}",
                        event.data
                    );
                }
                None
            }

            _ => {
                log::debug!("Unknown event type in ExecutionState: {}", event.event_type);
                None
            }
        }
    }

    /// Get the current phase.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Get a specific sub-query by ID.
    pub fn sub_query(&self, id: usize) -> Option<&SubQueryState> {
        self.sub_queries.get(id)
    }

    /// Get all sub-queries.
    pub fn sub_queries(&self) -> &[SubQueryState] {
        &self.sub_queries
    }

    /// Get the total elapsed time since start.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get the final result data, if available.
    pub fn final_result(&self) -> Option<&FinalResultData> {
        self.final_result.as_ref()
    }

    /// Calculate the total time if sub-queries had run sequentially.
    ///
    /// This is the sum of all individual sub-query durations.
    /// Returns None if no sub-queries have completed with timing data.
    pub fn sequential_time(&self) -> Option<Duration> {
        let total: Duration = self.sub_queries.iter().filter_map(|sq| sq.duration).sum();

        if total.is_zero() {
            None
        } else {
            Some(total)
        }
    }
}

impl Default for ExecutionState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gemicro_core::AgentUpdate;
    use serde_json::json;

    #[test]
    fn test_initial_state() {
        let state = ExecutionState::new();
        assert_eq!(state.phase(), Phase::NotStarted);
        assert!(state.sub_queries().is_empty());
        assert!(state.final_result().is_none());
    }

    #[test]
    fn test_decomposition_started() {
        let mut state = ExecutionState::new();
        let event = AgentUpdate::decomposition_started();

        state.update(&event);

        assert_eq!(state.phase(), Phase::Decomposing);
    }

    #[test]
    fn test_decomposition_complete() {
        let mut state = ExecutionState::new();
        let event = AgentUpdate::decomposition_complete(vec![
            "Query 1".to_string(),
            "Query 2".to_string(),
            "Query 3".to_string(),
        ]);

        state.update(&event);

        assert_eq!(state.phase(), Phase::Executing);
        assert_eq!(state.sub_queries().len(), 3);

        let sq = state.sub_query(0).unwrap();
        assert_eq!(sq.query, "Query 1");
        assert_eq!(sq.status, SubQueryStatus::Pending);
    }

    #[test]
    fn test_sub_query_started() {
        let mut state = ExecutionState::new();
        state.update(&AgentUpdate::decomposition_complete(vec!["Q1".to_string()]));

        let event = AgentUpdate::sub_query_started(0, "Q1".to_string());
        let updated_id = state.update(&event);

        assert_eq!(updated_id, Some(0));
        let sq = state.sub_query(0).unwrap();
        assert_eq!(sq.status, SubQueryStatus::InProgress);
        assert!(sq.start_time.is_some());
    }

    #[test]
    fn test_sub_query_completed() {
        let mut state = ExecutionState::new();
        state.update(&AgentUpdate::decomposition_complete(vec!["Q1".to_string()]));
        state.update(&AgentUpdate::sub_query_started(0, "Q1".to_string()));

        let event = AgentUpdate::sub_query_completed(0, "This is the result.".to_string(), 42);
        let updated_id = state.update(&event);

        assert_eq!(updated_id, Some(0));
        let sq = state.sub_query(0).unwrap();
        match &sq.status {
            SubQueryStatus::Completed {
                result_preview,
                tokens,
            } => {
                assert!(result_preview.contains("This is the result"));
                assert_eq!(*tokens, 42);
            }
            _ => panic!("Expected Completed status"),
        }
        assert!(sq.duration.is_some());
    }

    #[test]
    fn test_sub_query_failed() {
        let mut state = ExecutionState::new();
        state.update(&AgentUpdate::decomposition_complete(vec!["Q1".to_string()]));
        state.update(&AgentUpdate::sub_query_started(0, "Q1".to_string()));

        let event = AgentUpdate::sub_query_failed(0, "Timeout".to_string());
        let updated_id = state.update(&event);

        assert_eq!(updated_id, Some(0));
        let sq = state.sub_query(0).unwrap();
        match &sq.status {
            SubQueryStatus::Failed { error } => {
                assert_eq!(error, "Timeout");
            }
            _ => panic!("Expected Failed status"),
        }
    }

    #[test]
    fn test_synthesis_started() {
        let mut state = ExecutionState::new();
        state.update(&AgentUpdate::decomposition_complete(vec!["Q1".to_string()]));

        let event = AgentUpdate::synthesis_started();
        state.update(&event);

        assert_eq!(state.phase(), Phase::Synthesizing);
    }

    #[test]
    fn test_final_result() {
        let mut state = ExecutionState::new();

        let metadata = gemicro_core::ResultMetadata {
            total_tokens: 100,
            tokens_unavailable_count: 0,
            duration_ms: 5000,
            sub_queries_succeeded: 3,
            sub_queries_failed: 1,
        };
        let event = AgentUpdate::final_result("Final answer".to_string(), metadata);
        state.update(&event);

        assert_eq!(state.phase(), Phase::Complete);
        let result = state.final_result().unwrap();
        assert_eq!(result.answer, "Final answer");
        assert_eq!(result.total_tokens, 100);
        assert_eq!(result.sub_queries_succeeded, 3);
    }

    #[test]
    fn test_unknown_event_ignored() {
        let mut state = ExecutionState::new();
        let event = AgentUpdate {
            event_type: "unknown_event".to_string(),
            message: "Unknown".to_string(),
            timestamp: std::time::SystemTime::now(),
            data: json!({}),
        };

        let result = state.update(&event);

        assert!(result.is_none());
        assert_eq!(state.phase(), Phase::NotStarted);
    }

    #[test]
    fn test_sequential_time_no_queries() {
        let state = ExecutionState::new();
        assert!(state.sequential_time().is_none());
    }

    #[test]
    fn test_sequential_time_with_completed_queries() {
        let mut state = ExecutionState::new();

        // Set up sub-queries with durations
        state.update(&AgentUpdate::decomposition_complete(vec![
            "Q1".to_string(),
            "Q2".to_string(),
        ]));

        // Manually set durations since we can't wait for real time in tests
        state.sub_queries[0].duration = Some(Duration::from_secs(2));
        state.sub_queries[1].duration = Some(Duration::from_secs(3));

        let seq_time = state.sequential_time();
        assert!(seq_time.is_some());
        assert_eq!(seq_time.unwrap(), Duration::from_secs(5));
    }

    #[test]
    fn test_sequential_time_partial_completion() {
        let mut state = ExecutionState::new();

        state.update(&AgentUpdate::decomposition_complete(vec![
            "Q1".to_string(),
            "Q2".to_string(),
            "Q3".to_string(),
        ]));

        // Only two have durations
        state.sub_queries[0].duration = Some(Duration::from_secs(2));
        state.sub_queries[2].duration = Some(Duration::from_secs(4));

        let seq_time = state.sequential_time();
        assert!(seq_time.is_some());
        assert_eq!(seq_time.unwrap(), Duration::from_secs(6));
    }

    #[test]
    fn test_out_of_bounds_sub_query_returns_none() {
        let mut state = ExecutionState::new();
        state.update(&AgentUpdate::decomposition_complete(vec!["Q1".to_string()]));

        // Try to start a sub-query that doesn't exist (id=99)
        let event = AgentUpdate::sub_query_started(99, "Invalid".to_string());
        let result = state.update(&event);

        // Should return None since id 99 doesn't exist
        assert!(result.is_none());

        // Original sub-query should still be pending
        let sq = state.sub_query(0).unwrap();
        assert_eq!(sq.status, SubQueryStatus::Pending);
    }
}
