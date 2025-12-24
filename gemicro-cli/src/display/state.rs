//! Terminal-agnostic state tracking for research progress.
//!
//! This module contains pure state with no terminal dependencies,
//! making it easy to test and enabling renderer swappability.

use crate::format::first_sentence;
use gemicro_core::{
    AgentUpdate, EVENT_DECOMPOSITION_COMPLETE, EVENT_DECOMPOSITION_STARTED, EVENT_FINAL_RESULT,
    EVENT_SUB_QUERY_COMPLETED, EVENT_SUB_QUERY_FAILED, EVENT_SUB_QUERY_STARTED,
    EVENT_SYNTHESIS_STARTED,
};
use std::time::{Duration, Instant};

/// Current execution phase of the research agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Initial state before any events
    NotStarted,
    /// Query is being decomposed into sub-queries
    Decomposing,
    /// Sub-queries are being executed in parallel
    Executing,
    /// Results are being synthesized
    Synthesizing,
    /// Research is complete
    Complete,
}

/// Status of an individual sub-query.
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone)]
pub struct SubQueryState {
    pub id: usize,
    pub query: String,
    pub status: SubQueryStatus,
    pub start_time: Option<Instant>,
    pub duration: Option<Duration>,
}

/// Data from the final result event.
#[derive(Debug, Clone)]
pub struct FinalResultData {
    pub answer: String,
    pub total_tokens: u32,
    pub tokens_unavailable_count: usize,
    // Note: duration_ms from metadata is not stored here.
    // Use DisplayState::elapsed() for wall-clock timing instead.
    pub sub_queries_succeeded: usize,
    pub sub_queries_failed: usize,
}

/// Terminal-agnostic state machine for research progress.
///
/// Tracks the current phase, sub-query statuses, timing information,
/// and final results. Updated via `update()` from `AgentUpdate` events.
pub struct DisplayState {
    phase: Phase,
    sub_queries: Vec<SubQueryState>,
    start_time: Instant,
    final_result: Option<FinalResultData>,
}

impl DisplayState {
    /// Create a new DisplayState.
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
                    }
                    return Some(id);
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
                log::debug!("Unknown event type in DisplayState: {}", event.event_type);
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
}

impl Default for DisplayState {
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
        let state = DisplayState::new();
        assert_eq!(state.phase(), Phase::NotStarted);
        assert!(state.sub_queries().is_empty());
        assert!(state.final_result().is_none());
    }

    #[test]
    fn test_decomposition_started() {
        let mut state = DisplayState::new();
        let event = AgentUpdate::decomposition_started();

        state.update(&event);

        assert_eq!(state.phase(), Phase::Decomposing);
    }

    #[test]
    fn test_decomposition_complete() {
        let mut state = DisplayState::new();
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
        let mut state = DisplayState::new();
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
        let mut state = DisplayState::new();
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
        let mut state = DisplayState::new();
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
        let mut state = DisplayState::new();
        state.update(&AgentUpdate::decomposition_complete(vec!["Q1".to_string()]));

        let event = AgentUpdate::synthesis_started();
        state.update(&event);

        assert_eq!(state.phase(), Phase::Synthesizing);
    }

    #[test]
    fn test_final_result() {
        let mut state = DisplayState::new();

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
        let mut state = DisplayState::new();
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
}
