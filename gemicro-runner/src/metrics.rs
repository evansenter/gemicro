//! Execution metrics for programmatic consumption.
//!
//! Provides structured, serializable metrics from agent execution,
//! useful for logging, telemetry, and evaluation frameworks.

use crate::state::{ExecutionState, Phase, SubQueryStatus};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Rich execution metrics structure for programmatic consumption.
///
/// This is an immutable snapshot of execution state, suitable for:
/// - Logging and telemetry
/// - Evaluation framework benchmarking
/// - A/B testing of prompts
/// - Cost analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    /// Total wall-clock duration of execution
    pub total_duration: Duration,

    /// Estimated sequential time (sum of sub-query durations)
    pub sequential_time: Option<Duration>,

    /// Parallel speedup factor (sequential_time / total_duration)
    pub parallel_speedup: Option<f64>,

    /// Total number of sub-queries
    pub sub_queries_total: usize,

    /// Number of successful sub-queries
    pub sub_queries_succeeded: usize,

    /// Number of failed sub-queries
    pub sub_queries_failed: usize,

    /// Per-sub-query timing information
    pub sub_query_timings: Vec<SubQueryTiming>,

    /// Total tokens used across all LLM calls
    pub total_tokens: u32,

    /// Number of calls where token count was unavailable
    pub tokens_unavailable_count: usize,

    /// The final synthesized answer, if execution completed
    pub final_answer: Option<String>,

    /// The phase at which metrics were captured
    pub completion_phase: Phase,
}

/// Timing information for a single sub-query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubQueryTiming {
    /// Sub-query index
    pub id: usize,

    /// The sub-query text
    pub query: String,

    /// Whether the sub-query succeeded or failed
    pub succeeded: bool,

    /// Duration of the sub-query execution
    pub duration: Option<Duration>,

    /// Tokens used by this sub-query (if available)
    pub tokens_used: Option<u32>,
}

impl From<&ExecutionState> for ExecutionMetrics {
    fn from(state: &ExecutionState) -> Self {
        let total_duration = state.elapsed();
        let sequential_time = state.sequential_time();

        let parallel_speedup = sequential_time.map(|seq| {
            if !total_duration.is_zero() {
                seq.as_secs_f64() / total_duration.as_secs_f64()
            } else {
                1.0
            }
        });

        let sub_query_timings: Vec<SubQueryTiming> = state
            .sub_queries()
            .iter()
            .map(|sq| {
                let (succeeded, tokens_used) = match &sq.status {
                    SubQueryStatus::Completed { tokens, .. } => (true, Some(*tokens)),
                    SubQueryStatus::Failed { .. } => (false, None),
                    SubQueryStatus::Pending | SubQueryStatus::InProgress => (false, None),
                };

                SubQueryTiming {
                    id: sq.id,
                    query: sq.query.clone(),
                    succeeded,
                    duration: sq.duration,
                    tokens_used,
                }
            })
            .collect();

        let sub_queries_succeeded = sub_query_timings.iter().filter(|t| t.succeeded).count();
        let sub_queries_failed = sub_query_timings
            .iter()
            .filter(|t| {
                !t.succeeded
                    && state
                        .sub_query(t.id)
                        .map(|sq| matches!(sq.status, SubQueryStatus::Failed { .. }))
                        .unwrap_or(false)
            })
            .count();

        let (total_tokens, tokens_unavailable_count, final_answer) =
            if let Some(result) = state.final_result() {
                (
                    result.total_tokens,
                    result.tokens_unavailable_count,
                    Some(result.answer.clone()),
                )
            } else {
                // Calculate from sub-queries if no final result yet
                let tokens: u32 = sub_query_timings.iter().filter_map(|t| t.tokens_used).sum();
                let unavailable = sub_query_timings
                    .iter()
                    .filter(|t| t.succeeded && t.tokens_used.is_none())
                    .count();
                (tokens, unavailable, None)
            };

        ExecutionMetrics {
            total_duration,
            sequential_time,
            parallel_speedup,
            sub_queries_total: state.sub_queries().len(),
            sub_queries_succeeded,
            sub_queries_failed,
            sub_query_timings,
            total_tokens,
            tokens_unavailable_count,
            final_answer,
            completion_phase: state.phase(),
        }
    }
}

impl ExecutionMetrics {
    /// Create metrics from an ExecutionState.
    pub fn from_state(state: &ExecutionState) -> Self {
        Self::from(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ExecutionState;
    use gemicro_core::{AgentUpdate, ResultMetadata};

    fn create_completed_state() -> ExecutionState {
        let mut state = ExecutionState::new();
        state.update(&AgentUpdate::decomposition_started());
        state.update(&AgentUpdate::decomposition_complete(vec![
            "Q1".to_string(),
            "Q2".to_string(),
            "Q3".to_string(),
        ]));

        // Simulate sub-query execution
        state.update(&AgentUpdate::sub_query_started(0, "Q1".to_string()));
        state.update(&AgentUpdate::sub_query_completed(
            0,
            "Result 1".to_string(),
            50,
        ));

        state.update(&AgentUpdate::sub_query_started(1, "Q2".to_string()));
        state.update(&AgentUpdate::sub_query_completed(
            1,
            "Result 2".to_string(),
            60,
        ));

        state.update(&AgentUpdate::sub_query_started(2, "Q3".to_string()));
        state.update(&AgentUpdate::sub_query_failed(2, "Timeout".to_string()));

        state.update(&AgentUpdate::synthesis_started());

        let metadata = ResultMetadata {
            total_tokens: 150,
            tokens_unavailable_count: 0,
            duration_ms: 5000,
            sub_queries_succeeded: 2,
            sub_queries_failed: 1,
        };
        state.update(&AgentUpdate::final_result(
            "Final answer".to_string(),
            metadata,
        ));

        state
    }

    #[test]
    fn test_metrics_from_completed_state() {
        let state = create_completed_state();
        let metrics = ExecutionMetrics::from(&state);

        assert_eq!(metrics.completion_phase, Phase::Complete);
        assert_eq!(metrics.sub_queries_total, 3);
        assert_eq!(metrics.sub_queries_succeeded, 2);
        assert_eq!(metrics.sub_queries_failed, 1);
        assert_eq!(metrics.total_tokens, 150);
        assert_eq!(metrics.tokens_unavailable_count, 0);
        assert_eq!(metrics.final_answer, Some("Final answer".to_string()));
    }

    #[test]
    fn test_metrics_sub_query_timings() {
        let state = create_completed_state();
        let metrics = ExecutionMetrics::from(&state);

        assert_eq!(metrics.sub_query_timings.len(), 3);

        // First two succeeded
        assert!(metrics.sub_query_timings[0].succeeded);
        assert_eq!(metrics.sub_query_timings[0].tokens_used, Some(50));

        assert!(metrics.sub_query_timings[1].succeeded);
        assert_eq!(metrics.sub_query_timings[1].tokens_used, Some(60));

        // Third failed
        assert!(!metrics.sub_query_timings[2].succeeded);
        assert_eq!(metrics.sub_query_timings[2].tokens_used, None);
    }

    #[test]
    fn test_metrics_from_partial_state() {
        let mut state = ExecutionState::new();
        state.update(&AgentUpdate::decomposition_started());
        state.update(&AgentUpdate::decomposition_complete(vec![
            "Q1".to_string(),
            "Q2".to_string(),
        ]));

        let metrics = ExecutionMetrics::from(&state);

        assert_eq!(metrics.completion_phase, Phase::Executing);
        assert_eq!(metrics.sub_queries_total, 2);
        assert_eq!(metrics.sub_queries_succeeded, 0);
        assert!(metrics.final_answer.is_none());
    }

    #[test]
    fn test_metrics_serialization() {
        let state = create_completed_state();
        let metrics = ExecutionMetrics::from(&state);

        // Serialize to JSON
        let json = serde_json::to_string(&metrics).expect("Should serialize");

        // Deserialize back
        let deserialized: ExecutionMetrics =
            serde_json::from_str(&json).expect("Should deserialize");

        assert_eq!(deserialized.sub_queries_total, metrics.sub_queries_total);
        assert_eq!(deserialized.total_tokens, metrics.total_tokens);
        assert_eq!(deserialized.final_answer, metrics.final_answer);
    }

    #[test]
    fn test_metrics_from_state_method() {
        let state = create_completed_state();
        let metrics = ExecutionMetrics::from_state(&state);

        assert_eq!(metrics.completion_phase, Phase::Complete);
    }
}
