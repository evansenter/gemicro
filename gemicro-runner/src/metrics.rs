//! Execution metrics for programmatic consumption.
//!
//! Provides structured, serializable metrics from agent execution,
//! useful for logging, telemetry, and evaluation frameworks.

use crate::state::{phases, ExecutionState, StepStatus};
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

    /// Estimated sequential time (sum of step durations)
    pub sequential_time: Option<Duration>,

    /// Parallel speedup factor (sequential_time / total_duration)
    /// Only meaningful for agents that execute steps in parallel
    pub parallel_speedup: Option<f64>,

    /// Total number of execution steps
    pub steps_total: usize,

    /// Number of successful steps
    pub steps_succeeded: usize,

    /// Number of failed steps
    pub steps_failed: usize,

    /// Per-step timing information
    pub step_timings: Vec<StepTiming>,

    /// Total tokens used across all LLM calls
    pub total_tokens: u32,

    /// Number of calls where token count was unavailable
    pub tokens_unavailable_count: usize,

    /// The final synthesized answer, if execution completed
    pub final_answer: Option<String>,

    /// The phase at which metrics were captured (string for flexibility)
    pub completion_phase: String,
}

/// Timing information for a single execution step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepTiming {
    /// Step identifier
    pub id: String,

    /// The step label/description
    pub label: String,

    /// Whether the step succeeded or failed.
    ///
    /// - `Some(true)` = completed successfully
    /// - `Some(false)` = failed with error
    /// - `None` = still in progress (pending or executing)
    pub succeeded: Option<bool>,

    /// Duration of the step execution
    pub duration: Option<Duration>,

    /// Tokens used by this step (if available)
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

        let step_timings: Vec<StepTiming> = state
            .steps()
            .iter()
            .map(|step| {
                // Allow unreachable_patterns: StepStatus is #[non_exhaustive], but the
                // wildcard is unreachable within this crate. We keep it for consistency with
                // how downstream crates must handle this enum.
                #[allow(unreachable_patterns)]
                let (succeeded, tokens_used) = match &step.status {
                    StepStatus::Completed { tokens, .. } => (Some(true), *tokens),
                    StepStatus::Failed { .. } => (Some(false), None),
                    StepStatus::Pending | StepStatus::InProgress => (None, None),
                    _ => (None, None),
                };

                StepTiming {
                    id: step.id.clone(),
                    label: step.label.clone(),
                    succeeded,
                    duration: step.duration,
                    tokens_used,
                }
            })
            .collect();

        let steps_succeeded = step_timings
            .iter()
            .filter(|t| t.succeeded == Some(true))
            .count();
        let steps_failed = step_timings
            .iter()
            .filter(|t| t.succeeded == Some(false))
            .count();

        let (total_tokens, tokens_unavailable_count, final_answer) =
            if let Some(result) = state.final_result() {
                (
                    result.total_tokens,
                    result.tokens_unavailable_count,
                    Some(result.answer.clone()),
                )
            } else {
                // Calculate from steps if no final result yet
                let tokens: u32 = step_timings.iter().filter_map(|t| t.tokens_used).sum();
                let unavailable = step_timings
                    .iter()
                    .filter(|t| t.succeeded == Some(true) && t.tokens_used.is_none())
                    .count();
                (tokens, unavailable, None)
            };

        ExecutionMetrics {
            total_duration,
            sequential_time,
            parallel_speedup,
            steps_total: state.steps().len(),
            steps_succeeded,
            steps_failed,
            step_timings,
            total_tokens,
            tokens_unavailable_count,
            final_answer,
            completion_phase: state.phase().to_string(),
        }
    }
}

impl ExecutionMetrics {
    /// Create metrics from an ExecutionState.
    pub fn from_state(state: &ExecutionState) -> Self {
        Self::from(state)
    }

    /// Check if execution completed successfully.
    pub fn is_complete(&self) -> bool {
        self.completion_phase == phases::COMPLETE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{DeepResearchStateHandler, ExecutionState, StateHandler};
    use gemicro_core::{AgentUpdate, ResultMetadata};

    fn create_completed_state() -> ExecutionState {
        use serde_json::json;

        let handler = DeepResearchStateHandler;
        let mut state = ExecutionState::new();

        handler.handle(
            &mut state,
            &AgentUpdate::custom("decomposition_started", "Decomposing query", json!({})),
        );
        handler.handle(
            &mut state,
            &AgentUpdate::custom(
                "decomposition_complete",
                "Decomposed into 3 sub-queries",
                json!({ "sub_queries": ["Q1", "Q2", "Q3"] }),
            ),
        );

        // Simulate sub-query execution
        handler.handle(
            &mut state,
            &AgentUpdate::custom(
                "sub_query_started",
                "Sub-query 0 started",
                json!({ "id": 0, "query": "Q1" }),
            ),
        );
        handler.handle(
            &mut state,
            &AgentUpdate::custom(
                "sub_query_completed",
                "Sub-query 0 completed",
                json!({ "id": 0, "result": "Result 1", "tokens_used": 50 }),
            ),
        );

        handler.handle(
            &mut state,
            &AgentUpdate::custom(
                "sub_query_started",
                "Sub-query 1 started",
                json!({ "id": 1, "query": "Q2" }),
            ),
        );
        handler.handle(
            &mut state,
            &AgentUpdate::custom(
                "sub_query_completed",
                "Sub-query 1 completed",
                json!({ "id": 1, "result": "Result 2", "tokens_used": 60 }),
            ),
        );

        handler.handle(
            &mut state,
            &AgentUpdate::custom(
                "sub_query_started",
                "Sub-query 2 started",
                json!({ "id": 2, "query": "Q3" }),
            ),
        );
        handler.handle(
            &mut state,
            &AgentUpdate::custom(
                "sub_query_failed",
                "Sub-query 2 failed",
                json!({ "id": 2, "error": "Timeout" }),
            ),
        );

        handler.handle(
            &mut state,
            &AgentUpdate::custom("synthesis_started", "Synthesizing results", json!({})),
        );

        let metadata = ResultMetadata {
            total_tokens: 150,
            tokens_unavailable_count: 0,
            duration_ms: 5000,
            sub_queries_succeeded: 2,
            sub_queries_failed: 1,
        };
        handler.handle(
            &mut state,
            &AgentUpdate::final_result("Final answer".to_string(), metadata),
        );

        state
    }

    #[test]
    fn test_metrics_from_completed_state() {
        let state = create_completed_state();
        let metrics = ExecutionMetrics::from(&state);

        assert!(metrics.is_complete());
        assert_eq!(metrics.steps_total, 3);
        assert_eq!(metrics.steps_succeeded, 2);
        assert_eq!(metrics.steps_failed, 1);
        assert_eq!(metrics.total_tokens, 150);
        assert_eq!(metrics.tokens_unavailable_count, 0);
        assert_eq!(metrics.final_answer, Some("Final answer".to_string()));
    }

    #[test]
    fn test_metrics_step_timings() {
        let state = create_completed_state();
        let metrics = ExecutionMetrics::from(&state);

        assert_eq!(metrics.step_timings.len(), 3);

        // First two succeeded
        assert_eq!(metrics.step_timings[0].succeeded, Some(true));
        assert_eq!(metrics.step_timings[0].tokens_used, Some(50));

        assert_eq!(metrics.step_timings[1].succeeded, Some(true));
        assert_eq!(metrics.step_timings[1].tokens_used, Some(60));

        // Third failed
        assert_eq!(metrics.step_timings[2].succeeded, Some(false));
        assert_eq!(metrics.step_timings[2].tokens_used, None);
    }

    #[test]
    fn test_metrics_from_partial_state() {
        use serde_json::json;

        let handler = DeepResearchStateHandler;
        let mut state = ExecutionState::new();

        handler.handle(
            &mut state,
            &AgentUpdate::custom("decomposition_started", "Decomposing query", json!({})),
        );
        handler.handle(
            &mut state,
            &AgentUpdate::custom(
                "decomposition_complete",
                "Decomposed into 2 sub-queries",
                json!({ "sub_queries": ["Q1", "Q2"] }),
            ),
        );

        let metrics = ExecutionMetrics::from(&state);

        assert!(!metrics.is_complete());
        assert_eq!(metrics.completion_phase, "executing");
        assert_eq!(metrics.steps_total, 2);
        assert_eq!(metrics.steps_succeeded, 0);
        assert_eq!(metrics.steps_failed, 0);
        assert!(metrics.final_answer.is_none());

        // Pending steps should have succeeded: None (not false)
        assert_eq!(metrics.step_timings[0].succeeded, None);
        assert_eq!(metrics.step_timings[1].succeeded, None);
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

        assert_eq!(deserialized.steps_total, metrics.steps_total);
        assert_eq!(deserialized.total_tokens, metrics.total_tokens);
        assert_eq!(deserialized.final_answer, metrics.final_answer);
    }

    #[test]
    fn test_metrics_from_state_method() {
        let state = create_completed_state();
        let metrics = ExecutionMetrics::from_state(&state);

        assert!(metrics.is_complete());
    }
}
