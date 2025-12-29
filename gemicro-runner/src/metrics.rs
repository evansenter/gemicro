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
#[non_exhaustive]
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

    /// Create metrics from an ExecutionTracking trait object.
    ///
    /// This is the preferred method when using `agent.create_tracker()`.
    /// It provides simpler metrics without step-level details.
    pub fn from_tracker(tracker: &dyn gemicro_core::ExecutionTracking, duration: Duration) -> Self {
        let (total_tokens, tokens_unavailable_count, final_answer, completion_phase, extra) =
            if let Some(result) = tracker.final_result() {
                (
                    result.total_tokens,
                    result.tokens_unavailable_count,
                    Some(result.answer.clone()),
                    "complete".to_string(),
                    result.extra.clone(),
                )
            } else {
                let phase = if tracker.is_complete() {
                    log::debug!(
                        "Tracker reports complete but has no final_result - metrics will have zero values"
                    );
                    "complete"
                } else {
                    log::debug!("Creating metrics from incomplete tracker execution");
                    "unknown"
                };
                (0, 0, None, phase.to_string(), serde_json::Value::Null)
            };

        // Extract step counts from extra field if present
        let steps_succeeded = extra["steps_succeeded"].as_u64().unwrap_or(0) as usize;
        let steps_failed = extra["steps_failed"].as_u64().unwrap_or(0) as usize;
        let steps_total = steps_succeeded + steps_failed;

        ExecutionMetrics {
            total_duration: duration,
            sequential_time: None, // Not available without step timing
            parallel_speedup: None,
            steps_total,
            steps_succeeded,
            steps_failed,
            step_timings: Vec::new(), // Not available from basic tracker
            total_tokens,
            tokens_unavailable_count,
            final_answer,
            completion_phase,
        }
    }

    /// Check if execution completed successfully.
    pub fn is_complete(&self) -> bool {
        self.completion_phase == phases::COMPLETE
    }

    /// Create metrics for testing purposes.
    ///
    /// This constructor is provided for test code that needs to create
    /// `ExecutionMetrics` instances without going through the normal
    /// execution flow. It's marked `#[doc(hidden)]` since it's not
    /// part of the public API.
    #[doc(hidden)]
    pub fn for_testing(
        total_duration: Duration,
        total_tokens: u32,
        tokens_unavailable_count: usize,
        final_answer: Option<String>,
        steps_succeeded: usize,
        steps_failed: usize,
    ) -> Self {
        ExecutionMetrics {
            total_duration,
            sequential_time: None,
            parallel_speedup: None,
            steps_total: steps_succeeded + steps_failed,
            steps_succeeded,
            steps_failed,
            step_timings: Vec::new(),
            total_tokens,
            tokens_unavailable_count,
            final_answer,
            completion_phase: phases::COMPLETE.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{ExecutionState, ExecutionStep, FinalResultData, StepStatus};

    /// Create state directly for testing metrics conversion.
    /// This builds state without relying on any agent-specific handler.
    fn create_completed_state() -> ExecutionState {
        use std::time::Duration;

        let mut state = ExecutionState::new();

        // Add steps (simulating decomposition result)
        let mut step0 = ExecutionStep::new("0", "Q1");
        let mut step1 = ExecutionStep::new("1", "Q2");
        let mut step2 = ExecutionStep::new("2", "Q3");

        // Simulate execution: start and complete/fail each step
        step0.start();
        step0.duration = Some(Duration::from_millis(100));
        step0.status = StepStatus::Completed {
            result_preview: "Result 1".to_string(),
            tokens: Some(50),
        };

        step1.start();
        step1.duration = Some(Duration::from_millis(100));
        step1.status = StepStatus::Completed {
            result_preview: "Result 2".to_string(),
            tokens: Some(60),
        };

        step2.start();
        step2.duration = Some(Duration::from_millis(50));
        step2.status = StepStatus::Failed {
            error: "Timeout".to_string(),
        };

        state.add_steps(vec![step0, step1, step2]);

        // Set final result
        state.set_final_result(FinalResultData {
            answer: "Final answer".to_string(),
            total_tokens: 150,
            tokens_unavailable_count: 0,
            steps_succeeded: 2,
            steps_failed: 1,
        });

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
        let mut state = ExecutionState::new();

        // Simulate partial execution: steps added but not all completed
        state.set_phase(phases::EXECUTING);
        state.add_steps(vec![
            ExecutionStep::new("0", "Q1"),
            ExecutionStep::new("1", "Q2"),
        ]);

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

    #[test]
    fn test_metrics_from_tracker_with_final_result() {
        use gemicro_core::{AgentUpdate, DefaultTracker, ExecutionTracking, ResultMetadata};
        use serde_json::json;
        use std::time::Duration;

        let mut tracker = DefaultTracker::default();
        let metadata = ResultMetadata::with_extra(
            200,
            1,
            3000,
            json!({ "steps_succeeded": 5, "steps_failed": 2 }),
        );
        tracker.handle_event(&AgentUpdate::final_result("Answer".to_string(), metadata));

        let metrics = ExecutionMetrics::from_tracker(&tracker, Duration::from_secs(3));

        assert_eq!(metrics.total_tokens, 200);
        assert_eq!(metrics.tokens_unavailable_count, 1);
        assert_eq!(metrics.steps_succeeded, 5);
        assert_eq!(metrics.steps_failed, 2);
        assert_eq!(metrics.steps_total, 7);
        assert_eq!(metrics.final_answer, Some("Answer".to_string()));
        assert_eq!(metrics.completion_phase, "complete");
    }

    #[test]
    fn test_metrics_from_tracker_without_final_result() {
        use gemicro_core::{AgentUpdate, DefaultTracker, ExecutionTracking};
        use serde_json::json;
        use std::time::Duration;

        let mut tracker = DefaultTracker::default();
        // Only send intermediate event, no final_result
        tracker.handle_event(&AgentUpdate::custom("progress", "Working...", json!({})));

        let metrics = ExecutionMetrics::from_tracker(&tracker, Duration::from_secs(1));

        assert_eq!(metrics.total_tokens, 0);
        assert_eq!(metrics.steps_total, 0);
        assert!(metrics.final_answer.is_none());
        assert_eq!(metrics.completion_phase, "unknown");
    }

    #[test]
    fn test_metrics_from_tracker_missing_extra_fields() {
        use gemicro_core::{AgentUpdate, DefaultTracker, ExecutionTracking, ResultMetadata};
        use std::time::Duration;

        let mut tracker = DefaultTracker::default();
        // final_result with null extra (no steps_succeeded/failed)
        let metadata = ResultMetadata::new(100, 0, 2000);
        tracker.handle_event(&AgentUpdate::final_result("Answer".to_string(), metadata));

        let metrics = ExecutionMetrics::from_tracker(&tracker, Duration::from_secs(2));

        // Should default to 0 when extra fields are missing
        assert_eq!(metrics.steps_succeeded, 0);
        assert_eq!(metrics.steps_failed, 0);
        assert_eq!(metrics.steps_total, 0);
        assert_eq!(metrics.total_tokens, 100);
        assert_eq!(metrics.final_answer, Some("Answer".to_string()));
    }
}
