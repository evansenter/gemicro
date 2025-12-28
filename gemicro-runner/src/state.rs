//! Terminal-agnostic state tracking for agent execution.
//!
//! This module contains pure state with no terminal dependencies,
//! making it easy to test and enabling renderer swappability.
//!
//! ## Generic Design
//!
//! `ExecutionState` is agent-agnostic. It tracks:
//! - A string-based `phase` (agents define their own phase names)
//! - Generic `ExecutionStep`s (replaces DeepResearch-specific "sub-queries")
//! - Timing and final results
//!
//! To parse agent-specific events, use a `StateHandler` implementation.

use gemicro_core::AgentUpdate;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Well-known phase constants for convenience.
/// Agents may use these or define their own.
pub mod phases {
    pub const NOT_STARTED: &str = "not_started";
    pub const COMPLETE: &str = "complete";
    // DeepResearch phases (for backwards compatibility)
    pub const DECOMPOSING: &str = "decomposing";
    pub const EXECUTING: &str = "executing";
    pub const SYNTHESIZING: &str = "synthesizing";
    // ReAct phases
    pub const THINKING: &str = "thinking";
    pub const ACTING: &str = "acting";
    pub const OBSERVING: &str = "observing";
}

/// Status of an individual execution step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StepStatus {
    /// Waiting to start
    Pending,
    /// Currently executing
    InProgress,
    /// Completed successfully
    Completed {
        result_preview: String,
        tokens: Option<u32>,
    },
    /// Failed with error
    Failed { error: String },
}

/// State of an individual execution step.
///
/// This is a generic replacement for the previous DeepResearch-specific
/// "SubQueryState". Steps can represent sub-queries, ReAct iterations,
/// tool calls, or any other discrete unit of work.
///
/// # Serialization
///
/// `start_time` is skipped during serialization (`Instant` is not portable).
/// Timing data is preserved via `duration`, which is populated when the
/// step completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStep {
    /// Unique identifier for this step (string for flexibility)
    pub id: String,
    /// Human-readable label describing this step
    pub label: String,
    /// Current status of this step
    pub status: StepStatus,
    /// Skipped during serialization. See struct-level docs.
    #[serde(skip)]
    pub start_time: Option<Instant>,
    /// Duration of this step (populated on completion)
    pub duration: Option<Duration>,
}

impl ExecutionStep {
    /// Create a new pending step.
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            status: StepStatus::Pending,
            start_time: None,
            duration: None,
        }
    }

    /// Mark this step as in-progress.
    pub fn start(&mut self) {
        self.status = StepStatus::InProgress;
        self.start_time = Some(Instant::now());
    }

    /// Mark this step as completed.
    pub fn complete(&mut self, result_preview: String, tokens: Option<u32>) {
        self.duration = self.start_time.map(|s| s.elapsed());
        self.status = StepStatus::Completed {
            result_preview,
            tokens,
        };
    }

    /// Mark this step as failed.
    pub fn fail(&mut self, error: String) {
        self.duration = self.start_time.map(|s| s.elapsed());
        self.status = StepStatus::Failed { error };
    }
}

/// Data from the final result event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalResultData {
    pub answer: String,
    pub total_tokens: u32,
    pub tokens_unavailable_count: usize,
    /// Number of steps that succeeded (generic, was sub_queries_succeeded)
    pub steps_succeeded: usize,
    /// Number of steps that failed (generic, was sub_queries_failed)
    pub steps_failed: usize,
}

/// Terminal-agnostic state machine for agent execution.
///
/// Tracks the current phase, execution steps, timing information,
/// and final results. This is a generic state container - use a
/// `StateHandler` to parse agent-specific events.
#[derive(Clone)]
pub struct ExecutionState {
    phase: String,
    steps: Vec<ExecutionStep>,
    start_time: Instant,
    final_result: Option<FinalResultData>,
}

impl ExecutionState {
    /// Create a new ExecutionState.
    pub fn new() -> Self {
        Self {
            phase: phases::NOT_STARTED.to_string(),
            steps: Vec::new(),
            start_time: Instant::now(),
            final_result: None,
        }
    }

    /// Set the current phase.
    pub fn set_phase(&mut self, phase: impl Into<String>) {
        self.phase = phase.into();
    }

    /// Get the current phase.
    pub fn phase(&self) -> &str {
        &self.phase
    }

    /// Check if execution is complete.
    pub fn is_complete(&self) -> bool {
        self.phase == phases::COMPLETE
    }

    /// Add a new execution step.
    pub fn add_step(&mut self, step: ExecutionStep) {
        self.steps.push(step);
    }

    /// Add multiple steps at once.
    pub fn add_steps(&mut self, steps: Vec<ExecutionStep>) {
        self.steps.extend(steps);
    }

    /// Get a mutable reference to a step by ID.
    pub fn step_mut(&mut self, id: &str) -> Option<&mut ExecutionStep> {
        self.steps.iter_mut().find(|s| s.id == id)
    }

    /// Get a step by ID.
    pub fn step(&self, id: &str) -> Option<&ExecutionStep> {
        self.steps.iter().find(|s| s.id == id)
    }

    /// Get a step by numeric index (for backwards compatibility).
    pub fn step_by_index(&self, index: usize) -> Option<&ExecutionStep> {
        self.steps.get(index)
    }

    /// Get a mutable step by numeric index.
    pub fn step_by_index_mut(&mut self, index: usize) -> Option<&mut ExecutionStep> {
        self.steps.get_mut(index)
    }

    /// Get all steps.
    pub fn steps(&self) -> &[ExecutionStep] {
        &self.steps
    }

    /// Get the total elapsed time since start.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Set the final result.
    pub fn set_final_result(&mut self, result: FinalResultData) {
        self.final_result = Some(result);
        self.phase = phases::COMPLETE.to_string();
    }

    /// Get the final result data, if available.
    pub fn final_result(&self) -> Option<&FinalResultData> {
        self.final_result.as_ref()
    }

    /// Calculate the total time if steps had run sequentially.
    ///
    /// This is the sum of all individual step durations.
    /// Returns None if no steps have completed with timing data.
    pub fn sequential_time(&self) -> Option<Duration> {
        let total: Duration = self.steps.iter().filter_map(|s| s.duration).sum();

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

/// Trait for handling agent-specific events and updating ExecutionState.
///
/// Implement this trait to parse events from a specific agent type
/// and update the generic ExecutionState accordingly.
pub trait StateHandler: Send + Sync {
    /// Process an event and update the state.
    ///
    /// Returns the ID of the step that was updated, if any.
    fn handle(&self, state: &mut ExecutionState, event: &AgentUpdate) -> Option<String>;
}

/// Default handler that processes common events.
///
/// This handles `final_result` events and logs unknown events.
/// For agent-specific events, use a specialized handler.
pub struct DefaultStateHandler;

impl StateHandler for DefaultStateHandler {
    fn handle(&self, state: &mut ExecutionState, event: &AgentUpdate) -> Option<String> {
        match event.event_type.as_str() {
            "final_result" => {
                if let Some(result) = event.as_final_result() {
                    // Extract step counts from extra field (agent-specific)
                    let steps_succeeded = result.metadata.extra["steps_succeeded"]
                        .as_u64()
                        .unwrap_or(0) as usize;
                    let steps_failed =
                        result.metadata.extra["steps_failed"].as_u64().unwrap_or(0) as usize;
                    state.set_final_result(FinalResultData {
                        answer: result.answer,
                        total_tokens: result.metadata.total_tokens,
                        tokens_unavailable_count: result.metadata.tokens_unavailable_count,
                        steps_succeeded,
                        steps_failed,
                    });
                } else {
                    log::warn!(
                        "Received final_result event with malformed data: {:?}",
                        event.data
                    );
                }
                None
            }
            _ => {
                log::debug!("Unhandled event type: {}", event.event_type);
                None
            }
        }
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
        assert_eq!(state.phase(), phases::NOT_STARTED);
        assert!(state.steps().is_empty());
        assert!(state.final_result().is_none());
    }

    #[test]
    fn test_set_phase() {
        let mut state = ExecutionState::new();
        state.set_phase("custom_phase");
        assert_eq!(state.phase(), "custom_phase");
    }

    #[test]
    fn test_add_steps() {
        let mut state = ExecutionState::new();
        state.add_step(ExecutionStep::new("0", "Step 1"));
        state.add_step(ExecutionStep::new("1", "Step 2"));

        assert_eq!(state.steps().len(), 2);
        assert_eq!(state.step("0").unwrap().label, "Step 1");
        assert_eq!(state.step("1").unwrap().label, "Step 2");
    }

    #[test]
    fn test_step_lifecycle() {
        let mut step = ExecutionStep::new("test", "Test step");
        assert!(matches!(step.status, StepStatus::Pending));

        step.start();
        assert!(matches!(step.status, StepStatus::InProgress));
        assert!(step.start_time.is_some());

        step.complete("Result preview".to_string(), Some(42));
        match &step.status {
            StepStatus::Completed {
                result_preview,
                tokens,
            } => {
                assert_eq!(result_preview, "Result preview");
                assert_eq!(*tokens, Some(42));
            }
            _ => panic!("Expected Completed status"),
        }
        assert!(step.duration.is_some());
    }

    #[test]
    fn test_step_failure() {
        let mut step = ExecutionStep::new("test", "Test step");
        step.start();
        step.fail("Something went wrong".to_string());

        match &step.status {
            StepStatus::Failed { error } => {
                assert_eq!(error, "Something went wrong");
            }
            _ => panic!("Expected Failed status"),
        }
    }

    #[test]
    fn test_default_handler_final_result() {
        let handler = DefaultStateHandler;
        let mut state = ExecutionState::new();

        let metadata = gemicro_core::ResultMetadata::with_extra(
            100,
            0,
            5000,
            json!({
                "steps_succeeded": 3,
                "steps_failed": 1,
            }),
        );
        let event = AgentUpdate::final_result("Final answer".to_string(), metadata);
        handler.handle(&mut state, &event);

        assert_eq!(state.phase(), phases::COMPLETE);
        let result = state.final_result().unwrap();
        assert_eq!(result.answer, "Final answer");
        assert_eq!(result.total_tokens, 100);
        assert_eq!(result.steps_succeeded, 3);
        assert_eq!(result.steps_failed, 1);
    }

    #[test]
    fn test_default_handler_unknown_event() {
        let handler = DefaultStateHandler;
        let mut state = ExecutionState::new();

        let event = AgentUpdate::custom("unknown_event", "Unknown", json!({}));
        let result = handler.handle(&mut state, &event);

        // Unknown events should return None and not change state
        assert!(result.is_none());
        assert_eq!(state.phase(), phases::NOT_STARTED);
    }

    #[test]
    fn test_default_handler_malformed_final_result() {
        let handler = DefaultStateHandler;
        let mut state = ExecutionState::new();

        // Create a malformed final_result event (missing required fields)
        let event = AgentUpdate::custom(
            "final_result",
            "Malformed",
            json!({ "answer": 123 }), // answer should be a string in proper format
        );
        let result = handler.handle(&mut state, &event);

        // Should log warning and not crash
        assert!(result.is_none());
        assert!(state.final_result().is_none());
    }

    #[test]
    fn test_sequential_time() {
        let mut state = ExecutionState::new();
        state.add_step(ExecutionStep::new("0", "Step 1"));
        state.add_step(ExecutionStep::new("1", "Step 2"));

        // Manually set durations
        state.step_by_index_mut(0).unwrap().duration = Some(Duration::from_secs(2));
        state.step_by_index_mut(1).unwrap().duration = Some(Duration::from_secs(3));

        let seq_time = state.sequential_time();
        assert!(seq_time.is_some());
        assert_eq!(seq_time.unwrap(), Duration::from_secs(5));
    }
}
