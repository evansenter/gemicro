//! Renderer trait for swappable display backends.
//!
//! This trait defines the interface that all renderers must implement,
//! enabling easy switching between indicatif, ratatui, or other backends.

use anyhow::Result;
use gemicro_runner::ExecutionState;

/// Trait for rendering agent execution progress to the terminal.
///
/// Implementors receive notifications when state changes and are
/// responsible for updating the terminal display accordingly.
///
/// This trait is agent-agnostic - it uses string-based phases and
/// generic "steps" instead of agent-specific concepts.
pub trait Renderer {
    /// Called when the execution phase changes.
    ///
    /// Common phases include: "not_started", "decomposing", "executing",
    /// "synthesizing", "complete". Agents may define additional phases.
    fn on_phase_change(&mut self, state: &ExecutionState) -> Result<()>;

    /// Called when an execution step's status changes.
    ///
    /// The `id` parameter indicates which step was updated.
    /// Query `state.step(id)` to get the current status.
    fn on_step_update(&mut self, state: &ExecutionState, id: &str) -> Result<()>;

    /// Called when the final result is available.
    ///
    /// Query `state.final_result()` to get the answer and metadata.
    fn on_final_result(&mut self, state: &ExecutionState) -> Result<()>;

    /// Called when execution is interrupted by the user (Ctrl+C).
    ///
    /// Displays any partial results that are available.
    fn on_interrupted(&mut self, state: &ExecutionState) -> Result<()>;

    /// Called when the stream ends to clean up resources.
    fn finish(&mut self) -> Result<()>;
}
