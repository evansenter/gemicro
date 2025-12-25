//! Renderer trait for swappable display backends.
//!
//! This trait defines the interface that all renderers must implement,
//! enabling easy switching between indicatif, ratatui, or other backends.

use anyhow::Result;
use gemicro_runner::ExecutionState;

/// Trait for rendering research progress to the terminal.
///
/// Implementors receive notifications when state changes and are
/// responsible for updating the terminal display accordingly.
pub trait Renderer {
    /// Called when the research phase changes.
    ///
    /// Phases progress: NotStarted → Decomposing → Executing → Synthesizing → Complete
    fn on_phase_change(&mut self, state: &ExecutionState) -> Result<()>;

    /// Called when a sub-query's status changes.
    ///
    /// The `id` parameter indicates which sub-query was updated.
    /// Query `state.sub_query(id)` to get the current status.
    fn on_sub_query_update(&mut self, state: &ExecutionState, id: usize) -> Result<()>;

    /// Called when the final result is available.
    ///
    /// Query `state.final_result()` to get the answer and metadata.
    fn on_final_result(&mut self, state: &ExecutionState) -> Result<()>;

    /// Called when the research is interrupted by the user (Ctrl+C).
    ///
    /// Displays any partial results that are available.
    fn on_interrupted(&mut self, state: &ExecutionState) -> Result<()>;

    /// Called when the stream ends to clean up resources.
    fn finish(&mut self) -> Result<()>;
}
