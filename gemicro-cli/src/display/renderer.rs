//! Renderer trait for swappable display backends.
//!
//! This trait defines the interface that all renderers must implement,
//! enabling easy switching between indicatif, ratatui, or other backends.

use anyhow::Result;
use gemicro_core::ExecutionTracking;

/// Trait for rendering agent execution progress to the terminal.
///
/// Implementors receive status updates from the agent's tracker and are
/// responsible for updating the terminal display accordingly.
///
/// This trait is agent-agnostic - each agent provides its own tracker
/// implementation that supplies appropriate status messages.
pub trait Renderer {
    /// Called after each event to update the display.
    ///
    /// The tracker provides status messages via `status_message()`.
    fn on_status(&mut self, tracker: &dyn ExecutionTracking) -> Result<()>;

    /// Called when execution completes successfully.
    ///
    /// The tracker provides the final result via `final_result()`.
    fn on_complete(&mut self, tracker: &dyn ExecutionTracking) -> Result<()>;

    /// Called when execution is interrupted by the user (Ctrl+C).
    ///
    /// Displays any partial results that are available.
    fn on_interrupted(&mut self, tracker: &dyn ExecutionTracking) -> Result<()>;

    /// Called when the stream ends to clean up resources.
    fn finish(&mut self) -> Result<()>;
}
