//! Display module for CLI rendering.
//!
//! This module provides a state-renderer separation pattern:
//! - State tracking is provided by `gemicro-runner::ExecutionState`
//! - `renderer`: Trait for swappable backends
//! - `indicatif`: indicatif-based implementation

mod indicatif;
mod renderer;

pub use indicatif::IndicatifRenderer;
pub use renderer::Renderer;

// Re-export execution state types from gemicro-runner
// DisplayState is an alias for backwards compatibility
pub use gemicro_runner::ExecutionState as DisplayState;
pub use gemicro_runner::Phase;
