//! Display module for CLI rendering.
//!
//! This module provides a state-renderer separation pattern:
//! - `state`: Terminal-agnostic state tracking
//! - `renderer`: Trait for swappable backends
//! - `indicatif`: indicatif-based implementation

mod indicatif;
mod renderer;
mod state;

pub use indicatif::IndicatifRenderer;
pub use renderer::Renderer;
pub use state::{DisplayState, Phase};
