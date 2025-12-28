//! Display module for CLI rendering.
//!
//! This module provides rendering for agent execution:
//! - `renderer`: Trait for swappable backends
//! - `indicatif`: indicatif-based implementation

mod indicatif;
mod renderer;

pub use indicatif::IndicatifRenderer;
pub use renderer::Renderer;
