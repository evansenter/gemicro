//! Trajectory serialization for offline replay and evaluation
//!
//! This module provides types and utilities for capturing full LLM interaction
//! traces during agent execution, enabling:
//!
//! - **Offline replay**: Re-run agent logic without API calls
//! - **Evaluation datasets**: Build test sets from real production runs
//! - **Debugging**: Inspect exact LLM requests and responses
//!
//! # Architecture
//!
//! A `Trajectory` captures an entire agent execution:
//! - `steps`: Raw LLM request/response pairs with timing
//! - `events`: High-level `AgentUpdate` events for compatibility
//! - `metadata`: Execution summary (tokens, duration, etc.)
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::trajectory::Trajectory;
//!
//! // Save a trajectory
//! # fn example(trajectory: Trajectory) -> std::io::Result<()> {
//! trajectory.save("trajectories/run_001.json")?;
//!
//! // Load and inspect
//! let loaded = Trajectory::load("trajectories/run_001.json")?;
//! println!("Query: {}", loaded.query);
//! println!("Steps: {}", loaded.steps.len());
//! for step in &loaded.steps {
//!     println!("  Phase: {}, Duration: {}ms", step.phase, step.duration_ms);
//! }
//! # Ok(())
//! # }
//! ```

mod builder;
mod data;

pub use builder::TrajectoryBuilder;
pub use data::{
    LlmResponseData, SerializableStreamChunk, Trajectory, TrajectoryMetadata, TrajectoryStep,
};

/// Current schema version for trajectory files
///
/// Version history:
/// - 1.0.0: Initial release with dual Option pattern for response/stream_chunks
/// - 2.0.0: Changed to LlmResponseData enum for type-safe response mode
/// - 3.0.0: Changed TrajectoryStep.request from SerializableLlmRequest to serde_json::Value
pub const SCHEMA_VERSION: &str = "3.0.0";
