//! Streaming event types for agent updates.
//!
//! Re-exports types from `gemicro-streaming-events` crate for backwards compatibility.
//! External crates can depend directly on `gemicro-streaming-events` for minimal deps.

pub use gemicro_streaming_events::{AgentUpdate, FinalResult, ResultMetadata, EVENT_FINAL_RESULT};
