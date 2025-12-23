//! # Gemicro Core
//!
//! Platform-agnostic library for AI agent exploration patterns.
//!
//! This crate provides the core abstractions for building and running AI agents
//! with different implementation patterns (Deep Research, ReAct, Reflexion, etc.).
//!
//! ## Architecture
//!
//! - **Streaming-first**: Agents return async streams of updates for real-time observability
//! - **Soft-typed events**: `AgentUpdate` uses flexible JSON following Evergreen spec philosophy
//! - **Platform-agnostic**: Zero platform-specific dependencies for iOS compatibility
//!
//! ## Example
//!
//! ```no_run
//! use gemicro_core::{AgentUpdate, GemicroConfig};
//!
//! // Create configuration
//! let config = GemicroConfig::default();
//!
//! // Work with agent updates
//! let update = AgentUpdate::decomposition_started();
//! println!("{}", update.message);
//! ```

pub mod config;
pub mod error;
pub mod update;

// Re-export public API
pub use config::{GemicroConfig, LlmConfig, ResearchConfig, MODEL};
pub use error::{AgentError, GemicroError, LlmError};
pub use update::{AgentUpdate, FinalResult, ResultMetadata, SubQueryResult};
