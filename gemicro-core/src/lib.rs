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
//! - **Soft-typed events**: `AgentUpdate` uses flexible JSON following [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) philosophy
//! - **Platform-agnostic**: Zero platform-specific dependencies for iOS compatibility
//!
//! ## Agent Implementations
//!
//! Agents are in separate crates for hermetic isolation:
//! - `gemicro-deep-research`: Decomposes queries, executes sub-queries in parallel, synthesizes
//! - `gemicro-react`: Reasoning + Acting pattern with iterative tool use
//! - `gemicro-simple-qa`: Minimal single-call agent for reference/demonstration
//! - `gemicro-tool-agent`: Native function calling via rust-genai's `#[tool]` macro
//! - `gemicro-judge`: LLM-based semantic evaluation for scoring
//!
//! ## Example
//!
//! ```no_run
//! use gemicro_core::{Agent, AgentContext, AgentStream, AgentUpdate, AgentError, LlmClient, LlmConfig};
//!
//! struct MyAgent;
//!
//! impl Agent for MyAgent {
//!     fn name(&self) -> &str { "my_agent" }
//!     fn description(&self) -> &str { "A custom agent" }
//!     fn execute(&self, query: &str, context: AgentContext) -> AgentStream<'_> {
//!         // Implementation would return a stream of updates
//!         # todo!()
//!     }
//! }
//! ```

pub mod agent;
pub mod config;
pub mod error;
pub mod history;
pub mod llm;
pub mod update;
pub mod utils;

// Re-export public API - core infrastructure only
pub use agent::{
    remaining_time, timeout_error, with_timeout_and_cancellation, Agent, AgentContext, AgentStream,
};
pub use config::{GemicroConfig, LlmConfig, MODEL};
pub use error::{AgentError, GemicroError, LlmError};
pub use history::{ConversationHistory, HistoryEntry};
pub use llm::{LlmClient, LlmRequest, LlmStreamChunk};
// Re-export rust-genai types for convenience
pub use rust_genai::{InteractionResponse, UsageMetadata};
pub use update::{AgentUpdate, FinalResult, ResultMetadata};
pub use utils::{extract_total_tokens, first_sentence, truncate, truncate_with_count};
