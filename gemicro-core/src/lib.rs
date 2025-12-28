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
//! ## Example
//!
//! ```no_run
//! use gemicro_core::{AgentContext, DeepResearchAgent, ResearchConfig, LlmClient, LlmConfig};
//! use futures_util::StreamExt;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create LLM client and context
//! let genai_client = rust_genai::Client::builder("api-key".to_string()).build();
//! let llm = LlmClient::new(genai_client, LlmConfig::default());
//! let context = AgentContext::new(llm);
//!
//! // Create agent and execute
//! let agent = DeepResearchAgent::new(ResearchConfig::default())?;
//! let stream = agent.execute("What is quantum computing?", context);
//! futures_util::pin_mut!(stream);
//!
//! while let Some(update) = stream.next().await {
//!     let update = update?;
//!     println!("[{}] {}", update.event_type, update.message);
//! }
//! # Ok(())
//! # }
//! ```

pub mod agent;
pub mod config;
pub mod error;
pub mod history;
pub mod llm;
pub mod update;
pub mod utils;

// Re-export public API
pub use agent::{
    Agent, AgentContext, AgentStream, DeepResearchAgent, DeepResearchEventExt, ReactAgent,
    SimpleQaAgent, SimpleQaConfig, SubQueryResult, ToolAgent, ToolAgentConfig, ToolType,
};
pub use config::{
    GemicroConfig, LlmConfig, ReactConfig, ReactPrompts, ResearchConfig, ResearchPrompts, MODEL,
};
pub use error::{AgentError, GemicroError, LlmError};
pub use history::{ConversationHistory, HistoryEntry};
pub use llm::{LlmClient, LlmRequest, LlmStreamChunk};
// Re-export rust-genai types for convenience
pub use rust_genai::{InteractionResponse, UsageMetadata};
pub use update::{AgentUpdate, FinalResult, ResultMetadata};
pub use utils::{extract_total_tokens, first_sentence, truncate, truncate_with_count};
