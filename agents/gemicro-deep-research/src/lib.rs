//! Deep Research Agent
//!
//! Implements the Deep Research pattern: query decomposition, parallel sub-query
//! execution, and synthesis.
//!
//! # Architecture
//!
//! The agent follows a three-phase pattern:
//! 1. **Decomposition**: Break the query into focused sub-questions
//! 2. **Parallel Execution**: Answer each sub-question concurrently
//! 3. **Synthesis**: Combine findings into a comprehensive answer
//!
//! # Example
//!
//! ```no_run
//! use gemicro_deep_research::{DeepResearchAgent, ResearchConfig, DeepResearchEventExt};
//! use gemicro_core::{AgentContext, LlmClient, LlmConfig};
//! use futures_util::StreamExt;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let genai_client = genai_rs::Client::builder("api-key".to_string()).build().map_err(|e| AgentError::Other(e.to_string()))?;
//! let context = AgentContext::new(LlmClient::new(genai_client, LlmConfig::default()));
//! let agent = DeepResearchAgent::new(ResearchConfig::default())?;
//!
//! let stream = agent.execute("What is Rust?", context);
//! futures_util::pin_mut!(stream);
//! while let Some(update) = stream.next().await {
//!     println!("{:?}", update?);
//! }
//! # Ok(())
//! # }
//! ```

mod agent;
mod config;
mod events;

pub use agent::DeepResearchAgent;
pub use config::{ResearchConfig, ResearchPrompts};
pub use events::{DeepResearchEventExt, SubQueryResult};
