//! ReAct Agent
//!
//! Implements the ReAct (Reasoning + Acting) pattern: iterative loops of
//! Thought → Action → Observation until a final answer is reached.
//!
//! # Example
//!
//! ```no_run
//! use gemicro_react::{ReactAgent, ReactConfig};
//! use gemicro_core::{AgentContext, LlmClient, LlmConfig};
//! use futures_util::StreamExt;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let genai_client = rust_genai::Client::builder("api-key".to_string()).build();
//! let context = AgentContext::new(LlmClient::new(genai_client, LlmConfig::default()));
//! let agent = ReactAgent::new(ReactConfig::default())?;
//!
//! let stream = agent.execute("What is 25 * 4?", context);
//! futures_util::pin_mut!(stream);
//! while let Some(update) = stream.next().await {
//!     println!("{:?}", update?);
//! }
//! # Ok(())
//! # }
//! ```

mod agent;
mod config;

pub use agent::ReactAgent;
pub use config::{ReactConfig, ReactPrompts};

// Re-export core types for convenience
pub use gemicro_core::{Agent, AgentContext, AgentError, AgentStream, AgentUpdate};
