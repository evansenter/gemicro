//! LLM client wrapper for rust-genai
//!
//! Provides a simplified interface to the rust-genai Interactions API with:
//! - Automatic timeout enforcement from config
//! - Full access to `InteractionResponse` including `UsageMetadata`
//! - Both buffered (`generate`) and streaming (`generate_stream`) modes
//! - Automatic retry with exponential backoff on transient failures
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::{LlmClient, LlmRequest, LlmConfig};
//!
//! # async fn example() -> Result<(), gemicro_core::LlmError> {
//! let genai_client = genai_rs::Client::builder("api-key".to_string()).build()?;
//! let client = LlmClient::new(genai_client, LlmConfig::default());
//!
//! let request = LlmRequest::new("What is the capital of France?");
//! let response = client.generate(request).await?;
//!
//! println!("Response: {}", response.text().unwrap_or(""));
//! if let Some(usage) = &response.usage {
//!     println!("Tokens used: {:?}", usage.total_tokens);
//! }
//! # Ok(())
//! # }
//! ```

mod client;
mod request;

pub use client::LlmClient;
pub use request::{LlmRequest, LlmStreamChunk};

// Re-export Turn types from rust-genai for multi-turn conversation support
pub use genai_rs::{Role, Turn, TurnContent};
