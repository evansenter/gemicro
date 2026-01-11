//! LLM client wrapper for genai-rs
//!
//! Provides a simplified interface to the genai-rs Interactions API with:
//! - Automatic timeout enforcement from config
//! - Full access to `InteractionResponse` including `UsageMetadata`
//! - Both buffered (`generate`) and streaming (`generate_stream`) modes
//! - Automatic retry with exponential backoff on transient failures
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::{LlmClient, LlmConfig};
//!
//! # async fn example() -> Result<(), gemicro_core::LlmError> {
//! let genai_client = genai_rs::Client::builder("api-key".to_string()).build()?;
//! let client = LlmClient::new(genai_client, LlmConfig::default());
//!
//! // Build the request using genai-rs InteractionBuilder
//! let request = client.client().interaction()
//!     .user_text("What is the capital of France?")
//!     .build();
//!
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

pub use client::{GenerateWithToolsResponse, LlmClient, LlmStreamChunk};

// Re-export Turn types from genai-rs for multi-turn conversation support
pub use genai_rs::{Role, Turn, TurnContent};

// Re-export FunctionCallInfo for generate_with_tools callback signature
pub use genai_rs::FunctionCallInfo;
