//! LLM client wrapper for rust-genai
//!
//! Provides a simplified interface to the rust-genai Interactions API with:
//! - Automatic timeout enforcement from config
//! - Normalized token counting (returns 0 with warning if unavailable)
//! - Both buffered (`generate`) and streaming (`generate_stream`) modes
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::{LlmClient, LlmRequest, LlmConfig};
//!
//! # async fn example() -> Result<(), gemicro_core::LlmError> {
//! let genai_client = rust_genai::Client::builder("api-key".to_string()).build();
//! let client = LlmClient::new(genai_client, LlmConfig::default());
//!
//! let request = LlmRequest::new("What is the capital of France?");
//! let response = client.generate(request).await?;
//!
//! println!("Response: {}", response.text);
//! println!("Tokens used: {}", response.tokens_used);
//! # Ok(())
//! # }
//! ```

use crate::config::{LlmConfig, MODEL};
use crate::error::LlmError;
use futures_util::stream::{Stream, StreamExt};

/// LLM client wrapping rust-genai with timeout and configuration
pub struct LlmClient {
    /// Underlying rust-genai client
    client: rust_genai::Client,

    /// LLM configuration (timeout, tokens, temperature, etc.)
    config: LlmConfig,
}

/// Request to the LLM
#[derive(Debug, Clone)]
pub struct LlmRequest {
    /// User prompt
    pub prompt: String,

    /// Optional system instruction
    pub system_instruction: Option<String>,
}

/// Response from the LLM (buffered mode)
#[derive(Debug, Clone)]
pub struct LlmResponse {
    /// Generated text
    pub text: String,

    /// Number of tokens used (0 if unavailable)
    pub tokens_used: u32,

    /// Interaction ID for tracking
    pub interaction_id: String,
}

/// Chunk from streaming LLM response
#[derive(Debug, Clone)]
pub struct LlmStreamChunk {
    /// Text content of this chunk
    pub text: String,

    /// Whether this is the final chunk
    pub is_final: bool,
}

impl LlmRequest {
    /// Create a new LLM request with just a prompt
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            system_instruction: None,
        }
    }

    /// Create a new LLM request with prompt and system instruction
    pub fn with_system(prompt: impl Into<String>, system: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            system_instruction: Some(system.into()),
        }
    }
}

impl LlmClient {
    /// Create a new LLM client with the given rust-genai client and configuration
    pub fn new(client: rust_genai::Client, config: LlmConfig) -> Self {
        Self { client, config }
    }

    /// Generate a complete response (buffered mode)
    ///
    /// This method waits for the full response before returning.
    /// Use `generate_stream()` for real-time token-by-token output.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - `LlmError::Timeout` if the request exceeds `config.timeout`
    /// - `LlmError::GenAi` for underlying API errors
    /// - `LlmError::NoContent` if the response is empty
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use gemicro_core::{LlmClient, LlmRequest, LlmConfig};
    /// # async fn example() -> Result<(), gemicro_core::LlmError> {
    /// # let genai_client = rust_genai::Client::builder("key".to_string()).build();
    /// let client = LlmClient::new(genai_client, LlmConfig::default());
    /// let request = LlmRequest::new("Explain quantum computing");
    /// let response = client.generate(request).await?;
    /// println!("{}", response.text);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn generate(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
        self.validate_request(&request)?;

        let interaction = self.build_interaction(&request);

        // Execute with timeout
        let timeout_duration = self.config.timeout;
        let response = tokio::time::timeout(timeout_duration, interaction.create())
            .await
            .map_err(|_| LlmError::Timeout(timeout_duration.as_millis() as u64))?
            .map_err(LlmError::from)?;

        // Extract text
        let text = response.text().ok_or(LlmError::NoContent)?.to_string();

        // Extract tokens (graceful degradation)
        let tokens_used = Self::extract_token_count(&response);

        // Extract interaction ID
        let interaction_id = response.id.clone();

        Ok(LlmResponse {
            text,
            tokens_used,
            interaction_id,
        })
    }

    /// Generate a streaming response
    ///
    /// Returns a stream of text chunks as they arrive from the LLM.
    /// The final chunk has `is_final = true`. Each chunk read is subject
    /// to the configured timeout.
    ///
    /// # Errors
    ///
    /// Each stream item is a `Result<LlmStreamChunk, LlmError>`.
    /// Errors can occur at any point during streaming.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use gemicro_core::{LlmClient, LlmRequest, LlmConfig};
    /// # use futures_util::stream::StreamExt;
    /// # async fn example() -> Result<(), gemicro_core::LlmError> {
    /// # let genai_client = rust_genai::Client::builder("key".to_string()).build();
    /// let client = LlmClient::new(genai_client, LlmConfig::default());
    /// let request = LlmRequest::new("Count to 10");
    ///
    /// let stream = client.generate_stream(request);
    /// futures_util::pin_mut!(stream);
    /// while let Some(chunk) = stream.next().await {
    ///     let chunk = chunk?;
    ///     print!("{}", chunk.text);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn generate_stream(
        &self,
        request: LlmRequest,
    ) -> impl Stream<Item = Result<LlmStreamChunk, LlmError>> + Send + '_ {
        let timeout_duration = self.config.timeout;

        async_stream::try_stream! {
            self.validate_request(&request)?;

            let interaction = self.build_interaction(&request);

            // Get the stream (not async - returns immediately)
            let stream = interaction.create_stream();
            futures_util::pin_mut!(stream);

            // Process chunks with per-chunk timeout
            loop {
                let chunk = tokio::time::timeout(timeout_duration, stream.next())
                    .await
                    .map_err(|_| LlmError::Timeout(timeout_duration.as_millis() as u64))?;

                match chunk {
                    Some(Ok(response)) => {
                        if let Some(text) = response.text() {
                            yield LlmStreamChunk {
                                text: text.to_string(),
                                is_final: false,
                            };
                        }
                    }
                    Some(Err(e)) => {
                        Err(LlmError::from(e))?;
                    }
                    None => break,
                }
            }

            // Emit final marker
            yield LlmStreamChunk {
                text: String::new(),
                is_final: true,
            };
        }
    }

    /// Validate the request before processing
    fn validate_request(&self, request: &LlmRequest) -> Result<(), LlmError> {
        if request.prompt.is_empty() {
            return Err(LlmError::InvalidRequest(
                "Prompt cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    /// Build an interaction from the request (DRY helper)
    fn build_interaction(&self, request: &LlmRequest) -> rust_genai::InteractionBuilder<'_> {
        let mut interaction = self
            .client
            .interaction()
            .with_model(MODEL)
            .with_text(&request.prompt);

        if let Some(ref system) = request.system_instruction {
            interaction = interaction.with_system_instruction(system);
        }

        interaction
    }

    /// Extract token count from response, returning 0 with warning if unavailable
    fn extract_token_count(response: &rust_genai::InteractionResponse) -> u32 {
        response
            .usage
            .as_ref()
            .and_then(|u| u.total_tokens)
            .and_then(|t| u32::try_from(t).ok())
            .unwrap_or_else(|| {
                log::warn!(
                    "Token count unavailable for interaction {}, returning 0",
                    response.id
                );
                0
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_request_new() {
        let req = LlmRequest::new("Test prompt");
        assert_eq!(req.prompt, "Test prompt");
        assert!(req.system_instruction.is_none());
    }

    #[test]
    fn test_llm_request_with_system() {
        let req = LlmRequest::with_system("User prompt", "System instruction");
        assert_eq!(req.prompt, "User prompt");
        assert_eq!(
            req.system_instruction,
            Some("System instruction".to_string())
        );
    }

    #[test]
    fn test_llm_response_creation() {
        let response = LlmResponse {
            text: "Generated text".to_string(),
            tokens_used: 42,
            interaction_id: "test-123".to_string(),
        };

        assert_eq!(response.text, "Generated text");
        assert_eq!(response.tokens_used, 42);
        assert_eq!(response.interaction_id, "test-123");
    }

    #[test]
    fn test_llm_stream_chunk_creation() {
        let chunk = LlmStreamChunk {
            text: "Hello".to_string(),
            is_final: false,
        };
        assert_eq!(chunk.text, "Hello");
        assert!(!chunk.is_final);

        let final_chunk = LlmStreamChunk {
            text: String::new(),
            is_final: true,
        };
        assert!(final_chunk.is_final);
        assert!(final_chunk.text.is_empty());
    }
}
