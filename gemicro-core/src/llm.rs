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
//! let genai_client = rust_genai::Client::builder("api-key".to_string()).build();
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

use crate::config::{LlmConfig, MODEL};
use crate::error::LlmError;
use crate::trajectory::{
    LlmResponseData, SerializableLlmRequest, SerializableStreamChunk, TrajectoryStep,
};
use futures_util::stream::{Stream, StreamExt};
use rust_genai::GenerationConfig;
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime};
use tokio_util::sync::CancellationToken;

/// LLM client wrapping rust-genai with timeout and configuration
///
/// Optionally supports trajectory recording for offline replay and evaluation.
/// Recording is disabled by default for zero overhead. Use [`LlmClient::with_recording`]
/// to enable it.
pub struct LlmClient {
    /// Underlying rust-genai client
    client: rust_genai::Client,

    /// LLM configuration (timeout, tokens, temperature, etc.)
    config: LlmConfig,

    /// Optional recorder for trajectory capture
    ///
    /// When `Some`, all LLM interactions are recorded as `TrajectoryStep`s.
    /// Use `take_steps()` to retrieve recorded steps.
    recorder: Option<Arc<RwLock<Vec<TrajectoryStep>>>>,

    /// Current phase label for recorded steps
    ///
    /// This is a semantic identifier like "decomposition" or "sub_query_2"
    /// that helps identify what part of the agent's execution each LLM call
    /// belongs to. Call `set_phase()` before making LLM calls.
    current_phase: Arc<RwLock<String>>,
}

impl std::fmt::Debug for LlmClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmClient")
            .field("model", &MODEL)
            .field("client", &"[REDACTED]")
            .field("config", &self.config)
            .field("is_recording", &self.is_recording())
            .finish()
    }
}

/// Request to the LLM
#[derive(Debug, Clone)]
pub struct LlmRequest {
    /// User prompt
    pub prompt: String,

    /// Optional system instruction
    pub system_instruction: Option<String>,

    /// Enable Google Search grounding for real-time web data
    ///
    /// When enabled, the model can search the web and ground its responses
    /// in real-time information. This is useful for queries about current
    /// events, recent releases, or live data.
    ///
    /// Note: Grounded requests may have different pricing.
    pub use_google_search: bool,

    /// Optional JSON schema for structured output
    ///
    /// When provided, the model will generate a response that conforms to
    /// this JSON schema. Use this for reliable structured data extraction.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::LlmRequest;
    /// use serde_json::json;
    ///
    /// let schema = json!({
    ///     "type": "object",
    ///     "properties": {
    ///         "correct": {"type": "boolean"},
    ///         "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    ///     },
    ///     "required": ["correct", "confidence"]
    /// });
    ///
    /// let request = LlmRequest::new("Is Paris the capital of France?")
    ///     .with_response_format(schema);
    /// ```
    pub response_format: Option<serde_json::Value>,
}

/// Chunk from streaming LLM response
///
/// The stream ends naturally when no more chunks are available (yields `None`).
/// There is no artificial "final" marker - simply iterate until the stream ends.
#[derive(Debug, Clone)]
pub struct LlmStreamChunk {
    /// Text content of this chunk (may be empty for some chunks)
    pub text: String,
}

impl LlmRequest {
    /// Create a new LLM request with just a prompt
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            system_instruction: None,
            use_google_search: false,
            response_format: None,
        }
    }

    /// Create a new LLM request with prompt and system instruction
    pub fn with_system(prompt: impl Into<String>, system: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            system_instruction: Some(system.into()),
            use_google_search: false,
            response_format: None,
        }
    }

    /// Enable Google Search grounding for this request
    ///
    /// When enabled, the model can search the web to ground its responses
    /// in real-time information.
    pub fn with_google_search(mut self) -> Self {
        self.use_google_search = true;
        self
    }

    /// Set a JSON schema for structured output
    ///
    /// When provided, the model will generate a response that conforms to
    /// this JSON schema. The response text will be valid JSON matching the schema.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::LlmRequest;
    /// use serde_json::json;
    ///
    /// let schema = json!({
    ///     "type": "object",
    ///     "properties": {
    ///         "answer": {"type": "string"},
    ///         "confidence": {"type": "number"}
    ///     },
    ///     "required": ["answer", "confidence"]
    /// });
    ///
    /// let request = LlmRequest::new("What is the capital of France?")
    ///     .with_response_format(schema);
    /// ```
    pub fn with_response_format(mut self, schema: serde_json::Value) -> Self {
        self.response_format = Some(schema);
        self
    }
}

impl LlmClient {
    /// Create a new LLM client with the given rust-genai client and configuration
    ///
    /// Recording is disabled by default. Use [`with_recording`](Self::with_recording)
    /// to create a client that captures LLM interactions.
    pub fn new(client: rust_genai::Client, config: LlmConfig) -> Self {
        Self {
            client,
            config,
            recorder: None,
            current_phase: Arc::new(RwLock::new("default".to_string())),
        }
    }

    /// Create a new LLM client with trajectory recording enabled
    ///
    /// All LLM interactions will be recorded as `TrajectoryStep`s.
    /// Use [`set_phase`](Self::set_phase) to label steps with semantic context,
    /// and [`take_steps`](Self::take_steps) to retrieve recorded steps.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gemicro_core::{LlmClient, LlmRequest, LlmConfig};
    ///
    /// # async fn example() -> Result<(), gemicro_core::LlmError> {
    /// let genai_client = rust_genai::Client::builder("key".to_string()).build();
    /// let client = LlmClient::with_recording(genai_client, LlmConfig::default());
    ///
    /// client.set_phase("decomposition");
    /// let _response = client.generate(LlmRequest::new("Break this down")).await?;
    ///
    /// client.set_phase("synthesis");
    /// let _response = client.generate(LlmRequest::new("Combine these")).await?;
    ///
    /// let steps = client.take_steps();
    /// assert_eq!(steps.len(), 2);
    /// assert_eq!(steps[0].phase, "decomposition");
    /// assert_eq!(steps[1].phase, "synthesis");
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_recording(client: rust_genai::Client, config: LlmConfig) -> Self {
        Self {
            client,
            config,
            recorder: Some(Arc::new(RwLock::new(Vec::new()))),
            current_phase: Arc::new(RwLock::new("default".to_string())),
        }
    }

    /// Check if this client is recording LLM interactions
    pub fn is_recording(&self) -> bool {
        self.recorder.is_some()
    }

    /// Set the current phase label for recorded steps
    ///
    /// Call this before making LLM calls to label them with semantic context.
    /// The phase is a soft-typed string following Evergreen principles - use
    /// any identifier that makes sense for your agent (e.g., "decomposition",
    /// "sub_query_3", "synthesis", "reflection").
    pub fn set_phase(&self, phase: impl Into<String>) {
        let phase_str = phase.into();
        match self.current_phase.write() {
            Ok(mut current) => *current = phase_str,
            Err(poisoned) => {
                log::warn!("Phase lock poisoned while setting phase - recovering");
                *poisoned.into_inner() = phase_str;
            }
        }
    }

    /// Get the current phase label
    pub fn current_phase(&self) -> String {
        match self.current_phase.read() {
            Ok(p) => p.clone(),
            Err(poisoned) => {
                log::warn!("Phase lock poisoned while reading phase - recovering");
                poisoned.into_inner().clone()
            }
        }
    }

    /// Take all recorded steps, leaving the recorder empty
    ///
    /// Returns an empty vector if recording is disabled.
    /// After calling this, new steps will start accumulating again.
    pub fn take_steps(&self) -> Vec<TrajectoryStep> {
        let Some(ref recorder) = self.recorder else {
            return Vec::new();
        };

        match recorder.write() {
            Ok(mut steps) => std::mem::take(&mut *steps),
            Err(poisoned) => {
                log::warn!("Trajectory recorder lock poisoned - recovering recorded steps");
                std::mem::take(&mut *poisoned.into_inner())
            }
        }
    }

    /// Get the number of recorded steps without consuming them
    pub fn step_count(&self) -> usize {
        let Some(ref recorder) = self.recorder else {
            return 0;
        };

        match recorder.read() {
            Ok(steps) => steps.len(),
            Err(poisoned) => {
                log::warn!("Trajectory recorder lock poisoned while reading count - recovering");
                poisoned.into_inner().len()
            }
        }
    }

    /// Record a step if recording is enabled
    fn record_step(&self, step: TrajectoryStep) {
        let Some(ref recorder) = self.recorder else {
            return;
        };

        match recorder.write() {
            Ok(mut steps) => steps.push(step),
            Err(poisoned) => {
                log::warn!("Trajectory recorder lock poisoned - recovering and recording step");
                poisoned.into_inner().push(step);
            }
        }
    }

    /// Get a reference to the underlying rust-genai client.
    ///
    /// Use this for advanced operations like function calling that require
    /// direct access to the client.
    ///
    /// # Warning: Escape Hatch
    ///
    /// This method bypasses `LlmClient`'s automatic timeout enforcement, retry
    /// logic, and consistent error handling. When using this directly, you are
    /// responsible for implementing these guarantees yourself.
    ///
    /// Prefer using `generate()` or `generate_stream()` when possible.
    pub fn client(&self) -> &rust_genai::Client {
        &self.client
    }

    /// Get a reference to the LLM configuration.
    pub fn config(&self) -> &LlmConfig {
        &self.config
    }

    /// Generate a complete response (buffered mode)
    ///
    /// This method waits for the full response before returning.
    /// Use `generate_stream()` for real-time token-by-token output.
    ///
    /// Returns the full [`rust_genai::InteractionResponse`] with access to:
    /// - `response.text()` - the generated text
    /// - `response.usage` - full token usage metadata
    /// - `response.id` - interaction ID for tracking
    /// - `response.grounding_metadata` - Google Search sources (if enabled)
    ///
    /// # Retry Behavior
    ///
    /// Transient failures (timeouts, rate limits, temporary API errors) are
    /// automatically retried up to `config.max_retries` times with exponential
    /// backoff starting at `config.retry_base_delay_ms`.
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
    /// println!("{}", response.text().unwrap_or(""));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn generate(
        &self,
        request: LlmRequest,
    ) -> Result<rust_genai::InteractionResponse, LlmError> {
        self.validate_request(&request)?;

        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            match self.generate_once(&request).await {
                Ok(response) => return Ok(response),
                Err(e) if Self::is_retryable(&e) && attempt < self.config.max_retries => {
                    log::warn!(
                        "LLM request failed (attempt {}/{}): {}, retrying...",
                        attempt + 1,
                        self.config.max_retries + 1,
                        e
                    );
                    last_error = Some(e);
                    tokio::time::sleep(self.config.retry_delay(attempt)).await;
                }
                Err(e) => return Err(e),
            }
        }

        // This shouldn't be reachable, but just in case
        Err(last_error
            .unwrap_or_else(|| LlmError::Other("Retry loop exited unexpectedly".to_string())))
    }

    /// Generate a complete response with cancellation support
    ///
    /// Like `generate()`, but also checks the cancellation token before each
    /// attempt and during retry delays. Returns `LlmError::Cancelled` if the
    /// token is cancelled.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use gemicro_core::{LlmClient, LlmRequest, LlmConfig};
    /// # use tokio_util::sync::CancellationToken;
    /// # async fn example() -> Result<(), gemicro_core::LlmError> {
    /// # let genai_client = rust_genai::Client::builder("key".to_string()).build();
    /// let client = LlmClient::new(genai_client, LlmConfig::default());
    /// let token = CancellationToken::new();
    /// let request = LlmRequest::new("Explain quantum computing");
    ///
    /// // In another task: token.cancel() to abort
    /// let response = client.generate_with_cancellation(request, &token).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn generate_with_cancellation(
        &self,
        request: LlmRequest,
        cancellation_token: &CancellationToken,
    ) -> Result<rust_genai::InteractionResponse, LlmError> {
        self.validate_request(&request)?;

        // Check cancellation before starting
        if cancellation_token.is_cancelled() {
            return Err(LlmError::Cancelled);
        }

        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            // Check cancellation before each attempt
            if cancellation_token.is_cancelled() {
                return Err(LlmError::Cancelled);
            }

            // Race the LLM call against cancellation
            let result = tokio::select! {
                res = self.generate_once(&request) => res,
                _ = cancellation_token.cancelled() => {
                    return Err(LlmError::Cancelled);
                }
            };

            match result {
                Ok(response) => return Ok(response),
                Err(e) if Self::is_retryable(&e) && attempt < self.config.max_retries => {
                    log::warn!(
                        "LLM request failed (attempt {}/{}): {}, retrying...",
                        attempt + 1,
                        self.config.max_retries + 1,
                        e
                    );
                    last_error = Some(e);

                    // Race retry delay against cancellation
                    tokio::select! {
                        _ = tokio::time::sleep(self.config.retry_delay(attempt)) => {}
                        _ = cancellation_token.cancelled() => {
                            return Err(LlmError::Cancelled);
                        }
                    }
                }
                Err(e) => return Err(e),
            }
        }

        Err(last_error
            .unwrap_or_else(|| LlmError::Other("Retry loop exited unexpectedly".to_string())))
    }

    /// Execute a single generate request (no retries)
    async fn generate_once(
        &self,
        request: &LlmRequest,
    ) -> Result<rust_genai::InteractionResponse, LlmError> {
        let interaction = self.build_interaction(request);

        // Capture timing for recording
        let started_at = SystemTime::now();
        let start_instant = Instant::now();

        // Execute with timeout
        let timeout_duration = self.config.timeout;
        let response = tokio::time::timeout(timeout_duration, interaction.create())
            .await
            .map_err(|_| LlmError::Timeout(timeout_duration.as_millis() as u64))?
            .map_err(LlmError::from)?;

        // Validate response has content
        if response.text().is_none() {
            return Err(LlmError::NoContent);
        }

        // Record the step if recording is enabled
        if self.is_recording() {
            let duration_ms = start_instant.elapsed().as_millis() as u64;

            // Serialize the response to JSON for storage
            let response_data = serde_json::to_value(&response)
                .map(LlmResponseData::Buffered)
                .unwrap_or_else(|e| {
                    log::warn!("Failed to serialize LLM response: {}", e);
                    LlmResponseData::Buffered(serde_json::json!({"error": "serialization_failed"}))
                });

            self.record_step(TrajectoryStep {
                phase: self.current_phase(),
                request: SerializableLlmRequest::from(request),
                response: response_data,
                duration_ms,
                started_at,
            });
        }

        Ok(response)
    }

    /// Determine if an error is retryable
    fn is_retryable(error: &LlmError) -> bool {
        match error {
            // Transient failures that may succeed on retry
            LlmError::Timeout(_) => true,
            LlmError::RateLimit(_) => true,
            // API errors may be transient (network issues, server overload)
            LlmError::GenAi(_) => true,
            // These are not retryable
            LlmError::InvalidRequest(_) => false,
            LlmError::ResponseProcessing(_) => false,
            LlmError::NoContent => false,
            LlmError::Cancelled => false,
            LlmError::Other(_) => false,
        }
    }

    /// Generate a streaming response
    ///
    /// Returns a stream of text chunks as they arrive from the LLM.
    /// The stream ends naturally when the response is complete (yields `None`).
    ///
    /// # Timeout Behavior
    ///
    /// Each chunk read is subject to the configured timeout. The timeout resets
    /// for each chunk, so slow streaming (where each chunk arrives within the
    /// timeout window) will not trigger an error. If no chunk arrives within
    /// the timeout period, `LlmError::Timeout` is returned.
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

            // Capture timing and chunks for recording
            let started_at = SystemTime::now();
            let start_instant = Instant::now();
            let mut recorded_chunks: Vec<SerializableStreamChunk> = Vec::new();
            let is_recording = self.is_recording();
            let serializable_request = SerializableLlmRequest::from(&request);
            let phase = self.current_phase();

            // Get the stream (not async - returns immediately)
            let stream = interaction.create_stream();
            futures_util::pin_mut!(stream);

            // Process chunks with per-chunk timeout
            loop {
                let chunk = tokio::time::timeout(timeout_duration, stream.next())
                    .await
                    .map_err(|_| LlmError::Timeout(timeout_duration.as_millis() as u64))?;

                match chunk {
                    Some(Ok(stream_chunk)) => {
                        use rust_genai::StreamChunk;
                        match stream_chunk {
                            StreamChunk::Delta(delta) => {
                                if let Some(text) = delta.text() {
                                    let text_str = text.to_string();

                                    // Record chunk if recording is enabled
                                    if is_recording {
                                        recorded_chunks.push(SerializableStreamChunk {
                                            text: text_str.clone(),
                                            offset_ms: start_instant.elapsed().as_millis() as u64,
                                        });
                                    }

                                    yield LlmStreamChunk {
                                        text: text_str,
                                    };
                                }
                                // Skip deltas with no text (e.g., thought deltas)
                            }
                            StreamChunk::Complete(_response) => {
                                // Final response - stream will end after this
                                // We could extract usage metadata here if needed
                            }
                            _ => {
                                // Handle future StreamChunk variants gracefully
                                log::debug!("Unknown StreamChunk variant received");
                            }
                        }
                    }
                    Some(Err(e)) => {
                        Err(LlmError::from(e))?;
                    }
                    None => break,
                }
            }

            // Record the step at the end of streaming
            if is_recording {
                let duration_ms = start_instant.elapsed().as_millis() as u64;
                self.record_step(TrajectoryStep {
                    phase,
                    request: serializable_request,
                    response: LlmResponseData::Streaming(recorded_chunks),
                    duration_ms,
                    started_at,
                });
            }
            // Stream ends naturally - no artificial final marker needed
        }
    }

    /// Generate a streaming response with cancellation support
    ///
    /// Like `generate_stream()`, but also checks the cancellation token at each
    /// chunk. Returns `LlmError::Cancelled` if the token is cancelled.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use gemicro_core::{LlmClient, LlmRequest, LlmConfig};
    /// # use futures_util::stream::StreamExt;
    /// # use tokio_util::sync::CancellationToken;
    /// # async fn example() -> Result<(), gemicro_core::LlmError> {
    /// # let genai_client = rust_genai::Client::builder("key".to_string()).build();
    /// let client = LlmClient::new(genai_client, LlmConfig::default());
    /// let token = CancellationToken::new();
    /// let request = LlmRequest::new("Count to 10");
    ///
    /// let stream = client.generate_stream_with_cancellation(request, token.clone());
    /// futures_util::pin_mut!(stream);
    /// // In another task: token.cancel() to abort
    /// while let Some(chunk) = stream.next().await {
    ///     let chunk = chunk?;
    ///     print!("{}", chunk.text);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn generate_stream_with_cancellation(
        &self,
        request: LlmRequest,
        cancellation_token: CancellationToken,
    ) -> impl Stream<Item = Result<LlmStreamChunk, LlmError>> + Send + '_ {
        let timeout_duration = self.config.timeout;

        async_stream::try_stream! {
            // Check cancellation before starting
            if cancellation_token.is_cancelled() {
                Err(LlmError::Cancelled)?;
            }

            self.validate_request(&request)?;

            let interaction = self.build_interaction(&request);

            // Capture timing and chunks for recording
            let started_at = SystemTime::now();
            let start_instant = Instant::now();
            let mut recorded_chunks: Vec<SerializableStreamChunk> = Vec::new();
            let is_recording = self.is_recording();
            let serializable_request = SerializableLlmRequest::from(&request);
            let phase = self.current_phase();

            // Get the stream (not async - returns immediately)
            let stream = interaction.create_stream();
            futures_util::pin_mut!(stream);

            // Process chunks with per-chunk timeout and cancellation
            loop {
                // Check cancellation at each iteration
                if cancellation_token.is_cancelled() {
                    Err(LlmError::Cancelled)?;
                }

                // Race chunk read against timeout and cancellation
                let chunk = tokio::select! {
                    result = tokio::time::timeout(timeout_duration, stream.next()) => {
                        result.map_err(|_| LlmError::Timeout(timeout_duration.as_millis() as u64))
                    }
                    _ = cancellation_token.cancelled() => {
                        Err(LlmError::Cancelled)
                    }
                }?;

                match chunk {
                    Some(Ok(stream_chunk)) => {
                        use rust_genai::StreamChunk;
                        match stream_chunk {
                            StreamChunk::Delta(delta) => {
                                if let Some(text) = delta.text() {
                                    let text_str = text.to_string();

                                    // Record chunk if recording is enabled
                                    if is_recording {
                                        recorded_chunks.push(SerializableStreamChunk {
                                            text: text_str.clone(),
                                            offset_ms: start_instant.elapsed().as_millis() as u64,
                                        });
                                    }

                                    yield LlmStreamChunk {
                                        text: text_str,
                                    };
                                }
                                // Skip deltas with no text (e.g., thought deltas)
                            }
                            StreamChunk::Complete(_response) => {
                                // Final response - stream will end after this
                            }
                            _ => {
                                // Handle future StreamChunk variants gracefully
                                log::debug!("Unknown StreamChunk variant received");
                            }
                        }
                    }
                    Some(Err(e)) => {
                        Err(LlmError::from(e))?;
                    }
                    None => break,
                }
            }

            // Record the step at the end of streaming
            if is_recording {
                let duration_ms = start_instant.elapsed().as_millis() as u64;
                self.record_step(TrajectoryStep {
                    phase,
                    request: serializable_request,
                    response: LlmResponseData::Streaming(recorded_chunks),
                    duration_ms,
                    started_at,
                });
            }
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
        let generation_config = GenerationConfig {
            temperature: Some(self.config.temperature),
            max_output_tokens: Some(self.config.max_tokens as i32),
            ..Default::default()
        };

        let mut interaction = self
            .client
            .interaction()
            .with_model(MODEL)
            .with_text(&request.prompt)
            .with_generation_config(generation_config);

        if let Some(ref system) = request.system_instruction {
            interaction = interaction.with_system_instruction(system);
        }

        if request.use_google_search {
            interaction = interaction.with_tools(vec![rust_genai::Tool::GoogleSearch]);
        }

        if let Some(ref schema) = request.response_format {
            interaction = interaction.with_response_format(schema.clone());
        }

        interaction
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
        assert!(!req.use_google_search);
        assert!(req.response_format.is_none());
    }

    #[test]
    fn test_llm_request_with_system() {
        let req = LlmRequest::with_system("User prompt", "System instruction");
        assert_eq!(req.prompt, "User prompt");
        assert_eq!(
            req.system_instruction,
            Some("System instruction".to_string())
        );
        assert!(!req.use_google_search);
        assert!(req.response_format.is_none());
    }

    #[test]
    fn test_llm_request_with_google_search() {
        let req = LlmRequest::new("Test prompt").with_google_search();
        assert_eq!(req.prompt, "Test prompt");
        assert!(req.use_google_search);
        assert!(req.response_format.is_none());
    }

    #[test]
    fn test_llm_request_with_system_and_google_search() {
        let req = LlmRequest::with_system("User prompt", "System instruction").with_google_search();
        assert_eq!(req.prompt, "User prompt");
        assert_eq!(
            req.system_instruction,
            Some("System instruction".to_string())
        );
        assert!(req.use_google_search);
        assert!(req.response_format.is_none());
    }

    #[test]
    fn test_llm_request_with_response_format() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "correct": {"type": "boolean"},
                "confidence": {"type": "number"}
            },
            "required": ["correct", "confidence"]
        });

        let req = LlmRequest::new("Test prompt").with_response_format(schema.clone());
        assert_eq!(req.prompt, "Test prompt");
        assert!(req.system_instruction.is_none());
        assert!(!req.use_google_search);
        assert_eq!(req.response_format, Some(schema));
    }

    #[test]
    fn test_llm_request_with_all_options() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"result": {"type": "string"}}
        });

        let req = LlmRequest::with_system("User prompt", "System instruction")
            .with_google_search()
            .with_response_format(schema.clone());

        assert_eq!(req.prompt, "User prompt");
        assert_eq!(
            req.system_instruction,
            Some("System instruction".to_string())
        );
        assert!(req.use_google_search);
        assert_eq!(req.response_format, Some(schema));
    }

    #[test]
    fn test_llm_stream_chunk_creation() {
        let chunk = LlmStreamChunk {
            text: "Hello".to_string(),
        };
        assert_eq!(chunk.text, "Hello");

        let empty_chunk = LlmStreamChunk {
            text: String::new(),
        };
        assert!(empty_chunk.text.is_empty());
    }

    #[test]
    fn test_is_retryable_timeout() {
        assert!(LlmClient::is_retryable(&LlmError::Timeout(5000)));
    }

    #[test]
    fn test_is_retryable_rate_limit() {
        assert!(LlmClient::is_retryable(&LlmError::RateLimit(
            "Too many requests".to_string()
        )));
    }

    #[test]
    fn test_is_not_retryable_invalid_request() {
        assert!(!LlmClient::is_retryable(&LlmError::InvalidRequest(
            "Bad prompt".to_string()
        )));
    }

    #[test]
    fn test_is_not_retryable_no_content() {
        assert!(!LlmClient::is_retryable(&LlmError::NoContent));
    }

    #[test]
    fn test_is_not_retryable_other() {
        assert!(!LlmClient::is_retryable(&LlmError::Other(
            "Unknown error".to_string()
        )));
    }

    #[test]
    fn test_is_not_retryable_cancelled() {
        assert!(!LlmClient::is_retryable(&LlmError::Cancelled));
    }

    #[test]
    fn test_validate_request_empty_prompt() {
        let genai_client = rust_genai::Client::builder("test-key".to_string()).build();
        let client = LlmClient::new(genai_client, LlmConfig::default());
        let request = LlmRequest::new("");

        let result = client.validate_request(&request);
        assert!(result.is_err());
        assert!(matches!(result, Err(LlmError::InvalidRequest(_))));
    }

    #[test]
    fn test_validate_request_valid_prompt() {
        let genai_client = rust_genai::Client::builder("test-key".to_string()).build();
        let client = LlmClient::new(genai_client, LlmConfig::default());
        let request = LlmRequest::new("Valid prompt");

        let result = client.validate_request(&request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generation_config_applied() {
        use std::time::Duration;

        let config = LlmConfig {
            temperature: 0.5,
            max_tokens: 1024,
            timeout: Duration::from_secs(10),
            max_retries: 3,
            retry_base_delay_ms: 500,
        };

        let genai_client = rust_genai::Client::builder("test-key".to_string()).build();
        let client = LlmClient::new(genai_client, config.clone());

        // Verify client stored config correctly
        assert_eq!(client.config.temperature, 0.5);
        assert_eq!(client.config.max_tokens, 1024);
        assert_eq!(client.config.max_retries, 3);
    }

    #[test]
    fn test_llm_client_debug_redacts_api_key() {
        // Create client with a secret API key
        let genai_client = rust_genai::Client::builder("secret-api-key-12345".to_string()).build();
        let client = LlmClient::new(genai_client, LlmConfig::default());

        // Get debug output
        let debug_output = format!("{:?}", client);

        // Should contain model name for debugging
        assert!(
            debug_output.contains("gemini"),
            "Debug output should contain model name"
        );

        // Should contain REDACTED marker
        assert!(
            debug_output.contains("[REDACTED]"),
            "Debug output should contain [REDACTED]"
        );

        // Should NOT leak the API key
        assert!(
            !debug_output.contains("secret-api-key"),
            "Debug output must not contain API key"
        );
        assert!(
            !debug_output.contains("12345"),
            "Debug output must not contain API key suffix"
        );
    }

    // Recording functionality tests

    #[test]
    fn test_llm_client_not_recording_by_default() {
        let genai_client = rust_genai::Client::builder("test-key".to_string()).build();
        let client = LlmClient::new(genai_client, LlmConfig::default());

        assert!(!client.is_recording());
        assert_eq!(client.step_count(), 0);
        assert!(client.take_steps().is_empty());
    }

    #[test]
    fn test_llm_client_with_recording() {
        let genai_client = rust_genai::Client::builder("test-key".to_string()).build();
        let client = LlmClient::with_recording(genai_client, LlmConfig::default());

        assert!(client.is_recording());
        assert_eq!(client.step_count(), 0);
    }

    #[test]
    fn test_set_and_get_phase() {
        let genai_client = rust_genai::Client::builder("test-key".to_string()).build();
        let client = LlmClient::new(genai_client, LlmConfig::default());

        // Default phase
        assert_eq!(client.current_phase(), "default");

        // Set a new phase
        client.set_phase("decomposition");
        assert_eq!(client.current_phase(), "decomposition");

        // Set another phase
        client.set_phase("synthesis");
        assert_eq!(client.current_phase(), "synthesis");
    }

    #[test]
    fn test_take_steps_clears_recorder() {
        let genai_client = rust_genai::Client::builder("test-key".to_string()).build();
        let client = LlmClient::with_recording(genai_client, LlmConfig::default());

        // Manually record a step for testing
        let step = TrajectoryStep {
            phase: "test".to_string(),
            request: SerializableLlmRequest {
                prompt: "Test prompt".to_string(),
                system_instruction: None,
                use_google_search: false,
                response_format: None,
            },
            response: LlmResponseData::Buffered(serde_json::json!({"text": "Test"})),
            duration_ms: 100,
            started_at: SystemTime::now(),
        };
        client.record_step(step);

        assert_eq!(client.step_count(), 1);

        // Take steps should return 1 and clear
        let steps = client.take_steps();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].phase, "test");

        // Now empty
        assert_eq!(client.step_count(), 0);
        assert!(client.take_steps().is_empty());
    }

    #[test]
    fn test_debug_shows_recording_status() {
        let genai_client = rust_genai::Client::builder("test-key".to_string()).build();
        let recording_client = LlmClient::with_recording(genai_client, LlmConfig::default());
        let debug_output = format!("{:?}", recording_client);

        assert!(
            debug_output.contains("is_recording: true"),
            "Debug output should show is_recording: true"
        );

        let genai_client2 = rust_genai::Client::builder("test-key".to_string()).build();
        let non_recording = LlmClient::new(genai_client2, LlmConfig::default());
        let debug_output2 = format!("{:?}", non_recording);

        assert!(
            debug_output2.contains("is_recording: false"),
            "Debug output should show is_recording: false"
        );
    }
}
