//! LLM client implementation with trajectory recording support.

use crate::config::LlmConfig;
use crate::error::LlmError;
use crate::trajectory::{LlmResponseData, SerializableStreamChunk, TrajectoryStep};
use futures_util::stream::{Stream, StreamExt};
use genai_rs::{function_result_content, FunctionCallInfo, InteractionRequest};
use std::future::Future;
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime};
use tokio_util::sync::CancellationToken;

/// Chunk from streaming LLM response
///
/// The stream ends naturally when no more chunks are available (yields `None`).
/// There is no artificial "final" marker - simply iterate until the stream ends.
#[derive(Debug, Clone)]
pub struct LlmStreamChunk {
    /// Text content of this chunk (may be empty for some chunks)
    pub text: String,
}

/// Result from [`LlmClient::generate_with_tools`]
#[derive(Debug)]
#[non_exhaustive]
pub struct GenerateWithToolsResponse {
    /// The final response from the model (after all tool calls completed)
    pub response: genai_rs::InteractionResponse,

    /// Total tokens used across all turns (initial + continuations)
    pub total_tokens: u64,

    /// Number of tool calling continuation rounds executed
    ///
    /// 0 means the model responded without calling any tools.
    /// Does not count the initial request.
    pub turns: usize,
}

/// LLM client wrapping genai-rs with timeout and configuration
///
/// Optionally supports trajectory recording for offline replay and evaluation.
/// Recording is disabled by default for zero overhead. Use [`LlmClient::with_recording`]
/// to enable it.
pub struct LlmClient {
    /// Underlying genai-rs client
    client: genai_rs::Client,

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
            .field("client", &"[REDACTED]")
            .field("config", &self.config)
            .field("is_recording", &self.is_recording())
            .finish()
    }
}

impl LlmClient {
    /// Create a new LLM client with the given genai-rs client and configuration
    ///
    /// Recording is disabled by default. Use [`with_recording`](Self::with_recording)
    /// to create a client that captures LLM interactions.
    pub fn new(client: genai_rs::Client, config: LlmConfig) -> Self {
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
    /// use gemicro_core::{LlmClient, LlmConfig};
    ///
    /// # async fn example() -> Result<(), gemicro_core::LlmError> {
    /// # let genai_client = genai_rs::Client::builder("key".to_string()).build().unwrap();
    /// let client = LlmClient::with_recording(genai_client, LlmConfig::default());
    ///
    /// client.set_phase("decomposition");
    /// let request = client.client().interaction()
    ///     .with_text("Break this down")
    ///     .build()?;
    /// let _response = client.generate(request).await?;
    ///
    /// client.set_phase("synthesis");
    /// let request = client.client().interaction()
    ///     .with_text("Combine these")
    ///     .build()?;
    /// let _response = client.generate(request).await?;
    ///
    /// let steps = client.take_steps();
    /// assert_eq!(steps.len(), 2);
    /// assert_eq!(steps[0].phase, "decomposition");
    /// assert_eq!(steps[1].phase, "synthesis");
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_recording(client: genai_rs::Client, config: LlmConfig) -> Self {
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

    /// Get a reference to the underlying genai_rs client.
    ///
    /// Use this to build `InteractionRequest`s via the genai-rs `InteractionBuilder`:
    ///
    /// ```no_run
    /// # use gemicro_core::{LlmClient, LlmConfig};
    /// # let genai_client = genai_rs::Client::builder("key".to_string()).build().unwrap();
    /// # let client = LlmClient::new(genai_client, LlmConfig::default());
    /// // Simple request
    /// let request = client.client().interaction()
    ///     .with_model("gemini-3-flash-preview")
    ///     .with_text("Hello")
    ///     .with_system_instruction("Be helpful")
    ///     .build()
    ///     .unwrap();
    /// ```
    ///
    /// For function calling and continuations:
    ///
    /// ```text
    /// // With function calling
    /// let request = client.client().interaction()
    ///     .with_model("gemini-3-flash-preview")
    ///     .with_text("What time is it?")
    ///     .with_functions(function_declarations)
    ///     .with_store_enabled()  // Required for chaining
    ///     .build()
    ///     .unwrap();
    ///
    /// // Continuation after function call
    /// let request = client.client().interaction()
    ///     .with_model("gemini-3-flash-preview")
    ///     .with_previous_interaction(&interaction_id)
    ///     .with_content(function_results)
    ///     .with_functions(function_declarations)
    ///     .with_store_enabled()
    ///     .build()
    ///     .unwrap();
    /// ```
    ///
    /// Then pass the built request to `generate()`, `generate_stream()`, etc.
    pub fn client(&self) -> &genai_rs::Client {
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
    /// Build requests using the genai-rs `InteractionBuilder` via `client()`:
    ///
    /// ```no_run
    /// # use gemicro_core::{LlmClient, LlmConfig};
    /// # async fn example() -> Result<(), gemicro_core::LlmError> {
    /// # let genai_client = genai_rs::Client::builder("key".to_string()).build().unwrap();
    /// let client = LlmClient::new(genai_client, LlmConfig::default());
    /// let request = client.client().interaction()
    ///     .with_text("Explain quantum computing")
    ///     .build()?;
    /// let response = client.generate(request).await?;
    /// println!("{}", response.text().unwrap_or(""));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Returns the full [`genai_rs::InteractionResponse`] with access to:
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
    pub async fn generate(
        &self,
        request: InteractionRequest,
    ) -> Result<genai_rs::InteractionResponse, LlmError> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            match self.generate_once(&request).await {
                Ok(response) => return Ok(response),
                Err(e) if e.is_retryable() && attempt < self.config.max_retries => {
                    log::warn!(
                        "LLM request failed (attempt {}/{}): {}, retrying...",
                        attempt + 1,
                        self.config.max_retries + 1,
                        e
                    );
                    // Use server-specified retry delay if available, otherwise exponential backoff
                    let delay = e
                        .retry_after()
                        .unwrap_or_else(|| self.config.retry_delay(attempt));
                    last_error = Some(e);
                    tokio::time::sleep(delay).await;
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
    /// # use gemicro_core::{LlmClient, LlmConfig};
    /// # use tokio_util::sync::CancellationToken;
    /// # async fn example() -> Result<(), gemicro_core::LlmError> {
    /// # let genai_client = genai_rs::Client::builder("key".to_string()).build().unwrap();
    /// let client = LlmClient::new(genai_client, LlmConfig::default());
    /// let token = CancellationToken::new();
    /// let request = client.client().interaction()
    ///     .with_text("Explain quantum computing")
    ///     .build()?;
    ///
    /// // In another task: token.cancel() to abort
    /// let response = client.generate_with_cancellation(request, &token).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn generate_with_cancellation(
        &self,
        request: InteractionRequest,
        cancellation_token: &CancellationToken,
    ) -> Result<genai_rs::InteractionResponse, LlmError> {
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
                Err(e) if e.is_retryable() && attempt < self.config.max_retries => {
                    log::warn!(
                        "LLM request failed (attempt {}/{}): {}, retrying...",
                        attempt + 1,
                        self.config.max_retries + 1,
                        e
                    );
                    // Use server-specified retry delay if available, otherwise exponential backoff
                    let delay = e
                        .retry_after()
                        .unwrap_or_else(|| self.config.retry_delay(attempt));
                    last_error = Some(e);

                    // Race retry delay against cancellation
                    tokio::select! {
                        _ = tokio::time::sleep(delay) => {}
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

    /// Execute a multi-turn function calling loop.
    ///
    /// This method handles the entire function calling loop internally:
    /// 1. Makes initial request with functions
    /// 2. If response contains function calls, executes them via the callback
    /// 3. Sends results back to the model
    /// 4. Repeats until model returns final text (no function calls) or max_turns reached
    ///
    /// # Arguments
    ///
    /// * `request` - Initial request (should include functions via `with_functions()`)
    /// * `function_declarations` - Function declarations for continuations
    /// * `tool_executor` - Async callback that executes function calls. Receives a reference
    ///   to each [`FunctionCallInfo`] and returns a [`serde_json::Value`] result. Errors should
    ///   be wrapped in the Value (e.g., `json!({"error": "message"})`).
    /// * `max_turns` - Maximum continuation rounds. Does not count the initial request.
    ///   Set to 10 for typical use cases.
    /// * `cancellation_token` - Token for cancelling the operation
    ///
    /// # Returns
    ///
    /// [`GenerateWithToolsResponse`] containing the final response, total tokens used,
    /// and number of tool-calling turns executed.
    ///
    /// # Errors
    ///
    /// Returns `LlmError::Other` if max_turns is exceeded, or propagates errors from
    /// the underlying LLM calls.
    ///
    /// # Example
    ///
    /// ```text
    /// let request = client.client().interaction()
    ///     .with_text("What time is it?")
    ///     .with_system_instruction("You have tools.")
    ///     .with_functions(function_declarations.clone())
    ///     .with_store_enabled()
    ///     .build()?;
    ///
    /// let result = client.generate_with_tools(
    ///     request,
    ///     function_declarations,
    ///     |fc| async move { json!({"time": "12:00 PM"}) },
    ///     10,  // max_turns
    ///     &cancellation_token,
    /// ).await?;
    ///
    /// println!("Answer: {}", result.response.text().unwrap_or(""));
    /// println!("Tokens: {}, Turns: {}", result.total_tokens, result.turns);
    /// ```
    ///
    /// For complete working examples, see `agents/gemicro-prompt-agent/src/lib.rs`.
    pub async fn generate_with_tools<F, Fut>(
        &self,
        request: InteractionRequest,
        function_declarations: Vec<genai_rs::FunctionDeclaration>,
        tool_executor: F,
        max_turns: usize,
        cancellation_token: &CancellationToken,
    ) -> Result<GenerateWithToolsResponse, LlmError>
    where
        F: Fn(&FunctionCallInfo) -> Fut,
        Fut: Future<Output = serde_json::Value>,
    {
        // Check cancellation before starting
        if cancellation_token.is_cancelled() {
            return Err(LlmError::Cancelled);
        }

        // Make initial request
        let mut response = self
            .generate_with_cancellation(request, cancellation_token)
            .await?;

        let mut total_tokens: u64 = crate::extract_total_tokens(&response)
            .map(|t| t as u64)
            .unwrap_or(0);
        let mut turns: usize = 0;

        // Function calling loop
        loop {
            // Check cancellation
            if cancellation_token.is_cancelled() {
                return Err(LlmError::Cancelled);
            }

            let function_calls = response.function_calls();

            // If no function calls, we're done
            if function_calls.is_empty() {
                break;
            }

            // Check max turns
            turns += 1;
            if turns > max_turns {
                return Err(LlmError::Other(format!(
                    "Max tool-calling turns ({}) exceeded",
                    max_turns
                )));
            }

            // Get interaction ID for chaining
            let interaction_id = response.id.clone().ok_or_else(|| {
                LlmError::Other("Missing interaction ID for function calling continuation".into())
            })?;

            // Execute each function call via the callback
            let mut results = Vec::with_capacity(function_calls.len());
            for fc in &function_calls {
                // Check cancellation before each tool call
                if cancellation_token.is_cancelled() {
                    return Err(LlmError::Cancelled);
                }

                let result = tool_executor(fc).await;
                let call_id = fc.id.unwrap_or("unknown");

                results.push(function_result_content(
                    fc.name.to_string(),
                    call_id.to_string(),
                    result,
                ));
            }

            // Build continuation request
            let continuation = self
                .client
                .interaction()
                .with_previous_interaction(&interaction_id)
                .with_content(results)
                .with_functions(function_declarations.clone())
                .with_store_enabled()
                .build()
                .map_err(LlmError::from)?;

            response = self
                .generate_with_cancellation(continuation, cancellation_token)
                .await?;

            // Track tokens
            if let Some(tokens) = crate::extract_total_tokens(&response) {
                total_tokens += tokens as u64;
            }
        }

        Ok(GenerateWithToolsResponse {
            response,
            total_tokens,
            turns,
        })
    }

    /// Execute a single generate request (no retries)
    async fn generate_once(
        &self,
        request: &InteractionRequest,
    ) -> Result<genai_rs::InteractionResponse, LlmError> {
        // Capture timing for recording
        let started_at = SystemTime::now();
        let start_instant = Instant::now();

        // Serialize request for recording before execution (in case of failure)
        let serialized_request = if self.is_recording() {
            serde_json::to_value(request).ok()
        } else {
            None
        };

        // Execute with explicit timeout wrapping
        let response =
            tokio::time::timeout(self.config.timeout, self.client.execute(request.clone()))
                .await
                .map_err(|_| LlmError::Timeout(self.config.timeout.as_millis() as u64))?
                .map_err(LlmError::from)?;

        // Validate response has content (text or function calls)
        // Function calling responses may have function_calls but no text
        if response.text().is_none() && response.function_calls().is_empty() {
            return Err(LlmError::NoContent);
        }

        // Record the step if recording is enabled
        if let Some(request_json) = serialized_request {
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
                request: request_json,
                response: response_data,
                duration_ms,
                started_at,
            });
        }

        Ok(response)
    }

    /// Generate a streaming response
    ///
    /// Returns a stream of text chunks as they arrive from the LLM.
    /// The stream ends naturally when the response is complete (yields `None`).
    ///
    /// Unlike `generate()`, streaming requires an `InteractionBuilder` (not a built
    /// `InteractionRequest`) because genai-rs creates streams from builders directly.
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
    /// # use gemicro_core::{LlmClient, LlmConfig};
    /// # use futures_util::stream::StreamExt;
    /// # async fn example() -> Result<(), gemicro_core::LlmError> {
    /// # let genai_client = genai_rs::Client::builder("key".to_string()).build().unwrap();
    /// let client = LlmClient::new(genai_client, LlmConfig::default());
    /// // Note: Don't call .build() - pass the builder directly for streaming
    /// let builder = client.client().interaction()
    ///     .with_text("Count to 10");
    ///
    /// let stream = client.generate_stream(builder);
    /// futures_util::pin_mut!(stream);
    /// while let Some(chunk) = stream.next().await {
    ///     let chunk = chunk?;
    ///     print!("{}", chunk.text);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn generate_stream<'a, S>(
        &'a self,
        builder: genai_rs::InteractionBuilder<'a, S>,
    ) -> impl Stream<Item = Result<LlmStreamChunk, LlmError>> + Send + 'a
    where
        S: Send + 'a,
    {
        let timeout_duration = self.config.timeout;

        async_stream::try_stream! {
            // Capture timing and chunks for recording
            let started_at = SystemTime::now();
            let start_instant = Instant::now();
            let mut recorded_chunks: Vec<SerializableStreamChunk> = Vec::new();
            let is_recording = self.is_recording();
            let phase = self.current_phase();

            // Accumulate response text for debug logging
            let mut accumulated_response = String::new();

            // Get the stream (not async - returns immediately)
            let stream = builder.create_stream();
            futures_util::pin_mut!(stream);

            // Process chunks with per-chunk timeout
            loop {
                let chunk = tokio::time::timeout(timeout_duration, stream.next())
                    .await
                    .map_err(|_| LlmError::Timeout(timeout_duration.as_millis() as u64))?;

                match chunk {
                    Some(Ok(stream_event)) => {
                        use genai_rs::StreamChunk;
                        match stream_event.chunk {
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

                                    // Accumulate for debug logging
                                    accumulated_response.push_str(&text_str);

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

            // Log accumulated response in debug mode
            if log::log_enabled!(log::Level::Debug) && !accumulated_response.is_empty() {
                log::debug!("Response Body (streamed):\n{}", accumulated_response);
            }

            // Record the step at the end of streaming
            // Note: For streaming, we store null for the request since we take a builder
            // (builders can't be serialized after they've created a stream)
            if is_recording {
                let duration_ms = start_instant.elapsed().as_millis() as u64;
                self.record_step(TrajectoryStep {
                    phase,
                    request: serde_json::Value::Null,
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
    /// # use gemicro_core::{LlmClient, LlmConfig};
    /// # use futures_util::stream::StreamExt;
    /// # use tokio_util::sync::CancellationToken;
    /// # async fn example() -> Result<(), gemicro_core::LlmError> {
    /// # let genai_client = genai_rs::Client::builder("key".to_string()).build().unwrap();
    /// let client = LlmClient::new(genai_client, LlmConfig::default());
    /// let token = CancellationToken::new();
    /// // Note: Don't call .build() - pass the builder directly for streaming
    /// let builder = client.client().interaction()
    ///     .with_text("Count to 10");
    ///
    /// let stream = client.generate_stream_with_cancellation(builder, token.clone());
    /// futures_util::pin_mut!(stream);
    /// // In another task: token.cancel() to abort
    /// while let Some(chunk) = stream.next().await {
    ///     let chunk = chunk?;
    ///     print!("{}", chunk.text);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn generate_stream_with_cancellation<'a, S>(
        &'a self,
        builder: genai_rs::InteractionBuilder<'a, S>,
        cancellation_token: CancellationToken,
    ) -> impl Stream<Item = Result<LlmStreamChunk, LlmError>> + Send + 'a
    where
        S: Send + 'a,
    {
        let timeout_duration = self.config.timeout;

        async_stream::try_stream! {
            // Check cancellation before starting
            if cancellation_token.is_cancelled() {
                Err(LlmError::Cancelled)?;
            }

            // Capture timing and chunks for recording
            let started_at = SystemTime::now();
            let start_instant = Instant::now();
            let mut recorded_chunks: Vec<SerializableStreamChunk> = Vec::new();
            let is_recording = self.is_recording();
            let phase = self.current_phase();

            // Accumulate response text for debug logging
            let mut accumulated_response = String::new();

            // Get the stream (not async - returns immediately)
            let stream = builder.create_stream();
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
                    Some(Ok(stream_event)) => {
                        use genai_rs::StreamChunk;
                        match stream_event.chunk {
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

                                    // Accumulate for debug logging
                                    accumulated_response.push_str(&text_str);

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

            // Log accumulated response in debug mode
            if log::log_enabled!(log::Level::Debug) && !accumulated_response.is_empty() {
                log::debug!("Response Body (streamed):\n{}", accumulated_response);
            }

            // Record the step at the end of streaming
            // Note: For streaming, we store null for the request since we take a builder
            if is_recording {
                let duration_ms = start_instant.elapsed().as_millis() as u64;
                self.record_step(TrajectoryStep {
                    phase,
                    request: serde_json::Value::Null,
                    response: LlmResponseData::Streaming(recorded_chunks),
                    duration_ms,
                    started_at,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_retryable_timeout() {
        assert!(LlmError::Timeout(5000).is_retryable());
    }

    #[test]
    fn test_is_not_retryable_invalid_request() {
        assert!(!LlmError::InvalidRequest("Bad prompt".to_string()).is_retryable());
    }

    #[test]
    fn test_is_not_retryable_no_content() {
        assert!(!LlmError::NoContent.is_retryable());
    }

    #[test]
    fn test_is_not_retryable_other() {
        assert!(!LlmError::Other("Unknown error".to_string()).is_retryable());
    }

    #[test]
    fn test_is_not_retryable_cancelled() {
        assert!(!LlmError::Cancelled.is_retryable());
    }

    // Tests for GenAi variant delegation to genai-rs is_retryable()
    #[test]
    fn test_is_retryable_genai_rate_limit() {
        let genai_err = genai_rs::GenaiError::Api {
            status_code: 429,
            message: "Rate limited".to_string(),
            request_id: None,
            retry_after: Some(std::time::Duration::from_secs(60)),
        };
        assert!(LlmError::GenAi(genai_err).is_retryable());
    }

    #[test]
    fn test_is_retryable_genai_server_error() {
        let genai_err = genai_rs::GenaiError::Api {
            status_code: 500,
            message: "Internal server error".to_string(),
            request_id: None,
            retry_after: None,
        };
        assert!(LlmError::GenAi(genai_err).is_retryable());
    }

    #[test]
    fn test_is_retryable_genai_timeout() {
        let genai_err = genai_rs::GenaiError::Timeout(std::time::Duration::from_secs(30));
        assert!(LlmError::GenAi(genai_err).is_retryable());
    }

    #[test]
    fn test_is_not_retryable_genai_bad_request() {
        let genai_err = genai_rs::GenaiError::Api {
            status_code: 400,
            message: "Invalid model".to_string(),
            request_id: None,
            retry_after: None,
        };
        assert!(!LlmError::GenAi(genai_err).is_retryable());
    }

    #[test]
    fn test_is_not_retryable_genai_parse_error() {
        let genai_err = genai_rs::GenaiError::Parse("Invalid SSE".to_string());
        assert!(!LlmError::GenAi(genai_err).is_retryable());
    }

    // Tests for retry_after() delegation
    #[test]
    fn test_retry_after_genai_with_duration() {
        let genai_err = genai_rs::GenaiError::Api {
            status_code: 429,
            message: "Rate limited".to_string(),
            request_id: None,
            retry_after: Some(std::time::Duration::from_secs(60)),
        };
        assert_eq!(
            LlmError::GenAi(genai_err).retry_after(),
            Some(std::time::Duration::from_secs(60))
        );
    }

    #[test]
    fn test_retry_after_genai_without_duration() {
        let genai_err = genai_rs::GenaiError::Api {
            status_code: 500,
            message: "Server error".to_string(),
            request_id: None,
            retry_after: None,
        };
        assert_eq!(LlmError::GenAi(genai_err).retry_after(), None);
    }

    #[test]
    fn test_retry_after_non_genai_returns_none() {
        assert_eq!(LlmError::Timeout(5000).retry_after(), None);
        assert_eq!(LlmError::NoContent.retry_after(), None);
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

        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let client = LlmClient::new(genai_client, config.clone());

        // Verify client stored config correctly
        assert_eq!(client.config.temperature, 0.5);
        assert_eq!(client.config.max_tokens, 1024);
        assert_eq!(client.config.max_retries, 3);
    }

    #[test]
    fn test_llm_client_debug_redacts_api_key() {
        // Create client with a secret API key
        let genai_client = genai_rs::Client::builder("secret-api-key-12345".to_string())
            .build()
            .unwrap();
        let client = LlmClient::new(genai_client, LlmConfig::default());

        // Get debug output
        let debug_output = format!("{:?}", client);

        // Should contain REDACTED marker for client
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
        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let client = LlmClient::new(genai_client, LlmConfig::default());

        assert!(!client.is_recording());
        assert_eq!(client.step_count(), 0);
        assert!(client.take_steps().is_empty());
    }

    #[test]
    fn test_llm_client_with_recording() {
        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let client = LlmClient::with_recording(genai_client, LlmConfig::default());

        assert!(client.is_recording());
        assert_eq!(client.step_count(), 0);
    }

    #[test]
    fn test_set_and_get_phase() {
        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
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
        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let client = LlmClient::with_recording(genai_client, LlmConfig::default());

        // Manually record a step for testing (request is now serde_json::Value)
        let step = TrajectoryStep {
            phase: "test".to_string(),
            request: serde_json::json!({"prompt": "Test prompt"}),
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
        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let recording_client = LlmClient::with_recording(genai_client, LlmConfig::default());
        let debug_output = format!("{:?}", recording_client);

        assert!(
            debug_output.contains("is_recording: true"),
            "Debug output should show is_recording: true"
        );

        let genai_client2 = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let non_recording = LlmClient::new(genai_client2, LlmConfig::default());
        let debug_output2 = format!("{:?}", non_recording);

        assert!(
            debug_output2.contains("is_recording: false"),
            "Debug output should show is_recording: false"
        );
    }

    // generate_with_tools tests

    #[tokio::test]
    async fn test_generate_with_tools_respects_cancellation_before_start() {
        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let client = LlmClient::new(genai_client, LlmConfig::default());
        let token = CancellationToken::new();

        // Cancel before calling
        token.cancel();

        // Build a valid request using genai-rs InteractionBuilder
        let request = client
            .client()
            .interaction()
            .with_model("gemini-3-flash-preview")
            .with_text("Test prompt")
            .build()
            .unwrap();

        let result = client
            .generate_with_tools(
                request,
                vec![],
                |_fc| async { serde_json::json!({}) },
                10,
                &token,
            )
            .await;

        assert!(result.is_err());
        assert!(
            matches!(result, Err(LlmError::Cancelled)),
            "Expected Cancelled error, got {:?}",
            result
        );
    }
}
