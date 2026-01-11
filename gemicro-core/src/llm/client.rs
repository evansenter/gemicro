//! LLM client implementation with trajectory recording support.

use super::request::{LlmRequest, LlmStreamChunk};
use crate::config::{LlmConfig, MODEL};
use crate::error::LlmError;
use crate::trajectory::{
    LlmResponseData, SerializableLlmRequest, SerializableStreamChunk, TrajectoryStep,
};
use futures_util::stream::{Stream, StreamExt};
use genai_rs::{function_result_content, FunctionCallInfo, GenerationConfig};
use std::future::Future;
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime};
use tokio_util::sync::CancellationToken;

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
            .field("model", &MODEL)
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
    /// use gemicro_core::{LlmClient, LlmRequest, LlmConfig};
    ///
    /// # async fn example() -> Result<(), gemicro_core::LlmError> {
    /// # let genai_client = genai_rs::Client::builder("key".to_string()).build().unwrap();
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
    /// # Warning: Escape Hatch
    ///
    /// This method bypasses `LlmClient`'s automatic timeout enforcement, retry
    /// logic, recording, and response logging. When using this directly, you are
    /// responsible for implementing these features yourself.
    ///
    /// **Prefer using `generate()` or `generate_stream()` with `LlmRequest` builders:**
    /// - Function calling: `LlmRequest::new(...).with_functions(declarations)`
    /// - Multi-turn continuations: `LlmRequest::continuation(prev_id, functions, results)`
    /// - Conversation history: `LlmRequest::new(...).with_turns(turns)`
    ///
    /// Use this only for operations that genuinely require custom interaction builders
    /// (e.g., advanced streaming patterns or features not yet in `LlmRequest`).
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
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use gemicro_core::{LlmClient, LlmRequest, LlmConfig};
    /// # async fn example() -> Result<(), gemicro_core::LlmError> {
    /// # let genai_client = genai_rs::Client::builder("key".to_string()).build().unwrap();
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
    ) -> Result<genai_rs::InteractionResponse, LlmError> {
        self.validate_request(&request)?;

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
    /// # use gemicro_core::{LlmClient, LlmRequest, LlmConfig};
    /// # use tokio_util::sync::CancellationToken;
    /// # async fn example() -> Result<(), gemicro_core::LlmError> {
    /// # let genai_client = genai_rs::Client::builder("key".to_string()).build().unwrap();
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
    ) -> Result<genai_rs::InteractionResponse, LlmError> {
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
    /// let result = client.generate_with_tools(
    ///     LlmRequest::with_system("What time is it?", "You have tools.")
    ///         .with_functions(function_declarations),
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
        request: LlmRequest,
        tool_executor: F,
        max_turns: usize,
        cancellation_token: &CancellationToken,
    ) -> Result<GenerateWithToolsResponse, LlmError>
    where
        F: Fn(&FunctionCallInfo) -> Fut,
        Fut: Future<Output = serde_json::Value>,
    {
        self.validate_request(&request)?;

        // Check cancellation before starting
        if cancellation_token.is_cancelled() {
            return Err(LlmError::Cancelled);
        }

        // Get function declarations from request (needed for continuations)
        let function_declarations = request.functions.clone().unwrap_or_default();

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
            let mut function_results = Vec::with_capacity(function_calls.len());
            for fc in &function_calls {
                // Check cancellation before each tool call
                if cancellation_token.is_cancelled() {
                    return Err(LlmError::Cancelled);
                }

                let result = tool_executor(fc).await;
                let call_id = fc.id.unwrap_or("unknown");

                function_results.push(function_result_content(
                    fc.name.to_string(),
                    call_id.to_string(),
                    result,
                ));
            }

            // Send results back to LLM
            let continuation = LlmRequest::continuation(
                &interaction_id,
                function_declarations.clone(),
                function_results,
            );

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
    ///
    /// Uses the build() + execute() pattern from genai-rs 0.6.0 for cleaner
    /// separation of request construction and execution.
    async fn generate_once(
        &self,
        request: &LlmRequest,
    ) -> Result<genai_rs::InteractionResponse, LlmError> {
        // Capture timing for recording
        let started_at = SystemTime::now();
        let start_instant = Instant::now();

        // Build InteractionRequest from LlmRequest
        // Uses different builders for initial vs continuation requests (genai_rs typestate)
        let interaction_request = if self.is_continuation(request) {
            self.build_continuation_interaction(request)?.build()
        } else {
            self.build_interaction(request).build()
        }
        .map_err(LlmError::from)?;

        // Execute with explicit timeout wrapping
        let response = tokio::time::timeout(
            self.config.timeout,
            self.client.execute(interaction_request),
        )
        .await
        .map_err(|_| LlmError::Timeout(self.config.timeout.as_millis() as u64))?
        .map_err(LlmError::from)?;

        // Response logging is handled by genai-rs upstream

        // Validate response has content (text or function calls)
        // Function calling responses may have function_calls but no text
        if response.text().is_none() && response.function_calls().is_empty() {
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
    /// # let genai_client = genai_rs::Client::builder("key".to_string()).build().unwrap();
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

            // Accumulate response text for debug logging
            let mut accumulated_response = String::new();

            // Get the stream (not async - returns immediately)
            let stream = interaction.create_stream();
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
            // Note: Styling differs from genai_rs request logs. For consistent styling,
            // response logging should be added to genai_rs upstream.
            if log::log_enabled!(log::Level::Debug) && !accumulated_response.is_empty() {
                log::debug!("Response Body (streamed):\n{}", accumulated_response);
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
    /// # let genai_client = genai_rs::Client::builder("key".to_string()).build().unwrap();
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

            // Accumulate response text for debug logging
            let mut accumulated_response = String::new();

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
            // Note: Styling differs from genai_rs request logs. For consistent styling,
            // response logging should be added to genai_rs upstream.
            if log::log_enabled!(log::Level::Debug) && !accumulated_response.is_empty() {
                log::debug!("Response Body (streamed):\n{}", accumulated_response);
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
        // Continuations (function result responses) don't require a prompt
        if request.prompt.is_empty() && !self.is_continuation(request) {
            return Err(LlmError::InvalidRequest(
                "Prompt cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    /// Build an interaction from the request (DRY helper for initial requests)
    fn build_interaction(&self, request: &LlmRequest) -> genai_rs::InteractionBuilder<'_> {
        let generation_config = GenerationConfig {
            temperature: Some(self.config.temperature),
            max_output_tokens: Some(self.config.max_tokens as i32),
            ..Default::default()
        };

        let mut interaction = self
            .client
            .interaction()
            .with_model(MODEL)
            .with_generation_config(generation_config);

        // Initial request: use turns or simple text prompt
        if let Some(ref turns) = request.turns {
            // Clone turns and append current prompt as final user turn
            let mut full_turns = turns.clone();
            full_turns.push(genai_rs::Turn::user(request.prompt.as_str()));
            interaction = interaction.with_turns(full_turns);
        } else if !request.prompt.is_empty() {
            interaction = interaction.with_text(&request.prompt);
        }

        if let Some(ref system) = request.system_instruction {
            interaction = interaction.with_system_instruction(system);
        }

        // Add function declarations if provided
        if let Some(ref functions) = request.functions {
            interaction = interaction.with_functions(functions.clone());
        }

        // Enable storage for interaction chaining
        if request.store_enabled {
            interaction = interaction.with_store_enabled();
        }

        if request.use_google_search {
            interaction = interaction.with_tools(vec![genai_rs::Tool::GoogleSearch]);
        }

        if let Some(ref schema) = request.response_format {
            interaction = interaction.with_response_format(schema.clone());
        }

        interaction
    }

    /// Build a continuation interaction from the request (for function calling chains)
    ///
    /// This uses genai_rs's Chained typestate which is distinct from initial FirstTurn requests.
    fn build_continuation_interaction(
        &self,
        request: &LlmRequest,
    ) -> Result<genai_rs::InteractionBuilder<'_, genai_rs::request_builder::Chained>, LlmError>
    {
        let generation_config = GenerationConfig {
            temperature: Some(self.config.temperature),
            max_output_tokens: Some(self.config.max_tokens as i32),
            ..Default::default()
        };

        // Start with previous interaction - this creates a Chained builder
        let prev_id = request.previous_interaction_id.as_ref().ok_or_else(|| {
            LlmError::InvalidRequest(
                "Continuation request missing previous_interaction_id".to_string(),
            )
        })?;

        let mut interaction = self
            .client
            .interaction()
            .with_model(MODEL)
            .with_generation_config(generation_config)
            .with_previous_interaction(prev_id);

        // Add function results
        if let Some(ref results) = request.function_results {
            interaction = interaction.with_content(results.clone());
        }

        // Add function declarations if provided
        if let Some(ref functions) = request.functions {
            interaction = interaction.with_functions(functions.clone());
        }

        // Enable storage for interaction chaining (usually already enabled for continuations)
        if request.store_enabled {
            interaction = interaction.with_store_enabled();
        }

        Ok(interaction)
    }

    /// Check if a request is a continuation (for validation bypass)
    fn is_continuation(&self, request: &LlmRequest) -> bool {
        request.previous_interaction_id.is_some() && request.function_results.is_some()
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
    fn test_is_retryable_rate_limit() {
        assert!(LlmError::RateLimit("Too many requests".to_string()).is_retryable());
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

    #[test]
    fn test_validate_request_empty_prompt() {
        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let client = LlmClient::new(genai_client, LlmConfig::default());
        let request = LlmRequest::new("");

        let result = client.validate_request(&request);
        assert!(result.is_err());
        assert!(matches!(result, Err(LlmError::InvalidRequest(_))));
    }

    #[test]
    fn test_validate_request_valid_prompt() {
        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let client = LlmClient::new(genai_client, LlmConfig::default());
        let request = LlmRequest::new("Valid prompt");

        let result = client.validate_request(&request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_request_continuation_empty_prompt_allowed() {
        use genai_rs::function_result_content;

        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let client = LlmClient::new(genai_client, LlmConfig::default());

        // A continuation request has empty prompt but previous_interaction_id + function_results
        let request = LlmRequest::continuation(
            "interaction_123",
            vec![], // Empty functions for this test
            vec![function_result_content(
                "get_time",
                "call_1",
                serde_json::json!({"time": "12:00"}),
            )],
        );

        let result = client.validate_request(&request);
        assert!(
            result.is_ok(),
            "Continuation with empty prompt should be valid"
        );
    }

    #[test]
    fn test_is_continuation() {
        use genai_rs::function_result_content;

        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let client = LlmClient::new(genai_client, LlmConfig::default());

        // Regular request is not a continuation
        let regular = LlmRequest::new("Hello");
        assert!(!client.is_continuation(&regular));

        // Request with only previous_interaction_id is not a continuation
        let mut partial = LlmRequest::new("");
        partial.previous_interaction_id = Some("interaction_123".to_string());
        assert!(!client.is_continuation(&partial));

        // Proper continuation request
        let continuation = LlmRequest::continuation(
            "interaction_123",
            vec![],
            vec![function_result_content(
                "get_time",
                "call_1",
                serde_json::json!({}),
            )],
        );
        assert!(client.is_continuation(&continuation));
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

        // Manually record a step for testing
        let step = TrajectoryStep {
            phase: "test".to_string(),
            request: SerializableLlmRequest {
                prompt: "Test prompt".to_string(),
                turns: None,
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
    async fn test_generate_with_tools_validates_empty_prompt() {
        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let client = LlmClient::new(genai_client, LlmConfig::default());
        let token = CancellationToken::new();

        // Request with empty prompt (not a continuation) should fail validation
        let request = LlmRequest::new("");

        let result = client
            .generate_with_tools(request, |_fc| async { serde_json::json!({}) }, 10, &token)
            .await;

        assert!(result.is_err());
        assert!(matches!(result, Err(LlmError::InvalidRequest(_))));
    }

    #[tokio::test]
    async fn test_generate_with_tools_respects_cancellation_before_start() {
        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let client = LlmClient::new(genai_client, LlmConfig::default());
        let token = CancellationToken::new();

        // Cancel before calling
        token.cancel();

        let request = LlmRequest::new("Test prompt");

        let result = client
            .generate_with_tools(request, |_fc| async { serde_json::json!({}) }, 10, &token)
            .await;

        assert!(result.is_err());
        assert!(
            matches!(result, Err(LlmError::Cancelled)),
            "Expected Cancelled error, got {:?}",
            result
        );
    }
}
