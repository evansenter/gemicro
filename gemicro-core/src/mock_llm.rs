//! Mock LLM client for replaying recorded trajectories
//!
//! This module provides a mock client that replays recorded LLM interactions
//! from a trajectory, enabling:
//!
//! - **Offline testing**: Run agents without API calls
//! - **Deterministic testing**: Replay exact sequences for reproducible tests
//! - **Cost-free evaluation**: Score agents on historical runs
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::mock_llm::MockLlmClient;
//! use gemicro_core::Trajectory;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Load a previously recorded trajectory
//! let trajectory = Trajectory::load("runs/example.json")?;
//!
//! // Create a mock client from the trajectory
//! let mock = MockLlmClient::from_trajectory(&trajectory);
//!
//! // Use like a regular LlmClient - returns recorded responses
//! let request = gemicro_core::LlmRequest::new("What is the capital of France?");
//! let response = mock.generate(request).await?;
//! # Ok(())
//! # }
//! ```

use crate::error::LlmError;
use crate::llm::{LlmRequest, LlmStreamChunk};
use crate::trajectory::{LlmResponseData, Trajectory, TrajectoryStep};
use futures_util::stream::Stream;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// Mock LLM client that replays recorded trajectory steps
///
/// This client is useful for:
/// - Unit testing agents without API calls
/// - Running evaluations on historical data
/// - Reproducing exact execution sequences for debugging
///
/// The mock advances through recorded steps sequentially. Each call to
/// `generate()` or `generate_stream()` returns the next recorded response.
#[derive(Debug)]
pub struct MockLlmClient {
    /// Recorded steps to replay
    steps: Vec<TrajectoryStep>,

    /// Current step index
    current_index: AtomicUsize,

    /// Whether to simulate realistic timing by adding delays
    simulate_timing: bool,
}

impl MockLlmClient {
    /// Create a mock client from trajectory steps
    ///
    /// The mock will replay these steps in order, returning each step's
    /// response when `generate()` is called, or streaming chunks when
    /// `generate_stream()` is called.
    pub fn from_steps(steps: Vec<TrajectoryStep>) -> Self {
        Self {
            steps,
            current_index: AtomicUsize::new(0),
            simulate_timing: false,
        }
    }

    /// Create a mock client from a trajectory
    ///
    /// Extracts the steps from the trajectory and creates a mock that
    /// will replay them in order.
    pub fn from_trajectory(trajectory: &Trajectory) -> Self {
        Self::from_steps(trajectory.steps.clone())
    }

    /// Enable timing simulation
    ///
    /// When enabled, the mock will add delays based on recorded `duration_ms`
    /// to simulate realistic timing. Useful for testing timeout behavior.
    ///
    /// Delays are scaled down by a factor of 100 to avoid long test times
    /// (e.g., a 1000ms recorded duration becomes a 10ms delay).
    pub fn with_timing_simulation(mut self) -> Self {
        self.simulate_timing = true;
        self
    }

    /// Get the current step index
    pub fn current_index(&self) -> usize {
        self.current_index.load(Ordering::SeqCst)
    }

    /// Get the total number of steps
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Check if all steps have been consumed
    pub fn is_exhausted(&self) -> bool {
        self.current_index() >= self.steps.len()
    }

    /// Reset to the beginning
    pub fn reset(&self) {
        self.current_index.store(0, Ordering::SeqCst);
    }

    /// Get the next step and advance the index
    fn next_step(&self) -> Option<TrajectoryStep> {
        let index = self.current_index.fetch_add(1, Ordering::SeqCst);
        self.steps.get(index).cloned()
    }

    /// Generate a response by returning the next recorded step
    ///
    /// This simulates `LlmClient::generate()` by returning the recorded
    /// response from the next step. The request is validated but not used
    /// to select the response - steps are returned in order.
    ///
    /// # Errors
    ///
    /// - `LlmError::NoContent` if no more steps are available or step was streaming
    /// - `LlmError::InvalidRequest` if the request prompt is empty
    pub async fn generate(&self, request: LlmRequest) -> Result<serde_json::Value, LlmError> {
        // Validate request (same as real client)
        if request.prompt.is_empty() {
            return Err(LlmError::InvalidRequest(
                "Prompt cannot be empty".to_string(),
            ));
        }

        let step = self.next_step().ok_or_else(|| LlmError::NoContent)?;

        // Optionally simulate timing
        if self.simulate_timing && step.duration_ms > 0 {
            let delay = Duration::from_millis(step.duration_ms / 100);
            tokio::time::sleep(delay).await;
        }

        // Return the recorded response (only for buffered mode)
        match step.response {
            LlmResponseData::Buffered(response) => Ok(response),
            LlmResponseData::Streaming(_) => Err(LlmError::NoContent),
        }
    }

    /// Generate a streaming response by replaying recorded chunks
    ///
    /// This simulates `LlmClient::generate_stream()` by yielding the
    /// recorded stream chunks from the next step.
    ///
    /// If timing simulation is enabled, chunks are yielded with delays
    /// proportional to their recorded offsets.
    pub fn generate_stream(
        &self,
        request: LlmRequest,
    ) -> impl Stream<Item = Result<LlmStreamChunk, LlmError>> + Send + '_ {
        async_stream::try_stream! {
            // Validate request
            if request.prompt.is_empty() {
                Err(LlmError::InvalidRequest("Prompt cannot be empty".to_string()))?;
            }

            let step = self
                .next_step()
                .ok_or_else(|| LlmError::NoContent)?;

            match step.response {
                LlmResponseData::Streaming(chunks) => {
                    let mut last_offset = 0u64;

                    for chunk in chunks {
                        // Optionally simulate timing between chunks
                        if self.simulate_timing && chunk.offset_ms > last_offset {
                            let delay = Duration::from_millis((chunk.offset_ms - last_offset) / 100);
                            tokio::time::sleep(delay).await;
                            last_offset = chunk.offset_ms;
                        }

                        yield LlmStreamChunk { text: chunk.text };
                    }
                }
                LlmResponseData::Buffered(response) => {
                    // Fallback: if buffered response, yield the full response as one chunk
                    if let Some(text) = response.get("text").and_then(|t| t.as_str()) {
                        yield LlmStreamChunk { text: text.to_string() };
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::SerializableLlmRequest;
    use crate::trajectory::SerializableStreamChunk;
    use futures_util::StreamExt;
    use serde_json::json;
    use std::time::SystemTime;

    fn sample_buffered_step() -> TrajectoryStep {
        TrajectoryStep {
            phase: "test".to_string(),
            request: SerializableLlmRequest {
                prompt: "What is 2+2?".to_string(),
                system_instruction: None,
                use_google_search: false,
                response_format: None,
            },
            response: LlmResponseData::Buffered(json!({
                "text": "The answer is 4.",
                "usage": {"total_tokens": 10}
            })),
            duration_ms: 100,
            started_at: SystemTime::now(),
        }
    }

    fn sample_streaming_step() -> TrajectoryStep {
        TrajectoryStep {
            phase: "streaming".to_string(),
            request: SerializableLlmRequest {
                prompt: "Count to 3".to_string(),
                system_instruction: None,
                use_google_search: false,
                response_format: None,
            },
            response: LlmResponseData::Streaming(vec![
                SerializableStreamChunk {
                    text: "One".to_string(),
                    offset_ms: 10,
                },
                SerializableStreamChunk {
                    text: " Two".to_string(),
                    offset_ms: 50,
                },
                SerializableStreamChunk {
                    text: " Three".to_string(),
                    offset_ms: 100,
                },
            ]),
            duration_ms: 100,
            started_at: SystemTime::now(),
        }
    }

    #[test]
    fn test_mock_client_creation() {
        let mock = MockLlmClient::from_steps(vec![sample_buffered_step()]);

        assert_eq!(mock.step_count(), 1);
        assert_eq!(mock.current_index(), 0);
        assert!(!mock.is_exhausted());
    }

    #[test]
    fn test_mock_client_from_trajectory() {
        let trajectory = Trajectory::builder()
            .query("Test")
            .agent_name("test_agent")
            .build(
                vec![sample_buffered_step(), sample_streaming_step()],
                vec![],
                200,
                None,
            );

        let mock = MockLlmClient::from_trajectory(&trajectory);

        assert_eq!(mock.step_count(), 2);
    }

    #[tokio::test]
    async fn test_generate_returns_recorded_response() {
        let mock = MockLlmClient::from_steps(vec![sample_buffered_step()]);

        let response = mock.generate(LlmRequest::new("Any prompt")).await.unwrap();

        assert_eq!(response["text"], "The answer is 4.");
        assert!(mock.is_exhausted());
    }

    #[tokio::test]
    async fn test_generate_advances_through_steps() {
        let step1 = TrajectoryStep {
            response: LlmResponseData::Buffered(json!({"text": "First response"})),
            ..sample_buffered_step()
        };
        let step2 = TrajectoryStep {
            response: LlmResponseData::Buffered(json!({"text": "Second response"})),
            ..sample_buffered_step()
        };

        let mock = MockLlmClient::from_steps(vec![step1, step2]);

        let r1 = mock.generate(LlmRequest::new("First")).await.unwrap();
        assert_eq!(r1["text"], "First response");
        assert_eq!(mock.current_index(), 1);

        let r2 = mock.generate(LlmRequest::new("Second")).await.unwrap();
        assert_eq!(r2["text"], "Second response");
        assert!(mock.is_exhausted());
    }

    #[tokio::test]
    async fn test_generate_returns_error_when_exhausted() {
        let mock = MockLlmClient::from_steps(vec![]);

        let result = mock.generate(LlmRequest::new("Test")).await;
        assert!(matches!(result, Err(LlmError::NoContent)));
    }

    #[tokio::test]
    async fn test_generate_validates_empty_prompt() {
        let mock = MockLlmClient::from_steps(vec![sample_buffered_step()]);

        let result = mock.generate(LlmRequest::new("")).await;
        assert!(matches!(result, Err(LlmError::InvalidRequest(_))));
    }

    #[tokio::test]
    async fn test_generate_stream_yields_chunks() {
        let mock = MockLlmClient::from_steps(vec![sample_streaming_step()]);

        let stream = mock.generate_stream(LlmRequest::new("Count"));
        futures_util::pin_mut!(stream);

        let mut collected = Vec::new();
        while let Some(chunk) = stream.next().await {
            collected.push(chunk.unwrap().text);
        }

        assert_eq!(collected, vec!["One", " Two", " Three"]);
        assert!(mock.is_exhausted());
    }

    #[tokio::test]
    async fn test_generate_stream_fallback_to_response_text() {
        // Step with response but no stream chunks
        let mock = MockLlmClient::from_steps(vec![sample_buffered_step()]);

        let stream = mock.generate_stream(LlmRequest::new("Test"));
        futures_util::pin_mut!(stream);

        let mut collected = Vec::new();
        while let Some(chunk) = stream.next().await {
            collected.push(chunk.unwrap().text);
        }

        assert_eq!(collected, vec!["The answer is 4."]);
    }

    #[test]
    fn test_reset() {
        let mock = MockLlmClient::from_steps(vec![sample_buffered_step()]);

        // Advance
        let _ = mock.next_step();
        assert_eq!(mock.current_index(), 1);

        // Reset
        mock.reset();
        assert_eq!(mock.current_index(), 0);
    }

    #[test]
    fn test_with_timing_simulation() {
        let mock = MockLlmClient::from_steps(vec![]).with_timing_simulation();

        assert!(mock.simulate_timing);
    }
}
