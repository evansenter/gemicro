//! Trajectory serialization for offline replay and evaluation
//!
//! This module provides types and utilities for capturing full LLM interaction
//! traces during agent execution, enabling:
//!
//! - **Offline replay**: Re-run agent logic without API calls
//! - **Evaluation datasets**: Build test sets from real production runs
//! - **Debugging**: Inspect exact LLM requests and responses
//!
//! # Architecture
//!
//! A `Trajectory` captures an entire agent execution:
//! - `steps`: Raw LLM request/response pairs with timing
//! - `events`: High-level `AgentUpdate` events for compatibility
//! - `metadata`: Execution summary (tokens, duration, etc.)
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::trajectory::Trajectory;
//!
//! // Save a trajectory
//! # fn example(trajectory: Trajectory) -> std::io::Result<()> {
//! trajectory.save("trajectories/run_001.json")?;
//!
//! // Load and inspect
//! let loaded = Trajectory::load("trajectories/run_001.json")?;
//! println!("Query: {}", loaded.query);
//! println!("Steps: {}", loaded.steps.len());
//! for step in &loaded.steps {
//!     println!("  Phase: {}, Duration: {}ms", step.phase, step.duration_ms);
//! }
//! # Ok(())
//! # }
//! ```

use crate::llm::LlmRequest;
use crate::update::AgentUpdate;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;
use std::time::SystemTime;

/// Current schema version for trajectory files
///
/// Version history:
/// - 1.0.0: Initial release with dual Option pattern for response/stream_chunks
/// - 2.0.0: Changed to LlmResponseData enum for type-safe response mode
pub const SCHEMA_VERSION: &str = "2.0.0";

/// Response data from an LLM call
///
/// This enum ensures exactly one response mode is represented - either buffered
/// (complete response) or streaming (sequence of chunks). This makes illegal
/// states (both modes or neither) unrepresentable at compile time.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", content = "data")]
pub enum LlmResponseData {
    /// Complete response from buffered mode
    ///
    /// Contains the full `InteractionResponse` serialized as JSON.
    Buffered(serde_json::Value),

    /// Stream chunks from streaming mode
    ///
    /// Each chunk contains text and timing offset.
    Streaming(Vec<SerializableStreamChunk>),
}

impl LlmResponseData {
    /// Get the response as a buffered value, if available
    pub fn as_buffered(&self) -> Option<&serde_json::Value> {
        match self {
            LlmResponseData::Buffered(v) => Some(v),
            LlmResponseData::Streaming(_) => None,
        }
    }

    /// Get the stream chunks, if available
    pub fn as_streaming(&self) -> Option<&[SerializableStreamChunk]> {
        match self {
            LlmResponseData::Buffered(_) => None,
            LlmResponseData::Streaming(chunks) => Some(chunks),
        }
    }

    /// Extract text content from either mode
    ///
    /// For buffered mode, extracts text from rust-genai's InteractionResponse structure
    /// (outputs array with type="text" items).
    /// For streaming mode, concatenates all chunk texts.
    pub fn text(&self) -> Option<String> {
        match self {
            LlmResponseData::Buffered(v) => {
                // rust-genai InteractionResponse structure:
                // { "outputs": [{"type": "thought"}, {"type": "text", "text": "..."}] }
                v.get("outputs")
                    .and_then(|outputs| outputs.as_array())
                    .and_then(|arr| {
                        arr.iter().find_map(|output| {
                            if output.get("type").and_then(|t| t.as_str()) == Some("text") {
                                output
                                    .get("text")
                                    .and_then(|t| t.as_str())
                                    .map(String::from)
                            } else {
                                None
                            }
                        })
                    })
            }
            LlmResponseData::Streaming(chunks) => {
                let text: String = chunks.iter().map(|c| c.text.as_str()).collect();
                if text.is_empty() {
                    None
                } else {
                    Some(text)
                }
            }
        }
    }
}

/// A single LLM interaction step within a trajectory
///
/// Each step captures:
/// - The phase/context of this LLM call
/// - The complete request sent to the LLM
/// - The response data (buffered or streaming mode)
/// - Timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryStep {
    /// Semantic phase identifier (e.g., "decomposition", "sub_query_2", "synthesis")
    ///
    /// This is a soft-typed string following Evergreen principles - agents can
    /// define their own phase names without modifying core types.
    pub phase: String,

    /// The request sent to the LLM
    pub request: SerializableLlmRequest,

    /// The response data from the LLM
    ///
    /// Uses an enum to ensure exactly one mode (buffered or streaming) is represented.
    pub response: LlmResponseData,

    /// Wall-clock duration of this LLM call in milliseconds
    pub duration_ms: u64,

    /// Timestamp when this step started
    #[serde(with = "system_time_serde")]
    pub started_at: SystemTime,
}

/// Serializable wrapper for `LlmRequest`
///
/// This captures all fields from `LlmRequest` in a serializable form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLlmRequest {
    /// User prompt
    pub prompt: String,

    /// Optional system instruction
    pub system_instruction: Option<String>,

    /// Whether Google Search grounding was enabled
    pub use_google_search: bool,

    /// Optional JSON schema for structured output
    pub response_format: Option<serde_json::Value>,
}

impl From<&LlmRequest> for SerializableLlmRequest {
    fn from(request: &LlmRequest) -> Self {
        Self {
            prompt: request.prompt.clone(),
            system_instruction: request.system_instruction.clone(),
            use_google_search: request.use_google_search,
            response_format: request.response_format.clone(),
        }
    }
}

impl From<LlmRequest> for SerializableLlmRequest {
    fn from(request: LlmRequest) -> Self {
        Self::from(&request)
    }
}

/// A serializable stream chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableStreamChunk {
    /// Text content of the chunk
    pub text: String,

    /// Milliseconds since step started when this chunk was received
    pub offset_ms: u64,
}

/// Complete execution trajectory for an agent run
///
/// A trajectory captures everything needed to replay or analyze an agent execution:
/// - The original query and agent configuration
/// - All LLM interaction steps with full request/response data
/// - High-level events for compatibility with existing consumers
/// - Summary metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    /// Unique trajectory identifier (UUID)
    pub id: String,

    /// The original user query
    pub query: String,

    /// Name of the agent that executed
    pub agent_name: String,

    /// Agent configuration (serialized as flexible JSON)
    ///
    /// This uses `serde_json::Value` for Evergreen-style flexibility -
    /// different agents can have different config shapes.
    pub agent_config: serde_json::Value,

    /// All LLM interaction steps in execution order
    pub steps: Vec<TrajectoryStep>,

    /// High-level events (for compatibility with existing consumers)
    ///
    /// This mirrors the events that would be yielded by the agent stream,
    /// allowing trajectory consumers to use the same event-based processing.
    pub events: Vec<AgentUpdate>,

    /// Metadata about the trajectory
    pub metadata: TrajectoryMetadata,
}

/// Metadata about a trajectory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryMetadata {
    /// When the trajectory was created
    #[serde(with = "system_time_serde")]
    pub created_at: SystemTime,

    /// Total execution duration in milliseconds
    pub total_duration_ms: u64,

    /// Total tokens used across all steps
    pub total_tokens: u32,

    /// Number of steps where token count was unavailable
    pub tokens_unavailable_count: usize,

    /// The final answer (if execution completed successfully)
    pub final_answer: Option<String>,

    /// Model used (from LlmConfig)
    pub model: String,

    /// Schema version for forward compatibility
    pub schema_version: String,
}

impl Trajectory {
    /// Create a new trajectory builder
    pub fn builder() -> TrajectoryBuilder {
        TrajectoryBuilder::default()
    }

    /// Save trajectory to a JSON file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(path, json)
    }

    /// Load trajectory from a JSON file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let json = fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Get the final answer from the trajectory, if available
    pub fn final_answer(&self) -> Option<&str> {
        self.metadata.final_answer.as_deref()
    }

    /// Get the number of LLM calls in this trajectory
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

/// Builder for constructing trajectories
#[derive(Debug, Default)]
pub struct TrajectoryBuilder {
    query: Option<String>,
    agent_name: Option<String>,
    agent_config: Option<serde_json::Value>,
    model: Option<String>,
}

impl TrajectoryBuilder {
    /// Set the query
    pub fn query(mut self, query: impl Into<String>) -> Self {
        self.query = Some(query.into());
        self
    }

    /// Set the agent name
    pub fn agent_name(mut self, name: impl Into<String>) -> Self {
        self.agent_name = Some(name.into());
        self
    }

    /// Set the agent configuration
    pub fn agent_config(mut self, config: serde_json::Value) -> Self {
        self.agent_config = Some(config);
        self
    }

    /// Set the agent configuration from a serializable type
    ///
    /// Logs a warning if serialization fails.
    pub fn agent_config_from<T: Serialize>(mut self, config: &T) -> Self {
        match serde_json::to_value(config) {
            Ok(v) => self.agent_config = Some(v),
            Err(e) => {
                log::warn!(
                    "Failed to serialize agent config for trajectory: {}. Config will be null.",
                    e
                );
            }
        }
        self
    }

    /// Set the model name
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Build the trajectory from recorded steps and events
    ///
    /// Logs warnings if required fields (query, agent_name) are not set.
    ///
    /// # Arguments
    ///
    /// * `steps` - The recorded LLM interaction steps
    /// * `events` - The high-level agent events
    /// * `total_duration_ms` - Total execution time in milliseconds
    /// * `final_answer` - The final answer, if available
    pub fn build(
        self,
        steps: Vec<TrajectoryStep>,
        events: Vec<AgentUpdate>,
        total_duration_ms: u64,
        final_answer: Option<String>,
    ) -> Trajectory {
        // Warn on missing required fields
        if self.query.is_none() {
            log::warn!("Building trajectory without query - this may indicate a bug");
        }
        if self.agent_name.is_none() {
            log::warn!("Building trajectory without agent_name - will be recorded as 'unknown'");
        }

        // Calculate total tokens from steps (only available for buffered responses)
        let (total_tokens, tokens_unavailable_count) =
            steps.iter().fold((0u32, 0usize), |acc, step| {
                match &step.response {
                    LlmResponseData::Buffered(response) => {
                        if let Some(tokens) = response
                            .get("usage")
                            .and_then(|u| u.get("total_tokens"))
                            .and_then(|t| t.as_u64())
                        {
                            (acc.0.saturating_add(tokens as u32), acc.1)
                        } else {
                            (acc.0, acc.1 + 1)
                        }
                    }
                    LlmResponseData::Streaming(_) => {
                        // Streaming responses don't include token counts
                        (acc.0, acc.1 + 1)
                    }
                }
            });

        Trajectory {
            id: uuid::Uuid::new_v4().to_string(),
            query: self.query.unwrap_or_default(),
            agent_name: self.agent_name.unwrap_or_else(|| "unknown".to_string()),
            agent_config: self.agent_config.unwrap_or(serde_json::Value::Null),
            steps,
            events,
            metadata: TrajectoryMetadata {
                created_at: SystemTime::now(),
                total_duration_ms,
                total_tokens,
                tokens_unavailable_count,
                final_answer,
                model: self
                    .model
                    .unwrap_or_else(|| crate::config::MODEL.to_string()),
                schema_version: SCHEMA_VERSION.to_string(),
            },
        }
    }
}

/// Serde helper for SystemTime (stores as seconds since UNIX_EPOCH)
mod system_time_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + Duration::from_secs(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::update::ResultMetadata;
    use serde_json::json;

    fn sample_request() -> SerializableLlmRequest {
        SerializableLlmRequest {
            prompt: "What is the capital of France?".to_string(),
            system_instruction: Some("You are a helpful assistant.".to_string()),
            use_google_search: false,
            response_format: None,
        }
    }

    fn sample_step() -> TrajectoryStep {
        TrajectoryStep {
            phase: "test_phase".to_string(),
            request: sample_request(),
            // Use rust-genai InteractionResponse structure
            response: LlmResponseData::Buffered(json!({
                "outputs": [
                    {"type": "text", "text": "Paris is the capital of France."}
                ],
                "usage": {"total_tokens": 42}
            })),
            duration_ms: 150,
            started_at: SystemTime::now(),
        }
    }

    fn sample_events() -> Vec<AgentUpdate> {
        vec![
            AgentUpdate::custom("test_started", "Starting test", json!({})),
            AgentUpdate::final_result("Paris".to_string(), ResultMetadata::new(42, 0, 150)),
        ]
    }

    #[test]
    fn test_serializable_request_from_llm_request() {
        let request = LlmRequest::with_system("Hello", "System")
            .with_google_search()
            .with_response_format(json!({"type": "object"}));

        let serializable = SerializableLlmRequest::from(&request);

        assert_eq!(serializable.prompt, "Hello");
        assert_eq!(serializable.system_instruction, Some("System".to_string()));
        assert!(serializable.use_google_search);
        assert_eq!(
            serializable.response_format,
            Some(json!({"type": "object"}))
        );
    }

    #[test]
    fn test_trajectory_step_serialization_roundtrip() {
        let step = sample_step();
        let json = serde_json::to_string(&step).unwrap();
        let restored: TrajectoryStep = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.phase, step.phase);
        assert_eq!(restored.request.prompt, step.request.prompt);
        assert_eq!(restored.duration_ms, step.duration_ms);
    }

    #[test]
    fn test_trajectory_builder() {
        let trajectory = Trajectory::builder()
            .query("What is Rust?")
            .agent_name("test_agent")
            .agent_config(json!({"temperature": 0.7}))
            .model("gemini-2.0-flash")
            .build(
                vec![sample_step()],
                sample_events(),
                150,
                Some("Rust is a programming language.".to_string()),
            );

        assert_eq!(trajectory.query, "What is Rust?");
        assert_eq!(trajectory.agent_name, "test_agent");
        assert_eq!(trajectory.steps.len(), 1);
        assert_eq!(trajectory.events.len(), 2);
        assert_eq!(trajectory.metadata.total_tokens, 42);
        assert_eq!(trajectory.metadata.tokens_unavailable_count, 0);
        assert_eq!(
            trajectory.final_answer(),
            Some("Rust is a programming language.")
        );
    }

    #[test]
    fn test_trajectory_serialization_roundtrip() {
        let trajectory = Trajectory::builder()
            .query("Test query")
            .agent_name("test_agent")
            .build(vec![sample_step()], sample_events(), 150, None);

        let json = serde_json::to_string_pretty(&trajectory).unwrap();
        let restored: Trajectory = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.id, trajectory.id);
        assert_eq!(restored.query, trajectory.query);
        assert_eq!(restored.agent_name, trajectory.agent_name);
        assert_eq!(restored.steps.len(), 1);
        assert_eq!(restored.events.len(), 2);
        assert_eq!(restored.metadata.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn test_trajectory_save_load() {
        let trajectory = Trajectory::builder()
            .query("Save/load test")
            .agent_name("test_agent")
            .build(vec![sample_step()], sample_events(), 150, None);

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!("trajectory_test_{}.json", trajectory.id));

        // Save
        trajectory.save(&path).unwrap();

        // Load
        let loaded = Trajectory::load(&path).unwrap();

        assert_eq!(loaded.id, trajectory.id);
        assert_eq!(loaded.query, trajectory.query);
        assert_eq!(loaded.steps.len(), 1);

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_trajectory_step_with_streaming() {
        let step = TrajectoryStep {
            phase: "streaming_test".to_string(),
            request: sample_request(),
            response: LlmResponseData::Streaming(vec![
                SerializableStreamChunk {
                    text: "Hello".to_string(),
                    offset_ms: 10,
                },
                SerializableStreamChunk {
                    text: " World".to_string(),
                    offset_ms: 50,
                },
            ]),
            duration_ms: 100,
            started_at: SystemTime::now(),
        };

        let json = serde_json::to_string(&step).unwrap();
        let restored: TrajectoryStep = serde_json::from_str(&json).unwrap();

        let chunks = restored
            .response
            .as_streaming()
            .expect("Expected streaming");
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "Hello");
    }

    #[test]
    fn test_schema_version_included() {
        let trajectory = Trajectory::builder()
            .query("Version test")
            .build(vec![], vec![], 0, None);

        assert_eq!(trajectory.metadata.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn test_step_count() {
        let trajectory = Trajectory::builder().query("Count test").build(
            vec![sample_step(), sample_step()],
            vec![],
            0,
            None,
        );

        assert_eq!(trajectory.step_count(), 2);
    }

    #[test]
    fn test_tokens_unavailable_count() {
        let step_without_tokens = TrajectoryStep {
            phase: "no_tokens".to_string(),
            request: sample_request(),
            // Response with no usage field
            response: LlmResponseData::Buffered(json!({
                "outputs": [{"type": "text", "text": "No usage data"}]
            })),
            duration_ms: 50,
            started_at: SystemTime::now(),
        };

        let trajectory = Trajectory::builder().query("Tokens test").build(
            vec![sample_step(), step_without_tokens],
            vec![],
            0,
            None,
        );

        assert_eq!(trajectory.metadata.total_tokens, 42);
        assert_eq!(trajectory.metadata.tokens_unavailable_count, 1);
    }

    #[test]
    fn test_llm_response_data_accessors() {
        // Use rust-genai InteractionResponse structure
        let buffered = LlmResponseData::Buffered(json!({
            "outputs": [{"type": "text", "text": "Hello"}]
        }));
        assert!(buffered.as_buffered().is_some());
        assert!(buffered.as_streaming().is_none());
        assert_eq!(buffered.text(), Some("Hello".to_string()));

        let streaming = LlmResponseData::Streaming(vec![
            SerializableStreamChunk {
                text: "Hello".to_string(),
                offset_ms: 0,
            },
            SerializableStreamChunk {
                text: " World".to_string(),
                offset_ms: 10,
            },
        ]);
        assert!(streaming.as_buffered().is_none());
        assert!(streaming.as_streaming().is_some());
        assert_eq!(streaming.text(), Some("Hello World".to_string()));

        let empty_streaming = LlmResponseData::Streaming(vec![]);
        assert_eq!(empty_streaming.text(), None);
    }
}
