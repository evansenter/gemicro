//! Trajectory data types and serialization.

use crate::trajectory::TrajectoryBuilder;
use crate::update::AgentUpdate;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;
use std::time::SystemTime;

/// Response data from an LLM call
///
/// This enum ensures exactly one response mode is represented - either buffered
/// (complete response) or streaming (sequence of chunks). This makes illegal
/// states (both modes or neither) unrepresentable at compile time.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", content = "data")]
#[non_exhaustive]
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
    /// For buffered mode, extracts text from genai-rs's InteractionResponse structure
    /// (outputs array with type="text" items).
    /// For streaming mode, concatenates all chunk texts.
    pub fn text(&self) -> Option<String> {
        match self {
            LlmResponseData::Buffered(v) => {
                // genai-rs InteractionResponse structure:
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
/// - The complete request sent to the LLM (serialized as JSON)
/// - The response data (buffered or streaming mode)
/// - Timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TrajectoryStep {
    /// Semantic phase identifier (e.g., "decomposition", "sub_query_2", "synthesis")
    ///
    /// This is a soft-typed string following Evergreen principles - agents can
    /// define their own phase names without modifying core types.
    pub phase: String,

    /// The request sent to the LLM (genai-rs `InteractionRequest` serialized as JSON)
    ///
    /// For streaming requests where the builder cannot be serialized, this will be
    /// `serde_json::Value::Null`.
    pub request: serde_json::Value,

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
#[non_exhaustive]
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

    /// The final result (if execution completed successfully)
    ///
    /// Uses `serde_json::Value` for flexibility - can be a string, structured data,
    /// or null for side-effect-only agents.
    pub final_result: Option<serde_json::Value>,

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

    /// Get the final result from the trajectory, if available
    pub fn final_result(&self) -> Option<&serde_json::Value> {
        self.metadata.final_result.as_ref()
    }

    /// Get the number of LLM calls in this trajectory
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

/// Serde helper for SystemTime (stores as seconds since UNIX_EPOCH)
pub(crate) mod system_time_serde {
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

    fn sample_request() -> serde_json::Value {
        json!({
            "prompt": "What is the capital of France?",
            "system_instruction": "You are a helpful assistant."
        })
    }

    fn sample_step() -> TrajectoryStep {
        TrajectoryStep {
            phase: "test_phase".to_string(),
            request: sample_request(),
            // Use genai-rs InteractionResponse structure
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
            AgentUpdate::final_result(json!("Paris"), ResultMetadata::new(42, 0, 150)),
        ]
    }

    #[test]
    fn test_trajectory_step_serialization_roundtrip() {
        let step = sample_step();
        let json = serde_json::to_string(&step).unwrap();
        let restored: TrajectoryStep = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.phase, step.phase);
        assert_eq!(restored.request, step.request);
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
                Some(json!("Rust is a programming language.")),
            );

        assert_eq!(trajectory.query, "What is Rust?");
        assert_eq!(trajectory.agent_name, "test_agent");
        assert_eq!(trajectory.steps.len(), 1);
        assert_eq!(trajectory.events.len(), 2);
        assert_eq!(trajectory.metadata.total_tokens, 42);
        assert_eq!(trajectory.metadata.tokens_unavailable_count, 0);
        assert_eq!(
            trajectory.final_result(),
            Some(&json!("Rust is a programming language."))
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
        assert_eq!(
            restored.metadata.schema_version,
            crate::trajectory::SCHEMA_VERSION
        );
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
    fn test_trajectory_roundtrip_with_turns() {
        // Request is now serde_json::Value (serialized InteractionRequest)
        let request_with_turns = json!({
            "prompt": "Follow up",
            "turns": [
                {"role": "user", "parts": [{"type": "text", "text": "Q1"}]},
                {"role": "model", "parts": [{"type": "text", "text": "A1"}]}
            ]
        });

        let step_with_turns = TrajectoryStep {
            phase: "multi_turn".to_string(),
            request: request_with_turns,
            response: LlmResponseData::Buffered(json!({
                "outputs": [{"type": "text", "text": "Response"}]
            })),
            duration_ms: 100,
            started_at: SystemTime::now(),
        };

        let trajectory = Trajectory::builder()
            .query("Multi-turn test")
            .agent_name("test_agent")
            .build(vec![step_with_turns], vec![], 100, None);

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!("trajectory_turns_test_{}.json", trajectory.id));

        // Save and reload
        trajectory.save(&path).unwrap();
        let loaded = Trajectory::load(&path).unwrap();

        // Verify turns survived the round-trip (now in request as JSON)
        let turns = loaded.steps[0].request.get("turns").unwrap();
        assert!(turns.is_array());
        assert_eq!(turns.as_array().unwrap().len(), 2);

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

        assert_eq!(
            trajectory.metadata.schema_version,
            crate::trajectory::SCHEMA_VERSION
        );
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
        // Use genai-rs InteractionResponse structure
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

    // ========================================================================
    // Comprehensive round-trip tests (issue #149)
    // ========================================================================

    /// Comprehensive file round-trip test that verifies ALL fields survive save/load.
    ///
    /// This catches subtle serialization issues like:
    /// - SystemTime precision loss
    /// - Optional field handling
    /// - Nested struct serialization
    #[test]
    fn test_trajectory_file_roundtrip_comprehensive() {
        use tempfile::NamedTempFile;

        // Build trajectory with ALL fields populated
        let trajectory = Trajectory::builder()
            .query("Comprehensive roundtrip test query")
            .agent_name("test_agent_comprehensive")
            .agent_config(json!({
                "temperature": 0.7,
                "nested": {"key": "value"},
                "array": [1, 2, 3]
            }))
            .model("test-model-v1")
            .build(
                vec![sample_step()],
                sample_events(),
                12345,
                Some(json!("The final answer is 42.")),
            );

        // Save to temp file
        let temp_file = NamedTempFile::new().unwrap();
        trajectory.save(temp_file.path()).unwrap();

        // Load back
        let loaded = Trajectory::load(temp_file.path()).unwrap();

        // Verify ALL top-level fields
        assert_eq!(loaded.id, trajectory.id, "id mismatch");
        assert_eq!(loaded.query, trajectory.query, "query mismatch");
        assert_eq!(
            loaded.agent_name, trajectory.agent_name,
            "agent_name mismatch"
        );
        assert_eq!(
            loaded.agent_config, trajectory.agent_config,
            "agent_config mismatch"
        );

        // Verify metadata
        assert_eq!(
            loaded.metadata.total_duration_ms, trajectory.metadata.total_duration_ms,
            "total_duration_ms mismatch"
        );
        assert_eq!(
            loaded.metadata.total_tokens, trajectory.metadata.total_tokens,
            "total_tokens mismatch"
        );
        assert_eq!(
            loaded.metadata.tokens_unavailable_count, trajectory.metadata.tokens_unavailable_count,
            "tokens_unavailable_count mismatch"
        );
        assert_eq!(
            loaded.metadata.final_result, trajectory.metadata.final_result,
            "final_result mismatch"
        );
        assert_eq!(
            loaded.metadata.model, trajectory.metadata.model,
            "model mismatch"
        );
        assert_eq!(
            loaded.metadata.schema_version, trajectory.metadata.schema_version,
            "schema_version mismatch"
        );

        // Verify steps
        assert_eq!(
            loaded.steps.len(),
            trajectory.steps.len(),
            "steps length mismatch"
        );
        let orig_step = &trajectory.steps[0];
        let loaded_step = &loaded.steps[0];
        assert_eq!(loaded_step.phase, orig_step.phase, "step phase mismatch");
        assert_eq!(
            loaded_step.duration_ms, orig_step.duration_ms,
            "step duration_ms mismatch"
        );
        // Request is now serde_json::Value
        assert_eq!(
            loaded_step.request, orig_step.request,
            "step request mismatch"
        );

        // Verify events
        assert_eq!(
            loaded.events.len(),
            trajectory.events.len(),
            "events length mismatch"
        );
        assert_eq!(
            loaded.events[0].event_type, trajectory.events[0].event_type,
            "event type mismatch"
        );
        assert_eq!(
            loaded.events[1].event_type, trajectory.events[1].event_type,
            "final_result event type mismatch"
        );

        // Verify timestamps survive round-trip (serialized as whole seconds)
        // Note: Sub-second precision is intentionally lost per system_time_serde
        // Compare at seconds granularity since nanoseconds are truncated
        use std::time::UNIX_EPOCH;
        let orig_created_secs = trajectory
            .metadata
            .created_at
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let loaded_created_secs = loaded
            .metadata
            .created_at
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert_eq!(
            loaded_created_secs, orig_created_secs,
            "trajectory created_at seconds mismatch"
        );

        let orig_started_secs = orig_step
            .started_at
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let loaded_started_secs = loaded_step
            .started_at
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert_eq!(
            loaded_started_secs, orig_started_secs,
            "step started_at seconds mismatch"
        );
    }

    /// Test file round-trip with streaming response data.
    #[test]
    fn test_trajectory_file_roundtrip_streaming() {
        use tempfile::NamedTempFile;

        let streaming_step = TrajectoryStep {
            phase: "streaming_phase".to_string(),
            request: sample_request(),
            response: LlmResponseData::Streaming(vec![
                SerializableStreamChunk {
                    text: "First ".to_string(),
                    offset_ms: 10,
                },
                SerializableStreamChunk {
                    text: "second ".to_string(),
                    offset_ms: 50,
                },
                SerializableStreamChunk {
                    text: "third".to_string(),
                    offset_ms: 100,
                },
            ]),
            duration_ms: 150,
            started_at: SystemTime::now(),
        };

        let trajectory = Trajectory::builder()
            .query("Streaming test")
            .agent_name("streaming_agent")
            .build(vec![streaming_step], vec![], 150, None);

        let temp_file = NamedTempFile::new().unwrap();
        trajectory.save(temp_file.path()).unwrap();
        let loaded = Trajectory::load(temp_file.path()).unwrap();

        // Verify streaming chunks survived
        let chunks = loaded.steps[0]
            .response
            .as_streaming()
            .expect("Expected streaming response");
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "First ");
        assert_eq!(chunks[0].offset_ms, 10);
        assert_eq!(chunks[1].text, "second ");
        assert_eq!(chunks[2].text, "third");
        assert_eq!(chunks[2].offset_ms, 100);

        // Verify text extraction works after round-trip
        assert_eq!(
            loaded.steps[0].response.text(),
            Some("First second third".to_string())
        );
    }

    /// Test that load() returns appropriate error for corrupted JSON.
    #[test]
    fn test_trajectory_load_corrupted_json() {
        use tempfile::NamedTempFile;

        let temp_file = NamedTempFile::new().unwrap();

        // Write invalid JSON
        std::fs::write(temp_file.path(), "{ invalid json }").unwrap();

        let result = Trajectory::load(temp_file.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    /// Test that load() returns appropriate error for valid JSON with wrong schema.
    #[test]
    fn test_trajectory_load_wrong_schema() {
        use tempfile::NamedTempFile;

        let temp_file = NamedTempFile::new().unwrap();

        // Write valid JSON but wrong schema (missing required fields)
        std::fs::write(temp_file.path(), r#"{"foo": "bar"}"#).unwrap();

        let result = Trajectory::load(temp_file.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    /// Test that load() returns appropriate error for nonexistent file.
    #[test]
    fn test_trajectory_load_nonexistent() {
        let result = Trajectory::load("/nonexistent/path/to/trajectory.json");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }
}
