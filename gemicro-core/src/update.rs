use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::SystemTime;

use crate::agent::EVENT_FINAL_RESULT;

/// Flexible event structure for agent updates.
///
/// Inspired by [Evergreen protocol](https://github.com/google-deepmind/evergreen-spec)'s
/// approach to extensibility through pragmatic flexibility over rigid typing.
///
/// # Design Philosophy
///
/// - **Extensibility**: New agents can define new event types without modifying gemicro-core
/// - **Flexibility**: Event data can be arbitrarily complex JSON structures
/// - **Type Safety**: Helper constructors prevent typos; typed accessors provide ergonomics
/// - **Forward Compatibility**: Unknown event types are gracefully handled (logged/ignored)
///
/// # Example
///
/// ```
/// use gemicro_core::AgentUpdate;
/// use serde_json::json;
///
/// // Create events using custom() - the universal constructor
/// let update = AgentUpdate::custom(
///     "my_agent_step",
///     "Step completed",
///     json!({"step": 1, "result": "success"}),
/// );
/// assert_eq!(update.event_type, "my_agent_step");
///
/// // Access event data directly via the flexible JSON field
/// assert_eq!(update.data["step"], 1);
/// assert_eq!(update.data["result"], "success");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AgentUpdate {
    /// Event type identifier (e.g., "decomposition_started", "sub_query_completed")
    ///
    /// Semantic meaning comes from this field, not from enum discriminants.
    /// This allows new agent types to define new events without protocol changes.
    pub event_type: String,

    /// Human-readable message describing the event
    pub message: String,

    /// When the event occurred
    pub timestamp: SystemTime,

    /// Event-specific data as flexible JSON
    ///
    /// Each event type defines its own schema. Use typed accessors
    /// (e.g., `as_sub_query_completed()`) for ergonomic access.
    pub data: serde_json::Value,
}

/// Constructors and accessors for AgentUpdate
impl AgentUpdate {
    /// Create a custom event with any event type
    ///
    /// This is the universal constructor for all agent events, following
    /// [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) philosophy.
    /// Agents define their own event types without modifying gemicro-core.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::AgentUpdate;
    /// use serde_json::json;
    ///
    /// let update = AgentUpdate::custom(
    ///     "my_custom_event",
    ///     "Something happened",
    ///     json!({"detail": "value"})
    /// );
    /// assert_eq!(update.event_type, "my_custom_event");
    /// ```
    pub fn custom(
        event_type: impl Into<String>,
        message: impl Into<String>,
        data: serde_json::Value,
    ) -> Self {
        Self {
            event_type: event_type.into(),
            message: message.into(),
            timestamp: SystemTime::now(),
            data,
        }
    }

    /// Create a final_result event (required by all agents per event contract).
    ///
    /// This is the only dedicated constructor besides `custom()` because
    /// `final_result` is the universal completion signal that ALL agents must emit.
    /// It is cross-agent, not agent-specific.
    ///
    /// # Arguments
    ///
    /// * `result` - The agent's output as flexible JSON:
    ///   - Q&A agents: `json!("The answer")` - a string
    ///   - Structured results: `json!({"summary": "...", "sources": [...]})` - an object
    ///   - Side-effect agents: `Value::Null` - no result, just completion signal
    /// * `metadata` - Execution metadata (tokens, duration, agent-specific extras)
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::{AgentUpdate, ResultMetadata};
    /// use serde_json::json;
    ///
    /// // Q&A agent with string result
    /// let update = AgentUpdate::final_result(json!("The answer is 42"), ResultMetadata::new(100, 0, 500));
    ///
    /// // Side-effect agent with no result
    /// let update = AgentUpdate::final_result(serde_json::Value::Null, ResultMetadata::new(50, 0, 250));
    /// ```
    pub fn final_result(result: serde_json::Value, metadata: ResultMetadata) -> Self {
        Self {
            event_type: EVENT_FINAL_RESULT.into(),
            message: "Query complete".into(),
            timestamp: SystemTime::now(),
            data: json!({
                "result": result,
                "metadata": metadata,
            }),
        }
    }

    /// Typed accessor for final_result events
    ///
    /// Returns `None` if this is not a final_result event
    /// or if the data doesn't match the expected schema.
    pub fn as_final_result(&self) -> Option<FinalResult> {
        if self.event_type == "final_result" {
            serde_json::from_value(self.data.clone()).ok().or_else(|| {
                log::warn!("Failed to parse final_result data: {:?}", self.data);
                None
            })
        } else {
            None
        }
    }
}

/// Strongly-typed result struct for final_result events.
///
/// The `result` field is flexible JSON to support various agent types:
/// - Q&A agents: `json!("The answer")` - a string
/// - Structured results: `json!({"summary": "...", "sources": [...]})` - an object
/// - Side-effect agents: `Value::Null` - no result, just completion signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalResult {
    /// The agent's output - can be any JSON value (string, object, array, or null).
    pub result: serde_json::Value,
    pub metadata: ResultMetadata,
}

/// Metadata about the overall result.
///
/// Contains generic fields plus an extensible `extra` for agent-specific data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ResultMetadata {
    /// Total tokens used across all LLM calls
    ///
    /// If `tokens_unavailable_count > 0`, this is a partial sum (lower bound).
    pub total_tokens: u32,

    /// Number of LLM calls that didn't report token usage
    ///
    /// When the API doesn't return token counts, we treat them as 0 for summation
    /// but track how many were missing here. If this is > 0, `total_tokens` is incomplete.
    pub tokens_unavailable_count: usize,

    /// Total execution time in milliseconds
    pub duration_ms: u64,

    /// Agent-specific metadata (steps, iterations, tool calls, etc.)
    ///
    /// Examples:
    /// - DeepResearch: `{"steps_succeeded": 3, "steps_failed": 1}`
    /// - ReAct: `{"iterations": 5, "tools_used": ["calculator", "search"]}`
    /// - SimpleQA: `null` or `{}`
    #[serde(default)]
    pub extra: serde_json::Value,
}

impl ResultMetadata {
    /// Create metadata with no extra fields.
    pub fn new(total_tokens: u32, tokens_unavailable_count: usize, duration_ms: u64) -> Self {
        Self {
            total_tokens,
            tokens_unavailable_count,
            duration_ms,
            extra: serde_json::Value::Null,
        }
    }

    /// Create metadata with agent-specific extra fields.
    pub fn with_extra(
        total_tokens: u32,
        tokens_unavailable_count: usize,
        duration_ms: u64,
        extra: serde_json::Value,
    ) -> Self {
        Self {
            total_tokens,
            tokens_unavailable_count,
            duration_ms,
            extra,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_event() {
        let update = AgentUpdate::custom("my_event", "Something happened", json!({"key": "value"}));
        assert_eq!(update.event_type, "my_event");
        assert_eq!(update.message, "Something happened");
        assert_eq!(update.data["key"], "value");
    }

    // Note: Tests for as_decomposition_complete() and as_sub_query_completed()
    // are in agent/deep_research.rs where those accessors now live.

    #[test]
    fn test_final_result_with_string() {
        let metadata = ResultMetadata::new(100, 0, 5000);
        let update = AgentUpdate::final_result(json!("Answer"), metadata);

        assert_eq!(update.event_type, "final_result");

        let result = update.as_final_result().unwrap();
        assert_eq!(result.result, json!("Answer"));
        assert_eq!(result.metadata.total_tokens, 100);
        assert_eq!(result.metadata.tokens_unavailable_count, 0);
    }

    #[test]
    fn test_final_result_with_structured_data() {
        let metadata = ResultMetadata::new(100, 0, 5000);
        let update = AgentUpdate::final_result(
            json!({"summary": "A summary", "sources": ["src1", "src2"]}),
            metadata,
        );

        let result = update.as_final_result().unwrap();
        assert_eq!(result.result["summary"], "A summary");
        assert_eq!(result.result["sources"][0], "src1");
    }

    #[test]
    fn test_final_result_with_null() {
        let metadata = ResultMetadata::new(100, 0, 5000);
        let update = AgentUpdate::final_result(serde_json::Value::Null, metadata);

        let result = update.as_final_result().unwrap();
        assert!(result.result.is_null());
    }

    #[test]
    fn test_final_result_with_extra() {
        let metadata = ResultMetadata::with_extra(
            100,
            0,
            5000,
            json!({"steps_succeeded": 3, "steps_failed": 1}),
        );
        let update = AgentUpdate::final_result(json!("Answer"), metadata);

        let result = update.as_final_result().unwrap();
        assert_eq!(result.metadata.extra["steps_succeeded"], 3);
        assert_eq!(result.metadata.extra["steps_failed"], 1);
    }

    #[test]
    fn test_final_result_with_incomplete_tokens() {
        let metadata = ResultMetadata::new(50, 2, 3000);
        let update = AgentUpdate::final_result(json!("Partial answer"), metadata);

        let result = update.as_final_result().unwrap();
        assert_eq!(result.metadata.tokens_unavailable_count, 2);
        // total_tokens is a partial sum when tokens_unavailable_count > 0
    }

    #[test]
    fn test_accessor_wrong_type() {
        let update = AgentUpdate::custom("some_other_event", "Message", json!({}));

        // Should return None for wrong event type
        assert!(update.as_final_result().is_none());
    }

    #[test]
    fn test_accessor_malformed_data() {
        // Test final_result with malformed data (missing required fields)
        let update = AgentUpdate::custom(
            "final_result",
            "Test",
            json!({ "wrong_field": 123 }), // missing result and metadata
        );

        // Should return None and log warning (not panic)
        assert!(update.as_final_result().is_none());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let metadata = ResultMetadata::with_extra(
            100,
            0,
            5000,
            json!({"steps_succeeded": 2, "steps_failed": 0}),
        );
        let original = AgentUpdate::final_result(json!("Test answer"), metadata);

        let json_str = serde_json::to_string(&original).unwrap();
        let deserialized: AgentUpdate = serde_json::from_str(&json_str).unwrap();

        assert_eq!(deserialized.event_type, original.event_type);
        assert_eq!(deserialized.message, original.message);

        let result = deserialized.as_final_result().unwrap();
        assert_eq!(result.result, json!("Test answer"));
        assert_eq!(result.metadata.total_tokens, 100);
        assert_eq!(result.metadata.extra["steps_succeeded"], 2);
    }

    #[test]
    fn test_result_metadata_default_extra_is_null() {
        let metadata = ResultMetadata::new(100, 0, 5000);

        // Default extra is null (valid per docstring: "SimpleQA: `null` or `{}`")
        assert!(metadata.extra.is_null());
    }

    #[test]
    fn test_final_result_with_empty_extra() {
        let metadata = ResultMetadata::with_extra(100, 0, 5000, json!({}));
        let update = AgentUpdate::final_result(json!("Answer"), metadata);

        let result = update.as_final_result().unwrap();
        assert!(result.metadata.extra.is_object());
        assert!(result.metadata.extra.as_object().unwrap().is_empty());
    }
}
