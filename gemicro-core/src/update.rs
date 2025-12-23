use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::SystemTime;

// Event type constants to prevent typos and improve refactorability
pub const EVENT_DECOMPOSITION_STARTED: &str = "decomposition_started";
pub const EVENT_DECOMPOSITION_COMPLETE: &str = "decomposition_complete";
pub const EVENT_SUB_QUERY_STARTED: &str = "sub_query_started";
pub const EVENT_SUB_QUERY_COMPLETED: &str = "sub_query_completed";
pub const EVENT_SUB_QUERY_FAILED: &str = "sub_query_failed";
pub const EVENT_SYNTHESIS_STARTED: &str = "synthesis_started";
pub const EVENT_FINAL_RESULT: &str = "final_result";

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
///
/// // Create event using helper
/// let update = AgentUpdate::sub_query_completed(0, "Result text".to_string(), 42);
///
/// // Access typed data
/// if let Some(result) = update.as_sub_query_completed() {
///     println!("Sub-query {} used {} tokens", result.id, result.tokens_used);
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Type-safe helper constructors for common Deep Research events
impl AgentUpdate {
    /// Create a decomposition_started event
    pub fn decomposition_started() -> Self {
        Self {
            event_type: EVENT_DECOMPOSITION_STARTED.into(),
            message: "Decomposing query into sub-queries".into(),
            timestamp: SystemTime::now(),
            data: json!({}),
        }
    }

    /// Create a decomposition_complete event
    pub fn decomposition_complete(sub_queries: Vec<String>) -> Self {
        Self {
            event_type: EVENT_DECOMPOSITION_COMPLETE.into(),
            message: format!("Decomposed into {} sub-queries", sub_queries.len()),
            timestamp: SystemTime::now(),
            data: json!({ "sub_queries": sub_queries }),
        }
    }

    /// Create a sub_query_started event
    pub fn sub_query_started(id: usize, query: String) -> Self {
        Self {
            event_type: EVENT_SUB_QUERY_STARTED.into(),
            message: format!("Sub-query {} started", id),
            timestamp: SystemTime::now(),
            data: json!({ "id": id, "query": query }),
        }
    }

    /// Create a sub_query_completed event
    pub fn sub_query_completed(id: usize, result: String, tokens_used: u32) -> Self {
        Self {
            event_type: EVENT_SUB_QUERY_COMPLETED.into(),
            message: format!("Sub-query {} completed", id),
            timestamp: SystemTime::now(),
            data: json!({
                "id": id,
                "result": result,
                "tokens_used": tokens_used,
            }),
        }
    }

    /// Create a sub_query_failed event
    pub fn sub_query_failed(id: usize, error: String) -> Self {
        Self {
            event_type: EVENT_SUB_QUERY_FAILED.into(),
            message: format!("Sub-query {} failed", id),
            timestamp: SystemTime::now(),
            data: json!({ "id": id, "error": error }),
        }
    }

    /// Create a synthesis_started event
    pub fn synthesis_started() -> Self {
        Self {
            event_type: EVENT_SYNTHESIS_STARTED.into(),
            message: "Synthesizing results".into(),
            timestamp: SystemTime::now(),
            data: json!({}),
        }
    }

    /// Create a final_result event
    pub fn final_result(answer: String, metadata: ResultMetadata) -> Self {
        Self {
            event_type: EVENT_FINAL_RESULT.into(),
            message: "Research complete".into(),
            timestamp: SystemTime::now(),
            data: json!({
                "answer": answer,
                "metadata": metadata,
            }),
        }
    }

    /// Typed accessor for decomposition_complete events
    ///
    /// Returns `None` if this is not a decomposition_complete event
    /// or if the data doesn't match the expected schema.
    pub fn as_decomposition_complete(&self) -> Option<Vec<String>> {
        if self.event_type == EVENT_DECOMPOSITION_COMPLETE {
            self.data
                .get("sub_queries")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .or_else(|| {
                    log::warn!(
                        "Failed to parse decomposition_complete data: {:?}",
                        self.data
                    );
                    None
                })
        } else {
            None
        }
    }

    /// Typed accessor for sub_query_completed events
    ///
    /// Returns `None` if this is not a sub_query_completed event
    /// or if the data doesn't match the expected schema.
    pub fn as_sub_query_completed(&self) -> Option<SubQueryResult> {
        if self.event_type == EVENT_SUB_QUERY_COMPLETED {
            serde_json::from_value(self.data.clone()).ok().or_else(|| {
                log::warn!("Failed to parse sub_query_completed data: {:?}", self.data);
                None
            })
        } else {
            None
        }
    }

    /// Typed accessor for final_result events
    ///
    /// Returns `None` if this is not a final_result event
    /// or if the data doesn't match the expected schema.
    pub fn as_final_result(&self) -> Option<FinalResult> {
        if self.event_type == EVENT_FINAL_RESULT {
            serde_json::from_value(self.data.clone()).ok().or_else(|| {
                log::warn!("Failed to parse final_result data: {:?}", self.data);
                None
            })
        } else {
            None
        }
    }
}

/// Strongly-typed result struct for sub_query_completed events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubQueryResult {
    pub id: usize,
    pub result: String,
    pub tokens_used: u32,
}

/// Strongly-typed result struct for final_result events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalResult {
    pub answer: String,
    pub metadata: ResultMetadata,
}

/// Metadata about the overall research result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultMetadata {
    pub total_tokens: u32,
    pub duration_ms: u64,
    pub sub_queries_succeeded: usize,
    pub sub_queries_failed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decomposition_started() {
        let update = AgentUpdate::decomposition_started();
        assert_eq!(update.event_type, EVENT_DECOMPOSITION_STARTED);
        assert_eq!(update.message, "Decomposing query into sub-queries");
    }

    #[test]
    fn test_decomposition_complete() {
        let queries = vec!["Q1".to_string(), "Q2".to_string()];
        let update = AgentUpdate::decomposition_complete(queries.clone());

        assert_eq!(update.event_type, EVENT_DECOMPOSITION_COMPLETE);
        assert!(update.message.contains("2 sub-queries"));

        let extracted = update.as_decomposition_complete().unwrap();
        assert_eq!(extracted, queries);
    }

    #[test]
    fn test_sub_query_completed() {
        let update = AgentUpdate::sub_query_completed(0, "Result".to_string(), 42);

        assert_eq!(update.event_type, EVENT_SUB_QUERY_COMPLETED);

        let result = update.as_sub_query_completed().unwrap();
        assert_eq!(result.id, 0);
        assert_eq!(result.result, "Result");
        assert_eq!(result.tokens_used, 42);
    }

    #[test]
    fn test_sub_query_failed() {
        let update = AgentUpdate::sub_query_failed(1, "Timeout".to_string());

        assert_eq!(update.event_type, EVENT_SUB_QUERY_FAILED);
        assert!(update.message.contains("failed"));
    }

    #[test]
    fn test_final_result() {
        let metadata = ResultMetadata {
            total_tokens: 100,
            duration_ms: 5000,
            sub_queries_succeeded: 3,
            sub_queries_failed: 1,
        };
        let update = AgentUpdate::final_result("Answer".to_string(), metadata);

        assert_eq!(update.event_type, EVENT_FINAL_RESULT);

        let result = update.as_final_result().unwrap();
        assert_eq!(result.answer, "Answer");
        assert_eq!(result.metadata.total_tokens, 100);
    }

    #[test]
    fn test_accessor_wrong_type() {
        let update = AgentUpdate::decomposition_started();

        // Should return None for wrong event type
        assert!(update.as_sub_query_completed().is_none());
        assert!(update.as_final_result().is_none());
    }

    #[test]
    fn test_accessor_malformed_data() {
        use serde_json::json;

        // Correct event_type but wrong data structure
        let update = AgentUpdate {
            event_type: EVENT_DECOMPOSITION_COMPLETE.into(),
            message: "Test".into(),
            timestamp: std::time::SystemTime::now(),
            data: json!({ "wrong_field": "not sub_queries" }),
        };

        // Should return None and log warning (not panic)
        assert!(update.as_decomposition_complete().is_none());

        // Test sub_query_completed with malformed data
        let update = AgentUpdate {
            event_type: EVENT_SUB_QUERY_COMPLETED.into(),
            message: "Test".into(),
            timestamp: std::time::SystemTime::now(),
            data: json!({ "id": "not_a_number" }), // id should be usize
        };

        assert!(update.as_sub_query_completed().is_none());

        // Test final_result with malformed data
        let update = AgentUpdate {
            event_type: EVENT_FINAL_RESULT.into(),
            message: "Test".into(),
            timestamp: std::time::SystemTime::now(),
            data: json!({ "answer": 123 }), // answer should be string
        };

        assert!(update.as_final_result().is_none());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let original = AgentUpdate::sub_query_completed(5, "Test".to_string(), 99);

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: AgentUpdate = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.event_type, original.event_type);
        assert_eq!(deserialized.message, original.message);

        let result = deserialized.as_sub_query_completed().unwrap();
        assert_eq!(result.id, 5);
        assert_eq!(result.tokens_used, 99);
    }
}
