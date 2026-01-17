//! Conversation history for REPL sessions
//!
//! Stores the full event stream from each query execution, enabling
//! context-aware multi-turn interactions.

use crate::update::{AgentUpdate, EVENT_FINAL_RESULT};
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// A single entry in the conversation history
///
/// Each entry captures:
/// - The user's query
/// - The agent that processed it
/// - The full stream of events (including final result)
/// - Timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// The user's original query
    pub query: String,

    /// Name of the agent that processed this query
    pub agent_name: String,

    /// Full stream of events from the agent
    ///
    /// This includes all intermediate events (sub_query_started, etc.)
    /// as well as the final_result. Preserves complete execution trace.
    pub events: Vec<AgentUpdate>,

    /// When the query was submitted
    #[serde(with = "system_time_serde")]
    pub timestamp: SystemTime,
}

impl HistoryEntry {
    /// Create a new history entry
    pub fn new(query: String, agent_name: String, events: Vec<AgentUpdate>) -> Self {
        Self {
            query,
            agent_name,
            events,
            timestamp: SystemTime::now(),
        }
    }

    /// Get the final result text, if available.
    ///
    /// Returns the result as a string if the result is a string value.
    /// For structured or null results, returns `None`.
    pub fn final_result(&self) -> Option<&str> {
        self.events
            .iter()
            .find(|e| e.event_type == EVENT_FINAL_RESULT)
            .and_then(|e| e.data.get("result"))
            .and_then(|v| v.as_str())
    }
}

/// Conversation history for a REPL session
///
/// Maintains an ordered list of query/response pairs with full event streams.
/// Designed to be serializable for hot-reload persistence.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConversationHistory {
    entries: Vec<HistoryEntry>,
}

impl ConversationHistory {
    /// Create an empty conversation history
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new entry to the history
    pub fn push(&mut self, entry: HistoryEntry) {
        self.entries.push(entry);
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if history is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over entries
    pub fn iter(&self) -> impl Iterator<Item = &HistoryEntry> {
        self.entries.iter()
    }

    /// Get the last N entries
    pub fn last_n(&self, n: usize) -> &[HistoryEntry] {
        let start = self.entries.len().saturating_sub(n);
        &self.entries[start..]
    }

    /// Get a specific entry by index
    pub fn get(&self, index: usize) -> Option<&HistoryEntry> {
        self.entries.get(index)
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Build a context string from recent history for agent prompts
    ///
    /// Returns a formatted string with query/answer pairs from the last N entries.
    /// This can be prepended to new queries to provide conversational context.
    pub fn context_for_prompt(&self, last_n: usize) -> String {
        if self.entries.is_empty() {
            return String::new();
        }

        let entries = self.last_n(last_n);
        let mut context = String::from("Previous conversation:\n\n");

        for entry in entries {
            context.push_str("User: ");
            context.push_str(&entry.query);
            context.push('\n');

            if let Some(result) = entry.final_result() {
                context.push_str("Assistant: ");
                context.push_str(result);
                context.push_str("\n\n");
            }
        }

        context
    }
}

/// Serde helper for SystemTime
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
    use crate::update::{AgentUpdate, ResultMetadata};

    fn sample_events() -> Vec<AgentUpdate> {
        use serde_json::json;
        vec![
            AgentUpdate::custom("decomposition_started", "Decomposing query", json!({})),
            AgentUpdate::custom(
                "decomposition_complete",
                "Decomposed into 2 sub-queries",
                json!({ "sub_queries": ["Q1", "Q2"] }),
            ),
            AgentUpdate::final_result(json!("The answer is 42"), ResultMetadata::new(100, 0, 1000)),
        ]
    }

    #[test]
    fn test_history_entry_creation() {
        let entry = HistoryEntry::new(
            "What is Rust?".to_string(),
            "deep_research".to_string(),
            sample_events(),
        );

        assert_eq!(entry.query, "What is Rust?");
        assert_eq!(entry.agent_name, "deep_research");
        assert_eq!(entry.events.len(), 3);
    }

    #[test]
    fn test_history_entry_final_result() {
        let entry = HistoryEntry::new("Question".to_string(), "agent".to_string(), sample_events());

        assert_eq!(entry.final_result(), Some("The answer is 42"));
    }

    #[test]
    fn test_conversation_history_push() {
        let mut history = ConversationHistory::new();
        assert!(history.is_empty());

        history.push(HistoryEntry::new(
            "Q1".to_string(),
            "agent".to_string(),
            vec![],
        ));
        assert_eq!(history.len(), 1);

        history.push(HistoryEntry::new(
            "Q2".to_string(),
            "agent".to_string(),
            vec![],
        ));
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_conversation_history_last_n() {
        let mut history = ConversationHistory::new();

        for i in 0..5 {
            history.push(HistoryEntry::new(
                format!("Q{}", i),
                "agent".to_string(),
                vec![],
            ));
        }

        let last_2 = history.last_n(2);
        assert_eq!(last_2.len(), 2);
        assert_eq!(last_2[0].query, "Q3");
        assert_eq!(last_2[1].query, "Q4");
    }

    #[test]
    fn test_conversation_history_last_n_exceeds_length() {
        let mut history = ConversationHistory::new();
        history.push(HistoryEntry::new(
            "Q1".to_string(),
            "agent".to_string(),
            vec![],
        ));

        let last_10 = history.last_n(10);
        assert_eq!(last_10.len(), 1);
    }

    #[test]
    fn test_context_for_prompt_empty() {
        let history = ConversationHistory::new();
        assert_eq!(history.context_for_prompt(3), "");
    }

    #[test]
    fn test_context_for_prompt_with_entries() {
        let mut history = ConversationHistory::new();

        history.push(HistoryEntry::new(
            "What is Rust?".to_string(),
            "agent".to_string(),
            sample_events(),
        ));

        let context = history.context_for_prompt(3);
        assert!(context.contains("User: What is Rust?"));
        assert!(context.contains("Assistant: The answer is 42"));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut history = ConversationHistory::new();
        history.push(HistoryEntry::new(
            "Test query".to_string(),
            "deep_research".to_string(),
            sample_events(),
        ));

        let json = serde_json::to_string(&history).unwrap();
        let restored: ConversationHistory = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.len(), 1);
        assert_eq!(restored.get(0).unwrap().query, "Test query");
        assert_eq!(
            restored.get(0).unwrap().final_result(),
            Some("The answer is 42")
        );
    }
}
