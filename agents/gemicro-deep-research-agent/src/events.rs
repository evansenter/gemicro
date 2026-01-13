//! Deep Research event accessors.
//!
//! Extension trait for parsing DeepResearch-specific events from AgentUpdate.
//! These accessors belong here (not in gemicro-core) per the "No Agent/Dataset Leakage" principle.

use gemicro_core::AgentUpdate;
use serde::{Deserialize, Serialize};

// Event type constants used by DeepResearchAgent
pub(crate) const EVENT_DECOMPOSITION_COMPLETE: &str = "decomposition_complete";
pub(crate) const EVENT_SUB_QUERY_COMPLETED: &str = "sub_query_completed";

/// Strongly-typed result struct for sub_query_completed events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubQueryResult {
    pub id: usize,
    pub result: String,
    pub tokens_used: u32,
}

/// Extension trait for parsing DeepResearch-specific events from AgentUpdate.
///
/// Import this trait to use the accessor methods on AgentUpdate:
///
/// ```
/// use gemicro_deep_research_agent::DeepResearchEventExt;
/// use gemicro_core::AgentUpdate;
/// use serde_json::json;
///
/// let update = AgentUpdate::custom(
///     "decomposition_complete",
///     "Decomposed into 2 sub-queries",
///     json!({ "sub_queries": ["Q1", "Q2"] }),
/// );
///
/// if let Some(queries) = update.as_decomposition_complete() {
///     println!("Got {} sub-queries", queries.len());
/// }
/// ```
pub trait DeepResearchEventExt {
    /// Parse decomposition_complete event data.
    ///
    /// Returns `None` if this is not a decomposition_complete event
    /// or if the data doesn't match the expected schema.
    fn as_decomposition_complete(&self) -> Option<Vec<String>>;

    /// Parse sub_query_completed event data.
    ///
    /// Returns `None` if this is not a sub_query_completed event
    /// or if the data doesn't match the expected schema.
    fn as_sub_query_completed(&self) -> Option<SubQueryResult>;
}

impl DeepResearchEventExt for AgentUpdate {
    fn as_decomposition_complete(&self) -> Option<Vec<String>> {
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

    fn as_sub_query_completed(&self) -> Option<SubQueryResult> {
        if self.event_type == EVENT_SUB_QUERY_COMPLETED {
            serde_json::from_value(self.data.clone()).ok().or_else(|| {
                log::warn!("Failed to parse sub_query_completed data: {:?}", self.data);
                None
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_as_decomposition_complete() {
        let update = AgentUpdate::custom(
            "decomposition_complete",
            "Decomposed",
            json!({ "sub_queries": ["Q1", "Q2", "Q3"] }),
        );

        let queries = update.as_decomposition_complete();
        assert!(queries.is_some());
        assert_eq!(queries.unwrap().len(), 3);
    }

    #[test]
    fn test_as_decomposition_complete_wrong_event() {
        let update = AgentUpdate::custom("other_event", "Other", json!({}));
        assert!(update.as_decomposition_complete().is_none());
    }

    #[test]
    fn test_as_sub_query_completed() {
        let update = AgentUpdate::custom(
            "sub_query_completed",
            "Completed",
            json!({ "id": 0, "result": "Answer", "tokens_used": 42 }),
        );

        let result = update.as_sub_query_completed();
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.id, 0);
        assert_eq!(r.result, "Answer");
        assert_eq!(r.tokens_used, 42);
    }
}
