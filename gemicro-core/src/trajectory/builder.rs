//! Trajectory builder for constructing trajectories.

use crate::trajectory::data::{LlmResponseData, Trajectory, TrajectoryMetadata, TrajectoryStep};
use crate::trajectory::SCHEMA_VERSION;
use crate::update::AgentUpdate;
use serde::Serialize;
use std::time::SystemTime;

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
    /// * `final_result` - The final result, if available (string, structured data, or null)
    pub fn build(
        self,
        steps: Vec<TrajectoryStep>,
        events: Vec<AgentUpdate>,
        total_duration_ms: u64,
        final_result: Option<serde_json::Value>,
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
                final_result,
                model: self
                    .model
                    .unwrap_or_else(|| crate::config::MODEL.to_string()),
                schema_version: SCHEMA_VERSION.to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::data::SerializableStreamChunk;
    use crate::update::ResultMetadata;
    use serde_json::json;

    fn sample_request() -> serde_json::Value {
        json!({
            "prompt": "Test prompt",
            "system_instruction": null,
            "use_google_search": false
        })
    }

    fn sample_step() -> TrajectoryStep {
        TrajectoryStep {
            phase: "test".to_string(),
            request: sample_request(),
            response: LlmResponseData::Buffered(json!({
                "outputs": [{"type": "text", "text": "Test response"}],
                "usage": {"total_tokens": 100}
            })),
            duration_ms: 50,
            started_at: SystemTime::now(),
        }
    }

    #[test]
    fn test_builder_defaults() {
        let trajectory = TrajectoryBuilder::default().build(vec![], vec![], 0, None);

        assert!(trajectory.query.is_empty());
        assert_eq!(trajectory.agent_name, "unknown");
        assert_eq!(trajectory.agent_config, serde_json::Value::Null);
    }

    #[test]
    fn test_builder_with_all_fields() {
        let trajectory = TrajectoryBuilder::default()
            .query("Test query")
            .agent_name("test_agent")
            .agent_config(json!({"key": "value"}))
            .model("test-model")
            .build(vec![], vec![], 1000, Some(json!("Answer")));

        assert_eq!(trajectory.query, "Test query");
        assert_eq!(trajectory.agent_name, "test_agent");
        assert_eq!(trajectory.agent_config, json!({"key": "value"}));
        assert_eq!(trajectory.metadata.model, "test-model");
        assert_eq!(trajectory.metadata.total_duration_ms, 1000);
        assert_eq!(trajectory.metadata.final_result, Some(json!("Answer")));
    }

    #[test]
    fn test_builder_agent_config_from() {
        #[derive(Serialize)]
        struct Config {
            temperature: f64,
        }

        let config = Config { temperature: 0.5 };
        let trajectory = TrajectoryBuilder::default()
            .agent_config_from(&config)
            .build(vec![], vec![], 0, None);

        assert_eq!(trajectory.agent_config, json!({"temperature": 0.5}));
    }

    #[test]
    fn test_builder_calculates_tokens() {
        let step_with_tokens = TrajectoryStep {
            phase: "phase1".to_string(),
            request: sample_request(),
            response: LlmResponseData::Buffered(json!({
                "outputs": [],
                "usage": {"total_tokens": 50}
            })),
            duration_ms: 10,
            started_at: SystemTime::now(),
        };

        let step_without_tokens = TrajectoryStep {
            phase: "phase2".to_string(),
            request: sample_request(),
            response: LlmResponseData::Buffered(json!({"outputs": []})),
            duration_ms: 10,
            started_at: SystemTime::now(),
        };

        let streaming_step = TrajectoryStep {
            phase: "phase3".to_string(),
            request: sample_request(),
            response: LlmResponseData::Streaming(vec![SerializableStreamChunk {
                text: "test".to_string(),
                offset_ms: 0,
            }]),
            duration_ms: 10,
            started_at: SystemTime::now(),
        };

        let trajectory = TrajectoryBuilder::default().build(
            vec![step_with_tokens, step_without_tokens, streaming_step],
            vec![],
            0,
            None,
        );

        assert_eq!(trajectory.metadata.total_tokens, 50);
        // 2 unavailable: one buffered without usage, one streaming
        assert_eq!(trajectory.metadata.tokens_unavailable_count, 2);
    }

    #[test]
    fn test_builder_generates_uuid() {
        let t1 = TrajectoryBuilder::default().build(vec![], vec![], 0, None);
        let t2 = TrajectoryBuilder::default().build(vec![], vec![], 0, None);

        assert_ne!(t1.id, t2.id);
        // UUID v4 format: 8-4-4-4-12
        assert_eq!(t1.id.len(), 36);
    }

    #[test]
    fn test_builder_sets_schema_version() {
        let trajectory = TrajectoryBuilder::default().build(vec![], vec![], 0, None);
        assert_eq!(trajectory.metadata.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn test_builder_preserves_events() {
        let events = vec![
            AgentUpdate::custom("start", "Starting", json!({})),
            AgentUpdate::final_result(json!("Done"), ResultMetadata::new(10, 0, 100)),
        ];

        let trajectory =
            TrajectoryBuilder::default().build(vec![sample_step()], events.clone(), 100, None);

        assert_eq!(trajectory.events.len(), 2);
        assert_eq!(trajectory.events[0].event_type, "start");
        assert_eq!(trajectory.events[1].event_type, "final_result");
    }
}
