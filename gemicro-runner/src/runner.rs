//! Headless agent execution runner.
//!
//! Provides `AgentRunner` for executing agents without terminal dependencies,
//! returning structured `ExecutionMetrics` for programmatic consumption.

use crate::metrics::ExecutionMetrics;
use futures_util::StreamExt;
use gemicro_core::{
    enforce_final_result_contract, Agent, AgentContext, AgentError, ExecutionTracking,
};
use std::time::Instant;

/// Headless agent runner for programmatic execution.
///
/// Use this for:
/// - Executing agents without terminal rendering
/// - Building custom display backends
/// - Testing agent behavior
/// - Batch evaluation/benchmarking
///
/// # Example
///
/// ```text
/// // Requires an agent crate like gemicro-deep-research
/// use gemicro_runner::AgentRunner;
/// use gemicro_core::{AgentContext, LlmClient, LlmConfig};
/// use gemicro_deep_research::{DeepResearchAgent, ResearchConfig};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let runner = AgentRunner::new();
/// let agent = DeepResearchAgent::new(ResearchConfig::default())?;
///
/// let genai_client = rust_genai::Client::builder("api-key".to_string()).build();
/// let llm = LlmClient::new(genai_client, LlmConfig::default());
/// let context = AgentContext::new(llm);
///
/// let metrics = runner.execute_with_tracking(
///     &agent,
///     "What is Rust?",
///     context,
///     |_tracker, status| println!("Status: {}", status),
/// ).await?;
/// println!("Completed in {:?}", metrics.total_duration);
/// println!("Tokens used: {}", metrics.total_tokens);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AgentRunner;

impl AgentRunner {
    /// Create a new headless runner.
    pub fn new() -> Self {
        Self
    }

    /// Execute an agent and return final metrics.
    ///
    /// This is the preferred method for new code. It uses the agent's
    /// `create_tracker()` method instead of requiring external state handlers.
    ///
    /// # Arguments
    ///
    /// * `agent` - The agent to execute
    /// * `query` - The user's query
    /// * `context` - Agent context with LLM client
    /// * `on_status` - Callback receiving status message updates
    ///
    /// # Example
    ///
    /// ```text
    /// use gemicro_runner::AgentRunner;
    /// use gemicro_core::{Agent, AgentContext};
    ///
    /// async fn example(agent: &dyn Agent, context: AgentContext) -> Result<(), Box<dyn std::error::Error>> {
    ///     let runner = AgentRunner::new();
    ///
    ///     let metrics = runner.execute_with_tracking(
    ///         agent,
    ///         "query",
    ///         context,
    ///         |tracker, status| {
    ///             println!("Status: {}", status);
    ///         },
    ///     ).await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn execute_with_tracking<F>(
        &self,
        agent: &dyn Agent,
        query: &str,
        context: AgentContext,
        mut on_status: F,
    ) -> Result<ExecutionMetrics, AgentError>
    where
        F: FnMut(&dyn ExecutionTracking, &str),
    {
        let mut tracker = agent.create_tracker();
        let stream = agent.execute(query, context);
        // Wrap stream with contract enforcement to detect violations
        let stream = enforce_final_result_contract(stream);
        futures_util::pin_mut!(stream);
        let start = Instant::now();

        while let Some(result) = stream.next().await {
            let update = result?;
            tracker.handle_event(&update);
            if let Some(msg) = tracker.status_message() {
                on_status(tracker.as_ref(), msg);
            }
        }

        Ok(ExecutionMetrics::from_tracker(
            tracker.as_ref(),
            start.elapsed(),
        ))
    }
}

impl Default for AgentRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_stream::stream;
    use gemicro_core::{AgentStream, AgentUpdate, ResultMetadata};
    use std::sync::Arc;

    struct MockAgent {
        events: Vec<AgentUpdate>,
    }

    impl MockAgent {
        fn new(events: Vec<AgentUpdate>) -> Self {
            Self { events }
        }
    }

    impl Agent for MockAgent {
        fn name(&self) -> &str {
            "mock_agent"
        }

        fn description(&self) -> &str {
            "Mock agent for testing"
        }

        fn execute(&self, _query: &str, _context: AgentContext) -> AgentStream<'_> {
            let events = self.events.clone();
            Box::pin(stream! {
                for event in events {
                    yield Ok(event);
                }
            })
        }

        fn create_tracker(&self) -> Box<dyn gemicro_core::ExecutionTracking> {
            Box::new(gemicro_core::DefaultTracker::default())
        }
    }

    fn create_mock_context() -> AgentContext {
        // Create a minimal context for testing
        // Note: In real tests, you'd use a proper test helper
        use gemicro_core::{LlmClient, LlmConfig};

        let genai_client = rust_genai::Client::builder("test-key".to_string()).build();
        let llm = LlmClient::new(genai_client, LlmConfig::default());
        AgentContext::new(llm)
    }

    fn create_successful_events() -> Vec<AgentUpdate> {
        use serde_json::json;

        let metadata = ResultMetadata::new(100, 0, 1000);

        vec![
            AgentUpdate::custom("decomposition_started", "Decomposing query", json!({})),
            AgentUpdate::custom(
                "decomposition_complete",
                "Decomposed into 1 sub-query",
                json!({ "sub_queries": ["Q1"] }),
            ),
            AgentUpdate::custom(
                "sub_query_started",
                "Sub-query 0 started",
                json!({ "id": 0, "query": "Q1" }),
            ),
            AgentUpdate::custom(
                "sub_query_completed",
                "Sub-query 0 completed",
                json!({ "id": 0, "result": "Result", "tokens_used": 50 }),
            ),
            AgentUpdate::custom("synthesis_started", "Synthesizing results", json!({})),
            AgentUpdate::final_result("Final answer".to_string(), metadata),
        ]
    }

    #[tokio::test]
    async fn test_runner_default() {
        let runner = AgentRunner;
        assert!(std::mem::size_of_val(&runner) == 0); // Zero-size struct
    }

    #[tokio::test]
    async fn test_runner_execute_with_tracking() {
        let runner = AgentRunner::new();
        let agent = MockAgent::new(create_successful_events());
        let context = create_mock_context();

        let status_messages = Arc::new(std::sync::Mutex::new(Vec::new()));
        let messages_clone = status_messages.clone();

        let metrics = runner
            .execute_with_tracking(&agent, "query", context, move |tracker, msg| {
                messages_clone.lock().unwrap().push(msg.to_string());
                // Verify tracker state is accessible during callback
                if tracker.is_complete() {
                    assert!(tracker.final_result().is_some());
                }
            })
            .await
            .unwrap();

        let messages = status_messages.lock().unwrap();
        // Should have received status updates (one per event with non-empty message)
        assert!(!messages.is_empty());
        // Metrics should reflect final result
        assert_eq!(metrics.total_tokens, 100);
        assert_eq!(metrics.final_answer, Some("Final answer".to_string()));
    }
}
