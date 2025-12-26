//! Headless agent execution runner.
//!
//! Provides `AgentRunner` for executing agents without terminal dependencies,
//! returning structured `ExecutionMetrics` for programmatic consumption.

use crate::metrics::ExecutionMetrics;
use crate::state::ExecutionState;
use futures_util::StreamExt;
use gemicro_core::{Agent, AgentContext, AgentError};

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
/// ```no_run
/// use gemicro_runner::AgentRunner;
/// use gemicro_core::{DeepResearchAgent, ResearchConfig, AgentContext, LlmClient, LlmConfig};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let runner = AgentRunner::new();
/// let agent = DeepResearchAgent::new(ResearchConfig::default())?;
///
/// let genai_client = rust_genai::Client::builder("api-key".to_string()).build();
/// let llm = LlmClient::new(genai_client, LlmConfig::default());
/// let context = AgentContext::new(llm);
///
/// let metrics = runner.execute(&agent, "What is Rust?", context).await?;
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
    /// Consumes the entire event stream, tracks state internally,
    /// and returns `ExecutionMetrics` when complete.
    ///
    /// # Arguments
    ///
    /// * `agent` - The agent to execute
    /// * `query` - The user's query
    /// * `context` - Agent context with LLM client
    ///
    /// # Returns
    ///
    /// `ExecutionMetrics` containing timing, token usage, and results.
    ///
    /// # Errors
    ///
    /// Returns `AgentError` if the agent fails during execution.
    pub async fn execute(
        &self,
        agent: &dyn Agent,
        query: &str,
        context: AgentContext,
    ) -> Result<ExecutionMetrics, AgentError> {
        self.execute_with_callback(agent, query, context, |_, _| {})
            .await
    }

    /// Execute an agent with a progress callback.
    ///
    /// Like `execute()`, but calls `on_update` for each event,
    /// allowing custom progress tracking or logging.
    ///
    /// # Arguments
    ///
    /// * `agent` - The agent to execute
    /// * `query` - The user's query
    /// * `context` - Agent context with LLM client
    /// * `on_update` - Callback receiving `(state, changed_sub_query_id)`
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gemicro_runner::{AgentRunner, ExecutionState};
    /// # use gemicro_core::{DeepResearchAgent, ResearchConfig, AgentContext};
    ///
    /// # async fn example(agent: &DeepResearchAgent, context: AgentContext) -> Result<(), Box<dyn std::error::Error>> {
    /// let runner = AgentRunner::new();
    ///
    /// let metrics = runner.execute_with_callback(
    ///     agent,
    ///     "query",
    ///     context,
    ///     |state: &ExecutionState, changed_id| {
    ///         println!("Phase: {:?}", state.phase());
    ///         if let Some(id) = changed_id {
    ///             println!("Sub-query {} updated", id);
    ///         }
    ///     },
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn execute_with_callback<F>(
        &self,
        agent: &dyn Agent,
        query: &str,
        context: AgentContext,
        mut on_update: F,
    ) -> Result<ExecutionMetrics, AgentError>
    where
        F: FnMut(&ExecutionState, Option<usize>),
    {
        let mut state = ExecutionState::new();
        let stream = agent.execute(query, context);
        futures_util::pin_mut!(stream);

        while let Some(result) = stream.next().await {
            let update = result?;
            let changed_id = state.update(&update);
            on_update(&state, changed_id);
        }

        Ok(ExecutionMetrics::from(&state))
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
    use std::sync::atomic::{AtomicUsize, Ordering};
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
        let metadata = ResultMetadata {
            total_tokens: 100,
            tokens_unavailable_count: 0,
            duration_ms: 1000,
            sub_queries_succeeded: 1,
            sub_queries_failed: 0,
        };

        vec![
            AgentUpdate::decomposition_started(),
            AgentUpdate::decomposition_complete(vec!["Q1".to_string()]),
            AgentUpdate::sub_query_started(0, "Q1".to_string()),
            AgentUpdate::sub_query_completed(0, "Result".to_string(), 50),
            AgentUpdate::synthesis_started(),
            AgentUpdate::final_result("Final answer".to_string(), metadata),
        ]
    }

    #[tokio::test]
    async fn test_runner_execute_success() {
        let runner = AgentRunner::new();
        let agent = MockAgent::new(create_successful_events());
        let context = create_mock_context();

        let metrics = runner.execute(&agent, "test query", context).await.unwrap();

        assert_eq!(metrics.sub_queries_total, 1);
        assert_eq!(metrics.sub_queries_succeeded, 1);
        assert_eq!(metrics.total_tokens, 100);
        assert_eq!(metrics.final_answer, Some("Final answer".to_string()));
    }

    #[tokio::test]
    async fn test_runner_with_callback() {
        let runner = AgentRunner::new();
        let agent = MockAgent::new(create_successful_events());
        let context = create_mock_context();

        let callback_count = Arc::new(AtomicUsize::new(0));
        let callback_count_clone = callback_count.clone();

        let metrics = runner
            .execute_with_callback(&agent, "query", context, move |_state, _changed_id| {
                callback_count_clone.fetch_add(1, Ordering::SeqCst);
            })
            .await
            .unwrap();

        // Should be called for each event
        assert_eq!(callback_count.load(Ordering::SeqCst), 6);
        assert!(metrics.final_answer.is_some());
    }

    #[tokio::test]
    async fn test_runner_with_failure() {
        let events = vec![
            AgentUpdate::decomposition_started(),
            AgentUpdate::decomposition_complete(vec!["Q1".to_string()]),
            AgentUpdate::sub_query_started(0, "Q1".to_string()),
            AgentUpdate::sub_query_failed(0, "Timeout".to_string()),
        ];

        let runner = AgentRunner::new();
        let agent = MockAgent::new(events);
        let context = create_mock_context();

        let metrics = runner.execute(&agent, "query", context).await.unwrap();

        assert_eq!(metrics.sub_queries_failed, 1);
        assert_eq!(metrics.sub_queries_succeeded, 0);
        assert!(metrics.final_answer.is_none());
    }

    #[tokio::test]
    async fn test_runner_default() {
        let runner = AgentRunner;
        assert!(std::mem::size_of_val(&runner) == 0); // Zero-size struct
    }
}
