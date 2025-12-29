//! Task tool for spawning subagents.
//!
//! This tool allows agents to delegate subtasks to other registered agents,
//! enabling hierarchical agent composition and specialization.

use async_trait::async_trait;
use futures_util::StreamExt;
use gemicro_core::tool::{Tool, ToolError, ToolResult};
use gemicro_core::{AgentContext, LlmClient, EVENT_FINAL_RESULT};
use gemicro_runner::AgentRegistry;
use serde_json::{json, Value};
use std::sync::Arc;

/// Task tool for spawning subagents.
///
/// Allows an agent to delegate subtasks to other registered agents.
/// The tool spawns the target agent, collects its stream output, and
/// returns the final result.
///
/// # Example
///
/// ```no_run
/// use gemicro_task::Task;
/// use gemicro_core::{LlmClient, LlmConfig, tool::Tool};
/// use gemicro_runner::AgentRegistry;
/// use serde_json::json;
/// use std::sync::Arc;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let genai_client = rust_genai::Client::builder("api-key".to_string()).build();
/// let llm = LlmClient::new(genai_client, LlmConfig::default());
/// let registry = Arc::new(AgentRegistry::new());
/// let task = Task::new(registry, Arc::new(llm));
///
/// let result = task.execute(json!({
///     "agent": "simple_qa",
///     "query": "What is 2+2?"
/// })).await?;
/// println!("Subagent result: {}", result.content);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Task {
    registry: Arc<AgentRegistry>,
    llm: Arc<LlmClient>,
}

impl std::fmt::Debug for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Task")
            .field("registry", &format!("{} agents", self.registry.len()))
            .field("llm", &"LlmClient")
            .finish()
    }
}

impl Task {
    /// Create a new Task tool with the given registry and LLM client.
    pub fn new(registry: Arc<AgentRegistry>, llm: Arc<LlmClient>) -> Self {
        Self { registry, llm }
    }

    /// List available agents in the registry.
    pub fn available_agents(&self) -> Vec<&str> {
        self.registry.list()
    }
}

#[async_trait]
impl Tool for Task {
    fn name(&self) -> &str {
        "task"
    }

    fn description(&self) -> &str {
        "Spawn a subagent to handle a specific task. Returns the subagent's \
         final result. Use this to delegate specialized work to other agents."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Name of the agent to spawn (e.g., 'simple_qa', 'deep_research')"
                },
                "query": {
                    "type": "string",
                    "description": "The query or task to send to the subagent"
                }
            },
            "required": ["agent", "query"]
        })
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let agent_name = input
            .get("agent")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'agent' field".into()))?;

        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'query' field".into()))?;

        // Get the agent from registry
        let agent = self.registry.get(agent_name).ok_or_else(|| {
            let available: Vec<_> = self.registry.list();
            ToolError::NotFound(format!(
                "Agent '{}' not found. Available: {:?}",
                agent_name, available
            ))
        })?;

        // Create context for the subagent
        let context = AgentContext::from_arc(Arc::clone(&self.llm));

        // Execute the agent and collect results
        let mut stream = agent.execute(query, context);
        let mut final_answer: Option<String> = None;

        while let Some(result) = stream.next().await {
            match result {
                Ok(update) => {
                    if update.event_type == EVENT_FINAL_RESULT {
                        // Extract the answer from final_result event
                        if let Some(answer) = update.data.get("answer").and_then(|v| v.as_str()) {
                            final_answer = Some(answer.to_string());
                        } else {
                            // Fallback to message if answer field isn't present
                            log::debug!(
                                "final_result event for subagent '{}' missing 'answer' field, falling back to message",
                                agent_name
                            );
                            final_answer = Some(update.message.clone());
                        }
                    }
                }
                Err(e) => {
                    return Err(ToolError::ExecutionFailed(format!(
                        "Subagent '{}' failed: {}",
                        agent_name, e
                    )));
                }
            }
        }

        final_answer.map(ToolResult::new).ok_or_else(|| {
            ToolError::ExecutionFailed(format!(
                "Subagent '{}' completed without producing a final_result",
                agent_name
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_stream::try_stream;
    use gemicro_core::update::ResultMetadata;
    use gemicro_core::{Agent, AgentStream, AgentUpdate, DefaultTracker, ExecutionTracking};

    // Mock agent for testing
    struct MockAgent {
        name: String,
        answer: String,
    }

    impl Agent for MockAgent {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "A mock agent for testing"
        }

        fn execute(&self, _query: &str, _context: AgentContext) -> AgentStream<'_> {
            let answer = self.answer.clone();
            Box::pin(try_stream! {
                yield AgentUpdate::custom("mock_started", "Starting mock", json!({}));
                yield AgentUpdate::final_result(
                    answer,
                    ResultMetadata::new(10, 0, 100),
                );
            })
        }

        fn create_tracker(&self) -> Box<dyn ExecutionTracking> {
            Box::new(DefaultTracker::default())
        }
    }

    fn create_test_registry() -> Arc<AgentRegistry> {
        let mut registry = AgentRegistry::new();
        registry.register("mock_agent", || {
            Box::new(MockAgent {
                name: "mock_agent".to_string(),
                answer: "Mock answer".to_string(),
            })
        });
        Arc::new(registry)
    }

    fn create_test_llm() -> Arc<LlmClient> {
        let genai_client = rust_genai::Client::builder("test-key".to_string()).build();
        Arc::new(LlmClient::new(
            genai_client,
            gemicro_core::LlmConfig::default(),
        ))
    }

    #[test]
    fn test_task_name_and_description() {
        let registry = create_test_registry();
        let task = Task::new(registry, create_test_llm());

        assert_eq!(task.name(), "task");
        assert!(!task.description().is_empty());
    }

    #[test]
    fn test_task_parameters_schema() {
        let registry = create_test_registry();
        let task = Task::new(registry, create_test_llm());

        let schema = task.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["agent"].is_object());
        assert!(schema["properties"]["query"].is_object());

        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("agent")));
        assert!(required.contains(&json!("query")));
    }

    #[tokio::test]
    async fn test_task_missing_agent() {
        let registry = create_test_registry();
        let task = Task::new(registry, create_test_llm());

        let result = task.execute(json!({"query": "test"})).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_task_missing_query() {
        let registry = create_test_registry();
        let task = Task::new(registry, create_test_llm());

        let result = task.execute(json!({"agent": "mock_agent"})).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_task_agent_not_found() {
        let registry = create_test_registry();
        let task = Task::new(registry, create_test_llm());

        let result = task
            .execute(json!({
                "agent": "nonexistent_agent",
                "query": "test"
            }))
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::NotFound(_)));
        if let ToolError::NotFound(msg) = err {
            assert!(msg.contains("nonexistent_agent"));
            assert!(msg.contains("mock_agent")); // Shows available agents
        }
    }

    #[test]
    fn test_task_available_agents() {
        let registry = create_test_registry();
        let task = Task::new(registry, create_test_llm());

        let agents = task.available_agents();
        assert!(agents.contains(&"mock_agent"));
    }

    #[test]
    fn test_task_debug() {
        let registry = create_test_registry();
        let task = Task::new(registry, create_test_llm());

        let debug = format!("{:?}", task);
        assert!(debug.contains("Task"));
        assert!(debug.contains("1 agents"));
    }

    // Integration test - requires actual agent registry setup
    #[tokio::test]
    #[ignore]
    async fn test_task_execute_real_agent() {
        // This would test with a real agent, but requires GEMINI_API_KEY
        // and proper agent registration
    }
}
