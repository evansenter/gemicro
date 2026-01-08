//! Task tool for spawning subagents.
//!
//! This tool allows agents to delegate subtasks to other registered agents
//! or define ephemeral prompt-based agents inline.
//!
//! ## Features
//!
//! - **Named agents**: Reference pre-registered agents by name
//! - **Ephemeral agents**: Define prompt-based agents inline via `PromptAgentDef`
//! - **Execution tracking**: Parent-child relationships via `ExecutionContext`
//! - **Resource isolation**: Control subagent tools/timeout via `SubagentConfig`

use async_trait::async_trait;
use futures_util::StreamExt;
use gemicro_core::agent::{
    AgentSpec, ExecutionContext, OrchestrationGuard, OrchestrationState, PromptAgentDef,
    SubagentConfig,
};
use gemicro_core::tool::{Tool, ToolError, ToolResult};
use gemicro_core::{Agent, AgentContext, LlmClient, ToolSet, EVENT_FINAL_RESULT};
use gemicro_runner::AgentRegistry;
use serde_json::{json, Value};
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Default maximum subagent depth to prevent infinite recursion.
pub const DEFAULT_MAX_DEPTH: usize = 3;

/// Task tool for spawning subagents.
///
/// Allows an agent to delegate subtasks to other registered agents or
/// define ephemeral prompt-based agents inline.
///
/// # Example
///
/// ```no_run
/// use gemicro_task::Task;
/// use gemicro_core::{LlmClient, LlmConfig, tool::Tool};
/// use gemicro_runner::AgentRegistry;
/// use serde_json::json;
/// use std::sync::{Arc, RwLock};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let genai_client = rust_genai::Client::builder("api-key".to_string()).build()?;
/// let llm = LlmClient::new(genai_client, LlmConfig::default());
/// let registry = Arc::new(RwLock::new(AgentRegistry::new()));
/// let task = Task::new(registry, Arc::new(llm));
///
/// // Named agent
/// let result = task.execute(json!({
///     "agent": "simple_qa",
///     "query": "What is 2+2?"
/// })).await?;
///
/// // Ephemeral prompt agent
/// let result = task.execute(json!({
///     "agent": {
///         "type": "prompt",
///         "description": "Math helper",
///         "system_prompt": "You are a math tutor."
///     },
///     "query": "Explain fractions"
/// })).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Task {
    /// Agent registry wrapped in RwLock for shared access.
    ///
    /// RwLock allows the registry to be mutated by its owner (e.g., Session on reload)
    /// while Task holds a read-only view for agent lookups.
    registry: Arc<RwLock<AgentRegistry>>,
    llm: Arc<LlmClient>,
    /// Parent execution context for tracking (if available)
    parent_context: Option<ExecutionContext>,
    /// Maximum subagent nesting depth (fallback if no orchestration)
    max_depth: usize,
    /// Orchestration state for concurrency control (if available)
    orchestration: Option<Arc<OrchestrationState>>,
}

impl std::fmt::Debug for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let agent_count = self.registry.read().map(|r| r.len()).unwrap_or(0);
        f.debug_struct("Task")
            .field("registry", &format!("{} agents", agent_count))
            .field("llm", &"LlmClient")
            .field(
                "parent_depth",
                &self.parent_context.as_ref().map(|c| c.depth),
            )
            .finish()
    }
}

impl Task {
    /// Create a new Task tool with the given registry and LLM client.
    ///
    /// The registry is wrapped in `RwLock` to allow shared access: the owner
    /// (e.g., CLI Session) can modify it on reload, while Task holds a read view.
    pub fn new(registry: Arc<RwLock<AgentRegistry>>, llm: Arc<LlmClient>) -> Self {
        Self {
            registry,
            llm,
            parent_context: None,
            max_depth: DEFAULT_MAX_DEPTH,
            orchestration: None,
        }
    }

    /// Set the parent execution context for subagent tracking.
    #[must_use]
    pub fn with_parent_context(mut self, context: ExecutionContext) -> Self {
        self.parent_context = Some(context);
        self
    }

    /// Set the maximum nesting depth for subagents.
    ///
    /// Note: If orchestration is set, the orchestration's max_depth takes precedence.
    /// This is a fallback for when orchestration is not configured.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set orchestration state for concurrency control.
    ///
    /// When orchestration is set, the Task tool will:
    /// - Use orchestration's max_depth instead of the local max_depth
    /// - Acquire permits before spawning subagents
    /// - Respect the total timeout budget
    #[must_use]
    pub fn with_orchestration(mut self, orchestration: Arc<OrchestrationState>) -> Self {
        self.orchestration = Some(orchestration);
        self
    }

    /// List available agents in the registry.
    pub fn available_agents(&self) -> Vec<String> {
        self.registry
            .read()
            .map(|r| r.list().iter().map(|s| s.to_string()).collect())
            .unwrap_or_default()
    }
}

/// Parse agent specification from JSON.
///
/// Supports two formats:
/// - String: `"simple_qa"` → `AgentSpec::Named("simple_qa")`
/// - Object: `{ "type": "prompt", "description": "...", "system_prompt": "..." }`
fn parse_agent_spec(value: &Value) -> Result<AgentSpec, ToolError> {
    match value {
        Value::String(name) => Ok(AgentSpec::Named(name.clone())),
        Value::Object(obj) => {
            let type_field = obj.get("type").and_then(|v| v.as_str()).unwrap_or("named");

            match type_field {
                "named" => {
                    let name = obj.get("name").and_then(|v| v.as_str()).ok_or_else(|| {
                        ToolError::InvalidInput("Agent type 'named' requires 'name' field".into())
                    })?;
                    Ok(AgentSpec::Named(name.to_string()))
                }
                "prompt" => {
                    let description = obj
                        .get("description")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Ephemeral agent");

                    let system_prompt = obj
                        .get("system_prompt")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| {
                            ToolError::InvalidInput(
                                "Prompt agent requires 'system_prompt' field".into(),
                            )
                        })?;

                    let tools = parse_tool_set(obj.get("tools"))?;
                    let model = obj.get("model").and_then(|v| v.as_str()).map(String::from);

                    // Use builder pattern for non-exhaustive struct
                    let mut def = PromptAgentDef::new(description)
                        .with_system_prompt(system_prompt)
                        .with_tools(tools);

                    if let Some(m) = model {
                        def = def.with_model(m);
                    }

                    Ok(AgentSpec::Prompt(def))
                }
                other => Err(ToolError::InvalidInput(format!(
                    "Unknown agent type: '{}'. Expected 'named' or 'prompt'",
                    other
                ))),
            }
        }
        _ => Err(ToolError::InvalidInput(
            "Agent must be a string (agent name) or object (agent spec)".into(),
        )),
    }
}

/// Parse tool set from JSON.
fn parse_tool_set(value: Option<&Value>) -> Result<ToolSet, ToolError> {
    match value {
        None => Ok(ToolSet::Inherit),
        Some(Value::String(s)) => match s.as_str() {
            "all" => Ok(ToolSet::All),
            "none" => Ok(ToolSet::None),
            "inherit" => Ok(ToolSet::Inherit),
            _ => Err(ToolError::InvalidInput(format!(
                "Unknown tool set string: '{}'. Expected 'all', 'none', or 'inherit'",
                s
            ))),
        },
        Some(Value::Array(arr)) => {
            let tools: Result<Vec<String>, _> = arr
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(String::from)
                        .ok_or_else(|| ToolError::InvalidInput("Tool names must be strings".into()))
                })
                .collect();
            Ok(ToolSet::Specific(tools?))
        }
        Some(_) => Err(ToolError::InvalidInput(
            "Tools must be a string ('all', 'none', 'inherit') or array of tool names".into(),
        )),
    }
}

/// Parse subagent config from JSON.
fn parse_subagent_config(value: Option<&Value>) -> Result<SubagentConfig, ToolError> {
    let config = SubagentConfig::default();

    let Some(obj) = value.and_then(|v| v.as_object()) else {
        return Ok(config);
    };

    let mut config = config;

    if let Some(tools) = obj.get("tools") {
        config = config.with_tools(parse_tool_set(Some(tools))?);
    }

    if let Some(inherit_hooks) = obj.get("inherit_hooks").and_then(|v| v.as_bool()) {
        config = config.with_inherit_hooks(inherit_hooks);
    }

    if let Some(allow_nested) = obj.get("allow_nested").and_then(|v| v.as_bool()) {
        config = config.with_allow_nested(allow_nested);
    }

    if let Some(timeout_secs) = obj.get("timeout_secs").and_then(|v| v.as_u64()) {
        config = config.with_timeout(Duration::from_secs(timeout_secs));
    }

    Ok(config)
}

#[async_trait]
impl Tool for Task {
    fn name(&self) -> &str {
        "task"
    }

    fn description(&self) -> &str {
        "Spawn a subagent to handle a specific task. Supports named agents \
         or inline prompt-based agents. Returns the subagent's final result."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "agent": {
                    "oneOf": [
                        {
                            "type": "string",
                            "description": "Name of a registered agent (e.g., 'simple_qa', 'deep_research')"
                        },
                        {
                            "type": "object",
                            "description": "Inline agent definition",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["named", "prompt"],
                                    "description": "Agent type"
                                },
                                "name": {
                                    "type": "string",
                                    "description": "For type='named': agent name"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "For type='prompt': human-readable description"
                                },
                                "system_prompt": {
                                    "type": "string",
                                    "description": "For type='prompt': system prompt for the agent"
                                },
                                "tools": {
                                    "description": "Tool access: 'all', 'none', 'inherit', or array of tool names"
                                },
                                "model": {
                                    "type": "string",
                                    "description": "Optional model override (e.g., 'gemini-1.5-flash')"
                                }
                            }
                        }
                    ]
                },
                "query": {
                    "type": "string",
                    "description": "The query or task to send to the subagent"
                },
                "config": {
                    "type": "object",
                    "description": "Optional subagent configuration",
                    "properties": {
                        "tools": {
                            "description": "Tool access control"
                        },
                        "inherit_hooks": {
                            "type": "boolean",
                            "description": "Whether to inherit parent's hooks (default: true)"
                        },
                        "allow_nested": {
                            "type": "boolean",
                            "description": "Whether subagent can spawn its own subagents (default: true)"
                        },
                        "timeout_secs": {
                            "type": "integer",
                            "description": "Execution timeout in seconds (default: 60)"
                        }
                    }
                }
            },
            "required": ["agent", "query"]
        })
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        // Parse agent specification
        let agent_value = input
            .get("agent")
            .ok_or_else(|| ToolError::InvalidInput("Missing 'agent' field".into()))?;
        let agent_spec = parse_agent_spec(agent_value)?;

        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'query' field".into()))?;

        // Parse config
        let config = parse_subagent_config(input.get("config"))?;

        // Create child execution context
        let parent_context = self
            .parent_context
            .clone()
            .unwrap_or_else(ExecutionContext::root);
        let child_context = parent_context.child(agent_spec.name());

        // Check depth limit - use orchestration's limit if available, else fallback
        let max_depth = self
            .orchestration
            .as_ref()
            .map(|o| o.config().max_depth)
            .unwrap_or(self.max_depth);

        if child_context.depth >= max_depth {
            return Err(ToolError::ExecutionFailed(format!(
                "Max subagent depth ({}) exceeded at depth {}. Path: {}",
                max_depth,
                child_context.depth,
                child_context.path_string()
            )));
        }

        // Check if nesting is allowed
        // Note: We check depth > 1 because allow_nested controls whether THIS subagent
        // (at depth 1) can spawn further subagents (depth 2+). A root agent (depth 0)
        // can always spawn its first level of subagents regardless of this setting.
        if !config.can_spawn_subagents() && child_context.depth > 1 {
            return Err(ToolError::ExecutionFailed(
                "Subagent nesting is disabled by configuration".into(),
            ));
        }

        // Acquire orchestration permits if orchestration is configured.
        // We use the parent's execution_id to key the per-parent semaphore, which controls
        // how many siblings can run concurrently under the same parent. Falls back to
        // child's own ID at root level (no parent).
        let _guard: Option<OrchestrationGuard> = if let Some(ref orch) = self.orchestration {
            let parent_for_permit = child_context
                .parent_id
                .as_ref()
                .unwrap_or(&child_context.execution_id);
            Some(orch.acquire_permits(parent_for_permit).await.map_err(|e| {
                ToolError::ExecutionFailed(format!(
                    "Failed to acquire orchestration permits for subagent '{}': {}",
                    agent_spec.name(),
                    e
                ))
            })?)
        } else {
            None
        };
        // Guard is held until end of function, releasing permits on drop

        // Capture metadata before consuming the context
        let execution_metadata = json!({
            "execution_id": child_context.execution_id.as_str(),
            "parent_id": child_context.parent_id.as_ref().map(|id| id.as_str()),
            "depth": child_context.depth,
        });

        // Determine timeout: use orchestration's remaining time if available,
        // otherwise fall back to config timeout
        let timeout = if let Some(ref orch) = self.orchestration {
            match orch.remaining_time() {
                Some(remaining) => remaining.min(config.timeout),
                None => {
                    return Err(ToolError::ExecutionFailed(format!(
                        "Orchestration total timeout already exceeded before spawning subagent '{}'",
                        agent_spec.name()
                    )));
                }
            }
        } else {
            config.timeout
        };

        // Execute based on agent type
        let final_answer = match &agent_spec {
            AgentSpec::Named(name) => {
                self.execute_named_agent(name, query, child_context, &config, timeout)
                    .await?
            }
            AgentSpec::Prompt(def) => {
                self.execute_prompt_agent(def, query, child_context, &config, timeout)
                    .await?
            }
            // Forward compatibility: handle unknown AgentSpec variants.
            // AgentSpec is #[non_exhaustive], so new variants may be added.
            #[allow(unreachable_patterns)]
            _ => {
                return Err(ToolError::InvalidInput(format!(
                    "Unsupported agent specification type for agent '{}'",
                    agent_spec.name()
                )));
            }
        };

        Ok(ToolResult::text(final_answer).with_metadata(execution_metadata))
    }
}

impl Task {
    /// Execute a named (pre-registered) agent.
    async fn execute_named_agent(
        &self,
        name: &str,
        query: &str,
        execution: ExecutionContext,
        _config: &SubagentConfig,
        timeout: Duration,
    ) -> Result<String, ToolError> {
        // Get the agent from registry (acquire read lock in a block to drop before await)
        let agent = {
            let registry = self
                .registry
                .read()
                .map_err(|_| ToolError::ExecutionFailed("Agent registry lock poisoned".into()))?;
            registry.get(name).ok_or_else(|| {
                let available: Vec<_> = registry.list();
                ToolError::NotFound(format!(
                    "Agent '{}' not found. Available: {:?}",
                    name, available
                ))
            })?
        };
        // Guard is dropped here at end of block, before any .await

        // Create context for the subagent with execution tracking and orchestration
        let mut context = AgentContext::from_arc(Arc::clone(&self.llm)).with_execution(execution);
        if let Some(ref orch) = self.orchestration {
            context = context.with_orchestration(Arc::clone(orch));
        }

        // Execute the agent with timeout enforcement
        tokio::time::timeout(
            timeout,
            self.collect_final_result(agent.execute(query, context), name),
        )
        .await
        .map_err(|_| {
            ToolError::ExecutionFailed(format!("Subagent '{}' timed out after {:?}", name, timeout))
        })?
    }

    /// Execute an ephemeral prompt-based agent.
    async fn execute_prompt_agent(
        &self,
        def: &PromptAgentDef,
        query: &str,
        execution: ExecutionContext,
        _config: &SubagentConfig,
        timeout: Duration,
    ) -> Result<String, ToolError> {
        // Validate the definition
        def.validate().map_err(|e| {
            ToolError::InvalidInput(format!("Invalid prompt agent definition: {}", e))
        })?;

        // Create the ephemeral agent using PromptAgent
        // Note: For now we use PromptAgent with the custom system prompt.
        // In the future, this could support different base agent types.
        let agent = gemicro_prompt_agent::PromptAgent::with_system_prompt(&def.system_prompt)
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to create agent: {}", e)))?;

        // Create context for the subagent with execution tracking and orchestration
        let mut context = AgentContext::from_arc(Arc::clone(&self.llm)).with_execution(execution);
        if let Some(ref orch) = self.orchestration {
            context = context.with_orchestration(Arc::clone(orch));
        }

        // Execute the agent with timeout enforcement
        tokio::time::timeout(
            timeout,
            self.collect_final_result(agent.execute(query, context), &def.description),
        )
        .await
        .map_err(|_| {
            ToolError::ExecutionFailed(format!(
                "Subagent '{}' timed out after {:?}",
                &def.description, timeout
            ))
        })?
    }

    /// Collect the final result from an agent stream.
    async fn collect_final_result(
        &self,
        mut stream: gemicro_core::AgentStream<'_>,
        agent_name: &str,
    ) -> Result<String, ToolError> {
        let mut final_answer: Option<String> = None;

        while let Some(result) = stream.next().await {
            match result {
                Ok(update) => {
                    if update.event_type == EVENT_FINAL_RESULT {
                        // Extract the result from final_result event
                        if let Some(result_val) = update.data.get("result") {
                            // Convert result to string: strings directly, others as JSON
                            final_answer = Some(match result_val {
                                Value::String(s) => s.clone(),
                                Value::Null => String::new(),
                                other => serde_json::to_string(other).unwrap_or_default(),
                            });
                        } else {
                            // Fallback to message if result field isn't present
                            log::debug!(
                                "final_result event for subagent '{}' missing 'result' field, falling back to message",
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

        final_answer.ok_or_else(|| {
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
                    json!(answer),
                    ResultMetadata::new(10, 0, 100),
                );
            })
        }

        fn create_tracker(&self) -> Box<dyn ExecutionTracking> {
            Box::new(DefaultTracker::default())
        }
    }

    fn create_test_registry() -> Arc<RwLock<AgentRegistry>> {
        let mut registry = AgentRegistry::new();
        registry.register("mock_agent", || {
            Box::new(MockAgent {
                name: "mock_agent".to_string(),
                answer: "Mock answer".to_string(),
            })
        });
        Arc::new(RwLock::new(registry))
    }

    fn create_test_llm() -> Arc<LlmClient> {
        let genai_client = rust_genai::Client::builder("test-key".to_string())
            .build()
            .unwrap();
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
        assert!(agents.contains(&"mock_agent".to_string()));
    }

    #[test]
    fn test_task_debug() {
        let registry = create_test_registry();
        let task = Task::new(registry, create_test_llm());

        let debug = format!("{:?}", task);
        assert!(debug.contains("Task"));
        assert!(debug.contains("1 agents"));
    }

    #[test]
    fn test_parse_agent_spec_string() {
        let spec = parse_agent_spec(&json!("simple_qa")).unwrap();
        assert!(matches!(spec, AgentSpec::Named(name) if name == "simple_qa"));
    }

    #[test]
    fn test_parse_agent_spec_named_object() {
        let spec = parse_agent_spec(&json!({
            "type": "named",
            "name": "deep_research"
        }))
        .unwrap();
        assert!(matches!(spec, AgentSpec::Named(name) if name == "deep_research"));
    }

    #[test]
    fn test_parse_agent_spec_prompt_object() {
        let spec = parse_agent_spec(&json!({
            "type": "prompt",
            "description": "Code reviewer",
            "system_prompt": "Review code for bugs."
        }))
        .unwrap();

        if let AgentSpec::Prompt(def) = spec {
            assert_eq!(def.description, "Code reviewer");
            assert_eq!(def.system_prompt, "Review code for bugs.");
            assert!(matches!(def.tools, ToolSet::Inherit));
            assert!(def.model.is_none());
        } else {
            panic!("Expected AgentSpec::Prompt");
        }
    }

    #[test]
    fn test_parse_agent_spec_prompt_with_tools() {
        let spec = parse_agent_spec(&json!({
            "type": "prompt",
            "description": "Test",
            "system_prompt": "Test prompt",
            "tools": ["file_read", "grep"],
            "model": "gemini-1.5-flash"
        }))
        .unwrap();

        if let AgentSpec::Prompt(def) = spec {
            assert!(matches!(def.tools, ToolSet::Specific(ref tools) if tools.len() == 2));
            assert_eq!(def.model, Some("gemini-1.5-flash".to_string()));
        } else {
            panic!("Expected AgentSpec::Prompt");
        }
    }

    #[test]
    fn test_parse_agent_spec_missing_system_prompt() {
        let result = parse_agent_spec(&json!({
            "type": "prompt",
            "description": "Test"
        }));
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_tool_set() {
        assert!(matches!(parse_tool_set(None).unwrap(), ToolSet::Inherit));
        assert!(matches!(
            parse_tool_set(Some(&json!("all"))).unwrap(),
            ToolSet::All
        ));
        assert!(matches!(
            parse_tool_set(Some(&json!("none"))).unwrap(),
            ToolSet::None
        ));
        assert!(matches!(
            parse_tool_set(Some(&json!(["a", "b"]))).unwrap(),
            ToolSet::Specific(_)
        ));
    }

    #[test]
    fn test_parse_subagent_config() {
        let config = parse_subagent_config(Some(&json!({
            "timeout_secs": 120,
            "allow_nested": false,
            "inherit_hooks": false
        })))
        .unwrap();

        assert_eq!(config.timeout, Duration::from_secs(120));
        assert!(!config.allow_nested);
        assert!(!config.inherit_hooks);
    }

    #[test]
    fn test_task_with_parent_context() {
        let registry = create_test_registry();
        let parent = ExecutionContext::root();
        let task = Task::new(registry, create_test_llm()).with_parent_context(parent.clone());

        assert!(task.parent_context.is_some());
        assert_eq!(task.parent_context.unwrap().depth, 0);
    }

    #[test]
    fn test_task_with_max_depth() {
        let registry = create_test_registry();
        let task = Task::new(registry, create_test_llm()).with_max_depth(5);

        assert_eq!(task.max_depth, 5);
    }

    // Integration test - requires actual agent registry setup
    #[tokio::test]
    #[ignore]
    async fn test_task_execute_real_agent() {
        // This would test with a real agent, but requires GEMINI_API_KEY
        // and proper agent registration
    }
}
