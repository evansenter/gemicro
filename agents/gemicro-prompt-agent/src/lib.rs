//! Prompt Agent - Executes prompts with optional tool support.
//!
//! This agent handles prompt-based execution with:
//! - Custom system prompts
//! - Optional tool filtering via `ToolSet`
//! - Timeout and cancellation handling
//! - Streaming execution with `async_stream`
//!
//! # Example
//!
//! ```no_run
//! use gemicro_prompt_agent::{PromptAgent, PromptAgentConfig};
//! use gemicro_core::{Agent, AgentContext, LlmClient, LlmConfig};
//! use futures_util::StreamExt;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let agent = PromptAgent::new(PromptAgentConfig::default())?;
//!
//! let genai = genai_rs::Client::builder("api-key".to_string()).build().map_err(|e| AgentError::Other(e.to_string()))?;
//! let context = AgentContext::new(LlmClient::new(genai, LlmConfig::default()));
//!
//! let stream = agent.execute("What is Rust?", context);
//! futures_util::pin_mut!(stream);
//!
//! while let Some(update) = stream.next().await {
//!     let update = update?;
//!     match update.event_type.as_str() {
//!         "prompt_agent_started" => println!("Processing..."),
//!         "prompt_agent_result" => println!("Answer: {}", update.message),
//!         _ => {} // Ignore unknown events
//!     }
//! }
//! # Ok(())
//! # }
//! ```

pub mod tools;

use gemicro_core::{
    remaining_time, timeout_error, tool::ToolCallableAdapter, truncate,
    with_timeout_and_cancellation, Agent, AgentContext, AgentError, AgentStream, AgentUpdate,
    ResultMetadata, ToolSet,
};

use async_stream::try_stream;
use genai_rs::{
    function_result_content, CallableFunction, FunctionDeclaration, InteractionContent,
};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ============================================================================
// Event Type Constants
// ============================================================================

/// Event emitted when the agent starts processing a query.
const EVENT_PROMPT_AGENT_STARTED: &str = "prompt_agent_started";

/// Event emitted when the agent produces its final result.
const EVENT_PROMPT_AGENT_RESULT: &str = "prompt_agent_result";

/// Event emitted before executing a tool call.
const EVENT_TOOL_CALL_STARTED: &str = "tool_call_started";

/// Event emitted after a tool call completes.
const EVENT_TOOL_RESULT: &str = "tool_result";

/// Maximum function calling continuation rounds to prevent infinite loops.
///
/// This limits how many times the agent can send function results back to the model.
/// The initial LLM call is not counted, so total LLM calls = 1 + MAX_ITERATIONS.
///
/// Note: A time-based limit would be more appropriate. See #249.
const MAX_ITERATIONS: usize = 100;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the Prompt agent.
///
/// # Example
///
/// ```
/// use gemicro_prompt_agent::PromptAgentConfig;
/// use gemicro_core::ToolSet;
/// use std::time::Duration;
///
/// let config = PromptAgentConfig::default()
///     .with_timeout(Duration::from_secs(30))
///     .with_system_prompt("You are a helpful assistant.")
///     .with_tool_filter(ToolSet::Specific(vec!["calculator".into()]));
///
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PromptAgentConfig {
    /// Model to use for LLM requests.
    ///
    /// Default: "gemini-3-flash-preview"
    pub model: String,

    /// Total timeout for the query execution.
    pub timeout: Duration,

    /// System prompt to use for the LLM request.
    pub system_prompt: String,

    /// Tool filter for function calling.
    ///
    /// Only used when `AgentContext::tools` is provided.
    /// Defaults to `ToolSet::All` (use all available tools).
    pub tool_filter: ToolSet,

    /// Custom description for the agent.
    ///
    /// If set, this description is returned by `Agent::description()` instead
    /// of the default. Used by markdown-defined agents.
    pub description: Option<String>,
}

impl Default for PromptAgentConfig {
    fn default() -> Self {
        Self {
            model: "gemini-3-flash-preview".to_string(),
            timeout: Duration::from_secs(30),
            system_prompt:
                "You are a helpful assistant. Answer questions concisely and accurately."
                    .to_string(),
            tool_filter: ToolSet::All,
            description: None,
        }
    }
}

impl PromptAgentConfig {
    /// Set the model to use for LLM requests.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the timeout for query execution.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the system prompt for the LLM request.
    #[must_use]
    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = system_prompt.into();
        self
    }

    /// Set the tool filter for function calling.
    ///
    /// This filter is applied when tools are provided via `AgentContext::tools`.
    /// If no tools are provided in context, this setting has no effect.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_prompt_agent::PromptAgentConfig;
    /// use gemicro_core::ToolSet;
    ///
    /// // Only allow specific tools
    /// let config = PromptAgentConfig::default()
    ///     .with_tool_filter(ToolSet::Specific(vec!["calculator".into(), "file_read".into()]));
    ///
    /// // Exclude dangerous tools
    /// let config = PromptAgentConfig::default()
    ///     .with_tool_filter(ToolSet::Except(vec!["bash".into(), "file_write".into()]));
    /// ```
    #[must_use]
    pub fn with_tool_filter(mut self, tool_filter: ToolSet) -> Self {
        self.tool_filter = tool_filter;
        self
    }

    /// Set a custom description for the agent.
    ///
    /// When set, this description is returned by `Agent::description()` instead
    /// of the default. This is used by markdown-defined agents to show the
    /// description from the markdown frontmatter.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_prompt_agent::PromptAgentConfig;
    ///
    /// let config = PromptAgentConfig::default()
    ///     .with_description("Answers questions about codebase structure");
    /// ```
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Validate the configuration.
    ///
    /// Returns an error if:
    /// - Timeout is zero
    /// - System prompt is empty or whitespace-only
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_prompt_agent::PromptAgentConfig;
    /// use std::time::Duration;
    ///
    /// let invalid = PromptAgentConfig::default().with_timeout(Duration::ZERO);
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), AgentError> {
        let mut errors = Vec::new();

        if self.timeout.is_zero() {
            errors.push("timeout must be greater than zero");
        }

        if self.system_prompt.trim().is_empty() {
            errors.push("system_prompt must not be empty");
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(AgentError::InvalidConfig(errors.join("; ")))
        }
    }
}

// ============================================================================
// Agent Implementation
// ============================================================================

/// An agent that executes prompts with optional tool support.
///
/// PromptAgent makes LLM calls with a custom system prompt. When tools are
/// provided via `AgentContext::tools`, it uses function calling.
///
/// # Tool Support
///
/// Tools are **explicitly provided** via `AgentContext::tools`. There are no
/// default tools - this follows the "Explicit Over Implicit" principle.
///
/// ```no_run
/// use gemicro_prompt_agent::{PromptAgent, PromptAgentConfig};
/// use gemicro_core::{AgentContext, Agent, LlmClient, LlmConfig, ToolRegistry, ToolSet};
/// use std::sync::Arc;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create agent with tool filter
/// let config = PromptAgentConfig::default()
///     .with_tool_filter(ToolSet::Specific(vec!["calculator".into()]));
/// let agent = PromptAgent::new(config)?;
///
/// // Create context WITH explicit tools
/// let genai = genai_rs::Client::builder("api-key".to_string()).build().map_err(|e| AgentError::Other(e.to_string()))?;
/// let registry = ToolRegistry::new();
/// // registry.register(Calculator); // Register your tools
///
/// let context = AgentContext::new(LlmClient::new(genai, LlmConfig::default()))
///     .with_tools(registry);
///
/// // Now execute - agent will use function calling
/// let stream = agent.execute("What is 25 * 4?", context);
/// # Ok(())
/// # }
/// ```
///
/// # Example (Without Tools)
///
/// ```no_run
/// use gemicro_prompt_agent::{PromptAgent, PromptAgentConfig};
/// use gemicro_core::{AgentContext, Agent, LlmClient, LlmConfig};
/// use futures_util::StreamExt;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let agent = PromptAgent::new(PromptAgentConfig::default())?;
///
/// let genai = genai_rs::Client::builder("api-key".to_string()).build().map_err(|e| AgentError::Other(e.to_string()))?;
/// let context = AgentContext::new(LlmClient::new(genai, LlmConfig::default()));
///
/// let stream = agent.execute("What is Rust?", context);
/// futures_util::pin_mut!(stream);
///
/// while let Some(update) = stream.next().await {
///     let update = update?;
///     match update.event_type.as_str() {
///         "prompt_agent_started" => println!("Processing..."),
///         "prompt_agent_result" => println!("Answer: {}", update.message),
///         _ => {} // Ignore unknown events
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub struct PromptAgent {
    config: PromptAgentConfig,
}

impl PromptAgent {
    /// Create a new Prompt agent with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::InvalidConfig` if the configuration is invalid.
    pub fn new(config: PromptAgentConfig) -> Result<Self, AgentError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Create a Prompt agent with a custom system prompt.
    ///
    /// This is a convenience constructor for creating ephemeral prompt-based agents,
    /// typically used by the Task tool when executing [`PromptAgentDef`].
    ///
    /// Uses default configuration with the specified system prompt.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::InvalidConfig` if the system prompt is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_prompt_agent::PromptAgent;
    ///
    /// let agent = PromptAgent::with_system_prompt(
    ///     "You are a Python security auditor. Review code for vulnerabilities."
    /// ).unwrap();
    /// ```
    ///
    /// [`PromptAgentDef`]: gemicro_core::agent::PromptAgentDef
    pub fn with_system_prompt(system_prompt: impl Into<String>) -> Result<Self, AgentError> {
        let config = PromptAgentConfig::default().with_system_prompt(system_prompt);
        Self::new(config)
    }

    /// Create a Prompt agent with a custom system prompt and timeout.
    ///
    /// For creating ephemeral agents with custom timeout constraints.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::InvalidConfig` if the system prompt is empty or timeout is zero.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_prompt_agent::PromptAgent;
    /// use std::time::Duration;
    ///
    /// let agent = PromptAgent::with_system_prompt_and_timeout(
    ///     "You are a code reviewer.",
    ///     Duration::from_secs(60),
    /// ).unwrap();
    /// ```
    pub fn with_system_prompt_and_timeout(
        system_prompt: impl Into<String>,
        timeout: Duration,
    ) -> Result<Self, AgentError> {
        let config = PromptAgentConfig::default()
            .with_system_prompt(system_prompt)
            .with_timeout(timeout);
        Self::new(config)
    }

    /// Create a Prompt agent with system prompt and tool filter.
    ///
    /// For creating ephemeral agents with explicit tool constraints.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::InvalidConfig` if the system prompt is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_prompt_agent::PromptAgent;
    /// use gemicro_core::ToolSet;
    ///
    /// let agent = PromptAgent::with_system_prompt_and_tools(
    ///     "You are a calculator assistant.",
    ///     ToolSet::Specific(vec!["calculator".into()]),
    /// ).unwrap();
    /// ```
    pub fn with_system_prompt_and_tools(
        system_prompt: impl Into<String>,
        tool_filter: ToolSet,
    ) -> Result<Self, AgentError> {
        let config = PromptAgentConfig::default()
            .with_system_prompt(system_prompt)
            .with_tool_filter(tool_filter);
        Self::new(config)
    }

    /// Create a Prompt agent from a [`PromptAgentDef`].
    ///
    /// This constructor is used by the markdown agent loader to create agents
    /// from parsed markdown definitions.
    ///
    /// # Note
    ///
    /// The `model` field in [`PromptAgentDef`] is not used directly by this constructor.
    /// Model selection is handled at the runner/context level. The caller should use
    /// the model information when configuring the `AgentContext`.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::InvalidConfig` if the definition fails validation.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_prompt_agent::PromptAgent;
    /// use gemicro_core::agent::PromptAgentDef;
    /// use gemicro_core::ToolSet;
    ///
    /// let def = PromptAgentDef::new("Code reviewer")
    ///     .with_system_prompt("You review code for quality issues.")
    ///     .with_tools(ToolSet::Specific(vec!["file_read".into()]));
    ///
    /// let agent = PromptAgent::with_definition(&def).unwrap();
    /// ```
    ///
    /// [`PromptAgentDef`]: gemicro_core::agent::PromptAgentDef
    pub fn with_definition(def: &gemicro_core::agent::PromptAgentDef) -> Result<Self, AgentError> {
        // Validate the definition first
        def.validate()
            .map_err(|e| AgentError::InvalidConfig(e.to_string()))?;

        let config = PromptAgentConfig::default()
            .with_system_prompt(&def.system_prompt)
            .with_tool_filter(def.tools.clone())
            .with_description(&def.description);

        Self::new(config)
    }
}

impl Agent for PromptAgent {
    fn name(&self) -> &str {
        "prompt_agent"
    }

    fn description(&self) -> &str {
        self.config
            .description
            .as_deref()
            .unwrap_or("An agent that executes prompts with optional tool support")
    }

    fn execute(&self, query: &str, context: AgentContext) -> AgentStream<'_> {
        let query = query.to_string();
        let config = self.config.clone();

        Box::pin(try_stream! {
            let start = Instant::now();

            // Emit start event
            yield AgentUpdate::custom(
                EVENT_PROMPT_AGENT_STARTED,
                format!("Processing query: {}", truncate(&query, 50)),
                json!({ "query": query }),
            );

            // Calculate remaining time
            let timeout = remaining_time(start, config.timeout, "query")?;

            // Branch based on whether tools are available
            let (answer, tokens_used) = if let Some(ref registry) = context.tools {
                // Tool path: function calling loop with events
                // Uses LlmClient's generate() for centralized logging/retry/recording
                let tools = registry.filter(&config.tool_filter);
                let function_declarations: Vec<FunctionDeclaration> = tools
                    .iter()
                    .map(|t| t.to_function_declaration())
                    .collect();

                let mut total_tokens: u64 = 0;

                // Initial request with query and function declarations
                let initial_request = context
                    .llm
                    .client()
                    .interaction()
                    .with_model(&config.model)
                    .with_system_instruction(&config.system_prompt)
                    .with_text(&query)
                    .with_functions(function_declarations.clone())
                    .with_store_enabled() // Enable storage for function calling chains
                    .build().map_err(|e| AgentError::Other(e.to_string()))?;

                let mut response = context
                    .llm
                    .generate_with_cancellation(initial_request, &context.cancellation_token)
                    .await
                    .map_err(|e| AgentError::Other(format!("LLM error: {}", e)))?;

                if let Some(tokens) = gemicro_core::extract_total_tokens(&response) {
                    total_tokens += tokens as u64;
                }

                // Function calling loop
                let mut iteration = 0usize;
                let final_text = loop {
                    iteration += 1;
                    if iteration > MAX_ITERATIONS {
                        log::warn!("PromptAgent hit max iterations ({})", MAX_ITERATIONS);
                        Err(AgentError::Other(format!(
                            "Max iterations ({}) reached",
                            MAX_ITERATIONS
                        )))?;
                    }

                    // Check for cancellation
                    if context.cancellation_token.is_cancelled() {
                        Err(AgentError::Cancelled)?;
                    }

                    let function_calls = response.function_calls();

                    if function_calls.is_empty() {
                        // No function calls - done, return final text
                        break response.text().unwrap_or("").to_string();
                    }

                    // Get interaction ID for chaining
                    let interaction_id = response.id.clone().ok_or_else(|| {
                        AgentError::Other("Missing interaction ID for function calling".into())
                    })?;

                    // Execute each function call with events
                    let mut function_results: Vec<InteractionContent> = Vec::new();

                    for fc in function_calls {
                        let tool_name = &fc.name;
                        let call_id = fc.id.unwrap_or("unknown");
                        let arguments = &fc.args;

                        // Emit tool_call_started event
                        yield AgentUpdate::custom(
                            EVENT_TOOL_CALL_STARTED,
                            format!("Calling tool: {}", tool_name),
                            json!({
                                "tool_name": tool_name,
                                "call_id": call_id,
                                "arguments": arguments,
                            }),
                        );

                        let tool_start = Instant::now();

                        // Execute tool - find it in the filtered tools list
                        let tool_result: Result<Value, String> = {
                            let tool_opt = tools.iter().find(|t| t.name() == *tool_name);
                            if let Some(tool) = tool_opt {
                                let mut adapter = ToolCallableAdapter::new(Arc::clone(tool));
                                if let Some(handler) = &context.confirmation_handler {
                                    adapter = adapter.with_confirmation_handler(Arc::clone(handler));
                                }
                                if let Some(interceptors) = &context.interceptors {
                                    adapter = adapter.with_interceptors(Arc::clone(interceptors));
                                }

                                match adapter.call((*arguments).clone()).await {
                                    Ok(value) => Ok(value),
                                    Err(e) => Err(e.to_string()),
                                }
                            } else {
                                Err(format!("Tool not found: {}", tool_name))
                            }
                        };

                        let tool_duration = tool_start.elapsed();
                        let (success, result_json) = match &tool_result {
                            Ok(value) => (true, value.clone()),
                            Err(e) => (false, json!({ "error": e })),
                        };

                        // Emit tool_result event
                        yield AgentUpdate::custom(
                            EVENT_TOOL_RESULT,
                            format!("Tool {} completed", tool_name),
                            json!({
                                "tool_name": tool_name,
                                "call_id": call_id,
                                "result": result_json,
                                "success": success,
                                "duration_ms": tool_duration.as_millis() as u64,
                            }),
                        );

                        // Add to results for next LLM call
                        function_results.push(function_result_content(
                            tool_name.to_string(),
                            call_id.to_string(),
                            result_json,
                        ));
                    }

                    // Send function results back to LLM using continuation request
                    let continuation = context
                        .llm
                        .client()
                        .interaction()
                        .with_model(&config.model)
                        .with_previous_interaction(&interaction_id)
                        .with_functions(function_declarations.clone())
                        .with_content(function_results)
                        .with_store_enabled()
                        .build().map_err(|e| AgentError::Other(e.to_string()))?;

                    response = context
                        .llm
                        .generate_with_cancellation(continuation, &context.cancellation_token)
                        .await
                        .map_err(|e| AgentError::Other(format!("LLM error: {}", e)))?;

                    // Track tokens (logging now handled by LlmClient)
                    if let Some(tokens) = gemicro_core::extract_total_tokens(&response) {
                        total_tokens += tokens as u64;
                    }

                    // Loop continues - check for more function calls at top of loop
                };

                (final_text, Some(total_tokens))
            } else {
                // Simple path: no tools, just a prompt
                let request = context
                    .llm
                    .client()
                    .interaction()
                    .with_model(&config.model)
                    .with_system_instruction(&config.system_prompt)
                    .with_text(&query)
                    .build().map_err(|e| AgentError::Other(e.to_string()))?;

                let generate_future = async {
                    context
                        .llm
                        .generate(request)
                        .await
                        .map_err(|e| AgentError::Other(format!("LLM error: {}", e)))
                };

                let response = with_timeout_and_cancellation(
                    generate_future,
                    timeout,
                    &context.cancellation_token,
                    || timeout_error(start, config.timeout, "query"),
                )
                .await?;

                let text = response.text().unwrap_or("").to_string();
                let tokens = gemicro_core::extract_total_tokens(&response).map(|t| t as u64);
                (text, tokens)
            };

            let duration_ms = start.elapsed().as_millis() as u64;

            // Emit agent-specific result event
            yield AgentUpdate::custom(
                EVENT_PROMPT_AGENT_RESULT,
                answer.clone(),
                json!({
                    "answer": answer,
                    "tokens_used": tokens_used,
                    "duration_ms": duration_ms,
                }),
            );

            // Emit standard final_result for ExecutionState/harness compatibility
            let metadata = ResultMetadata::new(
                tokens_used.unwrap_or(0) as u32,
                if tokens_used.is_none() { 1 } else { 0 },
                duration_ms,
            );
            yield AgentUpdate::final_result(json!(answer), metadata);
        })
    }

    fn create_tracker(&self) -> Box<dyn gemicro_core::ExecutionTracking> {
        Box::new(gemicro_core::DefaultTracker::default())
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = PromptAgentConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_rejects_zero_timeout() {
        let config = PromptAgentConfig::default().with_timeout(Duration::ZERO);
        let err = config.validate().unwrap_err();
        match err {
            AgentError::InvalidConfig(msg) => {
                assert!(msg.contains("timeout"));
            }
            _ => panic!("Expected InvalidConfig error"),
        }
    }

    #[test]
    fn test_config_rejects_empty_system_prompt() {
        let config = PromptAgentConfig::default().with_system_prompt("   ");
        let err = config.validate().unwrap_err();
        match err {
            AgentError::InvalidConfig(msg) => {
                assert!(msg.contains("system_prompt"));
            }
            _ => panic!("Expected InvalidConfig error"),
        }
    }

    #[test]
    fn test_config_collects_multiple_errors() {
        let config = PromptAgentConfig::default()
            .with_timeout(Duration::ZERO)
            .with_system_prompt("");
        let err = config.validate().unwrap_err();
        match err {
            AgentError::InvalidConfig(msg) => {
                assert!(msg.contains("timeout"));
                assert!(msg.contains("system_prompt"));
            }
            _ => panic!("Expected InvalidConfig error"),
        }
    }

    #[test]
    fn test_agent_creation_validates_config() {
        let invalid_config = PromptAgentConfig::default().with_timeout(Duration::ZERO);
        assert!(PromptAgent::new(invalid_config).is_err());
    }

    #[test]
    fn test_agent_name_and_description() {
        let agent = PromptAgent::new(PromptAgentConfig::default()).unwrap();
        assert_eq!(agent.name(), "prompt_agent");
        assert!(!agent.description().is_empty());
    }

    #[test]
    fn test_event_constants_are_unique() {
        assert_ne!(EVENT_PROMPT_AGENT_STARTED, EVENT_PROMPT_AGENT_RESULT);
    }

    #[test]
    fn test_with_system_prompt_constructor() {
        let agent = PromptAgent::with_system_prompt("You are a helpful bot.").unwrap();
        assert_eq!(agent.name(), "prompt_agent");
    }

    #[test]
    fn test_with_system_prompt_rejects_empty() {
        let result = PromptAgent::with_system_prompt("   ");
        assert!(result.is_err());
    }

    #[test]
    fn test_with_system_prompt_and_timeout_constructor() {
        let agent = PromptAgent::with_system_prompt_and_timeout(
            "You are a code reviewer.",
            Duration::from_secs(60),
        )
        .unwrap();
        assert_eq!(agent.name(), "prompt_agent");
    }

    #[test]
    fn test_with_system_prompt_and_timeout_rejects_zero() {
        let result = PromptAgent::with_system_prompt_and_timeout("Valid prompt", Duration::ZERO);
        assert!(result.is_err());
    }

    #[test]
    fn test_with_system_prompt_and_tools_constructor() {
        let agent = PromptAgent::with_system_prompt_and_tools(
            "You are a calculator.",
            ToolSet::Specific(vec!["calculator".into()]),
        )
        .unwrap();
        assert_eq!(agent.name(), "prompt_agent");
    }

    #[test]
    fn test_config_with_tool_filter() {
        let config =
            PromptAgentConfig::default().with_tool_filter(ToolSet::Except(vec!["bash".into()]));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_default_tool_filter_is_all() {
        let config = PromptAgentConfig::default();
        assert!(matches!(config.tool_filter, ToolSet::All));
    }

    #[test]
    fn test_with_definition_constructor() {
        use gemicro_core::agent::PromptAgentDef;

        let def = PromptAgentDef::new("Test agent")
            .with_system_prompt("You are a helpful assistant.")
            .with_tools(ToolSet::Specific(vec!["file_read".into()]));

        let agent = PromptAgent::with_definition(&def).unwrap();
        assert_eq!(agent.name(), "prompt_agent");
    }

    #[test]
    fn test_with_definition_rejects_invalid() {
        use gemicro_core::agent::PromptAgentDef;

        // Missing system prompt
        let def = PromptAgentDef::new("Test agent");
        let result = PromptAgent::with_definition(&def);
        assert!(result.is_err());
    }
}
