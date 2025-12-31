//! Tool Agent implementation using rust-genai's native function calling.
//!
//! Unlike the ReAct agent which explicitly reasons about tool usage,
//! this agent delegates tool selection and execution to rust-genai's
//! automatic function calling loop.
//!
//! # Example
//!
//! ```no_run
//! use gemicro_tool_agent::{ToolAgent, ToolAgentConfig};
//! use gemicro_core::{AgentContext, LlmClient, LlmConfig, ToolSet};
//! use futures_util::StreamExt;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Use only the calculator tool
//! let config = ToolAgentConfig::default()
//!     .with_tool_filter(ToolSet::Specific(vec!["calculator".into()]));
//! let agent = ToolAgent::new(config)?;
//!
//! let genai_client = rust_genai::Client::builder("api-key".to_string()).build();
//! let context = AgentContext::new(LlmClient::new(genai_client, LlmConfig::default()));
//!
//! let stream = agent.execute("What is 25 * 4?", context);
//! futures_util::pin_mut!(stream);
//! while let Some(update) = stream.next().await {
//!     println!("{:?}", update?);
//! }
//! # Ok(())
//! # }
//! ```

pub mod tools;

use gemicro_core::{
    remaining_time, timeout_error, with_timeout_and_cancellation, Agent, AgentContext, AgentError,
    AgentStream, AgentUpdate, GemicroToolService, ResultMetadata, ToolRegistry, ToolSet, MODEL,
};
use tools::default_registry;

use async_stream::try_stream;
use futures_util::Stream;
use rust_genai::AutoFunctionResult;
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};

// ============================================================================
// Event Type Constants
// ============================================================================

/// Emitted when the tool agent starts processing
const EVENT_TOOL_AGENT_STARTED: &str = "tool_agent_started";

/// Emitted when the agent completes successfully
const EVENT_TOOL_AGENT_COMPLETE: &str = "tool_agent_complete";

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the ToolAgent.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolAgentConfig {
    /// Total timeout for the entire operation
    pub timeout: Duration,

    /// System prompt for the agent
    pub system_prompt: String,

    /// Filter for which tools to enable.
    ///
    /// By default, all tools in the registry are enabled.
    pub tool_filter: ToolSet,
}

impl Default for ToolAgentConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            system_prompt: "You are a helpful assistant with access to tools. \
                Use tools when they help answer the user's question accurately. \
                For math questions, use the calculator tool. \
                Always provide a clear, concise final answer."
                .to_string(),
            tool_filter: ToolSet::All,
        }
    }
}

impl ToolAgentConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), AgentError> {
        let mut errors = Vec::new();

        if self.timeout.is_zero() {
            errors.push("timeout must be greater than zero");
        }
        if self.system_prompt.trim().is_empty() {
            errors.push("system_prompt must not be empty");
        }
        // Note: We don't validate tool_filter here because validation requires
        // knowing the available registry, which isn't available until execute().
        // Runtime will error if filtering results in zero tools.

        if errors.is_empty() {
            Ok(())
        } else {
            Err(AgentError::InvalidConfig(errors.join("; ")))
        }
    }

    /// Create a config with only the calculator tool enabled.
    pub fn calculator_only() -> Self {
        Self {
            tool_filter: ToolSet::Specific(vec!["calculator".into()]),
            system_prompt: "You are a math assistant. Use the calculator tool to solve \
                mathematical problems. Always show the calculation and provide the answer."
                .to_string(),
            ..Default::default()
        }
    }

    /// Set the tool filter.
    pub fn with_tool_filter(mut self, filter: ToolSet) -> Self {
        self.tool_filter = filter;
        self
    }

    /// Set the timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }
}

// ============================================================================
// Tool Agent
// ============================================================================

/// An agent that uses rust-genai's native function calling.
///
/// This agent uses the [`Tool`](gemicro_core::Tool) trait implementations
/// and rust-genai's `create_with_auto_functions()` for automatic tool execution.
///
/// # Available Tools
///
/// By default, the agent uses the built-in tool registry with:
/// - **calculator**: Evaluates mathematical expressions (e.g., "2 + 2", "sqrt(16)")
/// - **current_datetime**: Gets the current date and time (UTC)
///
/// You can provide a custom registry via [`AgentContext::with_tools()`] or
/// filter available tools via [`ToolAgentConfig::with_tool_filter()`].
///
/// # Example
///
/// ```no_run
/// use gemicro_tool_agent::{ToolAgent, ToolAgentConfig};
/// use gemicro_core::{AgentContext, LlmClient, LlmConfig};
/// use futures_util::StreamExt;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let genai_client = rust_genai::Client::builder("api-key".to_string()).build();
/// let context = AgentContext::new(LlmClient::new(genai_client, LlmConfig::default()));
/// let agent = ToolAgent::new(ToolAgentConfig::default())?;
///
/// let stream = agent.execute("What is 25 * 4?", context);
/// futures_util::pin_mut!(stream);
/// while let Some(update) = stream.next().await {
///     println!("{:?}", update?);
/// }
/// # Ok(())
/// # }
/// ```
pub struct ToolAgent {
    config: ToolAgentConfig,
    /// Default tool registry (used when context.tools is None)
    default_registry: Arc<ToolRegistry>,
}

impl ToolAgent {
    /// Create a new ToolAgent with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::InvalidConfig` if the configuration is invalid.
    pub fn new(config: ToolAgentConfig) -> Result<Self, AgentError> {
        config.validate()?;
        Ok(Self {
            config,
            default_registry: Arc::new(default_registry()),
        })
    }

    /// Execute the tool agent.
    ///
    /// Returns a stream of [`AgentUpdate`] events.
    pub fn execute(
        &self,
        query: &str,
        context: AgentContext,
    ) -> impl Stream<Item = Result<AgentUpdate, AgentError>> + Send + '_ {
        let query = query.to_string();
        let config = self.config.clone();
        let default_registry = Arc::clone(&self.default_registry);

        try_stream! {
            let start_time = Instant::now();
            let mut total_tokens: u32 = 0;
            let mut tokens_unavailable: usize = 0;

            // Use context tools if provided, otherwise use default registry
            let registry = context.tools.as_ref().unwrap_or(&default_registry);

            // Build the tool service with filtering and confirmation handler
            let mut service = GemicroToolService::new(Arc::clone(registry))
                .with_filter(config.tool_filter.clone());

            // Add confirmation handler if provided in context
            if let Some(handler) = &context.confirmation_handler {
                service = service.with_confirmation_handler(Arc::clone(handler));
            }

            // Get tool names for the started event
            let tool_names: Vec<String> = service.registry()
                .filter(&config.tool_filter)
                .iter()
                .map(|t| t.name().to_string())
                .collect();

            if tool_names.is_empty() {
                Err(AgentError::InvalidConfig("No tools available after filtering".into()))?;
            }

            yield AgentUpdate::custom(
                EVENT_TOOL_AGENT_STARTED,
                format!("Processing query with {} tools available", tool_names.len()),
                json!({
                    "query": query,
                    "tools": tool_names,
                }),
            );

            // Check timeout
            let timeout = remaining_time(start_time, config.timeout, "tool_agent")?;

            // Get the raw genai client from our LlmClient
            // Note: We need to use the client directly for function calling
            let client = context.llm.client();

            // Build the interaction with tool service (handles confirmation automatically)
            let interaction = client
                .interaction()
                .with_model(MODEL)
                .with_system_instruction(&config.system_prompt)
                .with_text(&query)
                .with_tool_service(Arc::new(service));

            // Execute with auto function calling and timeout
            let response_future = interaction.create_with_auto_functions();

            let result: AutoFunctionResult = with_timeout_and_cancellation(
                async {
                    response_future.await.map_err(|e| AgentError::Other(format!("Function calling failed: {}", e)))
                },
                timeout,
                &context.cancellation_token,
                || timeout_error(start_time, config.timeout, "tool_agent"),
            ).await?;

            // Extract the response and tool executions from AutoFunctionResult
            let response = result.response;
            let executions = result.executions;

            // Extract the response text, logging if empty (include query context for debugging)
            let answer: String = match response.text() {
                Some(text) if !text.is_empty() => text.to_string(),
                Some(_) => {
                    let query_preview: String = query.chars().take(100).collect();
                    log::warn!(
                        "LLM returned empty text response for tool agent query: '{}'",
                        query_preview
                    );
                    String::new()
                }
                None => {
                    let query_preview: String = query.chars().take(100).collect();
                    log::warn!(
                        "LLM response contained no text content for tool agent query: '{}'",
                        query_preview
                    );
                    String::new()
                }
            };

            // Try to get token usage from the response
            if let Some(usage) = response.usage.as_ref() {
                total_tokens = usage.total_tokens.unwrap_or(0) as u32;
            } else {
                tokens_unavailable += 1;
            }

            // Build tool calls summary for observability
            let tool_calls: Vec<_> = executions.iter().map(|e| {
                json!({
                    "name": e.name,
                    "call_id": e.call_id,
                    "result": e.result,
                    "duration_ms": e.duration.as_millis() as u64,
                })
            }).collect();

            // Emit completion event with tool execution details
            yield AgentUpdate::custom(
                EVENT_TOOL_AGENT_COMPLETE,
                format!("Completed with {} tool call(s)", executions.len()),
                json!({
                    "answer": answer,
                    "tool_calls": tool_calls,
                    "tool_call_count": executions.len(),
                    "duration_ms": start_time.elapsed().as_millis() as u64,
                }),
            );

            // Emit standard final_result for harness compatibility.
            // Tool call details are exposed in the tool_agent_complete event above.
            let metadata = ResultMetadata::new(
                total_tokens,
                tokens_unavailable,
                start_time.elapsed().as_millis() as u64,
            );
            yield AgentUpdate::final_result(json!(answer), metadata);
        }
    }
}

impl Agent for ToolAgent {
    fn name(&self) -> &str {
        "tool_agent"
    }

    fn description(&self) -> &str {
        "An agent that uses native function calling for tool execution"
    }

    fn execute(&self, query: &str, context: AgentContext) -> AgentStream<'_> {
        Box::pin(ToolAgent::execute(self, query, context))
    }

    fn create_tracker(&self) -> Box<dyn gemicro_core::ExecutionTracking> {
        Box::new(gemicro_core::DefaultTracker::default())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use gemicro_core::{tools_to_callables, Tool};
    use rust_genai::CallableFunction;
    use tools::{Calculator, CurrentDatetime};

    #[test]
    fn test_default_config_is_valid() {
        let config = ToolAgentConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_calculator_only_config_is_valid() {
        let config = ToolAgentConfig::calculator_only();
        assert!(config.validate().is_ok());
        assert!(matches!(config.tool_filter, ToolSet::Specific(_)));
    }

    #[test]
    fn test_config_rejects_zero_timeout() {
        let config = ToolAgentConfig {
            timeout: Duration::ZERO,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timeout"));
    }

    #[test]
    fn test_config_rejects_empty_system_prompt() {
        let config = ToolAgentConfig {
            system_prompt: "   ".to_string(),
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("system_prompt"));
    }

    #[test]
    fn test_config_collects_all_errors() {
        let config = ToolAgentConfig {
            timeout: Duration::ZERO,
            system_prompt: String::new(),
            tool_filter: ToolSet::All,
        };
        let result = config.validate();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("timeout"));
        assert!(err.contains("system_prompt"));
    }

    #[test]
    fn test_agent_creation_valid_config() {
        let agent = ToolAgent::new(ToolAgentConfig::default());
        assert!(agent.is_ok());
    }

    #[test]
    fn test_agent_creation_invalid_config() {
        let agent = ToolAgent::new(ToolAgentConfig {
            timeout: Duration::ZERO,
            ..Default::default()
        });
        assert!(agent.is_err());
    }

    #[test]
    fn test_agent_name_and_description() {
        let agent = ToolAgent::new(ToolAgentConfig::default()).unwrap();
        assert_eq!(agent.name(), "tool_agent");
        assert!(!agent.description().is_empty());
    }

    #[test]
    fn test_default_registry_function_declarations() {
        let registry = default_registry();
        let tools = registry.filter(&ToolSet::All);
        let adapters = tools_to_callables(&tools, None);
        let declarations: Vec<_> = adapters.iter().map(|a| a.declaration()).collect();

        assert_eq!(declarations.len(), 6);
        let names: Vec<&str> = declarations.iter().map(|d| d.name()).collect();
        assert!(names.contains(&"calculator"));
        assert!(names.contains(&"current_datetime"));
        assert!(names.contains(&"file_read"));
        assert!(names.contains(&"web_fetch"));
        assert!(names.contains(&"glob"));
        assert!(names.contains(&"grep"));
    }

    #[test]
    fn test_filtered_function_declarations() {
        let registry = default_registry();
        let tools = registry.filter(&ToolSet::Specific(vec!["calculator".into()]));
        let adapters = tools_to_callables(&tools, None);
        let declarations: Vec<_> = adapters.iter().map(|a| a.declaration()).collect();

        assert_eq!(declarations.len(), 1);
        assert_eq!(declarations[0].name(), "calculator");
    }

    #[test]
    fn test_calculator_tool_via_trait() {
        let calc = Calculator;
        assert_eq!(calc.name(), "calculator");
        assert!(!calc.description().is_empty());
    }

    #[test]
    fn test_current_datetime_tool_via_trait() {
        let dt = CurrentDatetime;
        assert_eq!(dt.name(), "current_datetime");
        assert!(!dt.description().is_empty());
    }

    #[test]
    fn test_event_constants_are_unique() {
        let events = vec![EVENT_TOOL_AGENT_STARTED, EVENT_TOOL_AGENT_COMPLETE];
        let mut unique = std::collections::HashSet::new();
        for event in &events {
            assert!(unique.insert(*event), "Duplicate event: {}", event);
        }
    }

    #[test]
    fn test_default_registry_has_expected_tools() {
        let registry = default_registry();
        assert!(registry.contains("calculator"));
        assert!(registry.contains("current_datetime"));
        assert!(registry.contains("file_read"));
        assert!(registry.contains("web_fetch"));
        assert!(registry.contains("glob"));
        assert!(registry.contains("grep"));
        assert_eq!(registry.len(), 6);
    }
}
