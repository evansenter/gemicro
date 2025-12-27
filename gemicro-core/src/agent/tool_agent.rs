//! Tool Agent implementation using rust-genai's native function calling.
//!
//! Unlike the ReAct agent which explicitly reasons about tool usage,
//! this agent delegates tool selection and execution to rust-genai's
//! automatic function calling loop.

use super::{remaining_time, timeout_error, with_timeout_and_cancellation, Agent, AgentContext};
use crate::config::MODEL;
use crate::error::AgentError;
use crate::update::{AgentUpdate, ResultMetadata};

use async_stream::try_stream;
use futures_util::Stream;
use rust_genai::{CallableFunction, FunctionDeclaration, InteractionResponse};
use rust_genai_macros::tool;
use serde_json::json;
use std::time::{Duration, Instant};

// ============================================================================
// Event Type Constants
// ============================================================================

/// Emitted when the tool agent starts processing
pub const EVENT_TOOL_AGENT_STARTED: &str = "tool_agent_started";

/// Emitted when the agent completes successfully
pub const EVENT_TOOL_AGENT_COMPLETE: &str = "tool_agent_complete";

// ============================================================================
// Tool Definitions using #[tool] macro
// ============================================================================

/// Calculator tool for evaluating mathematical expressions.
///
/// Supports basic arithmetic (+, -, *, /), exponents (^), parentheses,
/// and common functions (sqrt, sin, cos, tan, log, ln, abs).
#[tool(expression(
    description = "A mathematical expression to evaluate, e.g., '2 + 2', 'sqrt(16)', '3.14 * 2^3'"
))]
fn calculator(expression: String) -> String {
    match meval::eval_str(&expression) {
        Ok(result) => {
            if result.is_nan() {
                "Error: Result is not a number (NaN)".to_string()
            } else if result.is_infinite() {
                "Error: Result is infinite (division by zero or overflow)".to_string()
            } else if result.fract() == 0.0 && result.abs() < 1e15 {
                format!("{:.0}", result)
            } else {
                format!("{}", result)
            }
        }
        Err(e) => format!("Error: {}", e),
    }
}

/// Gets the current date and time in a specified timezone.
///
/// Note: This is a simplified implementation that returns UTC time.
/// In production, you might use chrono-tz for proper timezone handling.
#[tool(timezone(
    description = "The timezone to get the time for (e.g., 'UTC', 'EST', 'PST', 'JST'). Currently returns UTC regardless of input."
))]
fn current_datetime(timezone: String) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    // Simple date/time calculation (UTC only for now)
    let total_secs = now.as_secs();
    let days_since_epoch = total_secs / 86400;
    let secs_today = total_secs % 86400;

    let hours = secs_today / 3600;
    let minutes = (secs_today % 3600) / 60;
    let seconds = secs_today % 60;

    // Approximate date calculation (not accounting for leap seconds)
    let mut year = 1970;
    let mut remaining_days = days_since_epoch as i64;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let days_in_months: [i64; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1;
    for days in days_in_months.iter() {
        if remaining_days < *days {
            break;
        }
        remaining_days -= *days;
        month += 1;
    }
    let day = remaining_days + 1;

    format!(
        r#"{{"timezone": "{}", "date": "{:04}-{:02}-{:02}", "time": "{:02}:{:02}:{:02}", "note": "Times are in UTC regardless of requested timezone"}}"#,
        timezone, year, month, day, hours, minutes, seconds
    )
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

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

    /// Which tools to enable
    pub enabled_tools: Vec<ToolType>,
}

/// Available tool types for the ToolAgent.
///
/// New variants may be added in future versions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ToolType {
    /// Mathematical expression evaluator
    Calculator,
    /// Current date and time
    CurrentDateTime,
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
            enabled_tools: vec![ToolType::Calculator, ToolType::CurrentDateTime],
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
        if self.enabled_tools.is_empty() {
            errors.push("enabled_tools must not be empty");
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(AgentError::InvalidConfig(errors.join("; ")))
        }
    }

    /// Create a config with only the calculator tool enabled.
    pub fn calculator_only() -> Self {
        Self {
            enabled_tools: vec![ToolType::Calculator],
            system_prompt: "You are a math assistant. Use the calculator tool to solve \
                mathematical problems. Always show the calculation and provide the answer."
                .to_string(),
            ..Default::default()
        }
    }

    /// Set the enabled tools.
    pub fn with_tools(mut self, tools: Vec<ToolType>) -> Self {
        self.enabled_tools = tools;
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
/// This agent leverages the `#[tool]` macro to define tools and uses
/// `create_with_auto_functions()` for automatic tool execution.
///
/// # Available Tools
///
/// - **calculator**: Evaluates mathematical expressions (e.g., "2 + 2", "sqrt(16)")
/// - **current_datetime**: Gets the current date and time
///
/// # Example
///
/// ```no_run
/// use gemicro_core::{AgentContext, ToolAgent, ToolAgentConfig, LlmClient, LlmConfig};
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
}

impl ToolAgent {
    /// Create a new ToolAgent with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::InvalidConfig` if the configuration is invalid.
    pub fn new(config: ToolAgentConfig) -> Result<Self, AgentError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Get the function declarations for enabled tools.
    fn get_function_declarations(&self) -> Vec<FunctionDeclaration> {
        let mut declarations = Vec::new();

        for tool_type in &self.config.enabled_tools {
            match tool_type {
                ToolType::Calculator => {
                    declarations.push(CalculatorCallable.declaration());
                }
                ToolType::CurrentDateTime => {
                    declarations.push(CurrentDatetimeCallable.declaration());
                }
            }
        }

        declarations
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
        let functions = self.get_function_declarations();

        try_stream! {
            let start_time = Instant::now();
            let mut total_tokens: u32 = 0;
            let mut tokens_unavailable: usize = 0;

            yield AgentUpdate::custom(
                EVENT_TOOL_AGENT_STARTED,
                format!("Processing query with {} tools available", functions.len()),
                json!({
                    "query": query,
                    "tools": functions.iter().map(|f| f.name()).collect::<Vec<_>>(),
                }),
            );

            // Check timeout
            let timeout = remaining_time(start_time, config.timeout, "tool_agent")?;

            // Get the raw genai client from our LlmClient
            // Note: We need to use the client directly for function calling
            let client = context.llm.client();

            // Build the interaction with function calling
            let interaction = client
                .interaction()
                .with_model(MODEL)
                .with_system_instruction(&config.system_prompt)
                .with_text(&query)
                .with_functions(functions);

            // Execute with auto function calling and timeout
            let response_future = interaction.create_with_auto_functions();

            let response: InteractionResponse = with_timeout_and_cancellation(
                async {
                    response_future.await.map_err(|e| AgentError::Other(format!("Function calling failed: {}", e)))
                },
                timeout,
                &context.cancellation_token,
                || timeout_error(start_time, config.timeout, "tool_agent"),
            ).await?;

            // Extract the response text, logging if empty
            let answer = match response.text() {
                Some(text) if !text.is_empty() => text.to_string(),
                Some(_) => {
                    log::warn!("LLM returned empty text response for tool agent query");
                    String::new()
                }
                None => {
                    log::warn!("LLM response contained no text content for tool agent query");
                    String::new()
                }
            };

            // Try to get token usage from the response
            if let Some(usage) = response.usage.as_ref() {
                total_tokens = usage.total_tokens.unwrap_or(0) as u32;
            } else {
                tokens_unavailable += 1;
            }

            // Emit completion event
            yield AgentUpdate::custom(
                EVENT_TOOL_AGENT_COMPLETE,
                answer.clone(),
                json!({
                    "answer": answer,
                    "duration_ms": start_time.elapsed().as_millis() as u64,
                }),
            );

            // Emit standard final_result for ExecutionState/harness compatibility
            // Note: Tool call counts are not available with create_with_auto_functions() API
            let metadata = ResultMetadata {
                total_tokens,
                tokens_unavailable_count: tokens_unavailable,
                duration_ms: start_time.elapsed().as_millis() as u64,
                sub_queries_succeeded: 0,
                sub_queries_failed: 0,
            };
            yield AgentUpdate::final_result(answer, metadata);
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

    fn execute(&self, query: &str, context: AgentContext) -> super::AgentStream<'_> {
        Box::pin(ToolAgent::execute(self, query, context))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = ToolAgentConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_calculator_only_config_is_valid() {
        let config = ToolAgentConfig::calculator_only();
        assert!(config.validate().is_ok());
        assert_eq!(config.enabled_tools.len(), 1);
        assert_eq!(config.enabled_tools[0], ToolType::Calculator);
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
    fn test_config_rejects_empty_tools() {
        let config = ToolAgentConfig {
            enabled_tools: vec![],
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("enabled_tools"));
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
            enabled_tools: vec![],
        };
        let result = config.validate();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("timeout"));
        assert!(err.contains("system_prompt"));
        assert!(err.contains("enabled_tools"));
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
    fn test_get_function_declarations() {
        let agent = ToolAgent::new(ToolAgentConfig::default()).unwrap();
        let declarations = agent.get_function_declarations();
        assert_eq!(declarations.len(), 2);

        let names: Vec<&str> = declarations.iter().map(|d| d.name()).collect();
        assert!(names.contains(&"calculator"));
        assert!(names.contains(&"current_datetime"));
    }

    #[test]
    fn test_calculator_tool_directly() {
        // Test the calculator function directly
        assert_eq!(calculator("2 + 2".to_string()), "4");
        assert_eq!(calculator("10 / 4".to_string()), "2.5");
        assert_eq!(calculator("sqrt(16)".to_string()), "4");
        assert!(calculator("invalid".to_string()).starts_with("Error:"));
    }

    #[test]
    fn test_current_datetime_tool_directly() {
        let result = current_datetime("UTC".to_string());
        assert!(result.contains("timezone"));
        assert!(result.contains("date"));
        assert!(result.contains("time"));
    }

    #[test]
    fn test_event_constants_are_unique() {
        let events = vec![EVENT_TOOL_AGENT_STARTED, EVENT_TOOL_AGENT_COMPLETE];
        let mut unique = std::collections::HashSet::new();
        for event in &events {
            assert!(unique.insert(*event), "Duplicate event: {}", event);
        }
    }
}
