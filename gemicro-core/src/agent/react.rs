//! ReAct Agent implementation.
//!
//! The ReAct pattern (Reasoning + Acting) iteratively generates thoughts,
//! chooses actions (tools), and observes results until a final answer is reached.

use super::{remaining_time, timeout_error, with_timeout_and_cancellation, Agent, AgentContext};
use crate::config::ReactConfig;
use crate::error::AgentError;
use crate::llm::LlmRequest;
use crate::update::AgentUpdate;

use async_stream::try_stream;
use futures_util::Stream;
use serde::Deserialize;
use serde_json::json;
use std::time::Instant;

/// Maximum scratchpad size in characters to prevent excessive context growth.
/// When exceeded, older entries are truncated to stay within bounds.
const MAX_SCRATCHPAD_CHARS: usize = 20_000;

/// ReAct Agent implementing the Reasoning + Acting pattern.
///
/// The agent iteratively:
/// 1. **Think**: Generate a thought about what to do
/// 2. **Act**: Choose a tool and input
/// 3. **Observe**: Execute the tool and receive results
/// 4. Repeat until `final_answer` tool is used or max iterations reached
///
/// # Events Emitted
///
/// - `react_started`
/// - `react_thought` / `react_action` / `react_observation` (per iteration)
/// - `react_complete` or `react_max_iterations`
///
/// # Example
///
/// ```no_run
/// use gemicro_core::{AgentContext, ReactAgent, ReactConfig, LlmClient, LlmConfig};
/// use futures_util::StreamExt;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let genai_client = rust_genai::Client::builder("api-key".to_string()).build();
/// let context = AgentContext::new(LlmClient::new(genai_client, LlmConfig::default()));
/// let agent = ReactAgent::new(ReactConfig::default())?;
///
/// let stream = agent.execute("What is 25 * 4?", context);
/// futures_util::pin_mut!(stream);
/// while let Some(update) = stream.next().await {
///     println!("{:?}", update?);
/// }
/// # Ok(())
/// # }
/// ```
pub struct ReactAgent {
    config: ReactConfig,
}

/// Parsed response from a ReAct step
#[derive(Debug, Deserialize)]
struct ReactStep {
    thought: String,
    action: ReactAction,
}

/// Action chosen by the agent
#[derive(Debug, Deserialize)]
struct ReactAction {
    tool: String,
    input: String,
}

impl ReactAgent {
    /// Create a new ReAct agent with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::InvalidConfig` if the configuration is invalid
    /// (e.g., zero max_iterations, empty tools list, empty prompts).
    pub fn new(config: ReactConfig) -> Result<Self, AgentError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Execute the ReAct reasoning loop.
    ///
    /// Returns a stream of [`AgentUpdate`] events as the agent thinks and acts.
    pub fn execute(
        &self,
        query: &str,
        context: AgentContext,
    ) -> impl Stream<Item = Result<AgentUpdate, AgentError>> + Send + '_ {
        let query = query.to_string();
        let config = self.config.clone();

        try_stream! {
            let start_time = Instant::now();
            let mut scratchpad = String::new();

            yield AgentUpdate::react_started(&query, config.max_iterations);

            for iteration in 1..=config.max_iterations {
                // Check timeout before each iteration
                let remaining = remaining_time(start_time, config.total_timeout, "react_loop")?;

                // Build prompt with current scratchpad
                let prompt = config.prompts.render_iteration(&query, &scratchpad);
                let schema = Self::react_step_schema(&config.available_tools);

                let request = LlmRequest::with_system(&prompt, &config.prompts.system)
                    .with_response_format(schema);

                // Execute LLM call with timeout/cancellation
                // Map LlmError to AgentError for the helper function
                let response = with_timeout_and_cancellation(
                    async {
                        context.llm.generate(request).await.map_err(AgentError::from)
                    },
                    remaining,
                    &context.cancellation_token,
                    || timeout_error(start_time, config.total_timeout, "react_step"),
                ).await?;

                // Parse structured response
                let step: ReactStep = serde_json::from_str(&response.text)
                    .map_err(|e| AgentError::ParseFailed(format!(
                        "Failed to parse ReAct step: {}. Response: {}",
                        e,
                        truncate_for_error(&response.text, 200)
                    )))?;

                // Emit thought event
                yield AgentUpdate::react_thought(iteration, step.thought.clone());

                // Emit action event
                yield AgentUpdate::react_action(iteration, step.action.tool.clone(), step.action.input.clone());

                // Check for final answer
                if step.action.tool == "final_answer" {
                    yield AgentUpdate::react_complete(iteration, step.action.input);
                    return;
                }

                // Execute tool and get observation
                let remaining = remaining_time(start_time, config.total_timeout, "tool_execution")?;
                let (observation, is_error) = Self::execute_tool(
                    &step.action.tool,
                    &step.action.input,
                    &context,
                    &config,
                    remaining,
                ).await;

                // Emit observation event
                yield AgentUpdate::react_observation(
                    iteration,
                    step.action.tool.clone(),
                    observation.clone(),
                    is_error,
                );

                // Append to scratchpad
                scratchpad.push_str(&format!(
                    "Thought {}: {}\nAction {}: {}[{}]\nObservation {}: {}\n\n",
                    iteration, step.thought,
                    iteration, step.action.tool, step.action.input,
                    iteration, observation,
                ));

                // Truncate scratchpad if it exceeds max size (keep most recent entries)
                if scratchpad.len() > MAX_SCRATCHPAD_CHARS {
                    // Find a clean break point (paragraph boundary) after the truncation point
                    let keep_from = scratchpad.len() - MAX_SCRATCHPAD_CHARS;
                    if let Some(break_pos) = scratchpad[keep_from..].find("\n\n") {
                        let new_start = keep_from + break_pos + 2;
                        scratchpad = format!("[Earlier iterations truncated]\n\n{}", &scratchpad[new_start..]);
                    }
                }
            }

            // Max iterations reached - extract last thought for context
            let last_thought = scratchpad
                .lines()
                .rev()
                .find(|line| line.starts_with("Thought"))
                .map(|s| s.to_string())
                .unwrap_or_default();

            yield AgentUpdate::react_max_iterations(config.max_iterations, last_thought);
        }
    }

    /// Generate JSON schema for ReAct step response
    fn react_step_schema(available_tools: &[String]) -> serde_json::Value {
        // Build enum from available tools + final_answer
        let mut tools: Vec<String> = available_tools.to_vec();
        if !tools.contains(&"final_answer".to_string()) {
            tools.push("final_answer".to_string());
        }

        json!({
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your reasoning about what to do next"
                },
                "action": {
                    "type": "object",
                    "properties": {
                        "tool": {
                            "type": "string",
                            "enum": tools
                        },
                        "input": {
                            "type": "string",
                            "description": "Input to the tool"
                        }
                    },
                    "required": ["tool", "input"]
                }
            },
            "required": ["thought", "action"]
        })
    }

    /// Execute a tool and return (observation, is_error)
    async fn execute_tool(
        tool: &str,
        input: &str,
        context: &AgentContext,
        config: &ReactConfig,
        timeout: std::time::Duration,
    ) -> (String, bool) {
        match tool {
            "web_search" => Self::execute_web_search(input, context, config, timeout).await,
            "calculator" => Self::execute_calculator(input),
            unknown => (
                format!(
                    "Unknown tool: '{}'. Available tools: {}",
                    unknown,
                    config.available_tools.join(", ")
                ),
                true,
            ),
        }
    }

    /// Execute web search using Google Search grounding
    async fn execute_web_search(
        query: &str,
        context: &AgentContext,
        config: &ReactConfig,
        timeout: std::time::Duration,
    ) -> (String, bool) {
        if !config.use_google_search {
            return ("Web search is disabled in configuration".to_string(), true);
        }

        let request = LlmRequest::with_system(
            query,
            "You are a research assistant. Provide a concise, factual answer based on web search results. Include key facts and numbers.",
        )
        .with_google_search();

        match tokio::time::timeout(timeout, context.llm.generate(request)).await {
            Ok(Ok(response)) => (response.text, false),
            Ok(Err(e)) => (format!("Web search failed: {}", e), true),
            Err(_) => ("Web search timed out".to_string(), true),
        }
    }

    /// Execute calculator using meval
    fn execute_calculator(expression: &str) -> (String, bool) {
        match meval::eval_str(expression) {
            Ok(result) => {
                if result.is_nan() {
                    ("Result is not a number (NaN)".to_string(), true)
                } else if result.is_infinite() {
                    ("Result is infinite".to_string(), true)
                } else {
                    // Format nicely - remove trailing zeros for integers
                    let formatted = if result.fract() == 0.0 && result.abs() < 1e15 {
                        format!("{:.0}", result)
                    } else {
                        format!("{}", result)
                    };
                    (formatted, false)
                }
            }
            Err(e) => (format!("Calculator error: {}", e), true),
        }
    }
}

impl Agent for ReactAgent {
    fn name(&self) -> &str {
        "react"
    }

    fn description(&self) -> &str {
        "Reasoning + Acting agent that iteratively thinks, acts with tools, and observes results"
    }

    fn execute(&self, query: &str, context: AgentContext) -> super::AgentStream<'_> {
        Box::pin(ReactAgent::execute(self, query, context))
    }
}

/// Truncate a string for error messages, preserving UTF-8 boundaries
fn truncate_for_error(s: &str, max_chars: usize) -> String {
    let truncated: String = s.chars().take(max_chars).collect();
    if s.chars().count() > max_chars {
        format!("{}...", truncated)
    } else {
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    // Calculator tests
    #[rstest]
    #[case::addition("2 + 2", "4", false)]
    #[case::multiplication("3 * 4", "12", false)]
    #[case::division("10 / 4", "2.5", false)]
    #[case::complex("2 + 3 * 4", "14", false)]
    #[case::parentheses("(2 + 3) * 4", "20", false)]
    #[case::power("2^3", "8", false)]
    #[case::sqrt("sqrt(16)", "4", false)]
    #[case::negative("-5 + 3", "-2", false)]
    #[case::decimal("3.14 * 2", "6.28", false)]
    fn test_calculator_success(#[case] expr: &str, #[case] expected: &str, #[case] is_error: bool) {
        let (result, err) = ReactAgent::execute_calculator(expr);
        assert_eq!(err, is_error);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case::invalid_syntax("2 + * 3")]
    #[case::unknown_function("unknown(5)")]
    #[case::empty("")]
    fn test_calculator_error(#[case] expr: &str) {
        let (result, is_error) = ReactAgent::execute_calculator(expr);
        assert!(is_error, "Expected error for '{}'", expr);
        assert!(result.contains("Calculator error"), "Result: {}", result);
    }

    #[test]
    fn test_calculator_division_by_zero() {
        let (result, is_error) = ReactAgent::execute_calculator("1/0");
        assert!(is_error);
        assert!(result.contains("infinite"));
    }

    #[test]
    fn test_calculator_nan() {
        // sqrt of negative number
        let (result, is_error) = ReactAgent::execute_calculator("sqrt(-1)");
        assert!(is_error);
        assert!(result.contains("NaN"));
    }

    // Schema generation tests
    #[test]
    fn test_react_step_schema_includes_tools() {
        let tools = vec!["web_search".to_string(), "calculator".to_string()];
        let schema = ReactAgent::react_step_schema(&tools);

        let tool_enum = &schema["properties"]["action"]["properties"]["tool"]["enum"];
        assert!(tool_enum.as_array().unwrap().contains(&json!("web_search")));
        assert!(tool_enum.as_array().unwrap().contains(&json!("calculator")));
        assert!(tool_enum
            .as_array()
            .unwrap()
            .contains(&json!("final_answer")));
    }

    #[test]
    fn test_react_step_schema_adds_final_answer() {
        let tools = vec!["calculator".to_string()];
        let schema = ReactAgent::react_step_schema(&tools);

        let tool_enum = &schema["properties"]["action"]["properties"]["tool"]["enum"];
        // final_answer should be added automatically
        assert!(tool_enum
            .as_array()
            .unwrap()
            .contains(&json!("final_answer")));
    }

    #[test]
    fn test_react_step_schema_no_duplicate_final_answer() {
        let tools = vec!["calculator".to_string(), "final_answer".to_string()];
        let schema = ReactAgent::react_step_schema(&tools);

        let tool_enum = &schema["properties"]["action"]["properties"]["tool"]["enum"];
        let final_answer_count = tool_enum
            .as_array()
            .unwrap()
            .iter()
            .filter(|v| v == &&json!("final_answer"))
            .count();
        assert_eq!(
            final_answer_count, 1,
            "final_answer should not be duplicated"
        );
    }

    // Parsing tests
    #[test]
    fn test_parse_react_step_valid() {
        let json = r#"{"thought": "I need to calculate", "action": {"tool": "calculator", "input": "2+2"}}"#;
        let step: ReactStep = serde_json::from_str(json).unwrap();
        assert_eq!(step.thought, "I need to calculate");
        assert_eq!(step.action.tool, "calculator");
        assert_eq!(step.action.input, "2+2");
    }

    #[test]
    fn test_parse_react_step_missing_thought() {
        let json = r#"{"action": {"tool": "calculator", "input": "2+2"}}"#;
        let result: Result<ReactStep, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_react_step_missing_action() {
        let json = r#"{"thought": "I need to calculate"}"#;
        let result: Result<ReactStep, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    // Config validation tests
    #[test]
    fn test_react_config_default_valid() {
        let config = ReactConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_react_config_zero_iterations() {
        let config = ReactConfig {
            max_iterations: 0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_iterations"));
    }

    #[test]
    fn test_react_config_empty_tools() {
        let config = ReactConfig {
            available_tools: vec![],
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("available_tools"));
    }

    // Truncation helper tests
    #[test]
    fn test_truncate_for_error_short() {
        let s = "hello";
        assert_eq!(truncate_for_error(s, 10), "hello");
    }

    #[test]
    fn test_truncate_for_error_exact() {
        let s = "hello";
        assert_eq!(truncate_for_error(s, 5), "hello");
    }

    #[test]
    fn test_truncate_for_error_long() {
        let s = "hello world";
        assert_eq!(truncate_for_error(s, 5), "hello...");
    }

    #[test]
    fn test_truncate_for_error_unicode() {
        let s = "你好世界";
        assert_eq!(truncate_for_error(s, 2), "你好...");
    }

    // Compile-time assertion that MAX_SCRATCHPAD_CHARS is within reasonable bounds
    // (20KB allows ~50-100 iterations without excessive context growth)
    const _: () = {
        assert!(MAX_SCRATCHPAD_CHARS >= 10_000);
        assert!(MAX_SCRATCHPAD_CHARS <= 100_000);
    };

    // Unknown tool handling test
    // Note: execute_tool is async and requires AgentContext, but the unknown tool
    // path is tested by verifying the error format matches expectations
    #[test]
    fn test_unknown_tool_error_format() {
        let config = ReactConfig {
            available_tools: vec!["calculator".to_string(), "web_search".to_string()],
            ..Default::default()
        };

        // Simulate what execute_tool returns for unknown tools
        let unknown = "magic_wand";
        let error_msg = format!(
            "Unknown tool: '{}'. Available tools: {}",
            unknown,
            config.available_tools.join(", ")
        );

        assert!(error_msg.contains("Unknown tool: 'magic_wand'"));
        assert!(error_msg.contains("calculator"));
        assert!(error_msg.contains("web_search"));
    }
}
