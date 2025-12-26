//! LLM Judge Agent
//!
//! A simple agent that uses structured output to judge whether a predicted
//! answer is semantically correct compared to a ground truth.
//!
//! # Design Philosophy
//!
//! Following [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) principles:
//! - An LLM Judge is just an agent with a prompt
//! - Uses structured output for reliable JSON responses
//! - Can be registered in `AgentRegistry` for A/B testing of different judges
//!
//! This module lives in `gemicro-eval` (not `gemicro-core`) to demonstrate that
//! agents can be defined in any crate that depends on `gemicro-core`.
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::{Agent, AgentContext, LlmClient, LlmConfig};
//! use gemicro_eval::{LlmJudgeAgent, JudgeConfig, JudgeInput};
//! use futures_util::StreamExt;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let genai_client = rust_genai::Client::builder("api-key".to_string()).build();
//! let llm = LlmClient::new(genai_client, LlmConfig::default());
//! let context = AgentContext::new(llm);
//!
//! let judge = LlmJudgeAgent::new(JudgeConfig::default());
//!
//! // Query format: JSON with predicted and ground_truth
//! let input = JudgeInput::new("Paris", "Paris is the capital of France");
//! let stream = judge.execute(&input.to_query(), context);
//! futures_util::pin_mut!(stream);
//!
//! while let Some(update) = stream.next().await {
//!     let update = update?;
//!     if update.event_type == "judge_result" {
//!         let correct: bool = update.data["correct"].as_bool().unwrap_or(false);
//!         println!("Correct: {}", correct);
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use gemicro_core::agent::{Agent, AgentContext, AgentStream};
use gemicro_core::error::AgentError;
use gemicro_core::llm::LlmRequest;
use gemicro_core::update::AgentUpdate;

use async_stream::try_stream;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Duration;

/// Event type for judge starting evaluation
pub const EVENT_JUDGE_STARTED: &str = "judge_started";

/// Event type for judge result
pub const EVENT_JUDGE_RESULT: &str = "judge_result";

/// Configuration for the LLM Judge agent
#[derive(Debug, Clone)]
pub struct JudgeConfig {
    /// System instruction for the judge
    pub system_instruction: String,

    /// Timeout for the LLM call
    pub timeout: Duration,
}

impl Default for JudgeConfig {
    fn default() -> Self {
        Self {
            system_instruction: DEFAULT_JUDGE_SYSTEM.to_string(),
            timeout: Duration::from_secs(30),
        }
    }
}

impl JudgeConfig {
    /// Create a new config with a custom system instruction
    pub fn with_system_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.system_instruction = instruction.into();
        self
    }

    /// Set the timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Default system instruction for the judge
const DEFAULT_JUDGE_SYSTEM: &str = r#"You are an expert evaluator comparing predicted answers to ground truth answers.

Your task is to determine if the predicted answer is semantically correct - meaning it conveys the same essential information as the ground truth, even if worded differently.

Guidelines:
- Focus on semantic meaning, not exact wording
- A correct answer may be more or less detailed than the ground truth
- Case and punctuation differences don't matter
- Partial answers that contain the key information are correct
- Answers that are factually equivalent are correct

Respond with your evaluation in the specified JSON format."#;

/// Input format for the judge query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeInput {
    /// The predicted answer from an agent
    pub predicted: String,

    /// The expected ground truth answer
    pub ground_truth: String,
}

impl JudgeInput {
    /// Create a new judge input
    pub fn new(predicted: impl Into<String>, ground_truth: impl Into<String>) -> Self {
        Self {
            predicted: predicted.into(),
            ground_truth: ground_truth.into(),
        }
    }

    /// Serialize to JSON string for use as query
    pub fn to_query(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

/// Output format from the judge (structured output schema)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeOutput {
    /// Whether the predicted answer is semantically correct
    pub correct: bool,

    /// Brief reasoning for the judgment
    pub reasoning: String,
}

impl JudgeOutput {
    /// Get the JSON schema for structured output
    pub fn schema() -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "correct": {
                    "type": "boolean",
                    "description": "True if the predicted answer is semantically correct"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why the answer is correct or incorrect"
                }
            },
            "required": ["correct", "reasoning"]
        })
    }
}

/// LLM Judge Agent
///
/// Uses structured output to reliably judge semantic correctness of answers.
/// The query should be a JSON string with `predicted` and `ground_truth` fields.
///
/// # Events
///
/// - `judge_started`: Evaluation beginning
/// - `judge_result`: Final result with `correct` boolean and `reasoning`
pub struct LlmJudgeAgent {
    config: JudgeConfig,
}

impl LlmJudgeAgent {
    /// Create a new LLM Judge agent
    pub fn new(config: JudgeConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default_agent() -> Self {
        Self::new(JudgeConfig::default())
    }
}

impl Agent for LlmJudgeAgent {
    fn name(&self) -> &str {
        "llm_judge"
    }

    fn description(&self) -> &str {
        "Evaluates predicted answers against ground truth using semantic comparison"
    }

    fn execute(&self, query: &str, context: AgentContext) -> AgentStream<'_> {
        let query = query.to_string();
        let config = self.config.clone();

        Box::pin(try_stream! {
            // Parse input
            let input: JudgeInput = serde_json::from_str(&query)
                .map_err(|e| AgentError::InvalidConfig(format!(
                    "Invalid judge input format. Expected JSON with 'predicted' and 'ground_truth' fields. Error: {}",
                    e
                )))?;

            // Emit started event
            yield AgentUpdate::custom(
                EVENT_JUDGE_STARTED,
                "Starting semantic evaluation",
                json!({
                    "predicted_length": input.predicted.len(),
                    "ground_truth_length": input.ground_truth.len()
                })
            );

            // Build prompt
            let prompt = format!(
                "Compare these two answers:\n\n\
                Ground Truth: {}\n\n\
                Predicted: {}\n\n\
                Is the predicted answer semantically correct?",
                input.ground_truth,
                input.predicted
            );

            // Create request with structured output
            let request = LlmRequest::with_system(&prompt, &config.system_instruction)
                .with_response_format(JudgeOutput::schema());

            // Execute LLM call
            let response = context.llm.generate(request).await
                .map_err(AgentError::Llm)?;

            // Parse structured output
            let output: JudgeOutput = serde_json::from_str(&response.text)
                .map_err(|e| AgentError::ParseFailed(format!(
                    "Failed to parse judge output: {}. Response: {}",
                    e,
                    gemicro_core::truncate(&response.text, 200)
                )))?;

            // Emit result event
            yield AgentUpdate::custom(
                EVENT_JUDGE_RESULT,
                if output.correct { "Answer is correct" } else { "Answer is incorrect" },
                json!({
                    "correct": output.correct,
                    "reasoning": output.reasoning,
                    "tokens_used": response.tokens_used
                })
            );
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_judge_config_default() {
        let config = JudgeConfig::default();
        assert!(!config.system_instruction.is_empty());
        assert_eq!(config.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_judge_config_builder() {
        let config = JudgeConfig::default()
            .with_system_instruction("Custom instruction")
            .with_timeout(Duration::from_secs(60));

        assert_eq!(config.system_instruction, "Custom instruction");
        assert_eq!(config.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_judge_input_creation() {
        let input = JudgeInput::new("Paris", "The capital of France is Paris");
        assert_eq!(input.predicted, "Paris");
        assert_eq!(input.ground_truth, "The capital of France is Paris");
    }

    #[test]
    fn test_judge_input_to_query() {
        let input = JudgeInput::new("Paris", "Paris");
        let query = input.to_query();
        let parsed: JudgeInput = serde_json::from_str(&query).unwrap();
        assert_eq!(parsed.predicted, "Paris");
        assert_eq!(parsed.ground_truth, "Paris");
    }

    #[test]
    fn test_judge_output_schema() {
        let schema = JudgeOutput::schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["correct"].is_object());
        assert!(schema["properties"]["reasoning"].is_object());
        assert_eq!(schema["required"], json!(["correct", "reasoning"]));
    }

    #[test]
    fn test_judge_output_deserialization() {
        let json = r#"{"correct": true, "reasoning": "Both say Paris"}"#;
        let output: JudgeOutput = serde_json::from_str(json).unwrap();
        assert!(output.correct);
        assert_eq!(output.reasoning, "Both say Paris");
    }

    #[test]
    fn test_llm_judge_agent_name() {
        let agent = LlmJudgeAgent::default_agent();
        assert_eq!(agent.name(), "llm_judge");
    }

    #[test]
    fn test_llm_judge_agent_description() {
        let agent = LlmJudgeAgent::default_agent();
        assert!(!agent.description().is_empty());
    }
}
