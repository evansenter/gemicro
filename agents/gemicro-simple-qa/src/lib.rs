//! SimpleQA Agent - A minimal reference implementation for agent authoring.
//!
//! This agent demonstrates the essential patterns for implementing agents in Gemicro:
//! - Agent-specific configuration with validation
//! - Soft-typed events following Evergreen philosophy
//! - Timeout and cancellation handling
//! - Streaming execution with `async_stream`
//!
//! See `docs/AGENT_AUTHORING.md` for a detailed walkthrough of these patterns.
//!
//! # Example
//!
//! ```no_run
//! use gemicro_simple_qa::{SimpleQaAgent, SimpleQaConfig};
//! use gemicro_core::{Agent, AgentContext, LlmClient, LlmConfig};
//! use futures_util::StreamExt;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let agent = SimpleQaAgent::new(SimpleQaConfig::default())?;
//!
//! let genai = rust_genai::Client::builder("api-key".to_string()).build();
//! let context = AgentContext::new(LlmClient::new(genai, LlmConfig::default()));
//!
//! let stream = agent.execute("What is Rust?", context);
//! futures_util::pin_mut!(stream);
//!
//! while let Some(update) = stream.next().await {
//!     let update = update?;
//!     match update.event_type.as_str() {
//!         "simple_qa_started" => println!("Processing..."),
//!         "simple_qa_result" => println!("Answer: {}", update.message),
//!         _ => {} // Ignore unknown events
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use gemicro_core::{
    remaining_time, timeout_error, truncate, with_timeout_and_cancellation, Agent, AgentContext,
    AgentError, AgentStream, AgentUpdate, LlmRequest, ResultMetadata,
};

use async_stream::try_stream;
use serde_json::json;
use std::time::{Duration, Instant};

// ============================================================================
// Event Type Constants
// ============================================================================

/// Event emitted when the agent starts processing a query.
const EVENT_SIMPLE_QA_STARTED: &str = "simple_qa_started";

/// Event emitted when the agent produces its final result.
const EVENT_SIMPLE_QA_RESULT: &str = "simple_qa_result";

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the SimpleQA agent.
///
/// # Example
///
/// ```
/// use gemicro_simple_qa::SimpleQaConfig;
/// use std::time::Duration;
///
/// let config = SimpleQaConfig {
///     timeout: Duration::from_secs(30),
///     system_prompt: "You are a helpful assistant.".to_string(),
/// };
///
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct SimpleQaConfig {
    /// Total timeout for the query execution.
    pub timeout: Duration,

    /// System prompt to use for the LLM request.
    pub system_prompt: String,
}

impl Default for SimpleQaConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            system_prompt:
                "You are a helpful assistant. Answer questions concisely and accurately."
                    .to_string(),
        }
    }
}

impl SimpleQaConfig {
    /// Validate the configuration.
    ///
    /// Returns an error if:
    /// - Timeout is zero
    /// - System prompt is empty or whitespace-only
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_simple_qa::SimpleQaConfig;
    /// use std::time::Duration;
    ///
    /// let invalid = SimpleQaConfig {
    ///     timeout: Duration::ZERO,
    ///     system_prompt: "Valid prompt".to_string(),
    /// };
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

/// A minimal question-answering agent for demonstration.
///
/// SimpleQA makes a single LLM call to answer a question. It demonstrates:
/// - Config validation in constructor
/// - Soft-typed event emission
/// - Timeout and cancellation handling
///
/// # Example
///
/// ```no_run
/// use gemicro_simple_qa::{SimpleQaAgent, SimpleQaConfig};
/// use gemicro_core::{AgentContext, Agent, LlmClient, LlmConfig};
/// use futures_util::StreamExt;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let agent = SimpleQaAgent::new(SimpleQaConfig::default())?;
///
/// let genai = rust_genai::Client::builder("api-key".to_string()).build();
/// let context = AgentContext::new(LlmClient::new(genai, LlmConfig::default()));
///
/// let stream = agent.execute("What is Rust?", context);
/// futures_util::pin_mut!(stream);
///
/// while let Some(update) = stream.next().await {
///     let update = update?;
///     match update.event_type.as_str() {
///         "simple_qa_started" => println!("Processing..."),
///         "simple_qa_result" => println!("Answer: {}", update.message),
///         _ => {} // Ignore unknown events
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub struct SimpleQaAgent {
    config: SimpleQaConfig,
}

impl SimpleQaAgent {
    /// Create a new SimpleQA agent with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::InvalidConfig` if the configuration is invalid.
    pub fn new(config: SimpleQaConfig) -> Result<Self, AgentError> {
        config.validate()?;
        Ok(Self { config })
    }
}

impl Agent for SimpleQaAgent {
    fn name(&self) -> &str {
        "simple_qa"
    }

    fn description(&self) -> &str {
        "A simple question-answering agent that makes a single LLM call"
    }

    fn execute(&self, query: &str, context: AgentContext) -> AgentStream<'_> {
        let query = query.to_string();
        let config = self.config.clone();

        Box::pin(try_stream! {
            let start = Instant::now();

            // Emit start event
            yield AgentUpdate::custom(
                EVENT_SIMPLE_QA_STARTED,
                format!("Processing query: {}", truncate(&query, 50)),
                json!({ "query": query }),
            );

            // Calculate remaining time and execute LLM call
            let timeout = remaining_time(start, config.timeout, "query")?;

            let request = LlmRequest::with_system(&query, &config.system_prompt);

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

            let answer = response.text().unwrap_or("").to_string();
            let tokens_used = gemicro_core::extract_total_tokens(&response);
            let duration_ms = start.elapsed().as_millis() as u64;

            // Emit agent-specific result event
            yield AgentUpdate::custom(
                EVENT_SIMPLE_QA_RESULT,
                answer.clone(),
                json!({
                    "answer": answer,
                    "tokens_used": tokens_used,
                    "duration_ms": duration_ms,
                }),
            );

            // Emit standard final_result for ExecutionState/harness compatibility
            let metadata = ResultMetadata::new(
                tokens_used.unwrap_or(0),
                if tokens_used.is_none() { 1 } else { 0 },
                duration_ms,
            );
            yield AgentUpdate::final_result(answer, metadata);
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
        let config = SimpleQaConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_rejects_zero_timeout() {
        let config = SimpleQaConfig {
            timeout: Duration::ZERO,
            system_prompt: "Valid prompt".to_string(),
        };
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
        let config = SimpleQaConfig {
            timeout: Duration::from_secs(30),
            system_prompt: "   ".to_string(),
        };
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
        let config = SimpleQaConfig {
            timeout: Duration::ZERO,
            system_prompt: "".to_string(),
        };
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
        let invalid_config = SimpleQaConfig {
            timeout: Duration::ZERO,
            system_prompt: "Valid".to_string(),
        };
        assert!(SimpleQaAgent::new(invalid_config).is_err());
    }

    #[test]
    fn test_agent_name_and_description() {
        let agent = SimpleQaAgent::new(SimpleQaConfig::default()).unwrap();
        assert_eq!(agent.name(), "simple_qa");
        assert!(!agent.description().is_empty());
    }

    #[test]
    fn test_event_constants_are_unique() {
        assert_ne!(EVENT_SIMPLE_QA_STARTED, EVENT_SIMPLE_QA_RESULT);
    }
}
