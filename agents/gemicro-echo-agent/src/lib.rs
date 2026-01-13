//! Echo Agent - A minimal agent for CLI integration testing.
//!
//! This agent echoes the input query without making any LLM calls, enabling:
//! - **Fast tests**: No network latency or API costs
//! - **Deterministic output**: Always returns predictable results
//! - **Decoupled testing**: CLI validation doesn't depend on specific agent behavior
//!
//! # Example
//!
//! ```
//! use gemicro_echo_agent::EchoAgent;
//! use gemicro_core::{Agent, AgentContext, LlmClient, LlmConfig};
//! use futures_util::StreamExt;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let agent = EchoAgent;
//!
//! let genai = genai_rs::Client::builder("unused".to_string()).build()?;
//! let context = AgentContext::new(LlmClient::new(genai, LlmConfig::default()));
//!
//! let stream = agent.execute("Hello, world!", context);
//! futures_util::pin_mut!(stream);
//!
//! while let Some(update) = stream.next().await {
//!     let update = update?;
//!     if let Some(result) = update.as_final_result() {
//!         let answer = result.result.as_str().unwrap_or("");
//!         assert!(answer.contains("Hello, world!"));
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use async_stream::try_stream;
use gemicro_core::{Agent, AgentContext, AgentStream, AgentUpdate, DefaultTracker, ResultMetadata};
use serde_json::json;

/// A minimal agent that echoes input without LLM calls.
///
/// Designed for CLI integration testing where we want to verify:
/// - CLI argument parsing works correctly
/// - Agent execution pipeline functions end-to-end
/// - Output formatting and rendering behaves as expected
///
/// Does not require API keys or network access.
#[derive(Debug, Clone, Copy, Default)]
pub struct EchoAgent;

impl Agent for EchoAgent {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echoes input without LLM calls (for testing)"
    }

    fn execute(&self, query: &str, _context: AgentContext) -> AgentStream<'_> {
        let query = query.to_string();

        Box::pin(try_stream! {
            // Emit final result immediately - no processing needed
            let answer = format!("Echo: {}", query);
            yield AgentUpdate::final_result(
                json!(answer),
                ResultMetadata::new(0, 0, 0),
            );
        })
    }

    fn create_tracker(&self) -> Box<dyn gemicro_core::ExecutionTracking> {
        Box::new(DefaultTracker::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use gemicro_core::{LlmClient, LlmConfig};

    #[test]
    fn test_agent_name() {
        let agent = EchoAgent;
        assert_eq!(agent.name(), "echo");
    }

    #[test]
    fn test_agent_description() {
        let agent = EchoAgent;
        assert!(!agent.description().is_empty());
        assert!(agent.description().contains("test"));
    }

    #[tokio::test]
    async fn test_echo_output() {
        let agent = EchoAgent;

        let genai = genai_rs::Client::builder("unused".to_string())
            .build()
            .unwrap();
        let context = AgentContext::new(LlmClient::new(genai, LlmConfig::default()));

        let stream = agent.execute("Hello, world!", context);
        futures_util::pin_mut!(stream);

        let mut events = Vec::new();
        while let Some(result) = stream.next().await {
            events.push(result.expect("should not error"));
        }

        // Should emit exactly one event: final_result
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "final_result");

        // Verify echo content
        let result = events[0].as_final_result().expect("should be final_result");
        let answer = result.result.as_str().unwrap_or("");
        assert!(answer.contains("Echo: Hello, world!"));
    }

    #[tokio::test]
    async fn test_zero_tokens() {
        let agent = EchoAgent;

        let genai = genai_rs::Client::builder("unused".to_string())
            .build()
            .unwrap();
        let context = AgentContext::new(LlmClient::new(genai, LlmConfig::default()));

        let stream = agent.execute("test", context);
        futures_util::pin_mut!(stream);

        let event = stream.next().await.unwrap().unwrap();
        let result = event.as_final_result().unwrap();

        // No LLM calls = zero tokens
        assert_eq!(result.metadata.total_tokens, 0);
    }
}
