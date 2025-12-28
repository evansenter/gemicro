//! Shared test utilities for integration tests

#![allow(dead_code)]

use gemicro_core::{AgentContext, LlmClient, LlmConfig};
use std::env;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Helper to get the API key, or return None if not available.
pub fn get_api_key() -> Option<String> {
    env::var("GEMINI_API_KEY").ok()
}

/// Create a test AgentContext with appropriate settings for testing.
pub fn create_test_context(api_key: &str) -> AgentContext {
    let genai_client = rust_genai::Client::builder(api_key.to_string()).build();
    let config = LlmConfig {
        timeout: Duration::from_secs(60),
        max_tokens: 4096,
        temperature: 0.7,
        max_retries: 1,
        retry_base_delay_ms: 500,
    };
    let llm = LlmClient::new(genai_client, config);
    AgentContext::new(llm)
}

/// Create a test AgentContext with cancellation support.
pub fn create_test_context_with_cancellation(
    api_key: &str,
    cancellation_token: CancellationToken,
) -> AgentContext {
    let genai_client = rust_genai::Client::builder(api_key.to_string()).build();
    let config = LlmConfig {
        timeout: Duration::from_secs(60),
        max_tokens: 4096,
        temperature: 0.7,
        max_retries: 1,
        retry_base_delay_ms: 500,
    };
    let llm = LlmClient::new(genai_client, config);
    AgentContext::new_with_cancellation(llm, cancellation_token)
}
