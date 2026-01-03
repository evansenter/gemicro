//! Shared test utilities for integration tests

#![allow(dead_code)]

use gemicro_core::{AgentContext, LlmClient, LlmConfig};
use std::env;
use std::time::Duration;

/// Helper to get the API key, or return None if not available.
pub fn get_api_key() -> Option<String> {
    env::var("GEMINI_API_KEY").ok()
}

/// Create a test AgentContext with appropriate settings for testing.
pub fn create_test_context(api_key: &str) -> AgentContext {
    let genai_client = rust_genai::Client::builder(api_key.to_string())
        .build()
        .unwrap();
    let config = LlmConfig::default()
        .with_timeout(Duration::from_secs(60))
        .with_max_tokens(4096)
        .with_temperature(0.7)
        .with_max_retries(1)
        .with_retry_base_delay_ms(500);
    let llm = LlmClient::new(genai_client, config);
    AgentContext::new(llm)
}
