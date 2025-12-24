//! Shared test utilities for integration tests
//!
//! This module provides common helper functions used across integration test files.

// Allow unused code - each test file includes this module separately,
// so not all functions are used in every compilation unit.
#![allow(dead_code)]

use gemicro_core::{AgentContext, LlmClient, LlmConfig};
use std::env;
use std::time::Duration;

/// Helper to get the API key, or return None if not available.
/// Tests should check for None and skip if the key is not set.
pub fn get_api_key() -> Option<String> {
    env::var("GEMINI_API_KEY").ok()
}

/// Create a test LlmClient with appropriate timeouts for testing.
///
/// Uses shorter timeouts and lower token counts to keep tests fast.
pub fn create_test_client(api_key: &str) -> LlmClient {
    let genai_client = rust_genai::Client::builder(api_key.to_string()).build();
    let config = LlmConfig {
        timeout: Duration::from_secs(60),
        max_tokens: 256,
        temperature: 0.0, // Deterministic for testing
        max_retries: 1,
        retry_base_delay_ms: 500,
    };
    LlmClient::new(genai_client, config)
}

/// Create a test AgentContext with appropriate timeouts for testing.
///
/// Uses slightly higher token limit than `create_test_client` for agent tests.
pub fn create_test_context(api_key: &str) -> AgentContext {
    let genai_client = rust_genai::Client::builder(api_key.to_string()).build();
    let config = LlmConfig {
        timeout: Duration::from_secs(60),
        max_tokens: 512,
        temperature: 0.7,
        max_retries: 1,
        retry_base_delay_ms: 500,
    };
    let llm = LlmClient::new(genai_client, config);
    AgentContext::new(llm)
}
