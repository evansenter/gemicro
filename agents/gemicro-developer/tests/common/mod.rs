//! Shared test utilities for DeveloperAgent integration tests

#![allow(dead_code)]

use gemicro_core::tool::{AutoApprove, ToolRegistry};
use gemicro_core::{AgentContext, LlmClient, LlmConfig};
use gemicro_file_read::FileRead;
use gemicro_glob::Glob;
use std::env;
use std::sync::Arc;
use std::time::Duration;

/// Helper to get the API key, or return None if not available.
pub fn get_api_key() -> Option<String> {
    env::var("GEMINI_API_KEY").ok()
}

/// Create a test AgentContext with tools for DeveloperAgent testing.
///
/// Includes FileRead and Glob tools with AutoApprove confirmation handler.
pub fn create_test_context(api_key: &str) -> AgentContext {
    let genai_client = rust_genai::Client::builder(api_key.to_string()).build();
    let config = LlmConfig::default()
        .with_timeout(Duration::from_secs(60))
        .with_max_tokens(4096)
        .with_temperature(0.7)
        .with_max_retries(1)
        .with_retry_base_delay_ms(500);
    let llm = LlmClient::new(genai_client, config);

    // Create tool registry with read-only tools
    let mut tools = ToolRegistry::new();
    tools.register(FileRead);
    tools.register(Glob);

    AgentContext::new(llm)
        .with_tools(tools)
        .with_confirmation_handler(Arc::new(AutoApprove))
}
