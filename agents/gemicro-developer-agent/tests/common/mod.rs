//! Shared test utilities for DeveloperAgent integration tests

#![allow(dead_code)]

use async_trait::async_trait;
use gemicro_bash::Bash;
use gemicro_core::tool::{default_batch_confirm, AutoApprove, ConfirmationHandler, ToolRegistry};
use gemicro_core::{
    AgentContext, BatchApproval, BatchConfirmationHandler, LlmClient, LlmConfig, ToolBatch,
};
use gemicro_file_read::FileRead;
use gemicro_glob::Glob;
use gemicro_grep::Grep;
use serde_json::Value;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Helper to get the API key, or return None if not available.
pub fn get_api_key() -> Option<String> {
    env::var("GEMINI_API_KEY").ok()
}

/// Create a test AgentContext with tools for DeveloperAgent testing.
///
/// Includes FileRead, Glob, Grep, and Bash tools with AutoApprove confirmation handler.
pub fn create_test_context(api_key: &str) -> AgentContext {
    let genai_client = genai_rs::Client::builder(api_key.to_string())
        .build()
        .unwrap();
    let config = LlmConfig::default()
        .with_timeout(Duration::from_secs(60))
        .with_max_tokens(4096)
        .with_temperature(0.7)
        .with_max_retries(1)
        .with_retry_base_delay_ms(500);
    let llm = LlmClient::new(genai_client, config);

    // Create tool registry with tools for testing
    // Note: Bash requires confirmation (handled by AutoApprove)
    let mut tools = ToolRegistry::new();
    tools.register(FileRead);
    tools.register(Glob);
    tools.register(Grep);
    tools.register(Bash);

    AgentContext::new(llm)
        .with_tools(tools)
        .with_confirmation_handler(Arc::new(AutoApprove))
}

/// Create a test AgentContext with a custom confirmation handler.
pub fn create_test_context_with_handler(
    api_key: &str,
    handler: Arc<dyn BatchConfirmationHandler>,
) -> AgentContext {
    let genai_client = genai_rs::Client::builder(api_key.to_string())
        .build()
        .unwrap();
    let config = LlmConfig::default()
        .with_timeout(Duration::from_secs(60))
        .with_max_tokens(4096)
        .with_temperature(0.7)
        .with_max_retries(1)
        .with_retry_base_delay_ms(500);
    let llm = LlmClient::new(genai_client, config);

    // Create tool registry with tools for testing
    let mut tools = ToolRegistry::new();
    tools.register(FileRead);
    tools.register(Glob);
    tools.register(Grep);
    tools.register(Bash);

    AgentContext::new(llm)
        .with_tools(tools)
        .with_confirmation_handler(handler)
}

/// Configurable confirmation handler for testing.
///
/// Implements ConfirmationHandler with configurable approval behavior.
/// The BatchConfirmationHandler blanket impl provides default batch behavior
/// (falls back to individual confirms).
#[derive(Debug)]
pub struct TestConfirmationHandler {
    /// Whether to approve individual confirmations.
    pub approve: bool,
    /// Counter for confirm calls.
    pub confirm_count: AtomicUsize,
}

impl TestConfirmationHandler {
    /// Create handler that approves all confirmations.
    pub fn approve_all() -> Self {
        Self {
            approve: true,
            confirm_count: AtomicUsize::new(0),
        }
    }

    /// Create handler that denies all confirmations.
    pub fn deny_all() -> Self {
        Self {
            approve: false,
            confirm_count: AtomicUsize::new(0),
        }
    }

    /// Get the number of times confirm was called.
    pub fn confirm_calls(&self) -> usize {
        self.confirm_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl ConfirmationHandler for TestConfirmationHandler {
    async fn confirm(&self, _tool_name: &str, _message: &str, _args: &Value) -> bool {
        self.confirm_count.fetch_add(1, Ordering::SeqCst);
        self.approve
    }
}

#[async_trait]
impl BatchConfirmationHandler for TestConfirmationHandler {
    async fn confirm_batch(&self, batch: &ToolBatch) -> BatchApproval {
        // Use default behavior: falls back to individual confirms
        default_batch_confirm(self, batch, true).await
    }
}
