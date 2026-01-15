//! Shared test utilities for integration tests
//!
//! This module provides common helper functions used across integration test files.

// Allow unused code - each test file includes this module separately,
// so not all functions are used in every compilation unit.
#![allow(dead_code)]

use gemicro_core::{AgentContext, LlmClient, LlmConfig};
use serde_json::json;
use std::env;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Helper to get the API key, or return None if not available.
/// Tests should check for None and skip if the key is not set.
pub fn get_api_key() -> Option<String> {
    env::var("GEMINI_API_KEY").ok()
}

/// Create a test LlmClient with appropriate settings for testing.
///
/// Uses shorter timeouts and sufficient token limits for reliable responses.
pub fn create_test_client(api_key: &str) -> LlmClient {
    let genai_client = genai_rs::Client::builder(api_key.to_string())
        .build()
        .unwrap();
    let config = LlmConfig::default()
        .with_timeout(Duration::from_secs(60))
        .with_max_tokens(4096)
        .with_temperature(0.0) // Deterministic for testing
        .with_max_retries(1)
        .with_retry_base_delay_ms(500);
    LlmClient::new(genai_client, config)
}

/// Create a test AgentContext with appropriate settings for testing.
///
/// Uses sufficient token limits to avoid truncated responses during agent execution.
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
    AgentContext::new(llm)
}

/// Create a test AgentContext with cancellation support.
///
/// Returns both the context and the cancellation token so tests can trigger cancellation.
pub fn create_test_context_with_cancellation(
    api_key: &str,
    cancellation_token: CancellationToken,
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
    AgentContext::new_with_cancellation(llm, cancellation_token)
}

/// Validate an LLM response semantically using another LLM call.
///
/// Instead of brittle string matching (e.g., `assert!(response.contains("12"))`),
/// this asks an LLM to judge whether the response correctly answers the question.
/// This is more robust against variations in phrasing, formatting, or explanation.
///
/// # Arguments
///
/// * `client` - The LLM client to use for validation
/// * `context` - Description of what was asked (e.g., "User asked 'What is 4 * 3?'")
/// * `response_text` - The actual LLM response to validate
/// * `validation_question` - What to check (e.g., "Does this response correctly state that the answer is 12?")
///
/// # Returns
///
/// A tuple of (is_valid, reason) where is_valid is true if the response passes validation.
pub async fn validate_response_semantically(
    client: &LlmClient,
    context: &str,
    response_text: &str,
    validation_question: &str,
) -> Result<(bool, String), String> {
    let validation_prompt = format!(
        "You are a test validator. Judge whether an LLM response is appropriate.\n\n\
         Context: {}\n\n\
         Response to validate: {}\n\n\
         Question: {}\n\n\
         Respond with JSON: {{\"is_valid\": true/false, \"reason\": \"brief explanation\"}}",
        context, response_text, validation_question
    );

    let schema = json!({
        "type": "object",
        "properties": {
            "is_valid": {
                "type": "boolean",
                "description": "Whether the response is semantically valid"
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation of the judgment"
            }
        },
        "required": ["is_valid", "reason"]
    });

    let request = client
        .client()
        .interaction()
        .with_model("gemini-3.0-flash-preview")
        .with_text(&validation_prompt)
        .with_response_format(schema)
        .build()
        .map_err(|e| e.to_string())?;

    let response = client.generate(request).await.map_err(|e| e.to_string())?;
    let text = response.text().unwrap_or("{}");

    // Parse the structured response
    let json: serde_json::Value =
        serde_json::from_str(text).unwrap_or(json!({"is_valid": false, "reason": "Parse error"}));

    let is_valid = json
        .get("is_valid")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let reason = json
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown")
        .to_string();

    Ok((is_valid, reason))
}

/// Helper macro for semantic assertions in tests.
///
/// Validates a response semantically and provides a clear failure message.
#[macro_export]
macro_rules! assert_semantic {
    ($client:expr, $context:expr, $response:expr, $question:expr) => {{
        let (is_valid, reason) =
            common::validate_response_semantically($client, $context, $response, $question)
                .await
                .expect("Semantic validation failed");
        assert!(
            is_valid,
            "Semantic validation failed: {}\nResponse was: {}",
            reason, $response
        );
    }};
}
