//! Common test utilities for CLI integration tests.

use gemicro_core::{LlmClient, LlmConfig};
use serde_json::json;

/// Create an LlmClient for test validation.
///
/// Uses GEMINI_API_KEY from environment.
pub fn create_test_client() -> LlmClient {
    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY required");
    let genai = genai_rs::Client::builder(api_key)
        .build()
        .expect("Failed to create genai client");
    LlmClient::new(genai, LlmConfig::default())
}

/// Validate a response semantically using LLM-as-judge.
///
/// This makes a separate LLM call to evaluate whether a response is correct.
/// Use this instead of brittle string matching for tests where the exact
/// wording may vary but the meaning should be consistent.
///
/// # Arguments
///
/// * `client` - The LLM client to use for validation
/// * `context` - What the test is checking (e.g., "User asked model to remember 42")
/// * `response_text` - The actual CLI output to validate
/// * `validation_question` - What to check (e.g., "Does the response correctly recall 42?")
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
        .with_model("gemini-3-flash-preview")
        .with_text(&validation_prompt)
        .with_response_format(schema)
        .build()
        .map_err(|e| e.to_string())?;

    let response = client.generate(request).await.map_err(|e| e.to_string())?;
    let text = response.as_text().unwrap_or("{}");

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
