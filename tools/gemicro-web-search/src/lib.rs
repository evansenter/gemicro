//! WebSearch tool for real-time web search.
//!
//! This tool wraps Gemini's Google Search grounding capability, allowing
//! agents to search the web for real-time information about current events,
//! recent releases, or live data.

use async_trait::async_trait;
use gemicro_core::tool::{Tool, ToolError, ToolResult};
use gemicro_core::LlmClient;
use serde_json::{json, Value};
use std::sync::Arc;

/// WebSearch tool for real-time web search.
///
/// Uses Gemini's Google Search grounding to search the web and return
/// information grounded in real-time web data.
///
/// # Example
///
/// ```no_run
/// use gemicro_web_search::WebSearch;
/// use gemicro_core::{LlmClient, LlmConfig, tool::Tool};
/// use serde_json::json;
/// use std::sync::Arc;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let genai_client = genai_rs::Client::builder("api-key".to_string()).build()?;
/// let llm = Arc::new(LlmClient::new(genai_client, LlmConfig::default()));
/// let search = WebSearch::new(llm);
///
/// let result = search.execute(json!({
///     "query": "What are the latest developments in quantum computing?"
/// })).await?;
/// println!("Search result: {}", result.content);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct WebSearch {
    llm: Arc<LlmClient>,
}

impl std::fmt::Debug for WebSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebSearch")
            .field("llm", &"LlmClient")
            .finish()
    }
}

impl WebSearch {
    /// Create a new WebSearch tool with the given LLM client.
    ///
    /// The LLM client will be used to make grounded search requests.
    pub fn new(llm: Arc<LlmClient>) -> Self {
        Self { llm }
    }
}

/// System instruction for grounded web search queries.
const SEARCH_SYSTEM_INSTRUCTION: &str = "\
You are a helpful assistant that provides accurate, up-to-date information based on web search results. \
When answering, synthesize the search results into a clear, concise response. \
Include relevant facts, dates, and sources when available. \
If the search results are insufficient or conflicting, say so clearly.";

#[async_trait]
impl Tool for WebSearch {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web for real-time information. Uses Google Search grounding \
         to find current events, recent releases, live data, and up-to-date facts. \
         Best for queries that require information beyond the model's training data."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'query' field".into()))?;

        if query.is_empty() {
            return Err(ToolError::InvalidInput("Query cannot be empty".into()));
        }

        // Build request with Google Search grounding enabled
        let request = self
            .llm
            .client()
            .interaction()
            .with_system_instruction(SEARCH_SYSTEM_INSTRUCTION)
            .with_text(query)
            .with_google_search()
            .build()
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        // Execute the grounded search
        let response = self
            .llm
            .generate(request)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Search failed: {}", e)))?;

        // Extract the response text
        let text = response
            .text()
            .ok_or_else(|| ToolError::ExecutionFailed("Search returned no content".into()))?;

        Ok(ToolResult::text(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gemicro_core::LlmConfig;

    fn create_test_llm() -> Arc<LlmClient> {
        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        Arc::new(LlmClient::new(genai_client, LlmConfig::default()))
    }

    #[test]
    fn test_web_search_name_and_description() {
        let llm = create_test_llm();
        let search = WebSearch::new(llm);

        assert_eq!(search.name(), "web_search");
        assert!(!search.description().is_empty());
        assert!(search.description().contains("Search"));
    }

    #[test]
    fn test_web_search_parameters_schema() {
        let llm = create_test_llm();
        let search = WebSearch::new(llm);

        let schema = search.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["query"].is_object());

        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("query")));
    }

    #[tokio::test]
    async fn test_web_search_missing_query() {
        let llm = create_test_llm();
        let search = WebSearch::new(llm);

        let result = search.execute(json!({})).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_web_search_empty_query() {
        let llm = create_test_llm();
        let search = WebSearch::new(llm);

        let result = search.execute(json!({"query": ""})).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        if let ToolError::InvalidInput(msg) = err {
            assert!(msg.contains("empty"));
        }
    }

    #[test]
    fn test_web_search_debug() {
        let llm = create_test_llm();
        let search = WebSearch::new(llm);

        let debug = format!("{:?}", search);
        assert!(debug.contains("WebSearch"));
        assert!(debug.contains("LlmClient"));
    }

    // Integration test - requires GEMINI_API_KEY
    #[tokio::test]
    #[ignore]
    async fn test_web_search_real_query() {
        // This would test with a real API key
        // let llm = create_real_llm();
        // let search = WebSearch::new(llm);
        // let result = search.execute(json!({"query": "latest Rust release date"})).await;
        // assert!(result.is_ok());
    }
}
