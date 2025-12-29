//! WebFetch tool for fetching URL content.

use async_trait::async_trait;
use gemicro_core::tool::{Tool, ToolError, ToolResult};
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;

/// Default timeout for HTTP requests (30 seconds).
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Maximum response body size (5MB).
const MAX_RESPONSE_SIZE: usize = 5 * 1024 * 1024;

/// WebFetch tool for fetching URL content.
///
/// Performs HTTP GET requests to fetch content from URLs. Includes timeout
/// and size limits to prevent hanging or memory issues.
///
/// # Example
///
/// ```no_run
/// use gemicro_web_fetch::WebFetch;
/// use gemicro_core::tool::Tool;
/// use serde_json::json;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let tool = WebFetch::new();
/// let result = tool.execute(json!({"url": "https://example.com"})).await?;
/// println!("Content: {}", result.content);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct WebFetch {
    client: Client,
    timeout: Duration,
}

impl Default for WebFetch {
    fn default() -> Self {
        Self::new()
    }
}

impl WebFetch {
    /// Create a new WebFetch tool with default settings.
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        }
    }

    /// Create a WebFetch tool with a custom timeout.
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            client: Client::new(),
            timeout,
        }
    }

    /// Create a WebFetch tool with a custom reqwest Client.
    ///
    /// Useful for sharing a client across multiple tools or configuring
    /// custom settings like proxies or TLS.
    pub fn with_client(client: Client, timeout: Duration) -> Self {
        Self { client, timeout }
    }
}

#[async_trait]
impl Tool for WebFetch {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch content from a URL. Performs an HTTP GET request and returns \
         the response body as text. Has a 30-second timeout and 5MB size limit."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from (must be http:// or https://)"
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let url = input
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'url' field".into()))?;

        // Validate URL scheme
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(ToolError::InvalidInput(format!(
                "URL must start with http:// or https://, got: {}",
                url
            )));
        }

        // Make the request with timeout
        let response = self
            .client
            .get(url)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    ToolError::ExecutionFailed(format!(
                        "Request timed out after {:?}",
                        self.timeout
                    ))
                } else if e.is_connect() {
                    ToolError::ExecutionFailed(format!("Failed to connect: {}", e))
                } else {
                    ToolError::ExecutionFailed(format!("HTTP request failed: {}", e))
                }
            })?;

        // Check status code
        let status = response.status();
        if !status.is_success() {
            return Err(ToolError::ExecutionFailed(format!(
                "HTTP {} {}",
                status.as_u16(),
                status.canonical_reason().unwrap_or("Unknown")
            )));
        }

        // Check content length if available
        if let Some(content_length) = response.content_length() {
            if content_length > MAX_RESPONSE_SIZE as u64 {
                return Err(ToolError::InvalidInput(format!(
                    "Response too large ({} bytes, max {} bytes)",
                    content_length, MAX_RESPONSE_SIZE
                )));
            }
        }

        // Read the response body with size limit
        let bytes = response.bytes().await.map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to read response body: {}", e))
        })?;

        if bytes.len() > MAX_RESPONSE_SIZE {
            return Err(ToolError::InvalidInput(format!(
                "Response too large ({} bytes, max {} bytes)",
                bytes.len(),
                MAX_RESPONSE_SIZE
            )));
        }

        // Convert to string, logging a warning if lossy conversion occurs
        let content = match String::from_utf8(bytes.to_vec()) {
            Ok(s) => s,
            Err(e) => {
                log::warn!(
                    "Response from {} contained invalid UTF-8 at byte {}, using lossy conversion",
                    url,
                    e.utf8_error().valid_up_to()
                );
                String::from_utf8_lossy(e.as_bytes()).into_owned()
            }
        };

        Ok(ToolResult::new(content))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_fetch_name_and_description() {
        let tool = WebFetch::new();
        assert_eq!(tool.name(), "web_fetch");
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn test_web_fetch_parameters_schema() {
        let tool = WebFetch::new();
        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["url"].is_object());
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .contains(&json!("url")));
    }

    #[tokio::test]
    async fn test_web_fetch_missing_url() {
        let tool = WebFetch::new();
        let result = tool.execute(json!({})).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_web_fetch_invalid_scheme() {
        let tool = WebFetch::new();
        let result = tool.execute(json!({"url": "ftp://example.com"})).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        if let ToolError::InvalidInput(msg) = err {
            assert!(msg.contains("http://") || msg.contains("https://"));
        }
    }

    #[tokio::test]
    async fn test_web_fetch_invalid_url() {
        let tool = WebFetch::new();
        let result = tool.execute(json!({"url": "not-a-url"})).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[test]
    fn test_web_fetch_with_timeout() {
        let tool = WebFetch::with_timeout(Duration::from_secs(10));
        assert_eq!(tool.timeout, Duration::from_secs(10));
    }

    #[test]
    fn test_web_fetch_default() {
        let tool = WebFetch::default();
        assert_eq!(tool.timeout, Duration::from_secs(DEFAULT_TIMEOUT_SECS));
    }

    // Integration test - requires network access
    #[tokio::test]
    #[ignore]
    async fn test_web_fetch_real_request() {
        let tool = WebFetch::new();
        let result = tool
            .execute(json!({"url": "https://httpbin.org/get"}))
            .await;
        assert!(result.is_ok());
        let content = result.unwrap().content;
        assert!(content.contains("httpbin.org"));
    }
}
