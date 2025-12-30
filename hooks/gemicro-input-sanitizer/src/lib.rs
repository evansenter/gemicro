//! Input sanitization hook for gemicro tool execution.
//!
//! Enforces limits on tool input sizes to prevent resource exhaustion
//! and performance degradation from excessively large inputs.
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::tool::HookRegistry;
//! use gemicro_input_sanitizer::InputSanitizer;
//!
//! // Limit inputs to 1MB
//! let hook = InputSanitizer::new(1024 * 1024);
//!
//! let hooks = HookRegistry::new()
//!     .with_hook(hook);
//! ```

use async_trait::async_trait;
use gemicro_core::tool::{HookDecision, HookError, ToolHook, ToolResult};
use serde_json::Value;

/// Input sanitization hook that validates and limits tool input sizes.
///
/// # Size Estimation
///
/// The hook estimates size by serializing the input to JSON and measuring
/// the UTF-8 byte length. This provides a reasonable approximation for
/// preventing resource exhaustion.
///
/// **Note:** This measures serialized JSON size, not in-memory size or
/// LLM token count. For token-based limits, use a separate hook that
/// integrates with your tokenizer.
///
/// # Example
///
/// ```
/// use gemicro_input_sanitizer::InputSanitizer;
///
/// // Limit to 10KB
/// let hook = InputSanitizer::new(10 * 1024);
/// ```
#[derive(Debug, Clone)]
pub struct InputSanitizer {
    max_input_size_bytes: usize,
}

impl InputSanitizer {
    /// Create a new input sanitizer with the given max input size in bytes.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_input_sanitizer::InputSanitizer;
    ///
    /// // Common configurations:
    /// let small = InputSanitizer::new(1024);           // 1KB
    /// let medium = InputSanitizer::new(100 * 1024);    // 100KB
    /// let large = InputSanitizer::new(1024 * 1024);    // 1MB
    /// ```
    pub fn new(max_input_size_bytes: usize) -> Self {
        Self {
            max_input_size_bytes,
        }
    }

    /// Estimate size of JSON value in bytes.
    ///
    /// Serializes to compact JSON and measures UTF-8 byte length.
    fn estimate_size(&self, value: &Value) -> Result<usize, HookError> {
        serde_json::to_string(value).map(|s| s.len()).map_err(|e| {
            HookError::ExecutionFailed(format!("Failed to serialize input for size check: {}", e))
        })
    }
}

#[async_trait]
impl ToolHook for InputSanitizer {
    async fn pre_tool_use(
        &self,
        tool_name: &str,
        input: &Value,
    ) -> Result<HookDecision, HookError> {
        let size = self.estimate_size(input)?;

        if size > self.max_input_size_bytes {
            return Ok(HookDecision::Deny {
                reason: format!(
                    "Input too large for tool '{}': {} bytes (max: {} bytes, {}% over limit)",
                    tool_name,
                    size,
                    self.max_input_size_bytes,
                    ((size as f64 / self.max_input_size_bytes as f64 - 1.0) * 100.0) as usize
                ),
            });
        }

        Ok(HookDecision::Allow)
    }

    async fn post_tool_use(
        &self,
        _tool_name: &str,
        _input: &Value,
        _output: &ToolResult,
    ) -> Result<(), HookError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_allows_small_input() {
        let hook = InputSanitizer::new(1000);
        let input = json!({"small": "input"});
        let decision = hook.pre_tool_use("test", &input).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn test_blocks_large_input() {
        let hook = InputSanitizer::new(10); // Very small limit
        let large = "x".repeat(100);
        let input = json!({"large": large});
        let decision = hook.pre_tool_use("test", &input).await.unwrap();
        match decision {
            HookDecision::Deny { reason } => {
                assert!(reason.contains("too large"));
                assert!(reason.contains("bytes"));
            }
            _ => panic!("Expected deny"),
        }
    }

    #[tokio::test]
    async fn test_empty_input() {
        let hook = InputSanitizer::new(100);
        let input = json!({});
        let decision = hook.pre_tool_use("test", &input).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn test_exact_limit() {
        let hook = InputSanitizer::new(100);
        // Create input that's just under the limit
        // {"data":"xxxxx..."} = 8 bytes overhead + data length
        let input = json!({"data": "x".repeat(80)});
        let decision = hook.pre_tool_use("test", &input).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn test_nested_json() {
        let hook = InputSanitizer::new(1000);
        let input = json!({
            "nested": {
                "data": {
                    "values": [1, 2, 3, 4, 5]
                }
            }
        });
        let decision = hook.pre_tool_use("test", &input).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }
}
