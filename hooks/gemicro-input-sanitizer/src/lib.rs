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
#[non_exhaustive]
pub struct InputSanitizer {
    /// Maximum input size in bytes.
    pub max_input_size_bytes: usize,
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

    // Edge case tests

    #[tokio::test]
    async fn test_zero_byte_limit() {
        // Even tiny limits should work (deny everything)
        let hook = InputSanitizer::new(1);
        let input = json!({}); // Serializes to "{}" (2 bytes)
        let decision = hook.pre_tool_use("test", &input).await.unwrap();

        // Should deny - even empty JSON has size
        match decision {
            HookDecision::Deny { reason } => {
                assert!(reason.contains("too large"));
                assert!(reason.contains("100% over limit"));
            }
            _ => panic!("Expected deny for zero-byte limit"),
        }
    }

    #[tokio::test]
    async fn test_at_exact_boundary() {
        let hook = InputSanitizer::new(100);

        // Create input that serializes to EXACTLY 100 bytes
        // {"data":"xxxx..."} where total is 100 bytes
        let overhead = "{\"data\":\"\"}".len();
        let content_len = 100 - overhead;
        let input = json!({"data": "x".repeat(content_len)});

        let size = serde_json::to_string(&input).unwrap().len();
        assert_eq!(size, 100, "Test setup: should be exactly 100 bytes");

        let decision = hook.pre_tool_use("test", &input).await.unwrap();

        // At limit should ALLOW (only > limit is denied, see line 86)
        assert_eq!(
            decision,
            HookDecision::Allow,
            "Should allow at exact boundary"
        );
    }

    #[tokio::test]
    async fn test_one_over_boundary() {
        let hook = InputSanitizer::new(100);

        let overhead = "{\"data\":\"\"}".len();
        let content_len = 100 - overhead + 1; // One over
        let input = json!({"data": "x".repeat(content_len)});

        let decision = hook.pre_tool_use("test", &input).await.unwrap();

        match decision {
            HookDecision::Deny { reason } => {
                assert!(reason.contains("101 bytes"));
                assert!(reason.contains("max: 100 bytes"));
            }
            _ => panic!("Expected deny one byte over limit"),
        }
    }

    #[tokio::test]
    async fn test_deeply_nested_json() {
        let hook = InputSanitizer::new(100000); // Large limit

        // Create 100-level nested structure (not too deep to overflow stack)
        let mut nested = json!("base");
        for _ in 0..100 {
            nested = json!([nested]);
        }

        let input = json!({"data": nested});

        // Should either handle gracefully or return error
        let result = hook.pre_tool_use("test", &input).await;

        match result {
            Ok(HookDecision::Allow) => {
                // Successfully serialized and within limit
            }
            Ok(HookDecision::Deny { reason }) => {
                // Serialized but exceeded limit
                assert!(reason.contains("too large"));
            }
            Err(HookError::ExecutionFailed(msg)) => {
                // Serialization failed - acceptable
                assert!(msg.contains("serialize"));
            }
            _ => panic!("Unexpected result for deeply nested JSON"),
        }
    }

    #[tokio::test]
    async fn test_large_array() {
        let hook = InputSanitizer::new(100);

        // Large array should be blocked
        let large_array: Vec<i32> = (0..1000).collect();
        let input = json!({"numbers": large_array});

        let decision = hook.pre_tool_use("test", &input).await.unwrap();

        match decision {
            HookDecision::Deny { .. } => {
                // Expected - large array exceeds limit
            }
            _ => panic!("Expected deny for large array"),
        }
    }

    #[tokio::test]
    async fn test_unicode_strings() {
        let hook = InputSanitizer::new(100);

        // Unicode characters can be multiple bytes
        let input = json!({"text": "日本語 🎌 test"});
        let size = serde_json::to_string(&input).unwrap().len();

        let decision = hook.pre_tool_use("test", &input).await.unwrap();

        if size > 100 {
            assert!(matches!(decision, HookDecision::Deny { .. }));
        } else {
            assert_eq!(decision, HookDecision::Allow);
        }
    }

    // Property-based tests

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Property: Size <= max should always allow
            #[test]
            fn size_under_limit_always_allows(
                max_size in 10usize..10000,
                data in ".*",
            ) {
                // Only test inputs that are actually under the limit
                let input = json!({"data": data});
                let size = serde_json::to_string(&input).unwrap().len();

                // Skip cases where size exceeds limit (we test those separately)
                prop_assume!(size <= max_size);

                let hook = InputSanitizer::new(max_size);
                let rt = tokio::runtime::Runtime::new().unwrap();
                let decision = rt.block_on(hook.pre_tool_use("test", &input)).unwrap();

                prop_assert_eq!(decision, HookDecision::Allow);
            }

            /// Property: Size > max should always deny
            #[test]
            fn size_over_limit_always_denies(
                max_size in 10usize..100,
                data in ".*",
            ) {
                let input = json!({"data": data});
                let size = serde_json::to_string(&input).unwrap().len();

                // Only test cases where size exceeds limit
                prop_assume!(size > max_size);

                let hook = InputSanitizer::new(max_size);
                let rt = tokio::runtime::Runtime::new().unwrap();
                let decision = rt.block_on(hook.pre_tool_use("test", &input)).unwrap();

                match decision {
                    HookDecision::Deny { .. } => Ok(()),
                    _ => Err(TestCaseError::fail("Expected Deny for size over limit")),
                }?;
            }

            /// Property: Estimation is deterministic (same input = same size)
            #[test]
            fn estimation_is_deterministic(data in ".*") {
                let hook = InputSanitizer::new(1000);
                let input = json!({"data": data});

                let size1 = hook.estimate_size(&input).unwrap();
                let size2 = hook.estimate_size(&input).unwrap();

                prop_assert_eq!(size1, size2);
            }
        }
    }
}
