//! Input sanitization interceptor for gemicro tool execution.
//!
//! Enforces limits on tool input sizes to prevent resource exhaustion
//! and performance degradation from excessively large inputs.
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::interceptor::{InterceptorChain, ToolCall};
//! use gemicro_core::tool::ToolResult;
//! use gemicro_input_sanitizer::InputSanitizer;
//!
//! // Limit inputs to 1MB
//! let interceptor = InputSanitizer::new(1024 * 1024);
//!
//! let interceptors: InterceptorChain<ToolCall, ToolResult> = InterceptorChain::new()
//!     .with(interceptor);
//! ```

use async_trait::async_trait;
use gemicro_core::interceptor::{InterceptDecision, InterceptError, Interceptor, ToolCall};
use gemicro_core::tool::ToolResult;

/// Input sanitization interceptor that validates and limits tool input sizes.
///
/// # Size Estimation
///
/// The interceptor estimates size by serializing the input to JSON and measuring
/// the UTF-8 byte length. This provides a reasonable approximation for
/// preventing resource exhaustion.
///
/// **Note:** This measures serialized JSON size, not in-memory size or
/// LLM token count. For token-based limits, use a separate interceptor that
/// integrates with your tokenizer.
///
/// # Example
///
/// ```
/// use gemicro_input_sanitizer::InputSanitizer;
///
/// // Limit to 10KB
/// let interceptor = InputSanitizer::new(10 * 1024);
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
    fn estimate_size(&self, value: &serde_json::Value) -> Result<usize, InterceptError> {
        serde_json::to_string(value).map(|s| s.len()).map_err(|e| {
            InterceptError::ExecutionFailed(format!(
                "Failed to serialize input for size check: {}",
                e
            ))
        })
    }
}

#[async_trait]
impl Interceptor<ToolCall, ToolResult> for InputSanitizer {
    async fn intercept(
        &self,
        input: &ToolCall,
    ) -> Result<InterceptDecision<ToolCall>, InterceptError> {
        let size = self.estimate_size(&input.arguments)?;

        if size > self.max_input_size_bytes {
            // Calculate percentage over limit, handling zero max gracefully
            let percent_over = if self.max_input_size_bytes == 0 {
                "N/A".to_string()
            } else {
                format!(
                    "{}",
                    ((size as f64 / self.max_input_size_bytes as f64 - 1.0) * 100.0) as usize
                )
            };
            return Ok(InterceptDecision::Deny {
                reason: format!(
                    "Input too large for tool '{}': {} bytes (max: {} bytes, {}% over limit)",
                    input.name, size, self.max_input_size_bytes, percent_over
                ),
            });
        }

        Ok(InterceptDecision::Allow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_allows_small_input() {
        let interceptor = InputSanitizer::new(1000);
        let input = ToolCall::new("test", json!({"small": "input"}));
        let decision = interceptor.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);
    }

    #[tokio::test]
    async fn test_blocks_large_input() {
        let interceptor = InputSanitizer::new(10); // Very small limit
        let large = "x".repeat(100);
        let input = ToolCall::new("test", json!({"large": large}));
        let decision = interceptor.intercept(&input).await.unwrap();
        match decision {
            InterceptDecision::Deny { reason } => {
                assert!(reason.contains("too large"));
                assert!(reason.contains("bytes"));
            }
            _ => panic!("Expected deny"),
        }
    }

    #[tokio::test]
    async fn test_empty_input() {
        let interceptor = InputSanitizer::new(100);
        let input = ToolCall::new("test", json!({}));
        let decision = interceptor.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);
    }

    #[tokio::test]
    async fn test_exact_limit() {
        let interceptor = InputSanitizer::new(100);
        // Create input that's just under the limit
        // {"data":"xxxxx..."} = 8 bytes overhead + data length
        let input = ToolCall::new("test", json!({"data": "x".repeat(80)}));
        let decision = interceptor.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);
    }

    #[tokio::test]
    async fn test_nested_json() {
        let interceptor = InputSanitizer::new(1000);
        let input = ToolCall::new(
            "test",
            json!({
                "nested": {
                    "data": {
                        "values": [1, 2, 3, 4, 5]
                    }
                }
            }),
        );
        let decision = interceptor.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);
    }

    // Edge case tests

    #[tokio::test]
    async fn test_actual_zero_byte_limit() {
        // Zero-byte limit should deny everything (pathological but valid config)
        let interceptor = InputSanitizer::new(0);
        let input = ToolCall::new("test", json!({})); // Even empty object has size

        let decision = interceptor.intercept(&input).await.unwrap();

        // Should deny - size > 0 always true for any valid JSON
        // Should show "N/A% over limit" to avoid division by zero
        match decision {
            InterceptDecision::Deny { reason } => {
                assert!(reason.contains("too large"));
                assert!(
                    reason.contains("N/A% over limit"),
                    "Should show N/A for zero limit, got: {}",
                    reason
                );
            }
            _ => panic!("Expected deny for zero-byte limit"),
        }
    }

    #[tokio::test]
    async fn test_one_byte_limit() {
        // Even tiny limits should work (deny everything)
        let interceptor = InputSanitizer::new(1);
        let input = ToolCall::new("test", json!({})); // Serializes to "{}" (2 bytes)
        let decision = interceptor.intercept(&input).await.unwrap();

        // Should deny - even empty JSON has size
        match decision {
            InterceptDecision::Deny { reason } => {
                assert!(reason.contains("too large"));
                assert!(reason.contains("100% over limit"));
            }
            _ => panic!("Expected deny for one-byte limit"),
        }
    }

    #[tokio::test]
    async fn test_at_exact_boundary() {
        let interceptor = InputSanitizer::new(100);

        // Create input that serializes to EXACTLY 100 bytes
        // {"data":"xxxx..."} where total is 100 bytes
        let overhead = "{\"data\":\"\"}".len();
        let content_len = 100 - overhead;
        let input = ToolCall::new("test", json!({"data": "x".repeat(content_len)}));

        let size = serde_json::to_string(&input.arguments).unwrap().len();
        assert_eq!(size, 100, "Test setup: should be exactly 100 bytes");

        let decision = interceptor.intercept(&input).await.unwrap();

        // At limit should ALLOW (only > limit is denied)
        assert_eq!(
            decision,
            InterceptDecision::Allow,
            "Should allow at exact boundary"
        );
    }

    #[tokio::test]
    async fn test_one_over_boundary() {
        let interceptor = InputSanitizer::new(100);

        let overhead = "{\"data\":\"\"}".len();
        let content_len = 100 - overhead + 1; // One over
        let input = ToolCall::new("test", json!({"data": "x".repeat(content_len)}));

        let decision = interceptor.intercept(&input).await.unwrap();

        match decision {
            InterceptDecision::Deny { reason } => {
                assert!(reason.contains("101 bytes"));
                assert!(reason.contains("max: 100 bytes"));
            }
            _ => panic!("Expected deny one byte over limit"),
        }
    }

    #[tokio::test]
    async fn test_deeply_nested_json() {
        let interceptor = InputSanitizer::new(100000); // Large limit

        // Create 100-level nested structure (not too deep to overflow stack)
        let mut nested = json!("base");
        for _ in 0..100 {
            nested = json!([nested]);
        }

        let input = ToolCall::new("test", json!({"data": nested}));

        // Should either handle gracefully or return error
        let result = interceptor.intercept(&input).await;

        match result {
            Ok(InterceptDecision::Allow) => {
                // Successfully serialized and within limit
            }
            Ok(InterceptDecision::Deny { reason }) => {
                // Serialized but exceeded limit
                assert!(reason.contains("too large"));
            }
            Err(InterceptError::ExecutionFailed(msg)) => {
                // Serialization failed - acceptable
                assert!(msg.contains("serialize"));
            }
            _ => panic!("Unexpected result for deeply nested JSON"),
        }
    }

    #[tokio::test]
    async fn test_large_array() {
        let interceptor = InputSanitizer::new(100);

        // Large array should be blocked
        let large_array: Vec<i32> = (0..1000).collect();
        let input = ToolCall::new("test", json!({"numbers": large_array}));

        let decision = interceptor.intercept(&input).await.unwrap();

        match decision {
            InterceptDecision::Deny { .. } => {
                // Expected - large array exceeds limit
            }
            _ => panic!("Expected deny for large array"),
        }
    }

    #[tokio::test]
    async fn test_unicode_strings() {
        let interceptor = InputSanitizer::new(100);

        // Unicode characters can be multiple bytes
        let input = ToolCall::new("test", json!({"text": "日本語 🎌 test"}));
        let size = serde_json::to_string(&input.arguments).unwrap().len();

        let decision = interceptor.intercept(&input).await.unwrap();

        if size > 100 {
            assert!(matches!(decision, InterceptDecision::Deny { .. }));
        } else {
            assert_eq!(decision, InterceptDecision::Allow);
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
                let input = ToolCall::new("test", json!({"data": data}));
                let size = serde_json::to_string(&input.arguments).unwrap().len();

                // Skip cases where size exceeds limit (we test those separately)
                prop_assume!(size <= max_size);

                let interceptor = InputSanitizer::new(max_size);
                let rt = tokio::runtime::Runtime::new().unwrap();
                let decision = rt.block_on(interceptor.intercept(&input)).unwrap();

                prop_assert_eq!(decision, InterceptDecision::Allow);
            }

            /// Property: Size > max should always deny
            #[test]
            fn size_over_limit_always_denies(
                max_size in 10usize..100,
                data in ".*",
            ) {
                let input = ToolCall::new("test", json!({"data": data}));
                let size = serde_json::to_string(&input.arguments).unwrap().len();

                // Only test cases where size exceeds limit
                prop_assume!(size > max_size);

                let interceptor = InputSanitizer::new(max_size);
                let rt = tokio::runtime::Runtime::new().unwrap();
                let decision = rt.block_on(interceptor.intercept(&input)).unwrap();

                match decision {
                    InterceptDecision::Deny { .. } => Ok(()),
                    _ => Err(TestCaseError::fail("Expected Deny for size over limit")),
                }?;
            }

            /// Property: Estimation is deterministic (same input = same size)
            #[test]
            fn estimation_is_deterministic(data in ".*") {
                let interceptor = InputSanitizer::new(1000);
                let input = ToolCall::new("test", json!({"data": data}));

                let size1 = interceptor.estimate_size(&input.arguments).unwrap();
                let size2 = interceptor.estimate_size(&input.arguments).unwrap();

                prop_assert_eq!(size1, size2);
            }
        }
    }
}
