//! Audit logging interceptor for gemicro tool execution.
//!
//! This interceptor logs all tool invocations before and after execution,
//! providing a complete audit trail for compliance and debugging.
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::interceptor::{InterceptorChain, ToolCall};
//! use gemicro_core::tool::ToolResult;
//! use gemicro_audit_log::AuditLog;
//!
//! let interceptors: InterceptorChain<ToolCall, ToolResult> = InterceptorChain::new()
//!     .with(AuditLog);
//! ```

use async_trait::async_trait;
use gemicro_core::interceptor::{InterceptDecision, InterceptError, Interceptor, ToolCall};
use gemicro_core::tool::ToolResult;
use serde_json::Value;

/// Audit log interceptor that logs all tool invocations.
///
/// Logs both pre and post execution for complete audit trail.
/// Uses the `log` crate, so configure a logger to see output.
///
/// # Output Format
///
/// **Pre-execution:**
/// ```text
/// [INFO] Tool invoked: bash with input: {"command": "ls -la"}
/// ```
///
/// **Post-execution:**
/// ```text
/// [INFO] Tool completed: bash -> total 16
/// drwxr-xr-x  4 user  staff...
/// ```
///
/// Long outputs are truncated to 100 characters for readability.
#[derive(Debug, Clone, Copy, Default)]
pub struct AuditLog;

#[async_trait]
impl Interceptor<ToolCall, ToolResult> for AuditLog {
    async fn intercept(
        &self,
        input: &ToolCall,
    ) -> Result<InterceptDecision<ToolCall>, InterceptError> {
        log::info!(
            "Tool invoked: {} with input: {}",
            input.name,
            input.arguments
        );
        Ok(InterceptDecision::Allow)
    }

    async fn observe(&self, input: &ToolCall, output: &ToolResult) -> Result<(), InterceptError> {
        let content_preview = match &output.content {
            Value::String(s) => {
                // Use char-aware truncation to avoid panicking on UTF-8 boundaries
                if s.chars().count() > 100 {
                    let truncated: String = s.chars().take(100).collect();
                    format!("{}...", truncated)
                } else {
                    s.clone()
                }
            }
            other => {
                let formatted = format!("{:?}", other);
                if formatted.chars().count() > 100 {
                    let truncated: String = formatted.chars().take(100).collect();
                    format!("{}...", truncated)
                } else {
                    formatted
                }
            }
        };
        log::info!("Tool completed: {} -> {}", input.name, content_preview);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_audit_log_allows_execution() {
        let interceptor = AuditLog;
        let input = ToolCall::new("test", json!({}));
        let decision = interceptor.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);
    }

    #[tokio::test]
    async fn test_audit_log_post_execution() {
        let interceptor = AuditLog;
        let input = ToolCall::new("test", json!({}));
        let result = ToolResult::text("test output");
        let res = interceptor.observe(&input, &result).await;
        assert!(res.is_ok());
    }
}
