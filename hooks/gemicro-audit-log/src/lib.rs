//! Audit logging hook for gemicro tool execution.
//!
//! This hook logs all tool invocations before and after execution,
//! providing a complete audit trail for compliance and debugging.
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::tool::HookRegistry;
//! use gemicro_audit_log::AuditLog;
//!
//! let hooks = HookRegistry::new()
//!     .with_hook(AuditLog);
//! ```

use async_trait::async_trait;
use gemicro_core::tool::{HookDecision, HookError, ToolHook, ToolResult};
use serde_json::Value;

/// Audit log hook that logs all tool invocations.
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
impl ToolHook for AuditLog {
    async fn pre_tool_use(
        &self,
        tool_name: &str,
        input: &Value,
    ) -> Result<HookDecision, HookError> {
        log::info!("Tool invoked: {} with input: {}", tool_name, input);
        Ok(HookDecision::Allow)
    }

    async fn post_tool_use(
        &self,
        tool_name: &str,
        _input: &Value,
        output: &ToolResult,
    ) -> Result<(), HookError> {
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
        log::info!("Tool completed: {} -> {}", tool_name, content_preview);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_audit_log_allows_execution() {
        let hook = AuditLog;
        let decision = hook.pre_tool_use("test", &json!({})).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn test_audit_log_post_execution() {
        let hook = AuditLog;
        let result = ToolResult::text("test output");
        let res = hook.post_tool_use("test", &json!({}), &result).await;
        assert!(res.is_ok());
    }
}
