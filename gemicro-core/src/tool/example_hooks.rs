//! Example hook implementations.
//!
//! These examples demonstrate common hook patterns for logging, security,
//! and validation. They can be used as-is for simple cases or as templates
//! for building custom hooks.

use super::{HookDecision, HookError, ToolHook, ToolResult};
use async_trait::async_trait;
use serde_json::Value;
use std::path::PathBuf;

/// Audit log hook that logs all tool invocations.
///
/// Logs both pre and post execution for complete audit trail.
/// Uses the `log` crate, so configure a logger to see output.
///
/// # Example
///
/// ```
/// use gemicro_core::tool::{HookRegistry, example_hooks::AuditLogHook};
///
/// let hooks = HookRegistry::new()
///     .with_hook(AuditLogHook);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct AuditLogHook;

#[async_trait]
impl ToolHook for AuditLogHook {
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
                if s.len() > 100 {
                    format!("{}...", &s[..100])
                } else {
                    s.clone()
                }
            }
            other => format!("{:?}", other),
        };
        log::info!("Tool completed: {} -> {}", tool_name, content_preview);
        Ok(())
    }
}

/// File security hook that blocks writes to sensitive paths.
///
/// Prevents tools from writing to specific directories or files.
/// Configure blocked paths via the constructor.
///
/// # Example
///
/// ```
/// use gemicro_core::tool::{HookRegistry, example_hooks::FileSecurityHook};
/// use std::path::PathBuf;
///
/// let hook = FileSecurityHook::new(vec![
///     PathBuf::from("/etc"),
///     PathBuf::from("/var"),
///     PathBuf::from("/home/user/.ssh"),
/// ]);
///
/// let hooks = HookRegistry::new()
///     .with_hook(hook);
/// ```
#[derive(Debug, Clone)]
pub struct FileSecurityHook {
    blocked_paths: Vec<PathBuf>,
}

impl FileSecurityHook {
    /// Create a new file security hook with the given blocked paths.
    ///
    /// Any tool attempting to write to these paths (or their subdirectories)
    /// will be denied.
    pub fn new(blocked_paths: Vec<PathBuf>) -> Self {
        Self { blocked_paths }
    }

    /// Check if a path is blocked.
    fn is_blocked(&self, path: &str) -> bool {
        let path = PathBuf::from(path);
        for blocked in &self.blocked_paths {
            if path.starts_with(blocked) {
                return true;
            }
        }
        false
    }
}

#[async_trait]
impl ToolHook for FileSecurityHook {
    async fn pre_tool_use(
        &self,
        tool_name: &str,
        input: &Value,
    ) -> Result<HookDecision, HookError> {
        // Only check file write tools
        if !matches!(tool_name, "file_write" | "file_edit") {
            return Ok(HookDecision::Allow);
        }

        // Extract path from input
        let path = match input.get("path").and_then(|p| p.as_str()) {
            Some(p) => p,
            None => return Ok(HookDecision::Allow), // No path = let tool handle error
        };

        // Check if blocked
        if self.is_blocked(path) {
            return Ok(HookDecision::Deny {
                reason: format!("Writing to '{}' is blocked by security policy", path),
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
        // No post-execution logic needed
        Ok(())
    }
}

/// Input sanitization hook that validates and modifies tool inputs.
///
/// Enforces limits on input sizes and can normalize inputs.
///
/// # Example
///
/// ```
/// use gemicro_core::tool::{HookRegistry, example_hooks::InputSanitizerHook};
///
/// let hook = InputSanitizerHook::new(1024 * 1024); // 1MB max
/// let hooks = HookRegistry::new()
///     .with_hook(hook);
/// ```
#[derive(Debug, Clone)]
pub struct InputSanitizerHook {
    max_input_size_bytes: usize,
}

impl InputSanitizerHook {
    /// Create a new input sanitizer with the given max input size.
    pub fn new(max_input_size_bytes: usize) -> Self {
        Self {
            max_input_size_bytes,
        }
    }

    /// Calculate approximate size of JSON value in bytes.
    fn estimate_size(&self, value: &Value) -> usize {
        value.to_string().len()
    }
}

#[async_trait]
impl ToolHook for InputSanitizerHook {
    async fn pre_tool_use(
        &self,
        _tool_name: &str,
        input: &Value,
    ) -> Result<HookDecision, HookError> {
        let size = self.estimate_size(input);
        if size > self.max_input_size_bytes {
            return Ok(HookDecision::Deny {
                reason: format!(
                    "Input too large: {} bytes (max: {} bytes)",
                    size, self.max_input_size_bytes
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

/// Conditional permission hook that requests permission for specific operations.
///
/// Unlike tools that always or never require confirmation, this hook dynamically
/// requests permission based on the operation being performed.
///
/// # Example
///
/// ```
/// use gemicro_core::tool::{HookRegistry, example_hooks::ConditionalPermissionHook};
///
/// let hook = ConditionalPermissionHook::new(vec!["rm".to_string(), "delete".to_string()]);
/// let hooks = HookRegistry::new()
///     .with_hook(hook);
/// ```
#[derive(Debug, Clone)]
pub struct ConditionalPermissionHook {
    /// Commands that require permission
    dangerous_patterns: Vec<String>,
}

impl ConditionalPermissionHook {
    /// Create a new conditional permission hook.
    ///
    /// Operations containing any of the dangerous patterns will trigger
    /// a permission request.
    pub fn new(dangerous_patterns: Vec<String>) -> Self {
        Self { dangerous_patterns }
    }

    /// Check if input contains dangerous patterns.
    fn is_dangerous(&self, input: &Value) -> Option<String> {
        let input_str = input.to_string().to_lowercase();
        for pattern in &self.dangerous_patterns {
            if input_str.contains(&pattern.to_lowercase()) {
                return Some(pattern.clone());
            }
        }
        None
    }
}

#[async_trait]
impl ToolHook for ConditionalPermissionHook {
    async fn pre_tool_use(
        &self,
        tool_name: &str,
        input: &Value,
    ) -> Result<HookDecision, HookError> {
        if let Some(pattern) = self.is_dangerous(input) {
            return Ok(HookDecision::RequestPermission {
                message: format!(
                    "Tool '{}' wants to perform operation containing '{}'. Allow?",
                    tool_name, pattern
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

/// Metrics collection hook for tracking tool usage.
///
/// Collects basic metrics on tool invocations. In a real implementation,
/// this would send to a metrics backend.
///
/// # Example
///
/// ```
/// use gemicro_core::tool::{HookRegistry, example_hooks::MetricsHook};
///
/// let hooks = HookRegistry::new()
///     .with_hook(MetricsHook);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct MetricsHook;

#[async_trait]
impl ToolHook for MetricsHook {
    async fn pre_tool_use(
        &self,
        tool_name: &str,
        _input: &Value,
    ) -> Result<HookDecision, HookError> {
        // In a real implementation, increment a counter metric
        log::debug!("Metric: tool.{}.invocations += 1", tool_name);
        Ok(HookDecision::Allow)
    }

    async fn post_tool_use(
        &self,
        tool_name: &str,
        _input: &Value,
        output: &ToolResult,
    ) -> Result<(), HookError> {
        // In a real implementation, record success/failure metrics
        let success = output.metadata.get("error").is_none();
        log::debug!(
            "Metric: tool.{}.{} += 1",
            tool_name,
            if success { "success" } else { "failure" }
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_audit_log_hook() {
        let hook = AuditLogHook;
        let decision = hook.pre_tool_use("test", &json!({})).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);

        let result = ToolResult::text("test output");
        let res = hook.post_tool_use("test", &json!({}), &result).await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_file_security_hook_allows_safe_path() {
        let hook = FileSecurityHook::new(vec![PathBuf::from("/etc")]);
        let input = json!({"path": "/home/user/safe.txt"});
        let decision = hook.pre_tool_use("file_write", &input).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn test_file_security_hook_blocks_dangerous_path() {
        let hook = FileSecurityHook::new(vec![PathBuf::from("/etc")]);
        let input = json!({"path": "/etc/passwd"});
        let decision = hook.pre_tool_use("file_write", &input).await.unwrap();
        match decision {
            HookDecision::Deny { reason } => {
                assert!(reason.contains("blocked"));
            }
            _ => panic!("Expected deny"),
        }
    }

    #[tokio::test]
    async fn test_file_security_hook_ignores_non_write_tools() {
        let hook = FileSecurityHook::new(vec![PathBuf::from("/etc")]);
        let input = json!({"path": "/etc/passwd"});
        let decision = hook.pre_tool_use("file_read", &input).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn test_input_sanitizer_allows_small_input() {
        let hook = InputSanitizerHook::new(1000);
        let input = json!({"small": "input"});
        let decision = hook.pre_tool_use("test", &input).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn test_input_sanitizer_blocks_large_input() {
        let hook = InputSanitizerHook::new(10); // Very small limit
        let large = "x".repeat(100);
        let input = json!({"large": large});
        let decision = hook.pre_tool_use("test", &input).await.unwrap();
        match decision {
            HookDecision::Deny { reason } => {
                assert!(reason.contains("too large"));
            }
            _ => panic!("Expected deny"),
        }
    }

    #[tokio::test]
    async fn test_metrics_hook() {
        let hook = MetricsHook;
        let decision = hook.pre_tool_use("test", &json!({})).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);

        let result = ToolResult::text("test");
        let res = hook.post_tool_use("test", &json!({}), &result).await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_conditional_permission_allows_safe() {
        let hook = ConditionalPermissionHook::new(vec!["rm".to_string(), "delete".to_string()]);
        let input = json!({"command": "ls -la"});
        let decision = hook.pre_tool_use("bash", &input).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn test_conditional_permission_requests_for_dangerous() {
        let hook = ConditionalPermissionHook::new(vec!["rm".to_string(), "delete".to_string()]);
        let input = json!({"command": "rm -rf /"});
        let decision = hook.pre_tool_use("bash", &input).await.unwrap();
        match decision {
            HookDecision::RequestPermission { message } => {
                assert!(message.contains("rm"));
            }
            _ => panic!("Expected RequestPermission"),
        }
    }

    #[tokio::test]
    async fn test_conditional_permission_case_insensitive() {
        let hook = ConditionalPermissionHook::new(vec!["DELETE".to_string()]);
        let input = json!({"query": "delete from users"});
        let decision = hook.pre_tool_use("sql", &input).await.unwrap();
        match decision {
            HookDecision::RequestPermission { .. } => {}
            _ => panic!("Expected RequestPermission"),
        }
    }
}
