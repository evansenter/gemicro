//! File security hook for gemicro tool execution.
//!
//! Prevents tools from writing to specific directories or files by blocking
//! file_write and file_edit operations to configured paths.
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::tool::HookRegistry;
//! use gemicro_file_security::FileSecurity;
//! use std::path::PathBuf;
//!
//! let hook = FileSecurity::new(vec![
//!     PathBuf::from("/etc"),
//!     PathBuf::from("/var"),
//!     PathBuf::from("/home/user/.ssh"),
//! ]);
//!
//! let hooks = HookRegistry::new()
//!     .with_hook(hook);
//! ```

use async_trait::async_trait;
use gemicro_core::tool::{HookDecision, HookError, ToolHook, ToolResult};
use serde_json::Value;
use std::path::PathBuf;

/// File security hook that blocks writes to sensitive paths.
///
/// Prevents tools from writing to specific directories or files.
/// Configure blocked paths via the constructor.
///
/// # Behavior
///
/// - Only intercepts `file_write` and `file_edit` tools
/// - Blocks any operation where the `path` parameter starts with a blocked path
/// - Other tools and operations pass through unchanged
///
/// # Security Note
///
/// This hook provides defense-in-depth but should not be the only security
/// layer. Tools can still:
/// - Read from blocked paths (use separate read hook if needed)
/// - Use symlinks to bypass checks (resolve symlinks first if critical)
/// - Use relative paths that escape to blocked locations
#[derive(Debug, Clone)]
pub struct FileSecurity {
    blocked_paths: Vec<PathBuf>,
}

impl FileSecurity {
    /// Create a new file security hook with the given blocked paths.
    ///
    /// Any tool attempting to write to these paths (or their subdirectories)
    /// will be denied.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_file_security::FileSecurity;
    /// use std::path::PathBuf;
    ///
    /// let hook = FileSecurity::new(vec![
    ///     PathBuf::from("/etc"),
    ///     PathBuf::from("/sys"),
    /// ]);
    /// ```
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
impl ToolHook for FileSecurity {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_allows_safe_path() {
        let hook = FileSecurity::new(vec![PathBuf::from("/etc")]);
        let input = json!({"path": "/home/user/safe.txt"});
        let decision = hook.pre_tool_use("file_write", &input).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn test_blocks_dangerous_path() {
        let hook = FileSecurity::new(vec![PathBuf::from("/etc")]);
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
    async fn test_ignores_non_write_tools() {
        let hook = FileSecurity::new(vec![PathBuf::from("/etc")]);
        let input = json!({"path": "/etc/passwd"});
        let decision = hook.pre_tool_use("file_read", &input).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn test_allows_missing_path() {
        let hook = FileSecurity::new(vec![PathBuf::from("/etc")]);
        let input = json!({"other": "field"});
        let decision = hook.pre_tool_use("file_write", &input).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }
}
