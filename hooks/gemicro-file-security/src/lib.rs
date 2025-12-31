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
/// # Security Warnings
///
/// **This hook has known bypass vulnerabilities and should NOT be relied upon
/// as the sole security mechanism:**
///
/// ## Known Bypass Methods
///
/// 1. **Symlink Attacks**: If `/tmp/evil -> /etc/passwd`, writing to `/tmp/evil`
///    will NOT be blocked. The hook does not resolve symlinks.
///
/// 2. **Relative Path Traversal**: Paths like `../../etc/passwd` or
///    `/safe/../etc/passwd` may bypass blocks depending on current directory
///    and path resolution. The hook only checks literal prefix matching.
///
/// 3. **Case Sensitivity**: On case-insensitive filesystems (macOS, Windows),
///    `/ETC/passwd` might bypass `/etc` blocks depending on PathBuf behavior.
///
/// ## Recommendations for Production Use
///
/// - **Canonicalize paths** before passing to `new()` using
///   `path.canonicalize()` if paths exist on disk
/// - **Combine with OS-level controls** like file permissions, SELinux, or
///   sandboxing
/// - **Validate inputs** at multiple layers, not just this hook
/// - **Monitor and audit** tool execution with the audit-log hook
///
/// This hook is best used as **defense-in-depth**, not primary security.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FileSecurity {
    /// Paths that are blocked from write operations.
    pub blocked_paths: Vec<PathBuf>,
}

impl FileSecurity {
    /// Create a new file security hook with the given blocked paths.
    ///
    /// Any tool attempting to write to these paths (or their subdirectories)
    /// will be denied.
    ///
    /// # Panics
    ///
    /// Panics if `blocked_paths` is empty. Use at least one path to create
    /// a meaningful security policy.
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
        if blocked_paths.is_empty() {
            panic!("FileSecurity requires at least one blocked path");
        }
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

        // Extract path from input - deny if missing (fail-closed for security)
        let path = match input.get("path").and_then(|p| p.as_str()) {
            Some(p) => p,
            None => {
                log::error!(
                    "FileSecurity: {} tool invocation has missing or non-string 'path' parameter - DENYING for safety",
                    tool_name
                );
                return Ok(HookDecision::Deny {
                    reason: format!(
                        "Tool '{}' missing required 'path' parameter - cannot validate security policy",
                        tool_name
                    ),
                });
            }
        };

        // Check if blocked
        if self.is_blocked(path) {
            log::warn!(
                "FileSecurity: BLOCKED write attempt to protected path '{}' by tool '{}'",
                path,
                tool_name
            );
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
    async fn test_denies_missing_path() {
        // Security: fail-closed when path cannot be validated
        let hook = FileSecurity::new(vec![PathBuf::from("/etc")]);
        let input = json!({"other": "field"});
        let decision = hook.pre_tool_use("file_write", &input).await.unwrap();
        assert!(
            matches!(decision, HookDecision::Deny { .. }),
            "Should deny when path is missing (fail-closed for security)"
        );
    }

    #[tokio::test]
    async fn test_blocks_subdirectories() {
        let hook = FileSecurity::new(vec![PathBuf::from("/etc")]);
        let input = json!({"path": "/etc/systemd/system/evil.service"});
        let decision = hook.pre_tool_use("file_write", &input).await.unwrap();
        assert!(
            matches!(decision, HookDecision::Deny { .. }),
            "Should block subdirectories of blocked paths"
        );
    }

    #[test]
    #[should_panic(expected = "FileSecurity requires at least one blocked path")]
    fn test_panics_on_empty_paths() {
        FileSecurity::new(vec![]);
    }

    // Tests documenting known security limitations (bypasses)

    #[tokio::test]
    async fn test_relative_path_traversal_not_prevented() {
        // KNOWN LIMITATION: Relative path traversal can bypass blocks
        // This test documents the current behavior - NOT a bug fix
        let hook = FileSecurity::new(vec![PathBuf::from("/etc")]);

        // Path that escapes to /etc via relative traversal
        let input = json!({"path": "/home/user/../../etc/passwd"});
        let decision = hook.pre_tool_use("file_write", &input).await.unwrap();

        // Currently ALLOWS (starts_with check passes because path starts with /home)
        // This is the documented vulnerability - hook doesn't canonicalize
        assert_eq!(
            decision,
            HookDecision::Allow,
            "KNOWN BYPASS: Hook allows relative path traversal (no canonicalization)"
        );
    }

    #[tokio::test]
    async fn test_relative_path_from_cwd_not_prevented() {
        // KNOWN LIMITATION: Relative paths depend on CWD and aren't resolved
        let hook = FileSecurity::new(vec![PathBuf::from("/etc")]);

        let input = json!({"path": "../../etc/passwd"});
        let decision = hook.pre_tool_use("file_write", &input).await.unwrap();

        // Currently ALLOWS (doesn't start with /etc)
        assert_eq!(
            decision,
            HookDecision::Allow,
            "KNOWN BYPASS: Hook allows relative paths (no CWD resolution)"
        );
    }
}
