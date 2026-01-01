//! File security interceptor for gemicro tool execution.
//!
//! Prevents tools from writing to specific directories or files by blocking
//! file_write and file_edit operations to configured paths.
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::interceptor::{InterceptorChain, ToolCall};
//! use gemicro_core::tool::ToolResult;
//! use gemicro_file_security::FileSecurity;
//! use std::path::PathBuf;
//!
//! let interceptor = FileSecurity::new(vec![
//!     PathBuf::from("/etc"),
//!     PathBuf::from("/var"),
//!     PathBuf::from("/home/user/.ssh"),
//! ]);
//!
//! let interceptors: InterceptorChain<ToolCall, ToolResult> = InterceptorChain::new()
//!     .with(interceptor);
//! ```

use async_trait::async_trait;
use gemicro_core::interceptor::{InterceptDecision, InterceptError, Interceptor, ToolCall};
use gemicro_core::tool::ToolResult;
use std::path::PathBuf;

/// File security interceptor that blocks writes to sensitive paths.
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
/// **This interceptor has known bypass vulnerabilities and should NOT be relied upon
/// as the sole security mechanism:**
///
/// ## Known Bypass Methods
///
/// 1. **Symlink Attacks**: If `/tmp/evil -> /etc/passwd`, writing to `/tmp/evil`
///    will NOT be blocked. The interceptor does not resolve symlinks.
///
/// 2. **Relative Path Traversal**: Paths like `../../etc/passwd` or
///    `/safe/../etc/passwd` may bypass blocks depending on current directory
///    and path resolution. The interceptor only checks literal prefix matching.
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
/// - **Validate inputs** at multiple layers, not just this interceptor
/// - **Monitor and audit** tool execution with the audit-log interceptor
///
/// This interceptor is best used as **defense-in-depth**, not primary security.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FileSecurity {
    /// Paths that are blocked from write operations.
    pub blocked_paths: Vec<PathBuf>,
}

impl FileSecurity {
    /// Create a new file security interceptor with the given blocked paths.
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
    /// let interceptor = FileSecurity::new(vec![
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

    /// Create a new file security interceptor with canonicalized paths.
    ///
    /// This constructor resolves symlinks and normalizes paths, providing
    /// stronger protection against path traversal attacks. However, it
    /// requires all blocked paths to exist on disk at construction time.
    ///
    /// # Errors
    ///
    /// Returns an error if any path cannot be canonicalized (e.g., doesn't exist).
    ///
    /// # Panics
    ///
    /// Panics if `blocked_paths` is empty after filtering.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gemicro_file_security::FileSecurity;
    /// use std::path::PathBuf;
    ///
    /// let interceptor = FileSecurity::new_canonical(vec![
    ///     PathBuf::from("/etc"),
    ///     PathBuf::from("/var"),
    /// ]).expect("Paths must exist");
    /// ```
    pub fn new_canonical(blocked_paths: Vec<PathBuf>) -> std::io::Result<Self> {
        let canonical: Result<Vec<PathBuf>, std::io::Error> = blocked_paths
            .into_iter()
            .map(|p| p.canonicalize())
            .collect();

        let canonical = canonical?;

        if canonical.is_empty() {
            panic!("FileSecurity requires at least one blocked path");
        }

        Ok(Self {
            blocked_paths: canonical,
        })
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
impl Interceptor<ToolCall, ToolResult> for FileSecurity {
    async fn intercept(
        &self,
        input: &ToolCall,
    ) -> Result<InterceptDecision<ToolCall>, InterceptError> {
        // Only check file write tools
        if !matches!(input.name.as_str(), "file_write" | "file_edit") {
            return Ok(InterceptDecision::Allow);
        }

        // Extract path from input - deny if missing (fail-closed for security)
        let path = match input.arguments.get("path").and_then(|p| p.as_str()) {
            Some(p) => p,
            None => {
                log::error!(
                    "FileSecurity: {} tool invocation has missing or non-string 'path' parameter - DENYING for safety",
                    input.name
                );
                return Ok(InterceptDecision::Deny {
                    reason: format!(
                        "Tool '{}' missing required 'path' parameter - cannot validate security policy",
                        input.name
                    ),
                });
            }
        };

        // Check if blocked
        if self.is_blocked(path) {
            log::warn!(
                "FileSecurity: BLOCKED write attempt to protected path '{}' by tool '{}'",
                path,
                input.name
            );
            return Ok(InterceptDecision::Deny {
                reason: format!("Writing to '{}' is blocked by security policy", path),
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
    async fn test_allows_safe_path() {
        let interceptor = FileSecurity::new(vec![PathBuf::from("/etc")]);
        let input = ToolCall::new("file_write", json!({"path": "/home/user/safe.txt"}));
        let decision = interceptor.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);
    }

    #[tokio::test]
    async fn test_blocks_dangerous_path() {
        let interceptor = FileSecurity::new(vec![PathBuf::from("/etc")]);
        let input = ToolCall::new("file_write", json!({"path": "/etc/passwd"}));
        let decision = interceptor.intercept(&input).await.unwrap();
        match decision {
            InterceptDecision::Deny { reason } => {
                assert!(reason.contains("blocked"));
            }
            _ => panic!("Expected deny"),
        }
    }

    #[tokio::test]
    async fn test_ignores_non_write_tools() {
        let interceptor = FileSecurity::new(vec![PathBuf::from("/etc")]);
        let input = ToolCall::new("file_read", json!({"path": "/etc/passwd"}));
        let decision = interceptor.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);
    }

    #[tokio::test]
    async fn test_denies_missing_path() {
        // Security: fail-closed when path cannot be validated
        let interceptor = FileSecurity::new(vec![PathBuf::from("/etc")]);
        let input = ToolCall::new("file_write", json!({"other": "field"}));
        let decision = interceptor.intercept(&input).await.unwrap();
        assert!(
            matches!(decision, InterceptDecision::Deny { .. }),
            "Should deny when path is missing (fail-closed for security)"
        );
    }

    #[tokio::test]
    async fn test_blocks_subdirectories() {
        let interceptor = FileSecurity::new(vec![PathBuf::from("/etc")]);
        let input = ToolCall::new(
            "file_write",
            json!({"path": "/etc/systemd/system/evil.service"}),
        );
        let decision = interceptor.intercept(&input).await.unwrap();
        assert!(
            matches!(decision, InterceptDecision::Deny { .. }),
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
        let interceptor = FileSecurity::new(vec![PathBuf::from("/etc")]);

        // Path that escapes to /etc via relative traversal
        let input = ToolCall::new("file_write", json!({"path": "/home/user/../../etc/passwd"}));
        let decision = interceptor.intercept(&input).await.unwrap();

        // Currently ALLOWS (starts_with check passes because path starts with /home)
        // This is the documented vulnerability - interceptor doesn't canonicalize
        assert_eq!(
            decision,
            InterceptDecision::Allow,
            "KNOWN BYPASS: Interceptor allows relative path traversal (no canonicalization)"
        );
    }

    #[tokio::test]
    async fn test_relative_path_from_cwd_not_prevented() {
        // KNOWN LIMITATION: Relative paths depend on CWD and aren't resolved
        let interceptor = FileSecurity::new(vec![PathBuf::from("/etc")]);

        let input = ToolCall::new("file_write", json!({"path": "../../etc/passwd"}));
        let decision = interceptor.intercept(&input).await.unwrap();

        // Currently ALLOWS (doesn't start with /etc)
        assert_eq!(
            decision,
            InterceptDecision::Allow,
            "KNOWN BYPASS: Interceptor allows relative paths (no CWD resolution)"
        );
    }

    #[test]
    fn test_new_canonical_with_existing_paths() {
        // /tmp should exist on most Unix systems
        let result = FileSecurity::new_canonical(vec![PathBuf::from("/tmp")]);
        assert!(result.is_ok(), "Should succeed for existing path");

        let interceptor = result.unwrap();
        // Canonical path should resolve symlinks if any
        assert!(!interceptor.blocked_paths.is_empty());
    }

    #[test]
    fn test_new_canonical_with_nonexistent_path() {
        let result = FileSecurity::new_canonical(vec![PathBuf::from(
            "/this/path/definitely/does/not/exist/12345",
        )]);
        assert!(result.is_err(), "Should fail for non-existent path");
    }
}
