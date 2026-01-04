//! Path sandbox interceptor for gemicro tool execution.
//!
//! Restricts file operations to a whitelist of allowed directories, preventing
//! agents from accessing files outside the sandbox. Unlike FileSecurity which
//! blocks specific paths (blacklist), PathSandbox allows only explicitly
//! permitted paths (whitelist).
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::interceptor::{InterceptorChain, ToolCall};
//! use gemicro_core::tool::ToolResult;
//! use gemicro_path_sandbox::PathSandbox;
//! use std::path::PathBuf;
//!
//! // Allow only /tmp and /home/user/workspace
//! let sandbox = PathSandbox::new(vec![
//!     PathBuf::from("/tmp"),
//!     PathBuf::from("/home/user/workspace"),
//! ]);
//!
//! let interceptors: InterceptorChain<ToolCall, ToolResult> = InterceptorChain::new()
//!     .with(sandbox);
//! ```

use async_trait::async_trait;
use gemicro_core::interceptor::{InterceptDecision, InterceptError, Interceptor, ToolCall};
use gemicro_core::tool::ToolResult;
use std::path::PathBuf;

/// Path sandbox interceptor that restricts file operations to allowed paths.
///
/// Prevents tools from accessing files outside specified sandbox directories.
/// All file operations (read, write, edit, grep, glob) are restricted.
///
/// # Behavior
///
/// - Intercepts `file_read`, `file_write`, `file_edit`, `grep`, and `glob` tools
/// - Canonicalizes paths at interception time to prevent symlink/traversal attacks
/// - Denies any operation where the resolved path is not under an allowed directory
/// - Other tools pass through unchanged
///
/// # Security Model
///
/// **Whitelist-based**: Only operations within `allowed_paths` are permitted.
/// All other paths are denied. This is the inverse of FileSecurity's blacklist approach.
///
/// **Canonical path resolution**: All paths are resolved to their canonical form
/// before checking, which:
/// - Resolves symlinks to their targets
/// - Normalizes `..` and `.` components
/// - Converts relative paths to absolute
///
/// ## Protection Against Attacks
///
/// 1. **Symlink Attacks**: If `/tmp/evil -> /etc/passwd`, the interceptor resolves
///    `/tmp/evil` to `/etc/passwd` and denies (unless `/etc` is in allowed_paths).
///
/// 2. **Path Traversal**: Paths like `../../etc/passwd` or `/tmp/../etc/passwd`
///    are canonicalized to `/etc/passwd` and checked against allowed_paths.
///
/// 3. **Relative Paths**: Relative paths are resolved against the current working
///    directory before checking (e.g., `../../etc/passwd` -> `/etc/passwd`).
///
/// ## Limitations
///
/// - **I/O overhead**: Each path is canonicalized (filesystem operation) at
///   interception time. This is necessary for security but adds latency.
///
/// - **TOCTOU race**: Between path validation and actual tool execution, the
///   filesystem could change (symlink swap, etc.). This is inherent to any
///   pre-execution validation. Combine with OS-level controls for defense-in-depth.
///
/// - **Non-existent paths**: `canonicalize()` fails for non-existent paths. For
///   file_write to a new file, the interceptor canonicalizes the parent directory.
///
/// ## Recommendations for Production Use
///
/// - **Combine with FileSecurity**: Use PathSandbox for broad restrictions and
///   FileSecurity for specific dangerous paths within the sandbox
/// - **OS-level controls**: Supplement with file permissions, SELinux, AppArmor,
///   or container isolation
/// - **Monitor and audit**: Use the audit-log interceptor to track all file access
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PathSandbox {
    /// Paths that are allowed for file operations.
    pub allowed_paths: Vec<PathBuf>,
}

impl PathSandbox {
    /// Create a new path sandbox with the given allowed paths.
    ///
    /// Any tool attempting to access paths outside these directories (or their
    /// subdirectories) will be denied.
    ///
    /// # Panics
    ///
    /// Panics if `allowed_paths` is empty. Use at least one path to create
    /// a meaningful sandbox policy.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_path_sandbox::PathSandbox;
    /// use std::path::PathBuf;
    ///
    /// let sandbox = PathSandbox::new(vec![
    ///     PathBuf::from("/tmp"),
    ///     PathBuf::from("/home/user/workspace"),
    /// ]);
    /// ```
    pub fn new(allowed_paths: Vec<PathBuf>) -> Self {
        if allowed_paths.is_empty() {
            panic!("PathSandbox requires at least one allowed path");
        }
        Self { allowed_paths }
    }

    /// Create a new path sandbox with canonicalized paths.
    ///
    /// This constructor resolves symlinks and normalizes paths at construction
    /// time, ensuring the allowed paths are well-formed. All allowed paths
    /// must exist on disk.
    ///
    /// # Errors
    ///
    /// Returns an error if any path cannot be canonicalized (e.g., doesn't exist).
    ///
    /// # Panics
    ///
    /// Panics if `allowed_paths` is empty after filtering.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gemicro_path_sandbox::PathSandbox;
    /// use std::path::PathBuf;
    ///
    /// let sandbox = PathSandbox::new_canonical(vec![
    ///     PathBuf::from("/tmp"),
    ///     PathBuf::from("/home/user/workspace"),
    /// ]).expect("Paths must exist");
    /// ```
    pub fn new_canonical(allowed_paths: Vec<PathBuf>) -> std::io::Result<Self> {
        let canonical: Result<Vec<PathBuf>, std::io::Error> = allowed_paths
            .into_iter()
            .map(|p| p.canonicalize())
            .collect();

        let canonical = canonical?;

        if canonical.is_empty() {
            panic!("PathSandbox requires at least one allowed path");
        }

        Ok(Self {
            allowed_paths: canonical,
        })
    }

    /// Check if a path is allowed by the sandbox.
    ///
    /// Canonicalizes the path and checks if it starts with any allowed path.
    /// For non-existent paths (e.g., file_write to new file), canonicalizes
    /// the parent directory instead.
    async fn is_path_allowed(&self, path: &str) -> Result<bool, String> {
        let path_buf = PathBuf::from(path);

        // Try to canonicalize the path directly
        let canonical = match tokio::fs::canonicalize(&path_buf).await {
            Ok(p) => p,
            Err(_) => {
                // Path doesn't exist - try canonicalizing parent directory
                // This handles file_write to new files
                let parent = path_buf.parent().ok_or_else(|| {
                    format!("Cannot determine parent directory for path: {}", path)
                })?;

                // Handle empty parent (e.g., just a filename)
                let parent = if parent.as_os_str().is_empty() {
                    std::path::Path::new(".")
                } else {
                    parent
                };

                let canonical_parent = tokio::fs::canonicalize(parent).await.map_err(|e| {
                    format!(
                        "Cannot canonicalize parent directory '{}': {}",
                        parent.display(),
                        e
                    )
                })?;

                // Reconstruct path with canonical parent + filename
                if let Some(filename) = path_buf.file_name() {
                    canonical_parent.join(filename)
                } else {
                    return Err(format!("Path has no filename component: {}", path));
                }
            }
        };

        // Check if canonical path starts with any allowed path
        for allowed in &self.allowed_paths {
            // Canonicalize allowed path too for consistent comparison
            let canonical_allowed = match tokio::fs::canonicalize(allowed).await {
                Ok(p) => p,
                Err(e) => {
                    // Skip non-canonicalizable allowed paths with warning
                    // (don't fall back to non-canonical - could allow bypass)
                    log::warn!(
                        "PathSandbox: Skipping allowed path '{}' - cannot canonicalize: {}",
                        allowed.display(),
                        e
                    );
                    continue;
                }
            };

            if canonical.starts_with(&canonical_allowed) {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Extract path from tool arguments based on tool name.
    ///
    /// Different tools use different parameter names:
    /// - file_read, file_write, file_edit, grep: "path"
    /// - glob: "base_dir" (optional, defaults to CWD)
    fn extract_path<'a>(
        tool_name: &str,
        arguments: &'a serde_json::Value,
    ) -> Result<Option<&'a str>, String> {
        let param_name = match tool_name {
            "file_read" | "file_write" | "file_edit" | "grep" => "path",
            "glob" => "base_dir",
            _ => return Ok(None),
        };

        match arguments.get(param_name) {
            None => {
                if tool_name == "glob" {
                    // base_dir is optional for glob - defaults to CWD
                    Ok(Some("."))
                } else {
                    // path is required for other tools
                    Err(format!(
                        "Tool '{}' missing required '{}' parameter",
                        tool_name, param_name
                    ))
                }
            }
            Some(value) => value.as_str().map(Some).ok_or_else(|| {
                format!(
                    "Tool '{}' parameter '{}' must be a string, got: {:?}",
                    tool_name, param_name, value
                )
            }),
        }
    }
}

#[async_trait]
impl Interceptor<ToolCall, ToolResult> for PathSandbox {
    async fn intercept(
        &self,
        input: &ToolCall,
    ) -> Result<InterceptDecision<ToolCall>, InterceptError> {
        // Only intercept file-related tools
        if !matches!(
            input.name.as_str(),
            "file_read" | "file_write" | "file_edit" | "grep" | "glob"
        ) {
            return Ok(InterceptDecision::Allow);
        }

        // Extract path based on tool type
        let path = match Self::extract_path(&input.name, &input.arguments) {
            Ok(Some(p)) => p,
            Ok(None) => {
                // Tool doesn't have a path parameter - shouldn't happen
                return Ok(InterceptDecision::Allow);
            }
            Err(reason) => {
                log::error!("PathSandbox: DENIED {} - {}", input.name, reason);
                return Ok(InterceptDecision::Deny { reason });
            }
        };

        // Check if path is allowed (with canonicalization)
        match self.is_path_allowed(path).await {
            Ok(true) => {
                log::debug!(
                    "PathSandbox: ALLOWED {} to path '{}' (within sandbox)",
                    input.name,
                    path
                );
                Ok(InterceptDecision::Allow)
            }
            Ok(false) => {
                log::warn!(
                    "PathSandbox: DENIED {} to path '{}' (outside sandbox)",
                    input.name,
                    path
                );
                Ok(InterceptDecision::Deny {
                    reason: format!(
                        "Path '{}' is outside the allowed sandbox. Allowed paths: {:?}",
                        path, self.allowed_paths
                    ),
                })
            }
            Err(e) => {
                // Canonicalization or validation failed - fail closed
                log::error!(
                    "PathSandbox: DENIED {} to path '{}' due to validation error: {}",
                    input.name,
                    path,
                    e
                );
                Ok(InterceptDecision::Deny {
                    reason: format!("Cannot validate path '{}': {}", path, e),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::env;

    #[tokio::test]
    async fn test_allows_path_in_sandbox() {
        let sandbox = PathSandbox::new(vec![PathBuf::from("/tmp")]);
        let input = ToolCall::new("file_read", json!({"path": "/tmp/test.txt"}));
        let decision = sandbox.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);
    }

    #[tokio::test]
    async fn test_denies_path_outside_sandbox() {
        let sandbox = PathSandbox::new(vec![PathBuf::from("/tmp")]);
        let input = ToolCall::new("file_read", json!({"path": "/etc/passwd"}));
        let decision = sandbox.intercept(&input).await.unwrap();
        match decision {
            InterceptDecision::Deny { reason } => {
                assert!(reason.contains("outside the allowed sandbox"));
                assert!(reason.contains("/etc/passwd"));
            }
            _ => panic!("Expected deny, got: {:?}", decision),
        }
    }

    #[tokio::test]
    async fn test_allows_subdirectory_in_sandbox() {
        // Create a real subdirectory to test
        let test_dir = env::temp_dir().join("gemicro_sandbox_subdir_test");
        let subdir = test_dir.join("subdir");
        let _ = tokio::fs::create_dir_all(&subdir).await;

        let sandbox = PathSandbox::new(vec![test_dir.clone()]);
        let test_file = subdir.join("file.txt");
        let input = ToolCall::new("file_read", json!({"path": test_file.to_str().unwrap()}));
        let decision = sandbox.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&test_dir).await;
    }

    #[tokio::test]
    async fn test_intercepts_file_read() {
        let sandbox = PathSandbox::new(vec![PathBuf::from("/tmp")]);
        let input = ToolCall::new("file_read", json!({"path": "/etc/passwd"}));
        let decision = sandbox.intercept(&input).await.unwrap();
        assert!(matches!(decision, InterceptDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn test_intercepts_file_write() {
        let sandbox = PathSandbox::new(vec![PathBuf::from("/tmp")]);
        let input = ToolCall::new("file_write", json!({"path": "/etc/test"}));
        let decision = sandbox.intercept(&input).await.unwrap();
        assert!(matches!(decision, InterceptDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn test_intercepts_file_edit() {
        let sandbox = PathSandbox::new(vec![PathBuf::from("/tmp")]);
        let input = ToolCall::new("file_edit", json!({"path": "/var/log/test.log"}));
        let decision = sandbox.intercept(&input).await.unwrap();
        assert!(matches!(decision, InterceptDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn test_intercepts_grep() {
        let sandbox = PathSandbox::new(vec![PathBuf::from("/tmp")]);
        let input = ToolCall::new(
            "grep",
            json!({"path": "/etc/hosts", "pattern": "localhost"}),
        );
        let decision = sandbox.intercept(&input).await.unwrap();
        assert!(matches!(decision, InterceptDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn test_intercepts_glob() {
        let sandbox = PathSandbox::new(vec![PathBuf::from("/tmp")]);
        let input = ToolCall::new("glob", json!({"pattern": "*.txt", "base_dir": "/etc"}));
        let decision = sandbox.intercept(&input).await.unwrap();
        assert!(matches!(decision, InterceptDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn test_glob_without_base_dir_uses_cwd() {
        // glob without base_dir should use CWD
        let cwd = env::current_dir().unwrap();
        let sandbox = PathSandbox::new(vec![cwd.clone()]);
        let input = ToolCall::new("glob", json!({"pattern": "*.txt"}));
        let decision = sandbox.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);
    }

    #[tokio::test]
    async fn test_ignores_non_file_tools() {
        let sandbox = PathSandbox::new(vec![PathBuf::from("/tmp")]);
        let input = ToolCall::new("bash", json!({"command": "ls /"}));
        let decision = sandbox.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);
    }

    #[tokio::test]
    async fn test_multiple_allowed_paths() {
        let sandbox = PathSandbox::new(vec![PathBuf::from("/tmp"), PathBuf::from("/var/log")]);

        let input1 = ToolCall::new("file_read", json!({"path": "/tmp/test.txt"}));
        assert_eq!(
            sandbox.intercept(&input1).await.unwrap(),
            InterceptDecision::Allow
        );

        let input2 = ToolCall::new("file_read", json!({"path": "/var/log/test.log"}));
        assert_eq!(
            sandbox.intercept(&input2).await.unwrap(),
            InterceptDecision::Allow
        );

        let input3 = ToolCall::new("file_read", json!({"path": "/etc/passwd"}));
        assert!(matches!(
            sandbox.intercept(&input3).await.unwrap(),
            InterceptDecision::Deny { .. }
        ));
    }

    #[test]
    #[should_panic(expected = "PathSandbox requires at least one allowed path")]
    fn test_panics_on_empty_paths() {
        PathSandbox::new(vec![]);
    }

    // Security tests - demonstrate protection against attacks

    #[tokio::test]
    async fn test_prevents_path_traversal() {
        // PathSandbox should prevent traversal attacks via canonicalization
        let sandbox = PathSandbox::new(vec![PathBuf::from("/tmp")]);

        // Attempt to escape via ../..
        let input = ToolCall::new("file_read", json!({"path": "/tmp/../../etc/passwd"}));
        let decision = sandbox.intercept(&input).await.unwrap();

        // Should deny - canonicalize resolves to /etc/passwd
        assert!(
            matches!(decision, InterceptDecision::Deny { .. }),
            "Path traversal should be prevented by canonicalization"
        );
    }

    #[tokio::test]
    async fn test_prevents_symlink_escape() {
        // Create a test directory structure
        let test_dir = env::temp_dir().join("gemicro_sandbox_symlink_test");
        let _ = tokio::fs::remove_dir_all(&test_dir).await; // Clean up any previous run
        let _ = tokio::fs::create_dir(&test_dir).await;

        let safe_dir = test_dir.join("safe");
        let _ = tokio::fs::create_dir(&safe_dir).await;

        let target_file = test_dir.join("target.txt");
        let _ = tokio::fs::write(&target_file, "secret").await;

        let symlink_path = safe_dir.join("link");

        // Create symlink (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            if symlink(&target_file, &symlink_path).is_ok() {
                let sandbox = PathSandbox::new(vec![safe_dir.clone()]);

                // Try to access symlink - should be denied because it resolves outside safe_dir
                let input =
                    ToolCall::new("file_read", json!({"path": symlink_path.to_str().unwrap()}));
                let decision = sandbox.intercept(&input).await.unwrap();

                assert!(
                    matches!(decision, InterceptDecision::Deny { .. }),
                    "Symlink escape should be prevented - symlink resolves to {:?} which is outside sandbox {:?}",
                    target_file,
                    safe_dir
                );
            }
        }

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&test_dir).await;
    }

    #[test]
    fn test_new_canonical_with_existing_paths() {
        // /tmp should exist on most Unix systems
        let result = PathSandbox::new_canonical(vec![PathBuf::from("/tmp")]);
        assert!(result.is_ok(), "Should succeed for existing path");

        let sandbox = result.unwrap();
        assert!(!sandbox.allowed_paths.is_empty());
    }

    #[test]
    fn test_new_canonical_with_nonexistent_path() {
        let result = PathSandbox::new_canonical(vec![PathBuf::from(
            "/this/path/definitely/does/not/exist/12345",
        )]);
        assert!(result.is_err(), "Should fail for non-existent path");
    }

    #[tokio::test]
    async fn test_denies_missing_path_parameter() {
        let sandbox = PathSandbox::new(vec![PathBuf::from("/tmp")]);

        // file_read without path parameter
        let input = ToolCall::new("file_read", json!({"other": "field"}));
        let decision = sandbox.intercept(&input).await.unwrap();

        assert!(
            matches!(decision, InterceptDecision::Deny { .. }),
            "Should deny when path parameter is missing"
        );
    }

    #[tokio::test]
    async fn test_denies_non_string_path() {
        let sandbox = PathSandbox::new(vec![PathBuf::from("/tmp")]);

        // path is a number instead of string
        let input = ToolCall::new("file_read", json!({"path": 123}));
        let decision = sandbox.intercept(&input).await.unwrap();

        assert!(
            matches!(decision, InterceptDecision::Deny { .. }),
            "Should deny when path is not a string"
        );
    }

    #[tokio::test]
    async fn test_allows_write_to_new_file_in_sandbox() {
        // Test writing a new file (doesn't exist yet) to sandbox
        let test_dir = env::temp_dir().join("gemicro_sandbox_newfile_test");
        let _ = tokio::fs::create_dir_all(&test_dir).await;

        let sandbox = PathSandbox::new(vec![test_dir.clone()]);
        let new_file = test_dir.join("newfile.txt");
        let input = ToolCall::new("file_write", json!({"path": new_file.to_str().unwrap()}));
        let decision = sandbox.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&test_dir).await;
    }

    // Bare filename tests - verify CWD resolution

    #[tokio::test]
    async fn test_bare_filename_allowed_when_cwd_in_sandbox() {
        // Bare filename like "file.txt" should resolve against CWD
        let cwd = env::current_dir().unwrap();
        let sandbox = PathSandbox::new(vec![cwd.clone()]);
        let input = ToolCall::new("file_write", json!({"path": "newfile.txt"}));
        let decision = sandbox.intercept(&input).await.unwrap();
        assert_eq!(
            decision,
            InterceptDecision::Allow,
            "Bare filename should be allowed when CWD is in sandbox"
        );
    }

    #[tokio::test]
    async fn test_bare_filename_denied_when_cwd_not_in_sandbox() {
        // Bare filename should be denied when CWD is not in sandbox
        // Use a path that's definitely not CWD
        let sandbox = PathSandbox::new(vec![PathBuf::from(
            "/definitely/not/the/current/working/directory/12345",
        )]);
        let input = ToolCall::new("file_write", json!({"path": "newfile.txt"}));
        let decision = sandbox.intercept(&input).await.unwrap();
        assert!(
            matches!(decision, InterceptDecision::Deny { .. }),
            "Bare filename should be denied when CWD is not in sandbox"
        );
    }
}
