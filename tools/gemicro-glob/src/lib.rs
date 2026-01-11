//! Glob tool for finding files by pattern.
//!
//! This tool allows agents to search for files matching glob patterns,
//! useful for discovering files in a project structure.

use async_trait::async_trait;
use gemicro_core::tool::{Tool, ToolError, ToolResult};
use glob::glob;
use serde_json::{json, Value};
use std::path::PathBuf;

/// Maximum number of results to return to prevent overwhelming output.
const MAX_RESULTS: usize = 100;

/// Glob tool for finding files by pattern.
///
/// Searches for files matching a glob pattern (e.g., `**/*.rs`, `src/*.txt`).
/// Returns a list of matching file paths.
///
/// # Result Metadata
///
/// On success, the result includes metadata with these fields:
/// - `pattern`: The glob pattern used
/// - `match_count`: Number of files found
/// - `truncated`: Whether results were truncated (max 100 files)
/// - `errors`: Array of paths that could not be read (permissions, etc.)
///
/// # Example
///
/// ```no_run
/// use gemicro_glob::Glob;
/// use gemicro_core::tool::Tool;
/// use serde_json::json;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let glob_tool = Glob;
///
/// let result = glob_tool.execute(json!({
///     "pattern": "src/**/*.rs"
/// })).await?;
/// println!("Found files: {}", result.content);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Glob;

#[async_trait]
impl Tool for Glob {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern. Returns a list of file paths. \
         Supports patterns like '**/*.rs' (all Rust files), 'src/*.txt' (txt files in src), \
         or 'docs/**/*' (all files in docs). The base_dir can be absolute or relative \
         (relative paths are resolved against the current working directory)."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files (e.g., '**/*.rs', 'src/*.txt')"
                },
                "base_dir": {
                    "type": "string",
                    "description": "Optional base directory to search from. Can be absolute or relative \
                                    (relative paths are resolved against the current working directory). \
                                    Defaults to current directory if not specified."
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let pattern = input
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'pattern' field".into()))?;

        if pattern.is_empty() {
            return Err(ToolError::InvalidInput("Pattern cannot be empty".into()));
        }

        // Get optional base directory
        let base_dir = input
            .get("base_dir")
            .and_then(|v| v.as_str())
            .map(PathBuf::from);

        // Construct the full pattern
        let full_pattern = if let Some(ref base) = base_dir {
            // Resolve relative paths against CWD, absolute paths remain unchanged
            let resolved_base = if base.is_absolute() {
                base.clone()
            } else {
                let cwd = std::env::current_dir().map_err(|e| {
                    ToolError::ExecutionFailed(format!(
                        "Failed to get current working directory: {}",
                        e
                    ))
                })?;
                cwd.join(base)
            };

            // Canonicalize to resolve symlinks and normalize (also validates existence)
            let canonical_base = resolved_base.canonicalize().map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => ToolError::NotFound(format!(
                    "Base directory does not exist: {} (resolved from {})",
                    resolved_base.display(),
                    base.display()
                )),
                std::io::ErrorKind::PermissionDenied => ToolError::ExecutionFailed(format!(
                    "Permission denied accessing base directory {}: {}",
                    base.display(),
                    e
                )),
                _ => ToolError::ExecutionFailed(format!(
                    "Failed to resolve base directory {}: {}",
                    base.display(),
                    e
                )),
            })?;

            // Verify it's actually a directory
            if !canonical_base.is_dir() {
                return Err(ToolError::InvalidInput(format!(
                    "Base path is not a directory: {}",
                    canonical_base.display()
                )));
            }

            format!("{}/{}", canonical_base.display(), pattern)
        } else {
            pattern.to_string()
        };

        // Execute glob search
        let entries = glob(&full_pattern).map_err(|e| {
            ToolError::InvalidInput(format!("Invalid glob pattern '{}': {}", pattern, e))
        })?;

        let mut matches: Vec<String> = Vec::new();
        let mut errors: Vec<String> = Vec::new();

        // Get CWD for converting relative paths to absolute
        let cwd = std::env::current_dir().ok();

        for entry in entries {
            match entry {
                Ok(path) => {
                    if matches.len() >= MAX_RESULTS {
                        break;
                    }
                    // Always return absolute paths for consistency with file_read/grep requirements
                    let absolute_path = if path.is_absolute() {
                        path
                    } else if let Some(ref cwd) = cwd {
                        cwd.join(&path)
                    } else {
                        path
                    };
                    matches.push(absolute_path.display().to_string());
                }
                Err(e) => {
                    errors.push(format!("Error reading path: {}", e));
                }
            }
        }

        // Sort results for consistent output
        matches.sort();

        let truncated = matches.len() >= MAX_RESULTS;
        let result = if matches.is_empty() {
            let mut output = format!("No files found matching pattern '{}'", pattern);
            // Surface any errors that occurred during traversal
            if !errors.is_empty() {
                output.push_str(&format!("\n\n{} path(s) could not be read:", errors.len()));
                for err in &errors {
                    output.push_str(&format!("\n  - {}", err));
                }
            }
            output
        } else {
            let mut output = matches.join("\n");
            if truncated {
                output.push_str(&format!("\n\n(Results truncated to {} files)", MAX_RESULTS));
            }
            // Surface any errors that occurred during traversal
            if !errors.is_empty() {
                output.push_str(&format!(
                    "\n\nNote: {} path(s) could not be read (see metadata for details)",
                    errors.len()
                ));
            }
            output
        };

        let mut tool_result = ToolResult::text(result);

        // Add metadata
        tool_result = tool_result.with_metadata(json!({
            "pattern": pattern,
            "match_count": matches.len(),
            "truncated": truncated,
            "errors": errors,
        }));

        Ok(tool_result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_name_and_description() {
        let glob_tool = Glob;
        assert_eq!(glob_tool.name(), "glob");
        assert!(!glob_tool.description().is_empty());
    }

    #[test]
    fn test_glob_parameters_schema() {
        let glob_tool = Glob;
        let schema = glob_tool.parameters_schema();

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["pattern"].is_object());
        assert!(schema["properties"]["base_dir"].is_object());

        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("pattern")));
    }

    #[tokio::test]
    async fn test_glob_missing_pattern() {
        let glob_tool = Glob;
        let result = glob_tool.execute(json!({})).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_glob_empty_pattern() {
        let glob_tool = Glob;
        let result = glob_tool.execute(json!({"pattern": ""})).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_glob_relative_base_dir() {
        use std::fs::File;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.rs");
        File::create(&file_path).unwrap();

        // RAII guard to restore CWD even on panic
        struct CwdGuard(std::path::PathBuf);
        impl Drop for CwdGuard {
            fn drop(&mut self) {
                let _ = std::env::set_current_dir(&self.0);
            }
        }
        let _guard = CwdGuard(std::env::current_dir().unwrap());
        std::env::set_current_dir(&dir).unwrap();

        let glob_tool = Glob;
        let result = glob_tool
            .execute(json!({
                "pattern": "*.rs",
                "base_dir": "."  // Relative path - should now work
            }))
            .await;

        // Guard handles restoration automatically via Drop

        assert!(result.is_ok(), "Relative path should be supported");
        let tool_result = result.unwrap();
        assert!(tool_result.content.as_str().unwrap().contains("test.rs"));
    }

    #[tokio::test]
    async fn test_glob_nonexistent_base_dir() {
        let glob_tool = Glob;
        let result = glob_tool
            .execute(json!({
                "pattern": "*.rs",
                "base_dir": "/nonexistent/directory/that/does/not/exist"
            }))
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_glob_no_matches() {
        let glob_tool = Glob;
        let result = glob_tool
            .execute(json!({
                "pattern": "/tmp/nonexistent_pattern_xyz_123/*.nonexistent"
            }))
            .await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result
            .content
            .as_str()
            .unwrap()
            .contains("No files found"));
    }

    #[tokio::test]
    async fn test_glob_with_temp_files() {
        use std::fs::File;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        File::create(&file_path).unwrap();

        let glob_tool = Glob;
        let result = glob_tool
            .execute(json!({
                "pattern": "*.txt",
                "base_dir": dir.path().to_str().unwrap()
            }))
            .await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result.content.as_str().unwrap().contains("test.txt"));

        // Check metadata
        let metadata = tool_result.metadata;
        assert_eq!(metadata["match_count"], 1);
        assert_eq!(metadata["truncated"], false);
    }

    #[tokio::test]
    async fn test_glob_recursive_pattern() {
        use std::fs::{self, File};
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let subdir = dir.path().join("subdir");
        fs::create_dir(&subdir).unwrap();

        File::create(dir.path().join("root.rs")).unwrap();
        File::create(subdir.join("nested.rs")).unwrap();

        let glob_tool = Glob;
        let result = glob_tool
            .execute(json!({
                "pattern": "**/*.rs",
                "base_dir": dir.path().to_str().unwrap()
            }))
            .await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        let content = tool_result.content.as_str().unwrap();
        assert!(content.contains("root.rs"));
        assert!(content.contains("nested.rs"));

        let metadata = tool_result.metadata;
        assert_eq!(metadata["match_count"], 2);
    }

    #[tokio::test]
    async fn test_glob_invalid_pattern() {
        let glob_tool = Glob;
        let result = glob_tool
            .execute(json!({
                "pattern": "[invalid"
            }))
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_glob_relative_path_resolution() {
        use std::fs::{self, File};
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let subdir = dir.path().join("subdir");
        fs::create_dir(&subdir).unwrap();
        File::create(subdir.join("nested.txt")).unwrap();

        // RAII guard to restore CWD even on panic
        struct CwdGuard(std::path::PathBuf);
        impl Drop for CwdGuard {
            fn drop(&mut self) {
                let _ = std::env::set_current_dir(&self.0);
            }
        }
        let _guard = CwdGuard(std::env::current_dir().unwrap());
        std::env::set_current_dir(&dir).unwrap();

        let glob_tool = Glob;

        // Test relative path from CWD
        let result = glob_tool
            .execute(json!({
                "pattern": "*.txt",
                "base_dir": "subdir"  // Relative to CWD
            }))
            .await;

        // Guard handles restoration automatically via Drop

        assert!(result.is_ok(), "Should resolve relative path against CWD");
        let tool_result = result.unwrap();
        assert!(tool_result.content.as_str().unwrap().contains("nested.txt"));
    }

    #[tokio::test]
    async fn test_glob_relative_nonexistent_dir() {
        let glob_tool = Glob;
        let result = glob_tool
            .execute(json!({
                "pattern": "*.rs",
                "base_dir": "nonexistent/relative/path"
            }))
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ToolError::NotFound(_)),
            "Should return NotFound for nonexistent relative paths"
        );
    }
}
