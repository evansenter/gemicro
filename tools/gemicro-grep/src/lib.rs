//! Grep tool for searching file contents.
//!
//! This tool allows agents to search for patterns within files,
//! useful for finding code, configuration, or text content.

use async_trait::async_trait;
use gemicro_core::tool::{Tool, ToolError, ToolResult};
use regex::Regex;
use serde_json::{json, Value};
use std::path::Path;
use tokio::fs;
use tokio::io::{AsyncBufReadExt, BufReader};

/// Maximum number of matches to return per file.
const MAX_MATCHES_PER_FILE: usize = 50;

/// Maximum total matches across all files when searching directories.
const MAX_TOTAL_MATCHES: usize = 200;

/// Maximum file size to search (10MB).
const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024;

/// Grep tool for searching file contents.
///
/// Searches for a regex pattern within a file or files and returns
/// matching lines with their line numbers.
///
/// # Result Metadata
///
/// On success, the result includes metadata with these fields:
/// - `pattern`: The search pattern used
/// - `path`: The file path searched
/// - `match_count`: Number of matches found
/// - `truncated`: Whether results were truncated (max 50 matches per file)
/// - `case_insensitive`: Whether case-insensitive matching was used
///
/// # Example
///
/// ```no_run
/// use gemicro_grep::Grep;
/// use gemicro_core::tool::Tool;
/// use serde_json::json;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let grep = Grep;
///
/// let result = grep.execute(json!({
///     "pattern": "fn main",
///     "path": "/path/to/file.rs"
/// })).await?;
/// println!("Matches: {}", result.content);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Grep;

/// A single match result.
#[derive(Debug)]
struct Match {
    file: String,
    line_number: usize,
    content: String,
}

impl Grep {
    async fn search_file(
        &self,
        path: &Path,
        regex: &Regex,
        case_insensitive: bool,
        max_matches: usize,
    ) -> Result<Vec<Match>, ToolError> {
        let metadata = fs::metadata(path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ToolError::NotFound(format!("File not found: {}", path.display()))
            } else {
                ToolError::ExecutionFailed(format!("Cannot access file: {}", e))
            }
        })?;

        if !metadata.is_file() {
            return Ok(Vec::new()); // Skip non-files silently when called from directory search
        }

        if metadata.len() > MAX_FILE_SIZE {
            return Ok(Vec::new()); // Skip large files silently
        }

        let file = fs::File::open(path)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Cannot open file: {}", e)))?;

        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let mut matches = Vec::new();
        let mut line_number = 0;

        while let Some(line) = lines.next_line().await.unwrap_or(None) {
            line_number += 1;

            let line_to_match = if case_insensitive {
                line.to_lowercase()
            } else {
                line.clone()
            };

            if regex.is_match(&line_to_match) {
                matches.push(Match {
                    file: path.display().to_string(),
                    line_number,
                    content: line,
                });

                if matches.len() >= max_matches {
                    break;
                }
            }
        }

        Ok(matches)
    }

    /// Recursively search a directory for matches.
    async fn search_directory(
        &self,
        dir: &Path,
        regex: &Regex,
        case_insensitive: bool,
    ) -> Result<Vec<Match>, ToolError> {
        let mut all_matches = Vec::new();
        let mut dirs_to_visit = vec![dir.to_path_buf()];

        while let Some(current_dir) = dirs_to_visit.pop() {
            let mut entries = fs::read_dir(&current_dir).await.map_err(|e| {
                ToolError::ExecutionFailed(format!(
                    "Cannot read directory {}: {}",
                    current_dir.display(),
                    e
                ))
            })?;

            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();

                // Skip hidden files/directories
                if path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with('.'))
                    .unwrap_or(false)
                {
                    continue;
                }

                if path.is_dir() {
                    dirs_to_visit.push(path);
                } else if path.is_file() {
                    // Calculate remaining matches allowed
                    let remaining = MAX_TOTAL_MATCHES.saturating_sub(all_matches.len());
                    if remaining == 0 {
                        return Ok(all_matches);
                    }

                    let file_matches = self
                        .search_file(
                            &path,
                            regex,
                            case_insensitive,
                            remaining.min(MAX_MATCHES_PER_FILE),
                        )
                        .await
                        .unwrap_or_default();

                    all_matches.extend(file_matches);

                    if all_matches.len() >= MAX_TOTAL_MATCHES {
                        return Ok(all_matches);
                    }
                }
            }
        }

        Ok(all_matches)
    }
}

#[async_trait]
impl Tool for Grep {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search for a pattern in file or directory contents. If path is a directory, searches \
         recursively (skipping hidden files). Returns matching lines with file paths and line numbers. \
         Supports regex patterns."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Path to file or directory to search. Relative paths are resolved against CWD."
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Whether to ignore case (default: false)"
                }
            },
            "required": ["pattern", "path"]
        })
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let pattern = input
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'pattern' field".into()))?;

        let path_str = input
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'path' field".into()))?;

        let case_insensitive = input
            .get("case_insensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if pattern.is_empty() {
            return Err(ToolError::InvalidInput("Pattern cannot be empty".into()));
        }

        // Resolve relative paths against CWD
        let path = Path::new(path_str);
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|e| {
                    ToolError::ExecutionFailed(format!("Cannot get current directory: {}", e))
                })?
                .join(path)
        };

        // Compile regex - for case-insensitive, use lowercase pattern and match against
        // lowercased lines in search_file. Only compile once to avoid redundant work.
        let regex_pattern = if case_insensitive {
            pattern.to_lowercase()
        } else {
            pattern.to_string()
        };

        let regex = Regex::new(&regex_pattern).map_err(|e| {
            ToolError::InvalidInput(format!("Invalid regex pattern '{}': {}", pattern, e))
        })?;

        // Check if path is a directory or file
        let metadata = fs::metadata(&path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ToolError::NotFound(format!("Path not found: {}", path.display()))
            } else {
                ToolError::ExecutionFailed(format!("Cannot access path: {}", e))
            }
        })?;

        let (matches, is_directory) = if metadata.is_dir() {
            (
                self.search_directory(&path, &regex, case_insensitive)
                    .await?,
                true,
            )
        } else {
            (
                self.search_file(&path, &regex, case_insensitive, MAX_MATCHES_PER_FILE)
                    .await?,
                false,
            )
        };

        let match_count = matches.len();
        let truncated = if is_directory {
            match_count >= MAX_TOTAL_MATCHES
        } else {
            match_count >= MAX_MATCHES_PER_FILE
        };

        let result = if matches.is_empty() {
            format!("No matches found for pattern '{}' in {}", pattern, path_str)
        } else {
            let formatted: Vec<String> = matches
                .iter()
                .map(|m| format!("{}:{}: {}", m.file, m.line_number, m.content))
                .collect();
            let mut output = formatted.join("\n");
            if truncated {
                let max = if is_directory {
                    MAX_TOTAL_MATCHES
                } else {
                    MAX_MATCHES_PER_FILE
                };
                output.push_str(&format!("\n\n(Results truncated to {} matches)", max));
            }
            output
        };

        // Count unique files in matches
        let files_searched: std::collections::HashSet<_> =
            matches.iter().map(|m| &m.file).collect();

        let mut tool_result = ToolResult::text(result);
        tool_result = tool_result.with_metadata(json!({
            "pattern": pattern,
            "path": path_str,
            "match_count": match_count,
            "files_with_matches": files_searched.len(),
            "is_directory": is_directory,
            "truncated": truncated,
            "case_insensitive": case_insensitive,
        }));

        Ok(tool_result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_grep_name_and_description() {
        let grep = Grep;
        assert_eq!(grep.name(), "grep");
        assert!(!grep.description().is_empty());
    }

    #[test]
    fn test_grep_parameters_schema() {
        let grep = Grep;
        let schema = grep.parameters_schema();

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["pattern"].is_object());
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["properties"]["case_insensitive"].is_object());

        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("pattern")));
        assert!(required.contains(&json!("path")));
    }

    #[tokio::test]
    async fn test_grep_missing_pattern() {
        let grep = Grep;
        let result = grep.execute(json!({"path": "/tmp/test.txt"})).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_grep_missing_path() {
        let grep = Grep;
        let result = grep.execute(json!({"pattern": "test"})).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_grep_empty_pattern() {
        let grep = Grep;
        let result = grep
            .execute(json!({"pattern": "", "path": "/tmp/test.txt"}))
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_grep_relative_path() {
        // Relative paths are now resolved against CWD, so this should fail with NotFound
        let grep = Grep;
        let result = grep
            .execute(json!({"pattern": "test", "path": "relative/path.txt"}))
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_grep_file_not_found() {
        let grep = Grep;
        let result = grep
            .execute(json!({
                "pattern": "test",
                "path": "/nonexistent/file/path.txt"
            }))
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_grep_invalid_regex() {
        let grep = Grep;
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "test content").unwrap();

        let result = grep
            .execute(json!({
                "pattern": "[invalid",
                "path": file.path().to_str().unwrap()
            }))
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_grep_no_matches() {
        let grep = Grep;
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "hello world").unwrap();

        let result = grep
            .execute(json!({
                "pattern": "nonexistent_pattern",
                "path": file.path().to_str().unwrap()
            }))
            .await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result
            .content
            .as_str()
            .unwrap()
            .contains("No matches found"));
    }

    #[tokio::test]
    async fn test_grep_finds_matches() {
        let grep = Grep;
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "line 1: hello").unwrap();
        writeln!(file, "line 2: world").unwrap();
        writeln!(file, "line 3: hello again").unwrap();

        let result = grep
            .execute(json!({
                "pattern": "hello",
                "path": file.path().to_str().unwrap()
            }))
            .await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        let content = tool_result.content.as_str().unwrap();
        assert!(content.contains("line 1: hello"));
        assert!(content.contains("line 3: hello again"));
        assert!(!content.contains("line 2: world"));

        let metadata = tool_result.metadata;
        assert_eq!(metadata["match_count"], 2);
    }

    #[tokio::test]
    async fn test_grep_case_insensitive() {
        let grep = Grep;
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "HELLO").unwrap();
        writeln!(file, "hello").unwrap();
        writeln!(file, "Hello").unwrap();

        let result = grep
            .execute(json!({
                "pattern": "hello",
                "path": file.path().to_str().unwrap(),
                "case_insensitive": true
            }))
            .await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();

        let metadata = tool_result.metadata;
        assert_eq!(metadata["match_count"], 3);
        assert_eq!(metadata["case_insensitive"], true);
    }

    #[tokio::test]
    async fn test_grep_regex_pattern() {
        let grep = Grep;
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "fn main() {{").unwrap();
        writeln!(file, "fn helper() {{").unwrap();
        writeln!(file, "let x = 42;").unwrap();

        let result = grep
            .execute(json!({
                "pattern": r"fn \w+\(\)",
                "path": file.path().to_str().unwrap()
            }))
            .await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();

        let metadata = tool_result.metadata;
        assert_eq!(metadata["match_count"], 2);
    }

    #[tokio::test]
    async fn test_grep_directory_search() {
        use tempfile::tempdir;

        let grep = Grep;
        let dir = tempdir().unwrap();

        // Create a few files with content
        let file1 = dir.path().join("file1.txt");
        std::fs::write(&file1, "hello world\nfoo bar\nhello again").unwrap();

        let file2 = dir.path().join("file2.txt");
        std::fs::write(&file2, "goodbye world\nhello there").unwrap();

        // Create a subdirectory with a file
        let subdir = dir.path().join("subdir");
        std::fs::create_dir(&subdir).unwrap();
        let file3 = subdir.join("file3.txt");
        std::fs::write(&file3, "hello from subdir").unwrap();

        let result = grep
            .execute(json!({
                "pattern": "hello",
                "path": dir.path().to_str().unwrap()
            }))
            .await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        let metadata = &tool_result.metadata;

        // Should find 4 matches across 3 files
        assert_eq!(metadata["match_count"], 4);
        assert_eq!(metadata["files_with_matches"], 3);
        assert_eq!(metadata["is_directory"], true);
    }

    #[tokio::test]
    async fn test_grep_directory_skips_hidden() {
        use tempfile::tempdir;

        let grep = Grep;
        let dir = tempdir().unwrap();

        // Create a regular file
        let file1 = dir.path().join("visible.txt");
        std::fs::write(&file1, "hello visible").unwrap();

        // Create a hidden file (should be skipped)
        let hidden = dir.path().join(".hidden.txt");
        std::fs::write(&hidden, "hello hidden").unwrap();

        let result = grep
            .execute(json!({
                "pattern": "hello",
                "path": dir.path().to_str().unwrap()
            }))
            .await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        let content = tool_result.content.as_str().unwrap();

        // Should only find the visible file
        assert!(content.contains("visible.txt"));
        assert!(!content.contains(".hidden.txt"));
        assert_eq!(tool_result.metadata["match_count"], 1);
    }
}
