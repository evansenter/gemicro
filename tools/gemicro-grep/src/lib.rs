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

/// Maximum file size to search (10MB).
const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024;

/// Grep tool for searching file contents.
///
/// Searches for a regex pattern within a file or files and returns
/// matching lines with their line numbers.
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
    ) -> Result<Vec<Match>, ToolError> {
        // Check file exists and is a file
        let metadata = fs::metadata(path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ToolError::NotFound(format!("File not found: {}", path.display()))
            } else {
                ToolError::ExecutionFailed(format!("Cannot access file: {}", e))
            }
        })?;

        if !metadata.is_file() {
            return Err(ToolError::InvalidInput(format!(
                "Path is not a file: {}",
                path.display()
            )));
        }

        if metadata.len() > MAX_FILE_SIZE {
            return Err(ToolError::InvalidInput(format!(
                "File too large ({} bytes, max {} bytes)",
                metadata.len(),
                MAX_FILE_SIZE
            )));
        }

        let file = fs::File::open(path)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Cannot open file: {}", e)))?;

        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let mut matches = Vec::new();
        let mut line_number = 0;

        while let Some(line) = lines
            .next_line()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Error reading file: {}", e)))?
        {
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

                if matches.len() >= MAX_MATCHES_PER_FILE {
                    break;
                }
            }
        }

        Ok(matches)
    }
}

#[async_trait]
impl Tool for Grep {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search for a pattern in file contents. Returns matching lines with line numbers. \
         Supports regex patterns. Use for finding code, configuration values, or text."
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
                    "description": "Absolute path to the file to search"
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

        let path = Path::new(path_str);
        if !path.is_absolute() {
            return Err(ToolError::InvalidInput("Path must be absolute".into()));
        }

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

        let matches = self.search_file(path, &regex, case_insensitive).await?;

        let match_count = matches.len();
        let truncated = match_count >= MAX_MATCHES_PER_FILE;

        let result = if matches.is_empty() {
            format!("No matches found for pattern '{}' in {}", pattern, path_str)
        } else {
            let formatted: Vec<String> = matches
                .iter()
                .map(|m| format!("{}:{}: {}", m.file, m.line_number, m.content))
                .collect();
            let mut output = formatted.join("\n");
            if truncated {
                output.push_str(&format!(
                    "\n\n(Results truncated to {} matches)",
                    MAX_MATCHES_PER_FILE
                ));
            }
            output
        };

        let mut tool_result = ToolResult::new(result);
        tool_result = tool_result.with_metadata(json!({
            "pattern": pattern,
            "path": path_str,
            "match_count": match_count,
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
        let grep = Grep;
        let result = grep
            .execute(json!({"pattern": "test", "path": "relative/path.txt"}))
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
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
        assert!(tool_result.content.contains("No matches found"));
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
        assert!(tool_result.content.contains("line 1: hello"));
        assert!(tool_result.content.contains("line 3: hello again"));
        assert!(!tool_result.content.contains("line 2: world"));

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
}
