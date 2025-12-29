//! FileWrite tool for writing file contents.
//!
//! This tool allows agents to write content to files, creating new files
//! or overwriting existing ones. **Requires confirmation before execution.**

use async_trait::async_trait;
use gemicro_core::tool::{Tool, ToolError, ToolResult};
use serde_json::{json, Value};
use std::path::Path;
use tokio::fs;

/// Maximum file size that can be written (5MB).
const MAX_WRITE_SIZE: usize = 5 * 1024 * 1024;

/// FileWrite tool for writing content to files.
///
/// Writes content to a file at the specified path. Will create the file
/// if it doesn't exist, or overwrite if it does. Parent directories must exist.
///
/// **This tool requires confirmation before execution.**
///
/// # Example
///
/// ```no_run
/// use gemicro_file_write::FileWrite;
/// use gemicro_core::tool::Tool;
/// use serde_json::json;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let write_tool = FileWrite;
///
/// let result = write_tool.execute(json!({
///     "path": "/tmp/output.txt",
///     "content": "Hello, world!"
/// })).await?;
/// println!("{}", result.content);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct FileWrite;

#[async_trait]
impl Tool for FileWrite {
    fn name(&self) -> &str {
        "file_write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates the file if it doesn't exist, \
         or overwrites if it does. Parent directories must already exist. \
         Requires confirmation before execution."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    fn requires_confirmation(&self, _input: &Value) -> bool {
        true
    }

    fn confirmation_message(&self, input: &Value) -> String {
        let path = input
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("<unknown>");
        let content_len = input
            .get("content")
            .and_then(|v| v.as_str())
            .map(|s| s.len())
            .unwrap_or(0);

        format!("Write {} bytes to file: {}", content_len, path)
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let path_str = input
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'path' field".into()))?;

        let content = input
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'content' field".into()))?;

        let path = Path::new(path_str);

        // Validate path is absolute
        if !path.is_absolute() {
            return Err(ToolError::InvalidInput("Path must be absolute".into()));
        }

        // Check content size
        if content.len() > MAX_WRITE_SIZE {
            return Err(ToolError::InvalidInput(format!(
                "Content too large ({} bytes, max {} bytes)",
                content.len(),
                MAX_WRITE_SIZE
            )));
        }

        // Check parent directory exists
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                return Err(ToolError::NotFound(format!(
                    "Parent directory does not exist: {}",
                    parent.display()
                )));
            }
        }

        // Check if file already exists
        let existed = path.exists();

        // Write the file
        fs::write(path, content)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to write file: {}", e)))?;

        let action = if existed { "Overwrote" } else { "Created" };
        let result = format!("{} file: {} ({} bytes)", action, path_str, content.len());

        Ok(ToolResult::new(result).with_metadata(json!({
            "path": path_str,
            "bytes_written": content.len(),
            "existed": existed,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_file_write_name_and_description() {
        let tool = FileWrite;
        assert_eq!(tool.name(), "file_write");
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn test_file_write_parameters_schema() {
        let tool = FileWrite;
        let schema = tool.parameters_schema();

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["properties"]["content"].is_object());

        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("path")));
        assert!(required.contains(&json!("content")));
    }

    #[test]
    fn test_file_write_requires_confirmation() {
        let tool = FileWrite;
        assert!(tool.requires_confirmation(&json!({})));
    }

    #[test]
    fn test_file_write_confirmation_message() {
        let tool = FileWrite;
        let input = json!({
            "path": "/tmp/test.txt",
            "content": "hello world"
        });
        let msg = tool.confirmation_message(&input);
        assert!(msg.contains("11 bytes"));
        assert!(msg.contains("/tmp/test.txt"));
    }

    #[tokio::test]
    async fn test_file_write_missing_path() {
        let tool = FileWrite;
        let result = tool.execute(json!({"content": "test"})).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_file_write_missing_content() {
        let tool = FileWrite;
        let result = tool.execute(json!({"path": "/tmp/test.txt"})).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_file_write_relative_path() {
        let tool = FileWrite;
        let result = tool
            .execute(json!({
                "path": "relative/path.txt",
                "content": "test"
            }))
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_file_write_parent_not_exists() {
        let tool = FileWrite;
        let result = tool
            .execute(json!({
                "path": "/nonexistent/parent/dir/file.txt",
                "content": "test"
            }))
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_file_write_creates_new_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("new_file.txt");

        let tool = FileWrite;
        let result = tool
            .execute(json!({
                "path": file_path.to_str().unwrap(),
                "content": "hello world"
            }))
            .await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result.content.contains("Created"));

        // Verify file contents
        let contents = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(contents, "hello world");

        // Check metadata
        let metadata = tool_result.metadata;
        assert_eq!(metadata["bytes_written"], 11);
        assert_eq!(metadata["existed"], false);
    }

    #[tokio::test]
    async fn test_file_write_overwrites_existing() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("existing.txt");

        // Create existing file
        std::fs::write(&file_path, "old content").unwrap();

        let tool = FileWrite;
        let result = tool
            .execute(json!({
                "path": file_path.to_str().unwrap(),
                "content": "new content"
            }))
            .await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result.content.contains("Overwrote"));

        // Verify file contents changed
        let contents = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(contents, "new content");

        // Check metadata
        let metadata = tool_result.metadata;
        assert_eq!(metadata["existed"], true);
    }

    #[tokio::test]
    async fn test_file_write_content_too_large() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("large.txt");

        let large_content = "x".repeat(MAX_WRITE_SIZE + 1);

        let tool = FileWrite;
        let result = tool
            .execute(json!({
                "path": file_path.to_str().unwrap(),
                "content": large_content
            }))
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(err.to_string().contains("too large"));
    }
}
