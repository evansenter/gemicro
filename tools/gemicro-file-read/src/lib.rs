//! FileRead tool for reading file contents.

use async_trait::async_trait;
use gemicro_core::tool::{Tool, ToolError, ToolResult};
use serde_json::{json, Value};
use std::path::Path;

/// Maximum file size to read (1MB).
const MAX_FILE_SIZE: u64 = 1024 * 1024;

/// FileRead tool for reading file contents.
///
/// Reads the contents of a file at the specified path. Enforces a size limit
/// to prevent memory issues with very large files.
///
/// # Example
///
/// ```no_run
/// use gemicro_file_read::FileRead;
/// use gemicro_core::tool::Tool;
/// use serde_json::json;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let tool = FileRead;
/// let result = tool.execute(json!({"path": "/path/to/file.txt"})).await?;
/// println!("File contents: {}", result.content);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct FileRead;

#[async_trait]
impl Tool for FileRead {
    fn name(&self) -> &str {
        "file_read"
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Returns the file contents as text. \
         Has a 1MB size limit to prevent reading very large files."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to read"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let path_str = input
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'path' field".into()))?;

        let path = Path::new(path_str);

        // Check if path is absolute
        if !path.is_absolute() {
            return Err(ToolError::InvalidInput(format!(
                "Path must be absolute, got: {}",
                path_str
            )));
        }

        // Check if file exists
        if !path.exists() {
            return Err(ToolError::NotFound(format!("File not found: {}", path_str)));
        }

        // Check if it's a file (not a directory)
        if !path.is_file() {
            return Err(ToolError::InvalidInput(format!(
                "Path is not a file: {}",
                path_str
            )));
        }

        // Check file size before reading
        let metadata = tokio::fs::metadata(path).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to get file metadata: {}", e))
        })?;

        if metadata.len() > MAX_FILE_SIZE {
            return Err(ToolError::InvalidInput(format!(
                "File too large ({} bytes, max {} bytes)",
                metadata.len(),
                MAX_FILE_SIZE
            )));
        }

        // Read the file
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read file: {}", e)))?;

        Ok(ToolResult::new(content))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_file_read_success() {
        let mut temp = NamedTempFile::new().unwrap();
        writeln!(temp, "Hello, World!").unwrap();
        let path = temp.path().to_str().unwrap();

        let tool = FileRead;
        let result = tool.execute(json!({"path": path})).await.unwrap();
        assert!(result.content.contains("Hello, World!"));
    }

    #[tokio::test]
    async fn test_file_read_missing_path() {
        let tool = FileRead;
        let result = tool.execute(json!({})).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_file_read_relative_path() {
        let tool = FileRead;
        let result = tool.execute(json!({"path": "relative/path.txt"})).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_file_read_not_found() {
        let tool = FileRead;
        let result = tool.execute(json!({"path": "/nonexistent/file.txt"})).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_file_read_directory() {
        let tool = FileRead;
        let result = tool.execute(json!({"path": "/tmp"})).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[test]
    fn test_file_read_name_and_description() {
        let tool = FileRead;
        assert_eq!(tool.name(), "file_read");
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn test_file_read_parameters_schema() {
        let tool = FileRead;
        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .contains(&json!("path")));
    }
}
