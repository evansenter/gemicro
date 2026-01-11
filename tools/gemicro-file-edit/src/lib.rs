//! FileEdit tool for editing existing files.
//!
//! This tool allows agents to edit files by replacing specific text strings.
//! Unlike file_write, this modifies existing files rather than overwriting.
//! **Requires confirmation before execution.**

use async_trait::async_trait;
use gemicro_core::tool::{Tool, ToolError, ToolResult};
use gemicro_core::truncate;
use serde_json::{json, Value};
use std::path::Path;
use tokio::fs;

/// Maximum file size that can be edited (10MB).
const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024;

/// FileEdit tool for editing existing files.
///
/// Performs string replacement within a file. Requires the `old_string` to be
/// unique within the file (unless `replace_all` is set). This ensures precise,
/// predictable edits.
///
/// **This tool requires confirmation before execution.**
///
/// # Result Metadata
///
/// On success, the result includes metadata with these fields:
/// - `path`: The file path edited
/// - `occurrences_replaced`: Number of replacements made
/// - `old_string_len`: Length of the replaced string
/// - `new_string_len`: Length of the replacement string
///
/// # Example
///
/// ```no_run
/// use gemicro_file_edit::FileEdit;
/// use gemicro_core::tool::Tool;
/// use serde_json::json;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let edit_tool = FileEdit;
///
/// let result = edit_tool.execute(json!({
///     "path": "/path/to/file.rs",
///     "old_string": "fn old_name()",
///     "new_string": "fn new_name()"
/// })).await?;
/// println!("{}", result.content);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct FileEdit;

#[async_trait]
impl Tool for FileEdit {
    fn name(&self) -> &str {
        "file_edit"
    }

    fn description(&self) -> &str {
        "Edit an existing file by replacing text. Finds 'old_string' and replaces \
         it with 'new_string'. By default, old_string must be unique in the file. \
         Use 'replace_all: true' to replace all occurrences. Requires confirmation."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "Text to find and replace (must be unique unless replace_all is true)"
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false, requires unique match)"
                }
            },
            "required": ["path", "old_string", "new_string"]
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
        let old_string = input
            .get("old_string")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let new_string = input
            .get("new_string")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let replace_all = input
            .get("replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let old_preview = truncate(old_string, 40);
        let new_preview = truncate(new_string, 40);

        if replace_all {
            format!(
                "Replace all '{}' with '{}' in: {}",
                old_preview, new_preview, path
            )
        } else {
            format!(
                "Replace '{}' with '{}' in: {}",
                old_preview, new_preview, path
            )
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let path_str = input
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'path' field".into()))?;

        let old_string = input
            .get("old_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'old_string' field".into()))?;

        let new_string = input
            .get("new_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'new_string' field".into()))?;

        let replace_all = input
            .get("replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

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

        // Check old_string is not empty
        if old_string.is_empty() {
            return Err(ToolError::InvalidInput("old_string cannot be empty".into()));
        }

        // Check old != new
        if old_string == new_string {
            return Err(ToolError::InvalidInput(
                "old_string and new_string must be different".into(),
            ));
        }

        // Check file exists and get metadata
        let metadata = fs::metadata(&path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ToolError::NotFound(format!("File not found: {}", path_str))
            } else {
                ToolError::ExecutionFailed(format!("Cannot access file: {}", e))
            }
        })?;

        if !metadata.is_file() {
            return Err(ToolError::InvalidInput(format!(
                "Path is not a file: {}",
                path_str
            )));
        }

        if metadata.len() > MAX_FILE_SIZE {
            return Err(ToolError::InvalidInput(format!(
                "File too large ({} bytes, max {} bytes)",
                metadata.len(),
                MAX_FILE_SIZE
            )));
        }

        // Read file content
        let content = fs::read_to_string(&path)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read file: {}", e)))?;

        // Count occurrences
        let occurrences = content.matches(old_string).count();

        if occurrences == 0 {
            return Err(ToolError::InvalidInput(format!(
                "old_string '{}' not found in file",
                truncate(old_string, 50)
            )));
        }

        if !replace_all && occurrences > 1 {
            return Err(ToolError::InvalidInput(format!(
                "old_string '{}' found {} times - must be unique (or use replace_all: true)",
                truncate(old_string, 50),
                occurrences
            )));
        }

        // Perform replacement
        let new_content = if replace_all {
            content.replace(old_string, new_string)
        } else {
            content.replacen(old_string, new_string, 1)
        };

        // Write back
        fs::write(&path, &new_content)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to write file: {}", e)))?;

        let result = format!("Replaced {} occurrence(s) in: {}", occurrences, path_str);

        Ok(ToolResult::text(result).with_metadata(json!({
            "path": path_str,
            "occurrences_replaced": occurrences,
            "old_string_len": old_string.len(),
            "new_string_len": new_string.len(),
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_file_edit_name_and_description() {
        let tool = FileEdit;
        assert_eq!(tool.name(), "file_edit");
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn test_file_edit_parameters_schema() {
        let tool = FileEdit;
        let schema = tool.parameters_schema();

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["properties"]["old_string"].is_object());
        assert!(schema["properties"]["new_string"].is_object());
        assert!(schema["properties"]["replace_all"].is_object());

        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("path")));
        assert!(required.contains(&json!("old_string")));
        assert!(required.contains(&json!("new_string")));
    }

    #[test]
    fn test_file_edit_requires_confirmation() {
        let tool = FileEdit;
        assert!(tool.requires_confirmation(&json!({})));
    }

    #[test]
    fn test_file_edit_confirmation_message() {
        let tool = FileEdit;
        let input = json!({
            "path": "/tmp/test.txt",
            "old_string": "old text",
            "new_string": "new text"
        });
        let msg = tool.confirmation_message(&input);
        assert!(msg.contains("old text"));
        assert!(msg.contains("new text"));
        assert!(msg.contains("/tmp/test.txt"));
    }

    #[tokio::test]
    async fn test_file_edit_missing_path() {
        let tool = FileEdit;
        let result = tool
            .execute(json!({
                "old_string": "old",
                "new_string": "new"
            }))
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_file_edit_missing_old_string() {
        let tool = FileEdit;
        let result = tool
            .execute(json!({
                "path": "/tmp/test.txt",
                "new_string": "new"
            }))
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_file_edit_empty_old_string() {
        let tool = FileEdit;
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "test content").unwrap();

        let result = tool
            .execute(json!({
                "path": file.path().to_str().unwrap(),
                "old_string": "",
                "new_string": "new"
            }))
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(err.to_string().contains("cannot be empty"));
    }

    #[tokio::test]
    async fn test_file_edit_same_old_new() {
        let tool = FileEdit;
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "test content").unwrap();

        let result = tool
            .execute(json!({
                "path": file.path().to_str().unwrap(),
                "old_string": "same",
                "new_string": "same"
            }))
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(err.to_string().contains("must be different"));
    }

    #[tokio::test]
    async fn test_file_edit_relative_path() {
        // Relative paths are now resolved against CWD, so this should fail with NotFound
        let tool = FileEdit;
        let result = tool
            .execute(json!({
                "path": "relative/path.txt",
                "old_string": "old",
                "new_string": "new"
            }))
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_file_edit_file_not_found() {
        let tool = FileEdit;
        let result = tool
            .execute(json!({
                "path": "/nonexistent/file/path.txt",
                "old_string": "old",
                "new_string": "new"
            }))
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_file_edit_string_not_found() {
        let tool = FileEdit;
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "hello world").unwrap();

        let result = tool
            .execute(json!({
                "path": file.path().to_str().unwrap(),
                "old_string": "nonexistent",
                "new_string": "replacement"
            }))
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(err.to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_file_edit_multiple_occurrences_error() {
        let tool = FileEdit;
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "hello world hello").unwrap();

        let result = tool
            .execute(json!({
                "path": file.path().to_str().unwrap(),
                "old_string": "hello",
                "new_string": "hi"
            }))
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(err.to_string().contains("2 times"));
    }

    #[tokio::test]
    async fn test_file_edit_single_replacement() {
        let tool = FileEdit;
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "hello world").unwrap();

        let result = tool
            .execute(json!({
                "path": file.path().to_str().unwrap(),
                "old_string": "hello",
                "new_string": "hi"
            }))
            .await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result
            .content
            .as_str()
            .unwrap()
            .contains("1 occurrence"));

        // Verify content
        let content = std::fs::read_to_string(file.path()).unwrap();
        assert_eq!(content, "hi world");

        // Check metadata
        let metadata = tool_result.metadata;
        assert_eq!(metadata["occurrences_replaced"], 1);
    }

    #[tokio::test]
    async fn test_file_edit_replace_all() {
        let tool = FileEdit;
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "hello world hello").unwrap();

        let result = tool
            .execute(json!({
                "path": file.path().to_str().unwrap(),
                "old_string": "hello",
                "new_string": "hi",
                "replace_all": true
            }))
            .await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result
            .content
            .as_str()
            .unwrap()
            .contains("2 occurrence"));

        // Verify content
        let content = std::fs::read_to_string(file.path()).unwrap();
        assert_eq!(content, "hi world hi");

        // Check metadata
        let metadata = tool_result.metadata;
        assert_eq!(metadata["occurrences_replaced"], 2);
    }
}
