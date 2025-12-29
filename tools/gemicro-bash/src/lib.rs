//! Bash tool for executing shell commands.
//!
//! This tool allows agents to execute shell commands and capture their output.
//! **Requires confirmation before execution due to security implications.**

use async_trait::async_trait;
use gemicro_core::tool::{Tool, ToolError, ToolResult};
use serde_json::{json, Value};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

/// Default timeout for command execution (30 seconds).
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Maximum timeout allowed (5 minutes).
const MAX_TIMEOUT_SECS: u64 = 300;

/// Maximum output size to capture (1MB).
const MAX_OUTPUT_SIZE: usize = 1024 * 1024;

/// Bash tool for executing shell commands.
///
/// Executes a command in a shell and returns the output. Both stdout and stderr
/// are captured. The command runs with a configurable timeout.
///
/// **This tool requires confirmation before execution.**
///
/// # Security Note
///
/// This tool can execute arbitrary commands. It should only be used in trusted
/// environments with appropriate confirmation mechanisms in place.
///
/// # Example
///
/// ```no_run
/// use gemicro_bash::Bash;
/// use gemicro_core::tool::Tool;
/// use serde_json::json;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let bash = Bash;
///
/// let result = bash.execute(json!({
///     "command": "echo 'Hello, World!'"
/// })).await?;
/// println!("{}", result.content);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Bash;

#[async_trait]
impl Tool for Bash {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return the output. Captures both stdout \
         and stderr. Commands run with a timeout (default 30s, max 5m). \
         Requires confirmation before execution."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory (must be absolute path)"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30, max: 300)"
                }
            },
            "required": ["command"]
        })
    }

    fn requires_confirmation(&self, _input: &Value) -> bool {
        true
    }

    fn confirmation_message(&self, input: &Value) -> String {
        let command = input
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("<unknown>");
        let working_dir = input.get("working_dir").and_then(|v| v.as_str());

        let truncated_cmd = if command.len() > 100 {
            format!("{}...", &command[..97])
        } else {
            command.to_string()
        };

        match working_dir {
            Some(dir) => format!("Execute in {}: {}", dir, truncated_cmd),
            None => format!("Execute: {}", truncated_cmd),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let command = input
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'command' field".into()))?;

        if command.trim().is_empty() {
            return Err(ToolError::InvalidInput("Command cannot be empty".into()));
        }

        let working_dir = input.get("working_dir").and_then(|v| v.as_str());

        let timeout_secs = input
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .min(MAX_TIMEOUT_SECS);

        // Validate working directory if provided
        if let Some(dir) = working_dir {
            let path = std::path::Path::new(dir);
            if !path.is_absolute() {
                return Err(ToolError::InvalidInput(
                    "working_dir must be an absolute path".into(),
                ));
            }
            if !path.exists() {
                return Err(ToolError::NotFound(format!(
                    "Working directory does not exist: {}",
                    dir
                )));
            }
            if !path.is_dir() {
                return Err(ToolError::InvalidInput(format!(
                    "working_dir is not a directory: {}",
                    dir
                )));
            }
        }

        // Build command
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(command);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        if let Some(dir) = working_dir {
            cmd.current_dir(dir);
        }

        // Execute with timeout
        let timeout_duration = Duration::from_secs(timeout_secs);
        let output = timeout(timeout_duration, cmd.output())
            .await
            .map_err(|_| ToolError::Timeout(timeout_secs * 1000))?
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to execute command: {}", e)))?;

        // Capture stdout and stderr
        let mut stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let mut stderr = String::from_utf8_lossy(&output.stderr).to_string();

        // Truncate if too large
        let stdout_truncated = stdout.len() > MAX_OUTPUT_SIZE;
        let stderr_truncated = stderr.len() > MAX_OUTPUT_SIZE;

        if stdout_truncated {
            stdout.truncate(MAX_OUTPUT_SIZE);
            stdout.push_str("\n... (stdout truncated)");
        }
        if stderr_truncated {
            stderr.truncate(MAX_OUTPUT_SIZE);
            stderr.push_str("\n... (stderr truncated)");
        }

        let exit_code = output.status.code().unwrap_or(-1);

        // Build result content
        let mut content = String::new();

        if !stdout.is_empty() {
            content.push_str(&stdout);
        }

        if !stderr.is_empty() {
            if !content.is_empty() {
                content.push_str("\n\n");
            }
            content.push_str("[stderr]\n");
            content.push_str(&stderr);
        }

        if content.is_empty() {
            content = format!("Command completed with exit code {}", exit_code);
        }

        let success = output.status.success();

        Ok(ToolResult::new(content).with_metadata(json!({
            "exit_code": exit_code,
            "success": success,
            "stdout_len": output.stdout.len(),
            "stderr_len": output.stderr.len(),
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "timeout_secs": timeout_secs,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bash_name_and_description() {
        let bash = Bash;
        assert_eq!(bash.name(), "bash");
        assert!(!bash.description().is_empty());
    }

    #[test]
    fn test_bash_parameters_schema() {
        let bash = Bash;
        let schema = bash.parameters_schema();

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["command"].is_object());
        assert!(schema["properties"]["working_dir"].is_object());
        assert!(schema["properties"]["timeout_secs"].is_object());

        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("command")));
    }

    #[test]
    fn test_bash_requires_confirmation() {
        let bash = Bash;
        assert!(bash.requires_confirmation(&json!({})));
    }

    #[test]
    fn test_bash_confirmation_message() {
        let bash = Bash;
        let input = json!({
            "command": "echo hello"
        });
        let msg = bash.confirmation_message(&input);
        assert!(msg.contains("echo hello"));
    }

    #[test]
    fn test_bash_confirmation_message_with_dir() {
        let bash = Bash;
        let input = json!({
            "command": "ls",
            "working_dir": "/tmp"
        });
        let msg = bash.confirmation_message(&input);
        assert!(msg.contains("/tmp"));
        assert!(msg.contains("ls"));
    }

    #[test]
    fn test_bash_confirmation_message_truncates_long_command() {
        let bash = Bash;
        let long_command = "x".repeat(150);
        let input = json!({
            "command": long_command
        });
        let msg = bash.confirmation_message(&input);
        assert!(msg.len() < 150);
        assert!(msg.contains("..."));
    }

    #[tokio::test]
    async fn test_bash_missing_command() {
        let bash = Bash;
        let result = bash.execute(json!({})).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_bash_empty_command() {
        let bash = Bash;
        let result = bash.execute(json!({"command": "   "})).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_bash_simple_echo() {
        let bash = Bash;
        let result = bash.execute(json!({"command": "echo hello"})).await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result.content.contains("hello"));

        let metadata = tool_result.metadata;
        assert_eq!(metadata["exit_code"], 0);
        assert_eq!(metadata["success"], true);
    }

    #[tokio::test]
    async fn test_bash_captures_stderr() {
        let bash = Bash;
        let result = bash.execute(json!({"command": "echo error >&2"})).await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result.content.contains("[stderr]"));
        assert!(tool_result.content.contains("error"));
    }

    #[tokio::test]
    async fn test_bash_captures_exit_code() {
        let bash = Bash;
        let result = bash.execute(json!({"command": "exit 42"})).await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();

        let metadata = tool_result.metadata;
        assert_eq!(metadata["exit_code"], 42);
        assert_eq!(metadata["success"], false);
    }

    #[tokio::test]
    async fn test_bash_relative_working_dir() {
        let bash = Bash;
        let result = bash
            .execute(json!({
                "command": "pwd",
                "working_dir": "relative/path"
            }))
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_bash_nonexistent_working_dir() {
        let bash = Bash;
        let result = bash
            .execute(json!({
                "command": "pwd",
                "working_dir": "/nonexistent/directory/path"
            }))
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_bash_with_working_dir() {
        let bash = Bash;
        let result = bash
            .execute(json!({
                "command": "pwd",
                "working_dir": "/tmp"
            }))
            .await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        // The output should contain /tmp (or a symlink to it on macOS)
        assert!(
            tool_result.content.contains("tmp") || tool_result.content.contains("/private/tmp")
        );
    }

    #[tokio::test]
    async fn test_bash_timeout() {
        let bash = Bash;
        let result = bash
            .execute(json!({
                "command": "sleep 10",
                "timeout_secs": 1
            }))
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::Timeout(_)));
    }

    #[tokio::test]
    async fn test_bash_timeout_capped_at_max() {
        let bash = Bash;
        // Request a very large timeout - should be capped
        let result = bash
            .execute(json!({
                "command": "echo fast",
                "timeout_secs": 99999
            }))
            .await;

        assert!(result.is_ok());
        let metadata = result.unwrap().metadata;
        assert_eq!(metadata["timeout_secs"], MAX_TIMEOUT_SECS);
    }
}
