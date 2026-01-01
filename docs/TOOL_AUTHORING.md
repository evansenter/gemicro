# Tool Authoring Guide

This guide walks you through implementing a new tool in Gemicro, following the [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) philosophy of extensible design.

## Overview

### What is a Tool?

A tool in Gemicro is an async function that agents can invoke to perform actions. Tools:
1. Have a unique name and description (for LLM decision-making)
2. Define a JSON Schema for input parameters
3. Execute logic and return structured results
4. Optionally require user confirmation for dangerous operations

### Design Philosophy

Following Evergreen principles:

| Principle | Description |
|-----------|-------------|
| **One tool per crate** | Each tool is an independent crate in `tools/` |
| **Minimal interface** | Tools implement a single `Tool` trait |
| **Async execution** | Tools are async for I/O-bound operations |
| **rust-genai integration** | Tools generate `FunctionDeclaration` for native function calling |

## Quick Start Checklist

- [ ] Create new tool crate: `tools/gemicro-{tool-name}/`
- [ ] Add crate to workspace `Cargo.toml` members
- [ ] Implement `Tool` trait (`name`, `description`, `parameters_schema`, `execute`)
- [ ] Add confirmation handling if the tool is dangerous
- [ ] Add unit tests for all code paths
- [ ] Add doc tests for public API

## Core Types

### Tool Trait

Location: `gemicro-core/src/tool/mod.rs`

```rust
#[async_trait]
pub trait Tool: Send + Sync + fmt::Debug {
    /// Unique identifier for this tool (e.g., "calculator", "file_read")
    fn name(&self) -> &str;

    /// Human-readable description shown to the LLM
    fn description(&self) -> &str;

    /// JSON Schema for input parameters
    fn parameters_schema(&self) -> Value;

    /// Execute the tool with the given input
    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError>;

    /// Whether this tool requires user confirmation (default: false)
    fn requires_confirmation(&self, _input: &Value) -> bool { false }

    /// Message to show in confirmation prompt
    fn confirmation_message(&self, input: &Value) -> String { ... }
}
```

### ToolResult

```rust
pub struct ToolResult {
    pub content: Value,    // Sent to LLM as tool response
    pub metadata: Value,   // For observability (NOT sent to LLM)
}

impl ToolResult {
    pub fn text(content: impl Into<String>) -> Self { ... }
    pub fn json(content: Value) -> Self { ... }
    pub fn with_metadata(self, metadata: Value) -> Self { ... }
}
```

### ToolError

```rust
pub enum ToolError {
    InvalidInput(String),         // Bad input from LLM
    ExecutionFailed(String),      // Tool logic failed
    NotFound(String),             // Resource not found
    Timeout(u64),                 // Operation timed out
    ConfirmationDenied(String),   // User declined
    InterceptorDenied(String),    // Interceptor blocked execution
    InterceptorFailed(String),    // Interceptor execution failed
    Other(String),                // Catch-all
}
```

### Error Signaling Convention

When a tool encounters an error but can still return a response to the LLM (as opposed to returning `Err(ToolError)`), use `metadata["error"]` to signal the failure:

```rust
// Recoverable error: return Ok with error metadata
Ok(ToolResult::text("Command failed with exit code 1")
    .with_metadata(json!({"error": "exit code 1"})))

// Fatal error: return Err
Err(ToolError::InvalidInput("Missing required 'path' field".into()))
```

**When to use each approach:**

| Scenario | Approach |
|----------|----------|
| Invalid input from LLM | `Err(ToolError::InvalidInput)` |
| Fatal execution failure | `Err(ToolError::ExecutionFailed)` |
| Non-zero exit code (LLM should see output) | `Ok(ToolResult)` + `metadata["error"]` |
| Partial success with warnings | `Ok(ToolResult)` + `metadata["error"]` |

**Why this matters:** Hooks like [`Metrics`](../hooks/gemicro-metrics/) detect failures by checking if `metadata["error"]` exists with a non-null value. Note that `{"error": null}` is treated as success, while `{"error": "message"}` signals failure.

## Complete Example: FileRead Tool

The `FileRead` tool is a reference implementation. See `tools/gemicro-file-read/src/lib.rs`.

### 1. Tool Struct

Tools are typically unit structs with `Default`:

```rust
// Location: tools/gemicro-file-read/src/lib.rs

/// Maximum file size to read (1MB).
const MAX_FILE_SIZE: u64 = 1024 * 1024;

/// FileRead tool for reading file contents.
#[derive(Debug, Clone, Copy, Default)]
pub struct FileRead;
```

### 2. Implement Tool Trait

```rust
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
        // 1. Extract and validate input
        let path_str = input
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'path' field".into()))?;

        let path = Path::new(path_str);

        // 2. Validate path is absolute
        if !path.is_absolute() {
            return Err(ToolError::InvalidInput(format!(
                "Path must be absolute, got: {}", path_str
            )));
        }

        // 3. Check file exists and get metadata (avoids TOCTOU)
        let metadata = tokio::fs::metadata(path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ToolError::NotFound(format!("File not found: {}", path_str))
            } else {
                ToolError::ExecutionFailed(format!("Failed to get metadata: {}", e))
            }
        })?;

        // 4. Validate it's a file and not too large
        if !metadata.is_file() {
            return Err(ToolError::InvalidInput(format!(
                "Path is not a file: {}", path_str
            )));
        }
        if metadata.len() > MAX_FILE_SIZE {
            return Err(ToolError::InvalidInput(format!(
                "File too large ({} bytes, max {} bytes)",
                metadata.len(), MAX_FILE_SIZE
            )));
        }

        // 5. Read and return content
        let content = tokio::fs::read_to_string(path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::InvalidData {
                ToolError::InvalidInput(format!(
                    "File is binary or contains invalid UTF-8: {}", path_str
                ))
            } else {
                ToolError::ExecutionFailed(format!("Failed to read file: {}", e))
            }
        })?;

        Ok(ToolResult::text(content))
    }
}
```

## Dangerous Tool Example: Bash

For tools that require user confirmation, implement `requires_confirmation()`:

```rust
// Location: tools/gemicro-bash/src/lib.rs

#[derive(Debug, Clone, Default)]
pub struct Bash {
    /// Commands that are always allowed without confirmation
    pub allowed_commands: Vec<String>,
}

#[async_trait]
impl Tool for Bash {
    fn name(&self) -> &str { "bash" }

    fn description(&self) -> &str {
        "Execute a bash command and return its output. \
         Requires confirmation for potentially dangerous commands."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                }
            },
            "required": ["command"]
        })
    }

    fn requires_confirmation(&self, input: &Value) -> bool {
        let command = input.get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Check if command is in allowed list
        !self.allowed_commands.iter().any(|allowed| {
            command.starts_with(allowed)
        })
    }

    fn confirmation_message(&self, input: &Value) -> String {
        let command = input.get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("<unknown>");

        format!("Execute bash command: {}", command)
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let command = input.get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'command'".into()))?;

        let output = tokio::process::Command::new("bash")
            .arg("-c")
            .arg(command)
            .output()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success() {
            Ok(ToolResult::text(stdout.to_string()))
        } else {
            Ok(ToolResult::text(format!(
                "Command failed with exit code {}\nstdout: {}\nstderr: {}",
                output.status.code().unwrap_or(-1), stdout, stderr
            )).with_metadata(json!({"error": true})))
        }
    }
}
```

## JSON Schema Best Practices

### Minimal Schema

```rust
fn parameters_schema(&self) -> Value {
    json!({
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "The input to process"
            }
        },
        "required": ["input"]
    })
}
```

### Schema with Optional Fields

```rust
fn parameters_schema(&self) -> Value {
    json!({
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch"
            },
            "timeout_ms": {
                "type": "integer",
                "description": "Timeout in milliseconds (default: 30000)"
            }
        },
        "required": ["url"]  // timeout_ms is optional
    })
}
```

### Schema with Enums

```rust
fn parameters_schema(&self) -> Value {
    json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "update", "delete"],
                "description": "The action to perform"
            }
        },
        "required": ["action"]
    })
}
```

## Registration and Usage

### Registering a Tool

```rust
use gemicro_core::tool::ToolRegistry;
use gemicro_file_read::FileRead;
use gemicro_bash::Bash;

let mut registry = ToolRegistry::new();
registry.register(FileRead);
registry.register(Bash::default());

// Access a tool
let tool = registry.get("file_read").unwrap();
```

### Filtering with ToolSet

```rust
use gemicro_core::tool::ToolSet;

// Use all tools
let all = ToolSet::All;

// Use only specific tools
let specific = ToolSet::Specific(vec!["file_read".into(), "calculator".into()]);

// Use all except dangerous ones
let safe = ToolSet::Except(vec!["bash".into()]);
```

### Integration with rust-genai

Tools automatically generate `FunctionDeclaration` for native function calling:

```rust
use gemicro_core::tool::{ToolRegistry, GemicroToolService, ToolSet};
use std::sync::Arc;

let registry = Arc::new(registry);
let service = GemicroToolService::new(registry)
    .with_filter(ToolSet::All);

// Use with rust-genai's interaction builder
client.interaction()
    .with_tool_service(Arc::new(service))
    .create_with_auto_functions()
    .await?;
```

## Testing Patterns

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[tokio::test]
    async fn test_file_read_success() {
        let mut temp = NamedTempFile::new().unwrap();
        writeln!(temp, "Hello, World!").unwrap();
        let path = temp.path().to_str().unwrap();

        let tool = FileRead;
        let result = tool.execute(json!({"path": path})).await.unwrap();
        assert!(result.content.as_str().unwrap().contains("Hello"));
    }

    #[tokio::test]
    async fn test_missing_required_field() {
        let tool = FileRead;
        let result = tool.execute(json!({})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_file_not_found() {
        let tool = FileRead;
        let result = tool.execute(json!({"path": "/nonexistent.txt"})).await;
        assert!(matches!(result, Err(ToolError::NotFound(_))));
    }

    #[test]
    fn test_name_and_description() {
        let tool = FileRead;
        assert_eq!(tool.name(), "file_read");
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn test_schema_structure() {
        let tool = FileRead;
        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&json!("path")));
    }
}
```

### Testing Confirmation Logic

```rust
#[test]
fn test_requires_confirmation() {
    let tool = Bash {
        allowed_commands: vec!["ls".into(), "pwd".into()],
    };

    // Allowed command - no confirmation
    assert!(!tool.requires_confirmation(&json!({"command": "ls -la"})));

    // Not allowed - requires confirmation
    assert!(tool.requires_confirmation(&json!({"command": "rm -rf /"})));
}
```

## Common Pitfalls

### 1. Blocking in async code

```rust
// ❌ DON'T - blocks the async runtime
let content = std::fs::read_to_string(path)?;

// ✅ DO - use async I/O
let content = tokio::fs::read_to_string(path).await?;
```

### 2. Missing error context

```rust
// ❌ DON'T - unhelpful error
return Err(ToolError::ExecutionFailed("Failed".into()));

// ✅ DO - include context
return Err(ToolError::ExecutionFailed(format!(
    "Failed to read {}: {}", path, e
)));
```

### 3. Not validating input types

```rust
// ❌ DON'T - panics on wrong type
let path = input["path"].as_str().unwrap();

// ✅ DO - return error on invalid input
let path = input.get("path")
    .and_then(|v| v.as_str())
    .ok_or_else(|| ToolError::InvalidInput("Missing 'path' field".into()))?;
```

### 4. TOCTOU vulnerabilities

```rust
// ❌ DON'T - race condition between check and use
if path.exists() {
    let content = tokio::fs::read_to_string(path).await?;
}

// ✅ DO - handle error from the actual operation
let content = tokio::fs::read_to_string(path).await.map_err(|e| {
    if e.kind() == std::io::ErrorKind::NotFound {
        ToolError::NotFound(format!("File not found: {}", path))
    } else {
        ToolError::ExecutionFailed(e.to_string())
    }
})?;
```

## File Structure for New Tools

Each tool gets its own crate in the `tools/` subdirectory:

```
tools/
└── gemicro-my-tool/
    ├── Cargo.toml           # Depends on gemicro-core only
    └── src/
        └── lib.rs           # Tool implementation
```

**Cargo.toml template:**

```toml
[package]
name = "gemicro-my-tool"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true

description = "Brief description of what the tool does"

[dependencies]
gemicro-core = { path = "../../gemicro-core" }
async-trait.workspace = true
serde_json.workspace = true
tokio = { workspace = true, features = ["fs", "process"] }  # as needed
log.workspace = true  # if using logging

[dev-dependencies]
tokio = { workspace = true, features = ["rt", "macros"] }
tempfile = "3"  # for file-based tests
```

**Add to workspace Cargo.toml:**

```toml
[workspace]
members = [
    # ... existing members
    "tools/gemicro-my-tool",
]
```

## See Also

- `tools/gemicro-file-read/src/lib.rs` - Simple read-only tool
- `tools/gemicro-bash/src/lib.rs` - Dangerous tool with confirmation
- `tools/gemicro-web-fetch/src/lib.rs` - Async I/O tool
- `gemicro-core/src/tool/mod.rs` - Core trait definitions
- `docs/HOOK_AUTHORING.md` - Creating hooks to intercept tools
- `CLAUDE.md` - Project design philosophy
