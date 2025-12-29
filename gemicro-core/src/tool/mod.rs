//! Tool abstraction for agent actions.
//!
//! This module provides the core [`Tool`] trait and [`ToolRegistry`] for
//! managing tools that agents can invoke. Following the
//! [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) philosophy,
//! tools are extensible without modifying gemicro-core.
//!
//! # Design
//!
//! - **Async execution**: Tools use async for I/O-bound operations (web fetch, file read)
//! - **Instance-based registry**: Tools are stateless, stored as instances not factories
//! - **Per-agent filtering**: [`ToolSet`] allows agents to specify which tools they need
//! - **rust-genai integration**: Tools can generate [`FunctionDeclaration`] for native function calling
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::tool::{Tool, ToolResult, ToolError, ToolRegistry, ToolSet};
//! use async_trait::async_trait;
//! use serde_json::{json, Value};
//!
//! #[derive(Debug)]
//! struct MyTool;
//!
//! #[async_trait]
//! impl Tool for MyTool {
//!     fn name(&self) -> &str { "my_tool" }
//!     fn description(&self) -> &str { "Does something useful" }
//!     fn parameters_schema(&self) -> Value {
//!         json!({
//!             "type": "object",
//!             "properties": {
//!                 "input": { "type": "string" }
//!             },
//!             "required": ["input"]
//!         })
//!     }
//!     async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
//!         let input_str = input["input"].as_str().unwrap_or("");
//!         Ok(ToolResult::new(format!("Processed: {}", input_str)))
//!     }
//! }
//!
//! // Register and use
//! let mut registry = ToolRegistry::new();
//! registry.register(MyTool);
//! let tool = registry.get("my_tool").unwrap();
//! ```

mod adapter;
mod registry;

pub use adapter::{tools_to_callables, ToolCallableAdapter};
pub use registry::ToolRegistry;

use async_trait::async_trait;
use rust_genai::{FunctionDeclaration, FunctionParameters};
use serde_json::Value;
use std::fmt;
use thiserror::Error;

/// Result returned by a tool execution.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolResult {
    /// The main content/output from the tool.
    pub content: String,
    /// Optional structured metadata for observability/logging.
    pub metadata: Value,
}

impl ToolResult {
    /// Create a result with just content.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            metadata: Value::Null,
        }
    }

    /// Create a result with content and metadata.
    pub fn with_metadata(content: impl Into<String>, metadata: Value) -> Self {
        Self {
            content: content.into(),
            metadata,
        }
    }
}

/// Errors that can occur during tool execution.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ToolError {
    /// Invalid input provided to the tool.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Tool execution failed.
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    /// Tool not found in registry.
    #[error("Tool not found: {0}")]
    NotFound(String),

    /// Tool execution timed out.
    #[error("Timeout after {0}ms")]
    Timeout(u64),

    /// Other error.
    #[error("{0}")]
    Other(String),
}

/// A tool that agents can invoke to perform actions.
///
/// Tools are the primary mechanism for agents to interact with the external world.
/// Each tool has a unique name, description, and parameter schema that the LLM
/// uses to decide when and how to invoke it.
///
/// # Async Execution
///
/// The [`execute`](Tool::execute) method is async to support I/O-bound tools like
/// web fetching or file operations. For synchronous tools (like calculator),
/// simply don't use `.await` in the implementation.
///
/// # rust-genai Integration
///
/// Tools can generate a [`FunctionDeclaration`] via [`to_function_declaration`](Tool::to_function_declaration)
/// for use with rust-genai's native function calling. The default implementation
/// builds this from the trait methods.
#[async_trait]
pub trait Tool: Send + Sync + fmt::Debug {
    /// Unique identifier for this tool (e.g., "calculator", "web_fetch").
    fn name(&self) -> &str;

    /// Human-readable description of what this tool does.
    /// This is shown to the LLM to help it decide when to use the tool.
    fn description(&self) -> &str;

    /// JSON Schema for the tool's input parameters.
    ///
    /// Should follow the JSON Schema specification. Example:
    /// ```json
    /// {
    ///   "type": "object",
    ///   "properties": {
    ///     "expression": {
    ///       "type": "string",
    ///       "description": "A mathematical expression"
    ///     }
    ///   },
    ///   "required": ["expression"]
    /// }
    /// ```
    fn parameters_schema(&self) -> Value;

    /// Execute the tool with the given input.
    ///
    /// The input is a JSON value matching the schema from [`parameters_schema`](Tool::parameters_schema).
    /// Returns a [`ToolResult`] on success or a [`ToolError`] on failure.
    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError>;

    /// Generate a rust-genai FunctionDeclaration for this tool.
    ///
    /// The default implementation builds from [`name`](Tool::name),
    /// [`description`](Tool::description), and [`parameters_schema`](Tool::parameters_schema).
    /// Override if you need custom declaration behavior.
    fn to_function_declaration(&self) -> FunctionDeclaration {
        let schema = self.parameters_schema();

        // Extract properties from schema (default to empty object)
        let properties = schema
            .get("properties")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));

        // Extract required fields from schema (default to empty)
        let required: Vec<String> = schema
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let params = FunctionParameters::new("object".to_string(), properties, required);

        FunctionDeclaration::new(
            self.name().to_string(),
            self.description().to_string(),
            params,
        )
    }
}

/// Per-agent filtering specification for tools.
///
/// Agents can specify which tools they want access to via their configuration.
/// This allows restricting agents to specific tool subsets for safety or focus.
///
/// # Example
///
/// ```
/// use gemicro_core::tool::ToolSet;
///
/// // Use all available tools
/// let all = ToolSet::All;
///
/// // Use no tools
/// let none = ToolSet::None;
///
/// // Use only specific tools
/// let specific = ToolSet::Specific(vec!["calculator".into(), "web_fetch".into()]);
///
/// // Use all except these
/// let except = ToolSet::Except(vec!["bash".into()]);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ToolSet {
    /// Use all tools from registry.
    #[default]
    All,

    /// Use no tools.
    None,

    /// Use only the specified tools by name.
    Specific(Vec<String>),

    /// Use all tools except the specified ones.
    Except(Vec<String>),
}

impl ToolSet {
    /// Check if a tool name matches this filter.
    pub fn matches(&self, tool_name: &str) -> bool {
        match self {
            ToolSet::All => true,
            ToolSet::None => false,
            ToolSet::Specific(names) => names.iter().any(|n| n == tool_name),
            ToolSet::Except(names) => !names.iter().any(|n| n == tool_name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_result_new() {
        let result = ToolResult::new("hello");
        assert_eq!(result.content, "hello");
        assert_eq!(result.metadata, Value::Null);
    }

    #[test]
    fn test_tool_result_with_metadata() {
        let result = ToolResult::with_metadata("hello", serde_json::json!({"key": "value"}));
        assert_eq!(result.content, "hello");
        assert_eq!(result.metadata["key"], "value");
    }

    #[test]
    fn test_tool_error_display() {
        assert_eq!(
            ToolError::InvalidInput("bad".into()).to_string(),
            "Invalid input: bad"
        );
        assert_eq!(
            ToolError::ExecutionFailed("oops".into()).to_string(),
            "Execution failed: oops"
        );
        assert_eq!(
            ToolError::NotFound("foo".into()).to_string(),
            "Tool not found: foo"
        );
        assert_eq!(ToolError::Timeout(1000).to_string(), "Timeout after 1000ms");
    }

    #[test]
    fn test_toolset_all() {
        let set = ToolSet::All;
        assert!(set.matches("anything"));
        assert!(set.matches("calculator"));
    }

    #[test]
    fn test_toolset_none() {
        let set = ToolSet::None;
        assert!(!set.matches("anything"));
        assert!(!set.matches("calculator"));
    }

    #[test]
    fn test_toolset_specific() {
        let set = ToolSet::Specific(vec!["calculator".into(), "web_fetch".into()]);
        assert!(set.matches("calculator"));
        assert!(set.matches("web_fetch"));
        assert!(!set.matches("bash"));
    }

    #[test]
    fn test_toolset_except() {
        let set = ToolSet::Except(vec!["bash".into()]);
        assert!(set.matches("calculator"));
        assert!(set.matches("web_fetch"));
        assert!(!set.matches("bash"));
    }

    #[test]
    fn test_toolset_default() {
        let set = ToolSet::default();
        assert_eq!(set, ToolSet::All);
    }
}
