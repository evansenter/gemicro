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
//!         Ok(ToolResult::text(format!("Processed: {}", input_str)))
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
mod service;

pub use adapter::{tools_to_callables, ToolCallableAdapter};
pub use registry::ToolRegistry;
pub use service::GemicroToolService;

use async_trait::async_trait;
use rust_genai::{FunctionDeclaration, FunctionParameters};
use serde_json::Value;
use std::fmt;
use thiserror::Error;

// ============================================================================
// Confirmation Handler
// ============================================================================

/// Handler for tool confirmation prompts.
///
/// Tools that perform potentially dangerous operations (file writes, shell commands,
/// etc.) can require confirmation before execution. This trait defines how that
/// confirmation is obtained.
///
/// # Implementations
///
/// - [`AutoApprove`]: Always approves - for testing and trusted contexts
/// - [`AutoDeny`]: Always denies - safe default when no handler configured
///
/// # Example
///
/// ```
/// use gemicro_core::tool::{ConfirmationHandler, AutoApprove, AutoDeny};
/// use async_trait::async_trait;
/// use serde_json::Value;
///
/// // For testing or trusted automation
/// let _handler = AutoApprove;
///
/// // Or create a custom handler
/// #[derive(Debug)]
/// struct LoggingHandler;
///
/// #[async_trait]
/// impl ConfirmationHandler for LoggingHandler {
///     async fn confirm(&self, tool_name: &str, message: &str, _args: &Value) -> bool {
///         println!("[{}] {}", tool_name, message);
///         true // or implement actual confirmation logic
///     }
/// }
/// ```
#[async_trait]
pub trait ConfirmationHandler: Send + Sync + fmt::Debug {
    /// Called when a tool requires confirmation before execution.
    ///
    /// # Arguments
    ///
    /// * `tool_name` - The name of the tool requesting confirmation
    /// * `message` - A human-readable description of what the tool will do
    /// * `args` - The arguments that will be passed to the tool
    ///
    /// # Returns
    ///
    /// * `true` to approve execution
    /// * `false` to deny (tool will return [`ToolError::ConfirmationDenied`])
    async fn confirm(&self, tool_name: &str, message: &str, args: &Value) -> bool;
}

/// Auto-approve all confirmations.
///
/// Use in testing or trusted automation contexts where manual confirmation
/// is not needed or desired.
///
/// # Example
///
/// ```
/// use gemicro_core::tool::AutoApprove;
///
/// let handler = AutoApprove;
/// // All tool confirmations will be approved automatically
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct AutoApprove;

#[async_trait]
impl ConfirmationHandler for AutoApprove {
    async fn confirm(&self, _tool_name: &str, _message: &str, _args: &Value) -> bool {
        true
    }
}

/// Auto-deny all confirmations.
///
/// Safe default for when no confirmation handler is configured.
/// Prevents any tool requiring confirmation from executing.
///
/// # Example
///
/// ```
/// use gemicro_core::tool::AutoDeny;
///
/// let handler = AutoDeny;
/// // All tool confirmations will be denied automatically
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct AutoDeny;

#[async_trait]
impl ConfirmationHandler for AutoDeny {
    async fn confirm(&self, _tool_name: &str, _message: &str, _args: &Value) -> bool {
        false
    }
}

// ============================================================================
// Tool Result
// ============================================================================

/// Result returned by a tool execution.
///
/// The `content` field is sent to the LLM as the tool's response.
/// The `metadata` field is for observability/logging and is NOT sent to the LLM.
///
/// # Error Signaling Convention
///
/// When a tool encounters an error but can still return a response to the LLM
/// (as opposed to returning `Err(ToolError)`), use `metadata["error"]` to signal
/// the failure. This allows hooks like [`Metrics`] to track tool failures.
///
/// **Recommended pattern:**
///
/// ```
/// use gemicro_core::tool::ToolResult;
/// use serde_json::json;
///
/// // Tool failed but wants to communicate the failure to the LLM
/// let result = ToolResult::text("File not found: config.toml")
///     .with_metadata(json!({"error": "File not found: config.toml"}));
/// ```
///
/// The value of `metadata["error"]` can be any non-null JSON value (string, object, etc.)
/// that provides context about the failure. Hooks detect errors by checking if the key
/// exists with a non-null value:
///
/// ```text
/// output.metadata.get("error").map(|v| !v.is_null()).unwrap_or(false)
/// ```
///
/// **Note:** `{"error": null}` is treated as success (no error), while
/// `{"error": "message"}` signals failure.
///
/// # When to Use Error Metadata vs ToolError
///
/// - **Return `Err(ToolError)`**: For fatal errors that should stop execution
///   (e.g., invalid input that can't be processed).
/// - **Return `Ok(ToolResult)` with `metadata["error"]`**: For recoverable errors
///   where the LLM should receive feedback and may retry or adjust its approach.
///
/// [`Metrics`]: ../../../gemicro_metrics/index.html
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolResult {
    /// The main content/output from the tool (sent to LLM).
    pub content: Value,
    /// Optional structured metadata for observability/logging (NOT sent to LLM).
    pub metadata: Value,
}

impl ToolResult {
    /// Create a result with string content.
    ///
    /// Use this for simple text responses. The string is wrapped as `Value::String`.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::tool::ToolResult;
    ///
    /// let result = ToolResult::text("Hello, world!");
    /// ```
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: Value::String(content.into()),
            metadata: Value::Null,
        }
    }

    /// Create a result with structured JSON content.
    ///
    /// Use this when the tool output is structured data that the LLM should interpret.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::tool::ToolResult;
    /// use serde_json::json;
    ///
    /// let result = ToolResult::json(json!({
    ///     "result": 42,
    ///     "expression": "6 * 7"
    /// }));
    /// ```
    pub fn json(content: Value) -> Self {
        Self {
            content,
            metadata: Value::Null,
        }
    }

    /// Builder method to add metadata to an existing result.
    ///
    /// Metadata is for observability and logging, NOT sent to the LLM.
    ///
    /// # Examples
    ///
    /// ```
    /// use gemicro_core::tool::ToolResult;
    /// use serde_json::json;
    ///
    /// // Add timing metadata for observability
    /// let result = ToolResult::text("42")
    ///     .with_metadata(json!({"execution_time_ms": 5}));
    /// ```
    ///
    /// To signal an error (see [`ToolResult`] for details):
    ///
    /// ```
    /// use gemicro_core::tool::ToolResult;
    /// use serde_json::json;
    ///
    /// // Signal failure via metadata["error"] so hooks can track it
    /// let result = ToolResult::text("Permission denied: /etc/shadow")
    ///     .with_metadata(json!({"error": "Permission denied"}));
    /// ```
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
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

    /// User denied confirmation for the operation.
    #[error("Confirmation denied: {0}")]
    ConfirmationDenied(String),

    /// Interceptor denied tool execution.
    #[error("Interceptor denied: {0}")]
    InterceptorDenied(String),

    /// Interceptor execution failed.
    #[error("Interceptor failed: {0}")]
    InterceptorFailed(String),

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

    /// Whether this tool requires user confirmation before execution.
    ///
    /// Override to return `true` for tools that perform potentially dangerous
    /// operations like writing files, executing shell commands, or making
    /// external API calls with side effects.
    ///
    /// The input is provided to allow conditional confirmation based on
    /// the specific operation (e.g., bash tool might require confirmation
    /// for some commands but not others).
    ///
    /// Default implementation returns `false` (no confirmation required).
    fn requires_confirmation(&self, _input: &Value) -> bool {
        false
    }

    /// Description of the operation for confirmation prompts.
    ///
    /// When [`requires_confirmation`](Tool::requires_confirmation) returns `true`,
    /// this method should return a human-readable description of what the tool
    /// is about to do, suitable for showing to users in a confirmation dialog.
    ///
    /// Default implementation returns a generic message using the tool name.
    fn confirmation_message(&self, input: &Value) -> String {
        format!(
            "Tool '{}' wants to execute with input: {}",
            self.name(),
            serde_json::to_string_pretty(input).unwrap_or_else(|_| input.to_string())
        )
    }

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
///
/// // Inherit from parent (for subagents)
/// let inherit = ToolSet::Inherit;
/// let inherit_except = ToolSet::InheritExcept(vec!["bash".into()]);
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

    /// Inherit parent's tool set (for subagents).
    ///
    /// When resolved against a parent ToolSet, becomes identical to the parent.
    /// Use [`resolve()`] to get a concrete ToolSet.
    Inherit,

    /// Inherit parent's tools except the specified ones.
    ///
    /// When resolved against a parent ToolSet, filters out the specified tools.
    /// Use [`resolve()`] to get a concrete ToolSet.
    InheritExcept(Vec<String>),
}

impl ToolSet {
    /// Check if a tool name matches this filter.
    ///
    /// # Panics
    ///
    /// Panics if called on `Inherit` or `InheritExcept` - these must be
    /// resolved first using [`resolve()`].
    pub fn matches(&self, tool_name: &str) -> bool {
        match self {
            ToolSet::All => true,
            ToolSet::None => false,
            ToolSet::Specific(names) => names.iter().any(|n| n == tool_name),
            ToolSet::Except(names) => !names.iter().any(|n| n == tool_name),
            ToolSet::Inherit | ToolSet::InheritExcept(_) => {
                panic!("Cannot call matches() on {:?} - call resolve() first", self)
            }
        }
    }

    /// Check if this ToolSet requires resolution against a parent.
    pub fn needs_resolution(&self) -> bool {
        matches!(self, ToolSet::Inherit | ToolSet::InheritExcept(_))
    }

    /// Resolve an inheriting ToolSet against a parent.
    ///
    /// - `Inherit` → clones the parent
    /// - `InheritExcept(names)` → parent with additional exclusions
    /// - Other variants → returns self unchanged
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::ToolSet;
    ///
    /// let parent = ToolSet::Except(vec!["file_write".into()]);
    ///
    /// // Inherit becomes the parent
    /// let child = ToolSet::Inherit.resolve(&parent);
    /// assert_eq!(child, parent);
    ///
    /// // InheritExcept adds more exclusions
    /// let child = ToolSet::InheritExcept(vec!["bash".into()]).resolve(&parent);
    /// assert!(!child.matches("file_write")); // From parent
    /// assert!(!child.matches("bash"));       // From child
    /// assert!(child.matches("file_read"));   // Allowed
    /// ```
    pub fn resolve(&self, parent: &ToolSet) -> ToolSet {
        match self {
            ToolSet::Inherit => parent.clone(),
            ToolSet::InheritExcept(additional_exclusions) => {
                match parent {
                    // Parent allows all: just exclude our list
                    ToolSet::All => ToolSet::Except(additional_exclusions.clone()),

                    // Parent allows none: stay as none
                    ToolSet::None => ToolSet::None,

                    // Parent has specific list: remove our exclusions from it
                    ToolSet::Specific(parent_names) => {
                        let filtered: Vec<String> = parent_names
                            .iter()
                            .filter(|name| !additional_exclusions.contains(name))
                            .cloned()
                            .collect();
                        if filtered.is_empty() {
                            ToolSet::None
                        } else {
                            ToolSet::Specific(filtered)
                        }
                    }

                    // Parent has exclusions: combine exclusions
                    ToolSet::Except(parent_exclusions) => {
                        let mut combined = parent_exclusions.clone();
                        for name in additional_exclusions {
                            if !combined.contains(name) {
                                combined.push(name.clone());
                            }
                        }
                        ToolSet::Except(combined)
                    }

                    // Parent is also inheriting - resolve parent first
                    // This shouldn't happen in practice (infinite loop prevention)
                    ToolSet::Inherit | ToolSet::InheritExcept(_) => {
                        log::warn!(
                            "Parent ToolSet is {:?}, cannot resolve InheritExcept",
                            parent
                        );
                        ToolSet::None // Fail safe
                    }
                }
            }
            // Already resolved variants return themselves
            other => other.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_result_text() {
        let result = ToolResult::text("hello");
        assert_eq!(result.content, Value::String("hello".into()));
        assert_eq!(result.metadata, Value::Null);
    }

    #[test]
    fn test_tool_result_json() {
        let result = ToolResult::json(serde_json::json!({"answer": 42}));
        assert_eq!(result.content["answer"], 42);
        assert_eq!(result.metadata, Value::Null);
    }

    #[test]
    fn test_tool_result_with_metadata() {
        let result = ToolResult::text("hello").with_metadata(serde_json::json!({"key": "value"}));
        assert_eq!(result.content, Value::String("hello".into()));
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

    #[tokio::test]
    async fn test_auto_approve_confirms() {
        let handler = AutoApprove;
        let result = handler
            .confirm(
                "bash",
                "Execute: ls -la",
                &serde_json::json!({"command": "ls"}),
            )
            .await;
        assert!(result);
    }

    #[tokio::test]
    async fn test_auto_deny_denies() {
        let handler = AutoDeny;
        let result = handler
            .confirm(
                "bash",
                "Execute: rm -rf /",
                &serde_json::json!({"command": "rm"}),
            )
            .await;
        assert!(!result);
    }

    #[test]
    fn test_auto_approve_debug() {
        let handler = AutoApprove;
        assert!(format!("{:?}", handler).contains("AutoApprove"));
    }

    #[test]
    fn test_auto_deny_debug() {
        let handler = AutoDeny;
        assert!(format!("{:?}", handler).contains("AutoDeny"));
    }

    // ToolSet inheritance tests

    #[test]
    fn test_toolset_needs_resolution() {
        assert!(!ToolSet::All.needs_resolution());
        assert!(!ToolSet::None.needs_resolution());
        assert!(!ToolSet::Specific(vec!["a".into()]).needs_resolution());
        assert!(!ToolSet::Except(vec!["a".into()]).needs_resolution());
        assert!(ToolSet::Inherit.needs_resolution());
        assert!(ToolSet::InheritExcept(vec!["a".into()]).needs_resolution());
    }

    #[test]
    fn test_toolset_resolve_inherit_from_all() {
        let child = ToolSet::Inherit;
        let parent = ToolSet::All;
        let resolved = child.resolve(&parent);
        assert_eq!(resolved, ToolSet::All);
    }

    #[test]
    fn test_toolset_resolve_inherit_from_none() {
        let child = ToolSet::Inherit;
        let parent = ToolSet::None;
        let resolved = child.resolve(&parent);
        assert_eq!(resolved, ToolSet::None);
    }

    #[test]
    fn test_toolset_resolve_inherit_from_specific() {
        let child = ToolSet::Inherit;
        let parent = ToolSet::Specific(vec!["a".into(), "b".into()]);
        let resolved = child.resolve(&parent);
        assert_eq!(resolved, ToolSet::Specific(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn test_toolset_resolve_inherit_from_except() {
        let child = ToolSet::Inherit;
        let parent = ToolSet::Except(vec!["bash".into()]);
        let resolved = child.resolve(&parent);
        assert_eq!(resolved, ToolSet::Except(vec!["bash".into()]));
    }

    #[test]
    fn test_toolset_resolve_inherit_except_from_all() {
        let child = ToolSet::InheritExcept(vec!["bash".into()]);
        let parent = ToolSet::All;
        let resolved = child.resolve(&parent);
        assert_eq!(resolved, ToolSet::Except(vec!["bash".into()]));
    }

    #[test]
    fn test_toolset_resolve_inherit_except_from_none() {
        // Excluding from nothing = nothing
        let child = ToolSet::InheritExcept(vec!["bash".into()]);
        let parent = ToolSet::None;
        let resolved = child.resolve(&parent);
        assert_eq!(resolved, ToolSet::None);
    }

    #[test]
    fn test_toolset_resolve_inherit_except_from_specific() {
        // Parent: [a, b, c], Child excludes [b] -> [a, c]
        let child = ToolSet::InheritExcept(vec!["b".into()]);
        let parent = ToolSet::Specific(vec!["a".into(), "b".into(), "c".into()]);
        let resolved = child.resolve(&parent);
        if let ToolSet::Specific(tools) = resolved {
            assert_eq!(tools.len(), 2);
            assert!(tools.contains(&"a".to_string()));
            assert!(tools.contains(&"c".to_string()));
            assert!(!tools.contains(&"b".to_string()));
        } else {
            panic!("Expected ToolSet::Specific");
        }
    }

    #[test]
    fn test_toolset_resolve_inherit_except_from_except() {
        // Parent excludes [a], Child excludes [b] -> excludes [a, b]
        let child = ToolSet::InheritExcept(vec!["b".into()]);
        let parent = ToolSet::Except(vec!["a".into()]);
        let resolved = child.resolve(&parent);
        if let ToolSet::Except(tools) = resolved {
            assert_eq!(tools.len(), 2);
            assert!(tools.contains(&"a".to_string()));
            assert!(tools.contains(&"b".to_string()));
        } else {
            panic!("Expected ToolSet::Except");
        }
    }

    #[test]
    fn test_toolset_resolve_already_resolved() {
        // Non-inherit variants return themselves
        let parent = ToolSet::All;
        assert_eq!(ToolSet::All.resolve(&parent), ToolSet::All);
        assert_eq!(ToolSet::None.resolve(&parent), ToolSet::None);
        assert_eq!(
            ToolSet::Specific(vec!["a".into()]).resolve(&parent),
            ToolSet::Specific(vec!["a".into()])
        );
        assert_eq!(
            ToolSet::Except(vec!["a".into()]).resolve(&parent),
            ToolSet::Except(vec!["a".into()])
        );
    }

    #[test]
    #[should_panic(expected = "call resolve() first")]
    fn test_toolset_matches_panics_on_inherit() {
        let set = ToolSet::Inherit;
        let _ = set.matches("anything");
    }

    #[test]
    #[should_panic(expected = "call resolve() first")]
    fn test_toolset_matches_panics_on_inherit_except() {
        let set = ToolSet::InheritExcept(vec!["bash".into()]);
        let _ = set.matches("anything");
    }

    #[test]
    fn test_toolset_resolved_inherit_can_match() {
        let child = ToolSet::Inherit;
        let parent = ToolSet::Specific(vec!["a".into(), "b".into()]);
        let resolved = child.resolve(&parent);
        assert!(resolved.matches("a"));
        assert!(resolved.matches("b"));
        assert!(!resolved.matches("c"));
    }
}
