//! Adapter for integrating Tool trait with rust-genai.
//!
//! This module provides [`ToolCallableAdapter`] which bridges the gap between
//! our async [`Tool`] trait and rust-genai's async `CallableFunction` trait.

use super::{ConfirmationHandler, Tool, ToolError};
use async_trait::async_trait;
use rust_genai::{CallableFunction, FunctionDeclaration, FunctionError};
use serde_json::Value;
use std::sync::Arc;

/// Adapter that wraps a [`Tool`] to implement rust-genai's `CallableFunction`.
///
/// This enables using our Tool implementations with rust-genai's automatic
/// function calling via `create_with_auto_functions()`.
///
/// Both `Tool::execute` and `CallableFunction::call` are async, so this adapter
/// simply delegates execution without any sync/async bridging.
///
/// # Confirmation Support
///
/// Tools can require user confirmation before execution. Use
/// [`with_confirmation_handler`](Self::with_confirmation_handler) to provide a handler.
/// If no handler is set, tools requiring confirmation will be denied by default for safety.
///
/// # Example
///
/// ```no_run
/// use gemicro_core::tool::{ToolCallableAdapter, ToolRegistry, AutoApprove};
/// use rust_genai::CallableFunction;
/// use std::sync::Arc;
///
/// // Assuming you have a registry with tools
/// let registry = ToolRegistry::new();
/// let tool = registry.get("calculator").unwrap();
///
/// // Create adapter for rust-genai integration
/// let adapter = ToolCallableAdapter::new(tool);
///
/// // With confirmation handler for dangerous tools
/// let handler = Arc::new(AutoApprove);
/// let adapter = adapter.with_confirmation_handler(handler);
///
/// // Use with rust-genai's function calling
/// let declaration = adapter.declaration();
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolCallableAdapter {
    tool: Arc<dyn Tool>,
    confirmation_handler: Option<Arc<dyn ConfirmationHandler>>,
}

impl ToolCallableAdapter {
    /// Create a new adapter wrapping the given tool.
    ///
    /// No confirmation handler is set by default. Tools requiring confirmation
    /// will be denied unless [`with_confirmation_handler`](Self::with_confirmation_handler)
    /// is called.
    pub fn new(tool: Arc<dyn Tool>) -> Self {
        Self {
            tool,
            confirmation_handler: None,
        }
    }

    /// Set a confirmation handler for this adapter.
    ///
    /// The handler is called when a tool's [`requires_confirmation`](super::Tool::requires_confirmation)
    /// returns `true`. If the handler returns `false`, the tool execution is denied.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gemicro_core::tool::{ToolCallableAdapter, AutoApprove};
    /// use std::sync::Arc;
    ///
    /// # fn example(tool: Arc<dyn gemicro_core::tool::Tool>) {
    /// let adapter = ToolCallableAdapter::new(tool)
    ///     .with_confirmation_handler(Arc::new(AutoApprove));
    /// # }
    /// ```
    pub fn with_confirmation_handler(mut self, handler: Arc<dyn ConfirmationHandler>) -> Self {
        self.confirmation_handler = Some(handler);
        self
    }

    /// Get the underlying tool.
    pub fn tool(&self) -> &Arc<dyn Tool> {
        &self.tool
    }

    /// Execute the tool asynchronously.
    ///
    /// This is a convenience method for when you want to execute the tool
    /// directly without going through the sync CallableFunction interface.
    ///
    /// Note: This method bypasses the confirmation check. For full confirmation
    /// support, use the [`CallableFunction::call`] method instead.
    pub async fn execute(
        &self,
        input: serde_json::Value,
    ) -> Result<super::ToolResult, super::ToolError> {
        self.tool.execute(input).await
    }
}

#[async_trait]
impl CallableFunction for ToolCallableAdapter {
    fn declaration(&self) -> FunctionDeclaration {
        self.tool.to_function_declaration()
    }

    async fn call(&self, args: Value) -> Result<Value, FunctionError> {
        // Check confirmation before execution for tools that require it
        if self.tool.requires_confirmation(&args) {
            let message = self.tool.confirmation_message(&args);

            match &self.confirmation_handler {
                Some(handler) => {
                    if !handler.confirm(self.tool.name(), &message, &args).await {
                        return Err(FunctionError::ExecutionError(Box::new(
                            ToolError::ConfirmationDenied(message),
                        )));
                    }
                }
                None => {
                    // No handler = deny by default for safety
                    return Err(FunctionError::ExecutionError(Box::new(
                        ToolError::ConfirmationDenied(format!(
                            "No confirmation handler configured for '{}': {}",
                            self.tool.name(),
                            message
                        )),
                    )));
                }
            }
        }

        // Execute the tool
        match self.tool.execute(args).await {
            Ok(result) => {
                // Return content directly - it's already a Value.
                // Note: ToolResult::metadata is intentionally not returned here because
                // rust-genai's CallableFunction expects a simple Value result that gets
                // passed back to the LLM. Metadata is for observability/logging within
                // gemicro, not for LLM consumption.
                Ok(result.content)
            }
            Err(e) => Err(FunctionError::ExecutionError(Box::new(e))),
        }
    }
}

/// Create adapters for all tools in a filtered set.
///
/// Convenience function for converting a set of tools to CallableFunction
/// implementations for use with rust-genai.
///
/// # Arguments
///
/// * `tools` - The tools to wrap
/// * `handler` - Optional confirmation handler for tools that require confirmation.
///   If `None`, tools requiring confirmation will be denied by default.
///
/// # Example
///
/// ```no_run
/// use gemicro_core::tool::{tools_to_callables, AutoApprove, Tool};
/// use std::sync::Arc;
///
/// # fn example(tools: Vec<Arc<dyn Tool>>) {
/// // Without confirmation handler (dangerous tools will be denied)
/// let adapters = tools_to_callables(&tools, None);
///
/// // With confirmation handler
/// let handler = Arc::new(AutoApprove);
/// let adapters = tools_to_callables(&tools, Some(handler));
/// # }
/// ```
pub fn tools_to_callables(
    tools: &[Arc<dyn Tool>],
    handler: Option<Arc<dyn ConfirmationHandler>>,
) -> Vec<ToolCallableAdapter> {
    tools
        .iter()
        .map(|t| {
            let mut adapter = ToolCallableAdapter::new(Arc::clone(t));
            if let Some(h) = &handler {
                adapter = adapter.with_confirmation_handler(Arc::clone(h));
            }
            adapter
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{AutoApprove, AutoDeny, ToolError, ToolResult};
    use async_trait::async_trait;
    use rust_genai::CallableFunction;
    use serde_json::{json, Value};

    #[derive(Debug)]
    struct TestTool;

    #[async_trait]
    impl Tool for TestTool {
        fn name(&self) -> &str {
            "test_tool"
        }

        fn description(&self) -> &str {
            "A test tool"
        }

        fn parameters_schema(&self) -> Value {
            json!({
                "type": "object",
                "properties": {
                    "input": { "type": "string" }
                }
            })
        }

        async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
            let input_str = input["input"].as_str().unwrap_or("default");
            Ok(ToolResult::text(format!("Received: {}", input_str)))
        }
    }

    /// A tool that requires confirmation for testing.
    #[derive(Debug)]
    struct DangerousTool;

    #[async_trait]
    impl Tool for DangerousTool {
        fn name(&self) -> &str {
            "dangerous_tool"
        }

        fn description(&self) -> &str {
            "A tool that requires confirmation"
        }

        fn parameters_schema(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }

        fn requires_confirmation(&self, _input: &Value) -> bool {
            true
        }

        fn confirmation_message(&self, _input: &Value) -> String {
            "This is dangerous!".to_string()
        }

        async fn execute(&self, _input: Value) -> Result<ToolResult, ToolError> {
            Ok(ToolResult::text("Executed dangerous operation"))
        }
    }

    #[test]
    fn test_adapter_declaration() {
        let tool = Arc::new(TestTool);
        let adapter = ToolCallableAdapter::new(tool);

        let decl = adapter.declaration();
        assert_eq!(decl.name(), "test_tool");
    }

    #[tokio::test]
    async fn test_adapter_execute() {
        let tool = Arc::new(TestTool);
        let adapter = ToolCallableAdapter::new(tool);

        let result = adapter.execute(json!({"input": "hello"})).await.unwrap();
        assert_eq!(result.content, Value::String("Received: hello".into()));
    }

    #[test]
    fn test_tools_to_callables() {
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(TestTool)];
        let callables = tools_to_callables(&tools, None);

        assert_eq!(callables.len(), 1);
        assert_eq!(callables[0].declaration().name(), "test_tool");
    }

    #[test]
    fn test_tools_to_callables_with_handler() {
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(TestTool)];
        let handler = Arc::new(AutoApprove);
        let callables = tools_to_callables(&tools, Some(handler));

        assert_eq!(callables.len(), 1);
    }

    #[tokio::test]
    async fn test_adapter_safe_tool_no_confirmation_needed() {
        // Safe tools execute without confirmation
        let tool = Arc::new(TestTool);
        let adapter = ToolCallableAdapter::new(tool);

        let result = adapter.call(json!({"input": "test"})).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_adapter_dangerous_tool_denied_without_handler() {
        // Dangerous tools are denied when no handler is configured
        let tool: Arc<dyn Tool> = Arc::new(DangerousTool);
        let adapter = ToolCallableAdapter::new(tool);

        let result = adapter.call(json!({})).await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        let err_msg = format!("{:?}", err);
        assert!(err_msg.contains("ConfirmationDenied"));
        assert!(err_msg.contains("No confirmation handler"));
    }

    #[tokio::test]
    async fn test_adapter_dangerous_tool_approved_with_auto_approve() {
        // Dangerous tools execute when auto-approved
        let tool: Arc<dyn Tool> = Arc::new(DangerousTool);
        let handler = Arc::new(AutoApprove);
        let adapter = ToolCallableAdapter::new(tool).with_confirmation_handler(handler);

        let result = adapter.call(json!({})).await;
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap().as_str().unwrap(),
            "Executed dangerous operation"
        );
    }

    #[tokio::test]
    async fn test_adapter_dangerous_tool_denied_with_auto_deny() {
        // Dangerous tools are denied with auto-deny handler
        let tool: Arc<dyn Tool> = Arc::new(DangerousTool);
        let handler = Arc::new(AutoDeny);
        let adapter = ToolCallableAdapter::new(tool).with_confirmation_handler(handler);

        let result = adapter.call(json!({})).await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        let err_msg = format!("{:?}", err);
        assert!(err_msg.contains("ConfirmationDenied"));
        assert!(err_msg.contains("This is dangerous!"));
    }
}
