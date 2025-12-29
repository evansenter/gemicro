//! Adapter for integrating Tool trait with rust-genai.
//!
//! This module provides [`ToolCallableAdapter`] which bridges the gap between
//! our async [`Tool`] trait and rust-genai's async `CallableFunction` trait.

use super::Tool;
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
/// # Example
///
/// ```no_run
/// use gemicro_core::tool::{ToolCallableAdapter, ToolRegistry};
/// use rust_genai::CallableFunction; // Required to use declaration()
///
/// // Assuming you have a registry with tools
/// let registry = ToolRegistry::new();
/// let tool = registry.get("calculator").unwrap();
///
/// // Create adapter for rust-genai integration
/// let adapter = ToolCallableAdapter::new(tool);
///
/// // Use with rust-genai's function calling
/// let declaration = adapter.declaration();
/// ```
#[derive(Debug, Clone)]
pub struct ToolCallableAdapter {
    tool: Arc<dyn Tool>,
}

impl ToolCallableAdapter {
    /// Create a new adapter wrapping the given tool.
    pub fn new(tool: Arc<dyn Tool>) -> Self {
        Self { tool }
    }

    /// Get the underlying tool.
    pub fn tool(&self) -> &Arc<dyn Tool> {
        &self.tool
    }

    /// Execute the tool asynchronously.
    ///
    /// This is a convenience method for when you want to execute the tool
    /// directly without going through the sync CallableFunction interface.
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
        match self.tool.execute(args).await {
            Ok(result) => {
                // Return content as a JSON string value for rust-genai compatibility.
                // Note: ToolResult::metadata is intentionally not returned here because
                // rust-genai's CallableFunction expects a simple Value result that gets
                // passed back to the LLM. Metadata is for observability/logging within
                // gemicro, not for LLM consumption.
                Ok(Value::String(result.content))
            }
            Err(e) => Err(FunctionError::ExecutionError(Box::new(e))),
        }
    }
}

/// Create adapters for all tools in a filtered set.
///
/// Convenience function for converting a set of tools to CallableFunction
/// implementations for use with rust-genai.
pub fn tools_to_callables(tools: &[Arc<dyn Tool>]) -> Vec<ToolCallableAdapter> {
    tools
        .iter()
        .map(|t| ToolCallableAdapter::new(Arc::clone(t)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{ToolError, ToolResult};
    use async_trait::async_trait;
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
            Ok(ToolResult::new(format!("Received: {}", input_str)))
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
        assert_eq!(result.content, "Received: hello");
    }

    #[test]
    fn test_tools_to_callables() {
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(TestTool)];
        let callables = tools_to_callables(&tools);

        assert_eq!(callables.len(), 1);
        assert_eq!(callables[0].declaration().name(), "test_tool");
    }
}
