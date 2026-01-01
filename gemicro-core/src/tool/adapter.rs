//! Adapter for integrating Tool trait with rust-genai.
//!
//! This module provides [`ToolCallableAdapter`] which bridges the gap between
//! our async [`Tool`] trait and rust-genai's async `CallableFunction` trait.
//!
//! # Architecture: Why Interceptors Live Here
//!
//! The adapter is the **critical interception point** for tool execution:
//!
//! ```text
//! LLM (via rust-genai)
//!     ↓ calls create_with_auto_functions()
//! CallableFunction::call()  ← ONLY INTERCEPTION POINT
//!     ↓ implemented by
//! ToolCallableAdapter
//!     ├─ Pre-interceptors (validation, security)
//!     ├─ Confirmation (user approval)
//!     ├─ Tool::execute()
//!     └─ Post-interceptors (logging, metrics)
//! ```
//!
//! When using rust-genai's automatic function calling, the LLM invokes
//! `CallableFunction::call()` directly. There is no opportunity to intercept
//! at the `Tool` or `ToolRegistry` level - those abstractions are bypassed.
//!
//! **Alternative designs considered:**
//! - Interceptors in `ToolRegistry::execute()` ❌ Bypassed by rust-genai
//! - Interceptors in `Tool::execute()` ❌ Couples all tools to interceptor logic
//! - Interceptors in `ToolCallableAdapter::call()` ✅ Single enforcement point
//!
//! **Trade-off:**
//! Direct calls to `tool.execute()` bypass interceptors. This is acceptable because:
//! - Direct calls are for testing or manual tool invocation
//! - LLM function calling (the primary use case) goes through the adapter
//! - Interceptors are opt-in via `with_interceptors()` builder method

use super::{ConfirmationHandler, Tool, ToolError, ToolResult};
use crate::interceptor::{InterceptDecision, InterceptorChain, ToolCall};
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
    interceptors: Option<Arc<InterceptorChain<ToolCall, ToolResult>>>,
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
            interceptors: None,
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

    /// Set an interceptor chain for this adapter.
    ///
    /// Interceptors are called before and after tool execution for validation,
    /// logging, and security controls.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gemicro_core::tool::ToolCallableAdapter;
    /// use gemicro_core::interceptor::InterceptorChain;
    /// use std::sync::Arc;
    ///
    /// # fn example(tool: Arc<dyn gemicro_core::tool::Tool>) {
    /// use gemicro_core::interceptor::ToolCall;
    /// use gemicro_core::tool::ToolResult;
    ///
    /// let interceptors: InterceptorChain<ToolCall, ToolResult> = InterceptorChain::new();
    /// let adapter = ToolCallableAdapter::new(tool)
    ///     .with_interceptors(Arc::new(interceptors));
    /// # }
    /// ```
    pub fn with_interceptors(
        mut self,
        interceptors: Arc<InterceptorChain<ToolCall, ToolResult>>,
    ) -> Self {
        self.interceptors = Some(interceptors);
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
    pub async fn execute(&self, input: serde_json::Value) -> Result<ToolResult, ToolError> {
        self.tool.execute(input).await
    }
}

#[async_trait]
impl CallableFunction for ToolCallableAdapter {
    fn declaration(&self) -> FunctionDeclaration {
        self.tool.to_function_declaration()
    }

    async fn call(&self, args: Value) -> Result<Value, FunctionError> {
        // Create ToolCall for interceptors
        let mut tool_call = ToolCall::new(self.tool.name(), args);

        // 1. PRE-INTERCEPTORS: Validate and potentially modify input
        if let Some(interceptors) = &self.interceptors {
            match interceptors.intercept(&tool_call).await {
                Ok(InterceptDecision::Allow) => {
                    // Continue with original args
                }
                Ok(InterceptDecision::Transform(modified)) => {
                    // Use modified ToolCall for execution
                    tool_call = modified;
                }
                Ok(InterceptDecision::Confirm { message }) => {
                    // Interceptor requests permission - use confirmation handler
                    match &self.confirmation_handler {
                        Some(handler) => {
                            if !handler
                                .confirm(self.tool.name(), &message, &tool_call.arguments)
                                .await
                            {
                                return Err(FunctionError::ExecutionError(Box::new(
                                    ToolError::ConfirmationDenied(message),
                                )));
                            }
                            // Permission granted - continue
                        }
                        None => {
                            // No handler = deny by default for safety
                            return Err(FunctionError::ExecutionError(Box::new(
                                ToolError::ConfirmationDenied(format!(
                                    "Interceptor requested permission but no confirmation handler configured: {}",
                                    message
                                )),
                            )));
                        }
                    }
                }
                Ok(InterceptDecision::Deny { reason }) => {
                    return Err(FunctionError::ExecutionError(Box::new(
                        ToolError::InterceptorDenied(reason),
                    )));
                }
                Err(e) => {
                    // Interceptor failure = deny for safety
                    return Err(FunctionError::ExecutionError(Box::new(
                        ToolError::InterceptorFailed(format!("Pre-interceptor failed: {}", e)),
                    )));
                }
            }
        }

        // 2. CONFIRMATION: Check if user approval is needed
        if self.tool.requires_confirmation(&tool_call.arguments) {
            let message = self.tool.confirmation_message(&tool_call.arguments);

            match &self.confirmation_handler {
                Some(handler) => {
                    if !handler
                        .confirm(self.tool.name(), &message, &tool_call.arguments)
                        .await
                    {
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

        // 3. EXECUTE: Run the tool
        let result = match self.tool.execute(tool_call.arguments.clone()).await {
            Ok(result) => result,
            Err(e) => return Err(FunctionError::ExecutionError(Box::new(e))),
        };

        // 4. POST-INTERCEPTORS: Observability (errors logged, don't fail execution)
        if let Some(interceptors) = &self.interceptors {
            interceptors.observe(&tool_call, &result).await;
        }

        // Return content directly - it's already a Value.
        // Note: ToolResult::metadata is intentionally not returned here because
        // rust-genai's CallableFunction expects a simple Value result that gets
        // passed back to the LLM. Metadata is for observability/logging within
        // gemicro, not for LLM consumption.
        Ok(result.content)
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
