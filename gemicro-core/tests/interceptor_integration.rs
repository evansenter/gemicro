//! Integration tests for interceptor system with adapters and tools.

use async_trait::async_trait;
use gemicro_audit_log::AuditLog;
use gemicro_core::interceptor::{
    InterceptDecision, InterceptError, Interceptor, InterceptorChain, ToolCall,
};
use gemicro_core::tool::{Tool, ToolCallableAdapter, ToolError, ToolResult};
use gemicro_file_security::FileSecurity;
use gemicro_input_sanitizer::InputSanitizer;
use gemicro_metrics::Metrics;
use genai_rs::CallableFunction;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;

// Test tool for integration testing
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
        Ok(ToolResult::text(format!("Result: {}", input_str)))
    }
}

#[tokio::test]
async fn test_interceptors_integrated_with_adapter() {
    // Create interceptors
    let interceptors = Arc::new(InterceptorChain::new().with(AuditLog).with(Metrics::new()));

    // Create adapter with interceptors
    let tool = Arc::new(TestTool);
    let adapter = ToolCallableAdapter::new(tool).with_interceptors(interceptors);

    // Execute via adapter (as genai-rs would)
    let result = adapter.call(json!({"input": "test"})).await.unwrap();
    assert_eq!(result.as_str().unwrap(), "Result: test");
}

#[tokio::test]
async fn test_interceptor_denies_execution() {
    #[derive(Debug)]
    struct DenyAllInterceptor;

    #[async_trait]
    impl Interceptor<ToolCall, ToolResult> for DenyAllInterceptor {
        async fn intercept(
            &self,
            _input: &ToolCall,
        ) -> Result<InterceptDecision<ToolCall>, InterceptError> {
            Ok(InterceptDecision::Deny {
                reason: "Denied by test interceptor".into(),
            })
        }
    }

    let interceptors = Arc::new(InterceptorChain::new().with(DenyAllInterceptor));
    let tool = Arc::new(TestTool);
    let adapter = ToolCallableAdapter::new(tool).with_interceptors(interceptors);

    let result = adapter.call(json!({"input": "test"})).await;
    assert!(result.is_err());

    let err = result.unwrap_err();
    let err_msg = format!("{:?}", err);
    assert!(err_msg.contains("InterceptorDenied"));
    assert!(err_msg.contains("Denied by test interceptor"));
}

#[tokio::test]
async fn test_interceptor_transforms_input() {
    #[derive(Debug)]
    struct TransformInputInterceptor;

    #[async_trait]
    impl Interceptor<ToolCall, ToolResult> for TransformInputInterceptor {
        async fn intercept(
            &self,
            input: &ToolCall,
        ) -> Result<InterceptDecision<ToolCall>, InterceptError> {
            let mut modified_args = input.arguments.clone();
            modified_args["input"] = json!("modified");
            Ok(InterceptDecision::Transform(ToolCall::new(
                input.name.clone(),
                modified_args,
            )))
        }
    }

    let interceptors = Arc::new(InterceptorChain::new().with(TransformInputInterceptor));
    let tool = Arc::new(TestTool);
    let adapter = ToolCallableAdapter::new(tool).with_interceptors(interceptors);

    let result = adapter.call(json!({"input": "original"})).await.unwrap();
    assert_eq!(result.as_str().unwrap(), "Result: modified");
}

#[tokio::test]
async fn test_file_security_interceptor_integration() {
    #[derive(Debug)]
    struct FileWriteTool;

    #[async_trait]
    impl Tool for FileWriteTool {
        fn name(&self) -> &str {
            "file_write"
        }

        fn description(&self) -> &str {
            "Write to a file"
        }

        fn parameters_schema(&self) -> Value {
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                }
            })
        }

        async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
            let path = input["path"].as_str().unwrap_or("");
            Ok(ToolResult::text(format!("Wrote to: {}", path)))
        }
    }

    let interceptors =
        Arc::new(InterceptorChain::new().with(FileSecurity::new(vec![PathBuf::from("/etc")])));

    let tool = Arc::new(FileWriteTool);
    let adapter = ToolCallableAdapter::new(tool).with_interceptors(interceptors);

    // Safe path should work
    let result = adapter
        .call(json!({"path": "/home/user/file.txt", "content": "test"}))
        .await;
    assert!(result.is_ok());

    // Blocked path should fail
    let result = adapter
        .call(json!({"path": "/etc/passwd", "content": "evil"}))
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_input_sanitizer_integration() {
    let interceptors = Arc::new(InterceptorChain::new().with(InputSanitizer::new(50)));

    let tool = Arc::new(TestTool);
    let adapter = ToolCallableAdapter::new(tool).with_interceptors(interceptors);

    // Small input should work
    let result = adapter.call(json!({"input": "small"})).await;
    assert!(result.is_ok());

    // Large input should fail
    let large_input = "x".repeat(100);
    let result = adapter.call(json!({"input": large_input})).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_multiple_interceptors_chain() {
    #[derive(Debug)]
    struct Interceptor1;
    #[derive(Debug)]
    struct Interceptor2;

    #[async_trait]
    impl Interceptor<ToolCall, ToolResult> for Interceptor1 {
        async fn intercept(
            &self,
            input: &ToolCall,
        ) -> Result<InterceptDecision<ToolCall>, InterceptError> {
            let mut modified_args = input.arguments.clone();
            modified_args["interceptor1"] = json!(true);
            Ok(InterceptDecision::Transform(ToolCall::new(
                input.name.clone(),
                modified_args,
            )))
        }
    }

    #[async_trait]
    impl Interceptor<ToolCall, ToolResult> for Interceptor2 {
        async fn intercept(
            &self,
            input: &ToolCall,
        ) -> Result<InterceptDecision<ToolCall>, InterceptError> {
            // Verify Interceptor1 ran first
            assert_eq!(input.arguments["interceptor1"], json!(true));
            let mut modified_args = input.arguments.clone();
            modified_args["interceptor2"] = json!(true);
            Ok(InterceptDecision::Transform(ToolCall::new(
                input.name.clone(),
                modified_args,
            )))
        }
    }

    let interceptors = Arc::new(
        InterceptorChain::new()
            .with(Interceptor1)
            .with(Interceptor2),
    );

    let tool = Arc::new(TestTool);
    let adapter = ToolCallableAdapter::new(tool).with_interceptors(interceptors);

    let result = adapter.call(json!({"input": "test"})).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_adapter_without_interceptors_still_works() {
    // Adapter should work fine with no interceptors configured
    let tool = Arc::new(TestTool);
    let adapter = ToolCallableAdapter::new(tool);

    let result = adapter.call(json!({"input": "test"})).await.unwrap();
    assert_eq!(result.as_str().unwrap(), "Result: test");
}
