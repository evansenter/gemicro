//! Integration tests for hook system with adapters and tools.

use async_trait::async_trait;
use gemicro_core::tool::{
    example_hooks::{AuditLogHook, FileSecurityHook, InputSanitizerHook, MetricsHook},
    HookDecision, HookError, HookRegistry, Tool, ToolCallableAdapter, ToolError, ToolHook,
    ToolResult,
};
use rust_genai::CallableFunction;
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
async fn test_hooks_integrated_with_adapter() {
    // Create hooks
    let hooks = Arc::new(
        HookRegistry::new()
            .with_hook(AuditLogHook)
            .with_hook(MetricsHook),
    );

    // Create adapter with hooks
    let tool = Arc::new(TestTool);
    let adapter = ToolCallableAdapter::new(tool).with_hooks(hooks);

    // Execute via adapter (as rust-genai would)
    let result = adapter.call(json!({"input": "test"})).await.unwrap();
    assert_eq!(result.as_str().unwrap(), "Result: test");
}

#[tokio::test]
async fn test_hook_denies_execution() {
    #[derive(Debug)]
    struct DenyAllHook;

    #[async_trait]
    impl ToolHook for DenyAllHook {
        async fn pre_tool_use(&self, _: &str, _: &Value) -> Result<HookDecision, HookError> {
            Ok(HookDecision::Deny {
                reason: "Denied by test hook".into(),
            })
        }

        async fn post_tool_use(&self, _: &str, _: &Value, _: &ToolResult) -> Result<(), HookError> {
            Ok(())
        }
    }

    let hooks = Arc::new(HookRegistry::new().with_hook(DenyAllHook));
    let tool = Arc::new(TestTool);
    let adapter = ToolCallableAdapter::new(tool).with_hooks(hooks);

    let result = adapter.call(json!({"input": "test"})).await;
    assert!(result.is_err());

    let err = result.unwrap_err();
    let err_msg = format!("{:?}", err);
    assert!(err_msg.contains("HookDenied"));
    assert!(err_msg.contains("Denied by test hook"));
}

#[tokio::test]
async fn test_hook_modifies_input() {
    #[derive(Debug)]
    struct ModifyInputHook;

    #[async_trait]
    impl ToolHook for ModifyInputHook {
        async fn pre_tool_use(&self, _: &str, input: &Value) -> Result<HookDecision, HookError> {
            let mut modified = input.clone();
            modified["input"] = json!("modified");
            Ok(HookDecision::AllowWithModifiedInput(modified))
        }

        async fn post_tool_use(&self, _: &str, _: &Value, _: &ToolResult) -> Result<(), HookError> {
            Ok(())
        }
    }

    let hooks = Arc::new(HookRegistry::new().with_hook(ModifyInputHook));
    let tool = Arc::new(TestTool);
    let adapter = ToolCallableAdapter::new(tool).with_hooks(hooks);

    let result = adapter.call(json!({"input": "original"})).await.unwrap();
    assert_eq!(result.as_str().unwrap(), "Result: modified");
}

#[tokio::test]
async fn test_file_security_hook_integration() {
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

    let hooks =
        Arc::new(HookRegistry::new().with_hook(FileSecurityHook::new(vec![PathBuf::from("/etc")])));

    let tool = Arc::new(FileWriteTool);
    let adapter = ToolCallableAdapter::new(tool).with_hooks(hooks);

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
    let hooks = Arc::new(HookRegistry::new().with_hook(InputSanitizerHook::new(50)));

    let tool = Arc::new(TestTool);
    let adapter = ToolCallableAdapter::new(tool).with_hooks(hooks);

    // Small input should work
    let result = adapter.call(json!({"input": "small"})).await;
    assert!(result.is_ok());

    // Large input should fail
    let large_input = "x".repeat(100);
    let result = adapter.call(json!({"input": large_input})).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_multiple_hooks_chain() {
    #[derive(Debug)]
    struct Hook1;
    #[derive(Debug)]
    struct Hook2;

    #[async_trait]
    impl ToolHook for Hook1 {
        async fn pre_tool_use(&self, _: &str, input: &Value) -> Result<HookDecision, HookError> {
            let mut modified = input.clone();
            modified["hook1"] = json!(true);
            Ok(HookDecision::AllowWithModifiedInput(modified))
        }

        async fn post_tool_use(&self, _: &str, _: &Value, _: &ToolResult) -> Result<(), HookError> {
            Ok(())
        }
    }

    #[async_trait]
    impl ToolHook for Hook2 {
        async fn pre_tool_use(&self, _: &str, input: &Value) -> Result<HookDecision, HookError> {
            assert_eq!(input["hook1"], json!(true)); // Verify Hook1 ran first
            let mut modified = input.clone();
            modified["hook2"] = json!(true);
            Ok(HookDecision::AllowWithModifiedInput(modified))
        }

        async fn post_tool_use(&self, _: &str, _: &Value, _: &ToolResult) -> Result<(), HookError> {
            Ok(())
        }
    }

    let hooks = Arc::new(HookRegistry::new().with_hook(Hook1).with_hook(Hook2));

    let tool = Arc::new(TestTool);
    let adapter = ToolCallableAdapter::new(tool).with_hooks(hooks);

    let result = adapter.call(json!({"input": "test"})).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_adapter_without_hooks_still_works() {
    // Adapter should work fine with no hooks configured
    let tool = Arc::new(TestTool);
    let adapter = ToolCallableAdapter::new(tool);

    let result = adapter.call(json!({"input": "test"})).await.unwrap();
    assert_eq!(result.as_str().unwrap(), "Result: test");
}
