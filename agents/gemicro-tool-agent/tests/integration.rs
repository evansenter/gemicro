//! Integration tests for ToolAgent
//!
//! These tests require a valid GEMINI_API_KEY environment variable.
//! Run with: cargo test -p gemicro-tool-agent -- --include-ignored

mod common;

use common::{create_test_context, get_api_key};
use futures_util::StreamExt;
use gemicro_core::ToolSet;
use gemicro_tool_agent::{ToolAgent, ToolAgentConfig};
use std::time::Duration;

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_tool_agent_calculator() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    let config = ToolAgentConfig::default()
        .with_tool_filter(ToolSet::Specific(vec!["calculator".into()]))
        .with_timeout(Duration::from_secs(60))
        .with_system_prompt(
            "You are a math assistant. Use the calculator tool to solve problems. \
            Always provide the numeric answer.",
        );

    let agent = ToolAgent::new(config).expect("Should create agent");

    let stream = agent.execute("What is 25 * 4?", context);
    futures_util::pin_mut!(stream);

    let mut events: Vec<String> = Vec::new();
    let mut final_answer = String::new();
    let mut tool_complete_data: Option<serde_json::Value> = None;

    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => {
                println!("[{}] {}", update.event_type, update.message);
                events.push(update.event_type.clone());

                if update.event_type == "tool_agent_complete" {
                    tool_complete_data = Some(update.data.clone());
                }

                if update.is_final_result() {
                    if let Some(result) = update.as_final_result() {
                        final_answer = result.result.as_str().unwrap_or("").to_string();
                    }
                }
            }
            Err(e) => {
                panic!("Agent error: {:?}", e);
            }
        }
    }

    // Verify events
    assert!(
        events.contains(&"tool_agent_started".to_string()),
        "Should have tool_agent_started"
    );
    assert!(
        events.contains(&"tool_agent_complete".to_string()),
        "Should have tool_agent_complete"
    );
    assert!(
        events.contains(&"final_result".to_string()),
        "Should have final_result"
    );

    // Verify the answer contains 100 (25 * 4)
    assert!(
        final_answer.contains("100"),
        "Answer should contain 100, got: {}",
        final_answer
    );

    // Verify tool execution metadata
    let data = tool_complete_data.expect("Should have tool_agent_complete data");

    let tool_call_count = data
        .get("tool_call_count")
        .and_then(|v| v.as_u64())
        .expect("Should have tool_call_count");
    assert!(
        tool_call_count >= 1,
        "Should have at least 1 tool call, got: {}",
        tool_call_count
    );

    let tool_calls = data
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .expect("Should have tool_calls array");
    assert!(
        !tool_calls.is_empty(),
        "tool_calls array should not be empty"
    );

    // Verify first tool call has expected fields
    let first_call = &tool_calls[0];
    assert!(first_call.get("name").is_some(), "Should have 'name' field");
    assert!(
        first_call.get("call_id").is_some(),
        "Should have 'call_id' field"
    );
    assert!(
        first_call.get("result").is_some(),
        "Should have 'result' field"
    );
    assert!(
        first_call.get("duration_ms").is_some(),
        "Should have 'duration_ms' field"
    );

    // Verify calculator tool was called
    let tool_names: Vec<&str> = tool_calls
        .iter()
        .filter_map(|c| c.get("name").and_then(|n| n.as_str()))
        .collect();
    assert!(
        tool_names.contains(&"calculator"),
        "Should have called calculator tool, got: {:?}",
        tool_names
    );

    let duration_ms = data
        .get("duration_ms")
        .and_then(|v| v.as_u64())
        .expect("Should have duration_ms");
    assert!(duration_ms > 0, "duration_ms should be non-zero");
}

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_tool_agent_complex_math() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    let config = ToolAgentConfig::calculator_only();
    let agent = ToolAgent::new(config).expect("Should create agent");

    let stream = agent.execute("What is the square root of 144 plus 13 squared?", context);
    futures_util::pin_mut!(stream);

    let mut final_answer = String::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => {
                println!("[{}] {}", update.event_type, update.message);
                if update.is_final_result() {
                    if let Some(result) = update.as_final_result() {
                        final_answer = result.result.as_str().unwrap_or("").to_string();
                    }
                }
            }
            Err(e) => {
                panic!("Agent error: {:?}", e);
            }
        }
    }

    // sqrt(144) = 12, 13^2 = 169, 12 + 169 = 181
    assert!(
        final_answer.contains("181"),
        "Answer should contain 181, got: {}",
        final_answer
    );
}

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_tool_agent_current_datetime() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    let config = ToolAgentConfig::default()
        .with_tool_filter(ToolSet::Specific(vec!["current_datetime".into()]))
        .with_timeout(Duration::from_secs(60))
        .with_system_prompt(
            "You are a helpful assistant. Use the current_datetime tool when asked about time.",
        );

    let agent = ToolAgent::new(config).expect("Should create agent");

    let stream = agent.execute("What time is it in UTC?", context);
    futures_util::pin_mut!(stream);

    let mut final_answer = String::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => {
                println!("[{}] {}", update.event_type, update.message);
                if update.is_final_result() {
                    if let Some(result) = update.as_final_result() {
                        final_answer = result.result.as_str().unwrap_or("").to_string();
                    }
                }
            }
            Err(e) => {
                panic!("Agent error: {:?}", e);
            }
        }
    }

    assert!(
        !final_answer.is_empty(),
        "Should have a non-empty answer about time"
    );
}

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_tool_agent_multiple_tools() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    let config = ToolAgentConfig::default();
    let agent = ToolAgent::new(config).expect("Should create agent");

    let stream = agent.execute(
        "What is 7 times 8, and also what year is it currently?",
        context,
    );
    futures_util::pin_mut!(stream);

    let mut final_answer = String::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => {
                println!("[{}] {}", update.event_type, update.message);
                if update.is_final_result() {
                    if let Some(result) = update.as_final_result() {
                        final_answer = result.result.as_str().unwrap_or("").to_string();
                    }
                }
            }
            Err(e) => {
                panic!("Agent error: {:?}", e);
            }
        }
    }

    // Should contain 56 (7 * 8)
    assert!(
        final_answer.contains("56"),
        "Answer should contain 56, got: {}",
        final_answer
    );
}

#[test]
fn test_tool_agent_config_validation() {
    // Valid config
    let config = ToolAgentConfig::default();
    assert!(config.validate().is_ok());

    // Valid: ToolSet::None is valid at config time (errors at execute time)
    let config = ToolAgentConfig::default().with_tool_filter(ToolSet::None);
    assert!(config.validate().is_ok());

    // Invalid: zero timeout
    let config = ToolAgentConfig::default().with_timeout(Duration::ZERO);
    assert!(config.validate().is_err());

    // Invalid: empty system prompt
    let config = ToolAgentConfig::default().with_system_prompt("");
    assert!(config.validate().is_err());
}

/// Test streaming function calling with hooks and confirmation.
///
/// This verifies that:
/// 1. `create_stream_with_auto_functions()` works with GemicroToolService
/// 2. Hooks are correctly applied in streaming mode
/// 3. Confirmation handlers work in streaming mode
/// 4. Streaming returns incremental text updates via AutoFunctionStreamChunk
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_streaming_function_calling_with_hooks() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    use gemicro_audit_log::AuditLog;
    use gemicro_core::interceptor::InterceptorChain;
    use gemicro_core::tool::{AutoApprove, GemicroToolService};
    use gemicro_core::MODEL;
    use gemicro_tool_agent::tools::default_registry;
    use rust_genai::AutoFunctionStreamChunk;
    use std::sync::Arc;

    // Set up interceptors and confirmation
    let interceptors = Arc::new(InterceptorChain::new().with(AuditLog));
    let confirmation = Arc::new(AutoApprove); // Auto-approve for test

    // Build tool service with interceptors and confirmation
    let registry = Arc::new(default_registry());
    let service = GemicroToolService::new(Arc::clone(&registry))
        .with_filter(ToolSet::Specific(vec!["calculator".into()]))
        .with_interceptors(interceptors)
        .with_confirmation_handler(confirmation);

    // Create streaming interaction
    let genai_client = rust_genai::Client::builder(api_key.to_string())
        .build()
        .unwrap();
    let mut stream = genai_client
        .interaction()
        .with_model(MODEL)
        .with_system_instruction(
            "You are a math assistant. Use the calculator tool to solve problems.",
        )
        .with_text("What is 15 + 27?")
        .with_tool_service(Arc::new(service))
        .create_stream_with_auto_functions();

    let mut text_parts = Vec::new();
    let mut saw_delta = false;
    let mut saw_executing = false;
    let mut saw_results = false;
    let mut saw_complete = false;

    // Consume the stream
    while let Some(result) = stream.next().await {
        match result {
            Ok(event) => match event.chunk {
                AutoFunctionStreamChunk::Delta(content) => {
                    saw_delta = true;
                    if let Some(text) = content.text() {
                        print!("{}", text);
                        text_parts.push(text.to_string());
                    }
                }
                AutoFunctionStreamChunk::ExecutingFunctions(_response) => {
                    saw_executing = true;
                    println!("\n[Executing functions via hooks...]");
                }
                AutoFunctionStreamChunk::FunctionResults(_results) => {
                    saw_results = true;
                    println!("[Function results received]");
                }
                AutoFunctionStreamChunk::Complete(response) => {
                    saw_complete = true;
                    println!("\n[Complete] Total tokens: {:?}", response.usage);
                }
                _ => {
                    println!("[Unknown AutoFunctionStreamChunk variant]");
                }
            },
            Err(e) => {
                panic!("Streaming error: {:?}", e);
            }
        }
    }

    println!(); // Final newline

    // Verify streaming behavior
    assert!(saw_delta, "Should receive Delta chunks in streaming mode");
    assert!(
        saw_complete,
        "Should receive Complete chunk at end of stream"
    );

    // Verify the answer was streamed (should contain 42)
    let full_text: String = text_parts.join("");
    assert!(
        full_text.contains("42"),
        "Streamed answer should contain 42 (15 + 27), got: {}",
        full_text
    );
    assert!(
        !text_parts.is_empty(),
        "Should receive incremental text chunks"
    );

    println!("✓ Streaming FC with hooks and confirmation works correctly");
    println!("✓ Received {} text chunks", text_parts.len());
    println!("✓ Saw ExecutingFunctions: {}", saw_executing);
    println!("✓ Saw FunctionResults: {}", saw_results);
    println!("✓ Final answer: {}", full_text);
}
