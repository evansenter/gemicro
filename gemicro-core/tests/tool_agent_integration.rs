//! Integration tests for ToolAgent
//!
//! These tests require a valid GEMINI_API_KEY environment variable.
//! They are marked with #[ignore] and run with `cargo test -- --include-ignored`.

mod common;

use common::{create_test_context, get_api_key};
use futures_util::StreamExt;
use gemicro_core::{ToolAgent, ToolAgentConfig, ToolType};
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
        .with_tools(vec![ToolType::Calculator])
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

                if update.event_type == "final_result" {
                    if let Some(result) = update.as_final_result() {
                        final_answer = result.answer.clone();
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

    // Verify tool execution metadata in tool_agent_complete event
    let data = tool_complete_data.expect("Should have tool_agent_complete data");

    // Verify tool_call_count exists and is at least 1
    let tool_call_count = data
        .get("tool_call_count")
        .and_then(|v| v.as_u64())
        .expect("Should have tool_call_count");
    assert!(
        tool_call_count >= 1,
        "Should have at least 1 tool call, got: {}",
        tool_call_count
    );

    // Verify tool_calls array exists and has entries
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
    assert!(
        first_call.get("name").is_some(),
        "Tool call should have 'name' field"
    );
    assert!(
        first_call.get("call_id").is_some(),
        "Tool call should have 'call_id' field"
    );
    assert!(
        first_call.get("result").is_some(),
        "Tool call should have 'result' field"
    );
    assert!(
        first_call.get("duration_ms").is_some(),
        "Tool call should have 'duration_ms' field"
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

    // Verify duration is non-zero
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

    // A problem that requires the calculator
    let stream = agent.execute("What is the square root of 144 plus 13 squared?", context);
    futures_util::pin_mut!(stream);

    let mut final_answer = String::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => {
                println!("[{}] {}", update.event_type, update.message);
                if update.event_type == "final_result" {
                    if let Some(result) = update.as_final_result() {
                        final_answer = result.answer.clone();
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
        "Answer should contain 181 (sqrt(144) + 13^2 = 12 + 169 = 181), got: {}",
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
        .with_tools(vec![ToolType::CurrentDateTime])
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
                if update.event_type == "final_result" {
                    if let Some(result) = update.as_final_result() {
                        final_answer = result.answer.clone();
                    }
                }
            }
            Err(e) => {
                panic!("Agent error: {:?}", e);
            }
        }
    }

    // Answer should contain time-related information
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

    // Default config has both tools
    let config = ToolAgentConfig::default();
    let agent = ToolAgent::new(config).expect("Should create agent");

    // This query might use both calculator and datetime
    let stream = agent.execute(
        "What is 7 times 8, and also what year is it currently?",
        context,
    );
    futures_util::pin_mut!(stream);

    let mut final_answer = String::new();
    let mut events: Vec<String> = Vec::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => {
                println!("[{}] {}", update.event_type, update.message);
                events.push(update.event_type.clone());
                if update.event_type == "final_result" {
                    if let Some(result) = update.as_final_result() {
                        final_answer = result.answer.clone();
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
        "Answer should contain 56 (7*8), got: {}",
        final_answer
    );
}

#[test]
fn test_tool_agent_config_validation() {
    // Valid config
    let config = ToolAgentConfig::default();
    assert!(config.validate().is_ok());

    // Invalid: no tools
    let config = ToolAgentConfig::default().with_tools(vec![]);
    assert!(config.validate().is_err());

    // Invalid: zero timeout
    let config = ToolAgentConfig::default().with_timeout(Duration::ZERO);
    assert!(config.validate().is_err());

    // Invalid: empty system prompt
    let config = ToolAgentConfig::default().with_system_prompt("");
    assert!(config.validate().is_err());
}
