//! Integration tests for DeveloperAgent
//!
//! These tests require a valid GEMINI_API_KEY environment variable.
//! Run with: cargo test -p gemicro-developer -- --include-ignored

mod common;

use common::{create_test_context, get_api_key};
use futures_util::StreamExt;
use gemicro_core::{Agent, AgentError};
use gemicro_developer::{DeveloperAgent, DeveloperConfig};
use std::time::Duration;

/// Test the full developer flow with file reading
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_developer_file_read_query() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    let config = DeveloperConfig::default()
        .with_max_iterations(5)
        .with_timeout(Duration::from_secs(90));

    let agent = DeveloperAgent::new(config).expect("Should create agent");
    let stream = agent.execute(
        "Read the Cargo.toml file and tell me the package name",
        context,
    );
    futures_util::pin_mut!(stream);

    let mut events: Vec<String> = Vec::new();
    let mut tool_calls: Vec<String> = Vec::new();
    let mut final_answer = String::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => {
                println!("[{}] {}", update.event_type, update.message);
                events.push(update.event_type.clone());

                match update.event_type.as_str() {
                    "tool_call_started" => {
                        if let Some(tool_name) = update.data["tool_name"].as_str() {
                            println!("  Tool: {}", tool_name);
                            tool_calls.push(tool_name.to_string());
                        }
                    }
                    "tool_result" => {
                        let success = update.data["success"].as_bool().unwrap_or(false);
                        println!("  Result: {}", if success { "OK" } else { "FAILED" });
                    }
                    "final_result" => {
                        if let Some(result) = update.as_final_result() {
                            if let Some(answer) = result.result.as_str() {
                                final_answer = answer.to_string();
                                println!("\n=== Final Answer ===");
                                println!("{}", answer);
                            }
                        }
                    }
                    _ => {}
                }
            }
            Err(e) => {
                panic!("Agent error: {:?}", e);
            }
        }
    }

    // Verify event sequence
    assert!(
        events.contains(&"developer_started".to_string()),
        "Should have developer_started"
    );
    assert!(
        events.contains(&"tool_call_started".to_string()),
        "Should have at least one tool_call_started"
    );
    assert!(
        events.contains(&"tool_result".to_string()),
        "Should have at least one tool_result"
    );
    assert!(
        events.contains(&"final_result".to_string()),
        "Should have final_result"
    );

    // Verify file_read was used (case-insensitive check since tool names vary)
    let used_file_read = tool_calls
        .iter()
        .any(|t| t.eq_ignore_ascii_case("file_read") || t.eq_ignore_ascii_case("FileRead"));
    assert!(used_file_read, "Should have used file_read tool");

    // Verify final answer mentions the package name
    assert!(
        final_answer.to_lowercase().contains("gemicro-developer")
            || final_answer.to_lowercase().contains("developer"),
        "Final answer should mention the package name: got {}",
        final_answer
    );
}

/// Test event ordering: developer_started first, final_result last
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_developer_event_ordering() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    let config = DeveloperConfig::default()
        .with_max_iterations(3)
        .with_timeout(Duration::from_secs(60));

    let agent = DeveloperAgent::new(config).expect("Should create agent");
    let stream = agent.execute(
        "List files matching *.toml in the current directory",
        context,
    );
    futures_util::pin_mut!(stream);

    let mut events: Vec<String> = Vec::new();

    while let Some(result) = stream.next().await {
        if let Ok(update) = result {
            events.push(update.event_type.clone());
        }
    }

    // Verify developer_started is first
    assert_eq!(
        events.first(),
        Some(&"developer_started".to_string()),
        "First event should be developer_started. Events: {:?}",
        events
    );

    // Verify final_result is last
    assert_eq!(
        events.last(),
        Some(&"final_result".to_string()),
        "Last event should be final_result. Events: {:?}",
        events
    );

    // Verify tool_call_started comes before tool_result for each pair
    let call_positions: Vec<_> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| *e == "tool_call_started")
        .map(|(i, _)| i)
        .collect();

    let result_positions: Vec<_> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| *e == "tool_result")
        .map(|(i, _)| i)
        .collect();

    assert_eq!(
        call_positions.len(),
        result_positions.len(),
        "Should have matching tool_call_started and tool_result counts"
    );

    for (call_pos, result_pos) in call_positions.iter().zip(result_positions.iter()) {
        assert!(
            call_pos < result_pos,
            "tool_call_started should come before tool_result"
        );
    }
}

/// Test max_iterations path emits final_result per event contract
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_developer_max_iterations_emits_final_result() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    // Use only 1 iteration to force hitting the limit
    let config = DeveloperConfig::default()
        .with_max_iterations(1)
        .with_timeout(Duration::from_secs(60));

    let agent = DeveloperAgent::new(config).expect("Should create agent");
    // Give a complex task that would need multiple iterations
    let stream = agent.execute(
        "Read Cargo.toml, then read README.md, then summarize both files",
        context,
    );
    futures_util::pin_mut!(stream);

    let mut events: Vec<String> = Vec::new();

    while let Some(result) = stream.next().await {
        if let Ok(update) = result {
            println!("[{}] {}", update.event_type, update.message);
            events.push(update.event_type.clone());
        }
    }

    // Verify final_result is emitted per event contract
    assert!(
        events.contains(&"final_result".to_string()),
        "Should emit final_result even on max_iterations path. Events: {:?}",
        events
    );

    // Verify final_result is last
    assert_eq!(
        events.last(),
        Some(&"final_result".to_string()),
        "final_result MUST be the last event. Events: {:?}",
        events
    );
}

/// Test invalid config - zero max_iterations
#[tokio::test]
async fn test_developer_invalid_config_zero_iterations() {
    let config = DeveloperConfig::default().with_max_iterations(0);

    let result = DeveloperAgent::new(config);
    assert!(result.is_err());
    assert!(matches!(result, Err(AgentError::InvalidConfig(_))));
}

/// Test invalid config - zero timeout
#[tokio::test]
async fn test_developer_invalid_config_zero_timeout() {
    let config = DeveloperConfig::default().with_timeout(Duration::ZERO);

    let result = DeveloperAgent::new(config);
    assert!(result.is_err());
    assert!(matches!(result, Err(AgentError::InvalidConfig(_))));
}

/// Test tool_call_started event contains required fields
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_tool_call_started_event_structure() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);
    let config = DeveloperConfig::default()
        .with_max_iterations(3)
        .with_timeout(Duration::from_secs(60));

    let agent = DeveloperAgent::new(config).expect("Should create agent");
    let stream = agent.execute("List *.toml files in the current directory", context);
    futures_util::pin_mut!(stream);

    let mut found_tool_call = false;

    while let Some(result) = stream.next().await {
        if let Ok(update) = result {
            if update.event_type == "tool_call_started" {
                found_tool_call = true;

                // Verify required fields exist
                assert!(
                    update.data.get("tool_name").is_some(),
                    "tool_call_started should have tool_name"
                );
                assert!(
                    update.data.get("call_id").is_some(),
                    "tool_call_started should have call_id"
                );
                assert!(
                    update.data.get("arguments").is_some(),
                    "tool_call_started should have arguments"
                );

                // Verify tool_name is a non-empty string
                let tool_name = update.data["tool_name"].as_str().unwrap();
                assert!(!tool_name.is_empty(), "tool_name should not be empty");

                // Verify arguments is an object
                assert!(
                    update.data["arguments"].is_object(),
                    "arguments should be a JSON object"
                );

                break;
            }
        }
    }

    assert!(
        found_tool_call,
        "Should have at least one tool_call_started event"
    );
}

/// Test tool_result event contains required fields
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_tool_result_event_structure() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);
    let config = DeveloperConfig::default()
        .with_max_iterations(3)
        .with_timeout(Duration::from_secs(60));

    let agent = DeveloperAgent::new(config).expect("Should create agent");
    let stream = agent.execute("List *.toml files in the current directory", context);
    futures_util::pin_mut!(stream);

    let mut found_tool_result = false;

    while let Some(result) = stream.next().await {
        if let Ok(update) = result {
            if update.event_type == "tool_result" {
                found_tool_result = true;

                // Verify required fields exist
                assert!(
                    update.data.get("tool_name").is_some(),
                    "tool_result should have tool_name"
                );
                assert!(
                    update.data.get("call_id").is_some(),
                    "tool_result should have call_id"
                );
                assert!(
                    update.data.get("success").is_some(),
                    "tool_result should have success"
                );
                assert!(
                    update.data.get("duration_ms").is_some(),
                    "tool_result should have duration_ms"
                );
                assert!(
                    update.data.get("result").is_some(),
                    "tool_result should have result"
                );

                // Verify types
                assert!(
                    update.data["success"].is_boolean(),
                    "success should be a boolean"
                );
                assert!(
                    update.data["duration_ms"].is_u64(),
                    "duration_ms should be a number"
                );

                break;
            }
        }
    }

    assert!(
        found_tool_result,
        "Should have at least one tool_result event"
    );
}

/// Test incomplete execution metadata on max_iterations
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_max_iterations_incomplete_metadata() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    // Use only 1 iteration to force hitting the limit
    let config = DeveloperConfig::default()
        .with_max_iterations(1)
        .with_timeout(Duration::from_secs(60));

    let agent = DeveloperAgent::new(config).expect("Should create agent");
    let stream = agent.execute(
        "Read all Cargo.toml files in the repository and summarize each one",
        context,
    );
    futures_util::pin_mut!(stream);

    let mut final_result_data = None;

    while let Some(result) = stream.next().await {
        if let Ok(update) = result {
            if update.event_type == "final_result" {
                final_result_data = Some(update);
            }
        }
    }

    let update = final_result_data.expect("Should have final_result");
    let result = update
        .as_final_result()
        .expect("Should parse as FinalResult");

    // Verify incomplete metadata
    let extra = &result.metadata.extra;
    assert!(
        extra
            .get("incomplete")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        "Should have incomplete: true in metadata. Extra: {:?}",
        extra
    );
    assert!(
        extra.get("reason").is_some(),
        "Should have reason in metadata when incomplete"
    );

    let reason = extra["reason"].as_str().unwrap();
    assert!(
        reason.contains("max iterations"),
        "Reason should mention max iterations: got {}",
        reason
    );
}
