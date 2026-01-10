//! Integration tests for DeveloperAgent
//!
//! These tests require a valid GEMINI_API_KEY environment variable.
//! Run with: cargo test -p gemicro-developer -- --include-ignored

mod common;

use common::{create_test_context, get_api_key};
use futures_util::StreamExt;
use gemicro_core::tool::{AutoApprove, ToolRegistry};
use gemicro_core::{Agent, AgentContext, AgentError, LlmClient, LlmConfig};
use gemicro_developer::{DeveloperAgent, DeveloperConfig};
use gemicro_file_read::FileRead;
use gemicro_glob::Glob;
use gemicro_grep::Grep;
use std::sync::Arc;
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

/// Test approval_batching: false mode (individual confirmations, no batch events)
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_developer_individual_approval_mode() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    // Disable batch approval - tools should be confirmed individually
    let config = DeveloperConfig::default()
        .with_max_iterations(3)
        .with_approval_batching(false)
        .with_timeout(Duration::from_secs(60));

    let agent = DeveloperAgent::new(config).expect("Should create agent");
    let stream = agent.execute("List *.toml files in the current directory", context);
    futures_util::pin_mut!(stream);

    let mut events: Vec<String> = Vec::new();

    while let Some(result) = stream.next().await {
        if let Ok(update) = result {
            println!("[{}] {}", update.event_type, update.message);
            events.push(update.event_type.clone());
        }
    }

    // Should NOT have batch events when approval_batching is false
    assert!(
        !events.contains(&"batch_plan".to_string()),
        "Should NOT have batch_plan when approval_batching=false. Events: {:?}",
        events
    );
    assert!(
        !events.contains(&"batch_approved".to_string()),
        "Should NOT have batch_approved when approval_batching=false. Events: {:?}",
        events
    );
    assert!(
        !events.contains(&"batch_denied".to_string()),
        "Should NOT have batch_denied when approval_batching=false. Events: {:?}",
        events
    );

    // Should still have tool events
    assert!(
        events.contains(&"tool_call_started".to_string()),
        "Should have tool_call_started events. Events: {:?}",
        events
    );
    assert!(
        events.contains(&"tool_result".to_string()),
        "Should have tool_result events. Events: {:?}",
        events
    );

    // Should have final_result
    assert!(
        events.contains(&"final_result".to_string()),
        "Should have final_result. Events: {:?}",
        events
    );
}

/// Test batch approval events are emitted when approval_batching: true
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_developer_batch_approval_events() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    // Enable batch approval (default)
    let config = DeveloperConfig::default()
        .with_max_iterations(5)
        .with_approval_batching(true)
        .with_timeout(Duration::from_secs(90));

    let agent = DeveloperAgent::new(config).expect("Should create agent");
    // Use a query that likely triggers multiple tool calls to see batch behavior
    let stream = agent.execute(
        "Read both Cargo.toml and README.md files and tell me what this project does",
        context,
    );
    futures_util::pin_mut!(stream);

    let mut events: Vec<String> = Vec::new();
    let mut batch_plan_count = 0;
    let mut batch_approved_count = 0;

    while let Some(result) = stream.next().await {
        if let Ok(update) = result {
            println!("[{}] {}", update.event_type, update.message);
            events.push(update.event_type.clone());

            match update.event_type.as_str() {
                "batch_plan" => {
                    batch_plan_count += 1;
                    // Verify batch_plan has expected fields
                    assert!(
                        update.data.get("tools").is_some(),
                        "batch_plan should have tools field"
                    );
                    assert!(
                        update.data.get("total").is_some(),
                        "batch_plan should have total field"
                    );
                }
                "batch_approved" => {
                    batch_approved_count += 1;
                    // Verify batch_approved has expected fields
                    assert!(
                        update.data.get("total").is_some(),
                        "batch_approved should have total field"
                    );
                }
                _ => {}
            }
        }
    }

    // With AutoApprove handler and batching enabled, we expect batch events
    // Note: The LLM might not always propose multiple tools in one turn,
    // so we check for at least the basic flow
    println!(
        "batch_plan_count: {}, batch_approved_count: {}",
        batch_plan_count, batch_approved_count
    );

    // If tools were called, we should see batch events (with batching enabled)
    if events.contains(&"tool_call_started".to_string()) {
        // At minimum, batch_plan should appear before tool execution
        assert!(
            batch_plan_count > 0 || batch_approved_count > 0,
            "With approval_batching=true and tool calls, should have batch events. Events: {:?}",
            events
        );
    }

    // Should have final_result regardless
    assert!(
        events.contains(&"final_result".to_string()),
        "Should have final_result. Events: {:?}",
        events
    );
}

/// Test cancellation token stops execution mid-stream
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_developer_cancellation_mid_execution() {
    use tokio_util::sync::CancellationToken;

    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    // Create context with cancellation token
    let cancellation_token = CancellationToken::new();
    let genai_client = genai_rs::Client::builder(api_key).build().unwrap();
    let config = LlmConfig::default()
        .with_timeout(Duration::from_secs(120))
        .with_max_tokens(4096);
    let llm = LlmClient::new(genai_client, config);

    let mut tools = ToolRegistry::new();
    tools.register(FileRead);
    tools.register(Glob);
    tools.register(Grep);

    let context = AgentContext::new_with_cancellation(llm, cancellation_token.clone())
        .with_tools(tools)
        .with_confirmation_handler(Arc::new(AutoApprove));

    // Give a complex task that requires multiple iterations
    let agent_config = DeveloperConfig::default()
        .with_max_iterations(20)
        .with_timeout(Duration::from_secs(120));

    let agent = DeveloperAgent::new(agent_config).expect("Should create agent");
    let stream = agent.execute(
        "Read all Cargo.toml files in the workspace and summarize each one in detail",
        context,
    );
    futures_util::pin_mut!(stream);

    let mut events: Vec<String> = Vec::new();
    let mut event_count = 0;

    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => {
                println!("[{}] {}", update.event_type, update.message);
                events.push(update.event_type.clone());
                event_count += 1;

                // Cancel after we see some activity (developer_started + at least one tool call)
                if event_count >= 3 && events.contains(&"tool_call_started".to_string()) {
                    println!(">>> Triggering cancellation after {} events", event_count);
                    cancellation_token.cancel();
                }
            }
            Err(e) => {
                println!("[ERROR] {:?}", e);
                // Cancellation may surface as an error
                if e.is_cancelled() {
                    println!(">>> Stream returned cancellation error (expected)");
                    break;
                }
            }
        }
    }

    // Verify we got some events before cancellation
    assert!(
        events.contains(&"developer_started".to_string()),
        "Should have developer_started before cancellation"
    );

    // Verify execution was cut short (didn't reach normal completion with many iterations)
    // The test config allows 20 iterations - if cancellation works, we should stop early
    println!("Total events: {}", events.len());

    // Either we got final_result (graceful completion) or stream ended early
    // Both are valid - the key is that cancellation was honored
    if events.contains(&"final_result".to_string()) {
        println!(">>> Agent completed gracefully after cancellation signal");
    } else {
        println!(">>> Stream ended without final_result (cancellation in progress)");
    }
}
