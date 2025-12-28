//! Integration tests for ReactAgent
//!
//! These tests require a valid GEMINI_API_KEY environment variable.
//! Run with: cargo test -p gemicro-react -- --include-ignored

mod common;

use common::{create_test_context, get_api_key};
use futures_util::StreamExt;
use gemicro_core::AgentError;
use gemicro_react::{ReactAgent, ReactConfig};
use std::time::Duration;

/// Test the full ReAct flow with a calculator query
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_react_calculator_query() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    let config = ReactConfig {
        max_iterations: 5,
        available_tools: vec!["calculator".to_string()],
        use_google_search: false,
        total_timeout: Duration::from_secs(60),
        ..Default::default()
    };

    let agent = ReactAgent::new(config).expect("Should create agent");
    let stream = agent.execute("What is 25 * 4?", context);
    futures_util::pin_mut!(stream);

    let mut events: Vec<String> = Vec::new();
    let mut used_calculator = false;
    let mut final_answer = String::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => {
                println!("[{}] {}", update.event_type, update.message);
                events.push(update.event_type.clone());

                match update.event_type.as_str() {
                    "react_action" => {
                        let tool = update.data.get("tool").and_then(|v| v.as_str());
                        if tool == Some("calculator") {
                            used_calculator = true;
                        }
                        println!(
                            "  Tool: {:?}, Input: {:?}",
                            tool,
                            update.data.get("input").and_then(|v| v.as_str())
                        );
                    }
                    "react_observation" => {
                        println!(
                            "  Observation: {:?}",
                            update.data.get("result").and_then(|v| v.as_str())
                        );
                    }
                    "react_complete" => {
                        if let Some(answer) =
                            update.data.get("final_answer").and_then(|v| v.as_str())
                        {
                            final_answer = answer.to_string();
                            println!("\n=== Final Answer ===");
                            println!("{}", answer);
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
        events.contains(&"react_started".to_string()),
        "Should have react_started"
    );
    assert!(
        events.contains(&"react_thought".to_string()),
        "Should have at least one thought"
    );
    assert!(
        events.contains(&"react_complete".to_string()),
        "Should have react_complete"
    );

    // Verify calculator was used
    assert!(used_calculator, "Should have used calculator tool");

    // Verify final answer contains the correct result
    assert!(
        final_answer.contains("100"),
        "Final answer should contain 100 (25 * 4)"
    );
}

/// Test multi-step calculation
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_react_multi_step_calculation() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    let config = ReactConfig {
        max_iterations: 10,
        available_tools: vec!["calculator".to_string()],
        use_google_search: false,
        total_timeout: Duration::from_secs(90),
        ..Default::default()
    };

    let agent = ReactAgent::new(config).expect("Should create agent");
    let stream = agent.execute(
        "First calculate 15 * 8, then add 20 to that result",
        context,
    );
    futures_util::pin_mut!(stream);

    let mut calculator_uses = 0;
    let mut final_answer = String::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => {
                println!("[{}] {}", update.event_type, update.message);

                if update.event_type == "react_action" {
                    let tool = update.data.get("tool").and_then(|v| v.as_str());
                    if tool == Some("calculator") {
                        calculator_uses += 1;
                    }
                }

                if update.event_type == "react_complete" {
                    if let Some(answer) = update.data.get("final_answer").and_then(|v| v.as_str()) {
                        final_answer = answer.to_string();
                    }
                }
            }
            Err(e) => {
                panic!("Agent error: {:?}", e);
            }
        }
    }

    assert!(
        calculator_uses >= 1,
        "Should have used calculator at least once"
    );

    // Final answer should contain 140 (15*8=120, 120+20=140)
    assert!(
        final_answer.contains("140"),
        "Final answer should contain 140: got {}",
        final_answer
    );
}

/// Test ReAct event ordering
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_react_event_ordering() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    let config = ReactConfig {
        max_iterations: 5,
        available_tools: vec!["calculator".to_string()],
        use_google_search: false,
        total_timeout: Duration::from_secs(60),
        ..Default::default()
    };

    let agent = ReactAgent::new(config).expect("Should create agent");
    let stream = agent.execute("What is 10 + 5?", context);
    futures_util::pin_mut!(stream);

    let mut events: Vec<String> = Vec::new();

    while let Some(result) = stream.next().await {
        if let Ok(update) = result {
            events.push(update.event_type.clone());
        }
    }

    // Verify react_started is first
    assert_eq!(
        events.first(),
        Some(&"react_started".to_string()),
        "First event should be react_started"
    );

    // Verify react_complete is present
    assert!(
        events.contains(&"react_complete".to_string()),
        "Events should contain react_complete"
    );

    // Verify final_result is last
    assert_eq!(
        events.last(),
        Some(&"final_result".to_string()),
        "Last event should be final_result"
    );

    // Verify thought comes before action in each iteration
    let thought_positions: Vec<_> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| *e == "react_thought")
        .map(|(i, _)| i)
        .collect();

    let action_positions: Vec<_> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| *e == "react_action")
        .map(|(i, _)| i)
        .collect();

    for (thought_pos, action_pos) in thought_positions.iter().zip(action_positions.iter()) {
        assert!(
            thought_pos < action_pos,
            "Thought should come before action"
        );
    }
}

/// Test max_iterations path emits final_result per event contract
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_react_max_iterations_emits_final_result() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    let config = ReactConfig {
        max_iterations: 1,
        available_tools: vec!["calculator".to_string()],
        use_google_search: false,
        total_timeout: Duration::from_secs(60),
        ..Default::default()
    };

    let agent = ReactAgent::new(config).expect("Should create agent");
    let stream = agent.execute(
        "Calculate the factorial of 5, then multiply by 2, then add 100",
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

    // Verify react_max_iterations was emitted
    assert!(
        events.contains(&"react_max_iterations".to_string()),
        "Should emit react_max_iterations when hitting limit. Events: {:?}",
        events
    );

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

/// Test invalid config
#[tokio::test]
async fn test_react_invalid_config() {
    let config = ReactConfig {
        max_iterations: 0,
        ..Default::default()
    };

    let result = ReactAgent::new(config);
    assert!(result.is_err());
    assert!(matches!(result, Err(AgentError::InvalidConfig(_))));
}

/// Test empty tools config
#[tokio::test]
async fn test_react_empty_tools_config() {
    let config = ReactConfig {
        available_tools: vec![],
        ..Default::default()
    };

    let result = ReactAgent::new(config);
    assert!(result.is_err());
    assert!(matches!(result, Err(AgentError::InvalidConfig(_))));
}
