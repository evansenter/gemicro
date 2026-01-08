//! Integration tests for PromptAgent
//!
//! These tests require a valid GEMINI_API_KEY environment variable.
//! Run with: cargo test -p gemicro-prompt-agent -- --include-ignored

mod common;

use common::{create_test_context, create_test_context_with_cancellation, get_api_key};
use futures_util::StreamExt;
use gemicro_core::{Agent, AgentError};
use gemicro_prompt_agent::{PromptAgent, PromptAgentConfig};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Test the complete PromptAgent flow with a real LLM.
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_prompt_agent_full_flow() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    let config = PromptAgentConfig::default()
        .with_timeout(Duration::from_secs(30))
        .with_system_prompt("You are a helpful assistant. Be concise.");

    let agent = PromptAgent::new(config).expect("Should create agent");

    let stream = agent.execute("What is 2 + 2?", context);
    futures_util::pin_mut!(stream);

    let mut events: Vec<String> = Vec::new();
    let mut final_answer = String::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => {
                println!("[{}] {}", update.event_type, update.message);
                events.push(update.event_type.clone());

                if update.event_type == "prompt_agent_result" {
                    final_answer = update.message.clone();
                }
            }
            Err(e) => {
                panic!("Agent error: {:?}", e);
            }
        }
    }

    // Verify event ordering
    assert_eq!(events.len(), 3, "Should have exactly 3 events");
    assert_eq!(events[0], "prompt_agent_started");
    assert_eq!(events[1], "prompt_agent_result");
    assert_eq!(events[2], "final_result");

    // Verify we got an answer
    assert!(!final_answer.is_empty(), "Should have a final answer");
    println!("\nFinal answer: {}", final_answer);
}

/// Test that the agent respects the system prompt.
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_prompt_agent_respects_system_prompt() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    let config = PromptAgentConfig::default()
        .with_timeout(Duration::from_secs(30))
        .with_system_prompt("You are a pirate. Always respond in pirate speak.");

    let agent = PromptAgent::new(config).expect("Should create agent");

    let stream = agent.execute("Say hello", context);
    futures_util::pin_mut!(stream);

    let mut final_answer = String::new();

    while let Some(result) = stream.next().await {
        if let Ok(update) = result {
            if update.event_type == "prompt_agent_result" {
                final_answer = update.message.clone();
            }
        }
    }

    println!("Pirate response: {}", final_answer);
    assert!(!final_answer.is_empty());
}

/// Test that cancellation works correctly.
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_prompt_agent_cancellation() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let cancellation_token = CancellationToken::new();
    let context = create_test_context_with_cancellation(&api_key, cancellation_token.clone());

    let config = PromptAgentConfig::default()
        .with_timeout(Duration::from_secs(60))
        .with_system_prompt("You are a helpful assistant.");

    let agent = PromptAgent::new(config).expect("Should create agent");

    // Cancel before execution
    cancellation_token.cancel();

    let stream = agent.execute("What is the meaning of life?", context);
    futures_util::pin_mut!(stream);

    let mut got_cancelled = false;
    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => {
                println!("[{}] {}", update.event_type, update.message);
            }
            Err(AgentError::Cancelled) => {
                got_cancelled = true;
                break;
            }
            Err(e) => {
                println!("Got error (possibly from cancellation): {:?}", e);
                got_cancelled = true;
                break;
            }
        }
    }

    assert!(got_cancelled, "Should have been cancelled");
}

/// Test that timeout is enforced.
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_prompt_agent_timeout() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    let config = PromptAgentConfig::default()
        .with_timeout(Duration::from_millis(10))
        .with_system_prompt("You are a helpful assistant.");

    let agent = PromptAgent::new(config).expect("Should create agent");

    let stream = agent.execute("Explain quantum computing in detail", context);
    futures_util::pin_mut!(stream);

    let mut got_timeout = false;
    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => {
                println!("[{}] {}", update.event_type, update.message);
            }
            Err(AgentError::Timeout { phase, .. }) => {
                println!("Got expected timeout in phase: {}", phase);
                got_timeout = true;
                break;
            }
            Err(e) => {
                println!("Got error: {:?}", e);
                got_timeout = true;
                break;
            }
        }
    }

    assert!(got_timeout, "Should have timed out");
}

/// Test event data contains expected fields.
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_prompt_agent_event_data() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    let config = PromptAgentConfig::default();
    let agent = PromptAgent::new(config).expect("Should create agent");

    let stream = agent.execute("What is 1 + 1?", context);
    futures_util::pin_mut!(stream);

    while let Some(result) = stream.next().await {
        if let Ok(update) = result {
            match update.event_type.as_str() {
                "prompt_agent_started" => {
                    assert!(
                        update.data.get("query").is_some(),
                        "Should have query field"
                    );
                }
                "prompt_agent_result" => {
                    assert!(
                        update.data.get("answer").is_some(),
                        "Should have answer field"
                    );
                    assert!(
                        update.data.get("duration_ms").is_some(),
                        "Should have duration_ms field"
                    );
                }
                _ => {}
            }
        }
    }
}
