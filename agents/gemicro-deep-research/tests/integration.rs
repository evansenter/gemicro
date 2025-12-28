//! Integration tests for DeepResearchAgent
//!
//! These tests require a valid GEMINI_API_KEY environment variable.
//! Run with: cargo test -p gemicro-deep-research -- --include-ignored

mod common;

use common::{create_test_context, create_test_context_with_cancellation, get_api_key};
use futures_util::StreamExt;
use gemicro_core::AgentError;
use gemicro_deep_research::{DeepResearchAgent, DeepResearchEventExt, ResearchConfig};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_deep_research_agent_full_flow() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    // Use minimal config for faster testing
    let config = ResearchConfig {
        min_sub_queries: 2,
        max_sub_queries: 3,
        continue_on_partial_failure: true,
        total_timeout: Duration::from_secs(120),
        ..Default::default()
    };

    let agent = DeepResearchAgent::new(config).expect("Should create agent");

    let stream = agent.execute(
        "What are the main benefits of the Rust programming language?",
        context,
    );
    futures_util::pin_mut!(stream);

    // Track event order
    let mut events: Vec<String> = Vec::new();
    let mut sub_query_count = 0;
    let mut sub_query_results = 0;
    let mut final_answer = String::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => {
                println!("[{}] {}", update.event_type, update.message);
                events.push(update.event_type.clone());

                match update.event_type.as_str() {
                    "decomposition_complete" => {
                        if let Some(sub_queries) = update.as_decomposition_complete() {
                            sub_query_count = sub_queries.len();
                            println!("  Sub-queries: {:?}", sub_queries);
                        }
                    }
                    "sub_query_completed" => {
                        sub_query_results += 1;
                        if let Some(result) = update.as_sub_query_completed() {
                            println!(
                                "  Sub-query {} completed ({} tokens)",
                                result.id, result.tokens_used
                            );
                        }
                    }
                    "sub_query_failed" => {
                        println!("  Sub-query failed: {:?}", update.data);
                    }
                    "final_result" => {
                        if let Some(result) = update.as_final_result() {
                            final_answer = result.answer.clone();
                            println!("\n=== Final Answer ===");
                            println!("{}", result.answer);
                            println!("\n=== Metadata ===");
                            println!("  Total tokens: {}", result.metadata.total_tokens);
                            println!(
                                "  Tokens unavailable: {}",
                                result.metadata.tokens_unavailable_count
                            );
                            println!("  Duration: {}ms", result.metadata.duration_ms);
                            println!(
                                "  Succeeded: {}, Failed: {}",
                                result.metadata.sub_queries_succeeded,
                                result.metadata.sub_queries_failed
                            );
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

    // Verify event order
    assert!(
        events.contains(&"decomposition_started".to_string()),
        "Should have decomposition_started"
    );
    assert!(
        events.contains(&"decomposition_complete".to_string()),
        "Should have decomposition_complete"
    );
    assert!(
        events.contains(&"synthesis_started".to_string()),
        "Should have synthesis_started"
    );
    assert!(
        events.contains(&"final_result".to_string()),
        "Should have final_result"
    );

    // Verify decomposition happened
    assert!(sub_query_count >= 2, "Should have at least 2 sub-queries");
    assert!(sub_query_count <= 3, "Should have at most 3 sub-queries");

    // Verify at least one sub-query succeeded
    assert!(
        sub_query_results > 0,
        "At least one sub-query should succeed"
    );

    // Verify final answer is not empty
    assert!(!final_answer.is_empty(), "Final answer should not be empty");
    assert!(
        final_answer.len() > 50,
        "Final answer should be substantive"
    );
}

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_agent_event_ordering() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let context = create_test_context(&api_key);

    let config = ResearchConfig {
        min_sub_queries: 2,
        max_sub_queries: 2,
        continue_on_partial_failure: true,
        total_timeout: Duration::from_secs(120),
        ..Default::default()
    };

    let agent = DeepResearchAgent::new(config).unwrap();
    let stream = agent.execute("What is 2+2 and what is 3+3?", context);
    futures_util::pin_mut!(stream);

    let mut events: Vec<String> = Vec::new();
    while let Some(result) = stream.next().await {
        if let Ok(update) = result {
            events.push(update.event_type.clone());
        }
    }

    // Verify strict ordering of phases
    let decomp_start = events.iter().position(|e| e == "decomposition_started");
    let decomp_complete = events.iter().position(|e| e == "decomposition_complete");
    let synth_start = events.iter().position(|e| e == "synthesis_started");
    let final_result = events.iter().position(|e| e == "final_result");

    assert!(decomp_start.is_some(), "Should have decomposition_started");
    assert!(
        decomp_complete.is_some(),
        "Should have decomposition_complete"
    );
    assert!(synth_start.is_some(), "Should have synthesis_started");
    assert!(final_result.is_some(), "Should have final_result");

    // Verify order: decomp_start < decomp_complete < synth_start < final_result
    let ds = decomp_start.unwrap();
    let dc = decomp_complete.unwrap();
    let ss = synth_start.unwrap();
    let fr = final_result.unwrap();

    assert!(
        ds < dc,
        "decomposition_started should come before decomposition_complete"
    );
    assert!(
        dc < ss,
        "decomposition_complete should come before synthesis_started"
    );
    assert!(ss < fr, "synthesis_started should come before final_result");

    // Verify sub_query_started events come after decomposition_complete
    let first_sub_query_started = events.iter().position(|e| e == "sub_query_started");
    if let Some(sq_start) = first_sub_query_started {
        assert!(
            sq_start > dc,
            "sub_query_started should come after decomposition_complete"
        );
        assert!(
            sq_start < ss,
            "sub_query_started should come before synthesis_started"
        );
    }
}

#[tokio::test]
async fn test_agent_invalid_config() {
    let config = ResearchConfig {
        min_sub_queries: 10,
        max_sub_queries: 5,
        ..Default::default()
    };

    let result = DeepResearchAgent::new(config);
    assert!(result.is_err());
    assert!(matches!(result, Err(AgentError::InvalidConfig(_))));
}

/// Test that cancellation works correctly and returns AgentError::Cancelled
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_cancellation_during_execution() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let cancellation_token = CancellationToken::new();
    let context = create_test_context_with_cancellation(&api_key, cancellation_token.clone());

    let config = ResearchConfig {
        min_sub_queries: 3,
        max_sub_queries: 5,
        total_timeout: Duration::from_secs(120),
        ..Default::default()
    };

    let agent = DeepResearchAgent::new(config).unwrap();
    let stream = agent.execute(
        "Explain the history, current state, and future of quantum computing",
        context,
    );
    futures_util::pin_mut!(stream);

    let mut seen_decomposition_started = false;
    let mut seen_sub_query_started = false;
    let mut cancelled_correctly = false;

    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => {
                println!("[{}] {}", update.event_type, update.message);

                if update.event_type == "decomposition_started" {
                    seen_decomposition_started = true;
                }

                if update.event_type == "sub_query_started" {
                    seen_sub_query_started = true;
                    println!("Triggering cancellation...");
                    cancellation_token.cancel();
                }
            }
            Err(AgentError::Cancelled) => {
                println!("Received AgentError::Cancelled as expected");
                cancelled_correctly = true;
                break;
            }
            Err(e) => {
                eprintln!(
                    "Test skipped due to LLM error (not a cancellation issue): {:?}",
                    e
                );
                return;
            }
        }
    }

    assert!(
        seen_decomposition_started,
        "Should have seen decomposition_started before cancellation"
    );
    assert!(
        seen_sub_query_started,
        "Should have seen sub_query_started before cancellation"
    );
    assert!(
        cancelled_correctly,
        "Should have received AgentError::Cancelled"
    );
}

/// Test that pre-cancelled token causes early termination
#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_immediate_cancellation() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let cancellation_token = CancellationToken::new();
    cancellation_token.cancel();

    let context = create_test_context_with_cancellation(&api_key, cancellation_token);

    let config = ResearchConfig {
        min_sub_queries: 2,
        max_sub_queries: 3,
        ..Default::default()
    };

    let agent = DeepResearchAgent::new(config).unwrap();
    let stream = agent.execute("What is 2+2?", context);
    futures_util::pin_mut!(stream);

    let mut got_cancelled = false;
    let mut event_count = 0;

    while let Some(result) = stream.next().await {
        event_count += 1;
        match result {
            Ok(update) => {
                println!("[{}] {}", update.event_type, update.message);
                assert!(
                    update.event_type == "decomposition_started",
                    "Should only see decomposition_started before cancellation, got {}",
                    update.event_type
                );
            }
            Err(AgentError::Cancelled) => {
                println!("Got AgentError::Cancelled");
                got_cancelled = true;
                break;
            }
            Err(e) => {
                panic!("Unexpected error: {:?}", e);
            }
        }
    }

    assert!(got_cancelled, "Should have received AgentError::Cancelled");
    assert!(
        event_count <= 2,
        "Should cancel quickly (got {} events)",
        event_count
    );
}
