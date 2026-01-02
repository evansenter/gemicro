//! Integration tests for CritiqueAgent (ground truth mode)
//!
//! These tests require a valid GEMINI_API_KEY environment variable.

use futures_util::StreamExt;
use gemicro_core::{Agent, AgentContext, LlmClient, LlmConfig};
use gemicro_critique::{CritiqueAgent, CritiqueConfig, CritiqueCriteria, CritiqueInput};
use std::time::Duration;

fn get_api_key() -> Option<String> {
    std::env::var("GEMINI_API_KEY").ok()
}

fn create_test_client(api_key: &str) -> LlmClient {
    let genai_client = rust_genai::Client::builder(api_key.to_string()).build();
    let config = LlmConfig::default()
        .with_timeout(Duration::from_secs(30))
        .with_max_tokens(1024)
        .with_temperature(0.7)
        .with_max_retries(1)
        .with_retry_base_delay_ms(500);
    LlmClient::new(genai_client, config)
}

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_critique_ground_truth_correct_answer() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let client = create_test_client(&api_key);
    let context = AgentContext::new(client);

    let agent = CritiqueAgent::new(CritiqueConfig::default()).unwrap();

    // Test: Correct answer
    let input = CritiqueInput::new("Paris").with_criteria(CritiqueCriteria::GroundTruth {
        expected: "The capital of France is Paris".into(),
    });
    let query = input.to_query();

    let stream = agent.execute(&query, context);
    futures_util::pin_mut!(stream);

    let mut found_result = false;
    while let Some(update) = stream.next().await {
        let update = update.expect("Critique should not fail");
        println!("[{}] {}", update.event_type, update.message);

        if update.event_type == "critique_result" {
            found_result = true;
            let verdict = update.data["verdict"].as_str();

            println!("  Verdict: {:?}", verdict);

            assert!(
                verdict == Some("Pass") || verdict == Some("PassWithWarnings"),
                "Paris should pass for 'capital of France', got: {:?}",
                verdict
            );
        }
    }

    assert!(found_result, "Should have received critique_result event");
}

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_critique_ground_truth_incorrect_answer() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let client = create_test_client(&api_key);
    let context = AgentContext::new(client);

    let agent = CritiqueAgent::new(CritiqueConfig::default()).unwrap();

    // Test: Incorrect answer
    let input = CritiqueInput::new("London").with_criteria(CritiqueCriteria::GroundTruth {
        expected: "The capital of France is Paris".into(),
    });
    let query = input.to_query();

    let stream = agent.execute(&query, context);
    futures_util::pin_mut!(stream);

    let mut found_result = false;
    while let Some(update) = stream.next().await {
        let update = update.expect("Critique should not fail");

        if update.event_type == "critique_result" {
            found_result = true;
            let verdict = update.data["verdict"].as_str();

            println!("London as capital of France - Verdict: {:?}", verdict);

            assert!(
                verdict == Some("NeedsRevision") || verdict == Some("Reject"),
                "London should fail for 'capital of France', got: {:?}",
                verdict
            );
        }
    }

    assert!(found_result, "Should have received critique_result event");
}

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_critique_ground_truth_semantic_equivalence() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let client = create_test_client(&api_key);
    let context = AgentContext::new(client);

    let agent = CritiqueAgent::new(CritiqueConfig::default()).unwrap();

    // Test: Semantically equivalent but differently worded
    let input = CritiqueInput::new("William Shakespeare wrote Romeo and Juliet").with_criteria(
        CritiqueCriteria::GroundTruth {
            expected: "Romeo and Juliet was written by Shakespeare".into(),
        },
    );
    let query = input.to_query();

    let stream = agent.execute(&query, context);
    futures_util::pin_mut!(stream);

    let mut found_result = false;
    while let Some(update) = stream.next().await {
        let update = update.expect("Critique should not fail");

        if update.event_type == "critique_result" {
            found_result = true;
            let verdict = update.data["verdict"].as_str();

            println!("Semantic equivalence test - Verdict: {:?}", verdict);

            assert!(
                verdict == Some("Pass") || verdict == Some("PassWithWarnings"),
                "Semantically equivalent answers should pass, got: {:?}",
                verdict
            );
        }
    }

    assert!(found_result, "Should have received critique_result event");
}
