//! Integration tests for CritiqueAgent (ground truth mode)
//!
//! These tests require a valid GEMINI_API_KEY environment variable.

use gemicro_core::{AgentContext, LlmClient, LlmConfig};
use gemicro_critique::{CritiqueAgent, CritiqueConfig, CritiqueCriteria, CritiqueInput};
use std::time::Duration;

fn get_api_key() -> Option<String> {
    std::env::var("GEMINI_API_KEY").ok()
}

fn create_test_client(api_key: &str) -> LlmClient {
    let genai_client = genai_rs::Client::builder(api_key.to_string())
        .build()
        .unwrap();
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

    // Use the typed critique() helper
    let output = agent
        .critique(&input, context)
        .await
        .expect("Critique should not fail");

    println!(
        "Verdict: {} (score: {:.2})",
        output.verdict,
        output.to_score()
    );

    assert!(
        output.verdict.is_passing(),
        "Paris should pass for 'capital of France', got: {}",
        output.verdict
    );
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

    let output = agent
        .critique(&input, context)
        .await
        .expect("Critique should not fail");

    println!(
        "London as capital of France - Verdict: {} (score: {:.2})",
        output.verdict,
        output.to_score()
    );

    assert!(
        !output.verdict.is_passing(),
        "London should fail for 'capital of France', got: {}",
        output.verdict
    );
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

    let output = agent
        .critique(&input, context)
        .await
        .expect("Critique should not fail");

    println!(
        "Semantic equivalence test - Verdict: {} (score: {:.2})",
        output.verdict,
        output.to_score()
    );

    assert!(
        output.verdict.is_passing(),
        "Semantically equivalent answers should pass, got: {}",
        output.verdict
    );
}
