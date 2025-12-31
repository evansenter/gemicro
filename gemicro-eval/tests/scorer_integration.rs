//! Integration tests for LlmJudgeScorer.
//!
//! These tests require a valid GEMINI_API_KEY environment variable.
//! Run with: `cargo test -p gemicro-eval -- --include-ignored`

use gemicro_core::{LlmClient, LlmConfig};
use gemicro_eval::{LlmJudgeScorer, Scorer};
use std::sync::Arc;
use std::time::Duration;

fn get_api_key() -> Option<String> {
    std::env::var("GEMINI_API_KEY").ok()
}

fn create_test_scorer(api_key: &str) -> LlmJudgeScorer {
    let genai_client = rust_genai::Client::builder(api_key.to_string()).build();
    let config = LlmConfig::default()
        .with_timeout(Duration::from_secs(30))
        .with_max_tokens(1024)
        .with_temperature(0.7)
        .with_max_retries(1)
        .with_retry_base_delay_ms(500);
    let llm = Arc::new(LlmClient::new(genai_client, config));
    LlmJudgeScorer::new(llm)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore] // Requires GEMINI_API_KEY
async fn test_llm_judge_scorer_correct_answer() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let scorer = create_test_scorer(&api_key);

    // Score "Paris" against ground truth "Paris"
    let score = scorer.score("Paris", "Paris");

    assert!(
        !score.is_nan(),
        "Scorer should not return NaN for valid inputs"
    );
    assert!(
        (score - 1.0).abs() < f64::EPSILON,
        "Exact match should return 1.0, got: {score}"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore] // Requires GEMINI_API_KEY
async fn test_llm_judge_scorer_incorrect_answer() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let scorer = create_test_scorer(&api_key);

    // Score "London" against ground truth "Paris"
    let score = scorer.score("London", "Paris");

    assert!(
        !score.is_nan(),
        "Scorer should not return NaN for valid inputs"
    );
    assert!(
        score.abs() < f64::EPSILON,
        "Wrong answer should return 0.0, got: {score}"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore] // Requires GEMINI_API_KEY
async fn test_llm_judge_scorer_semantic_match() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let scorer = create_test_scorer(&api_key);

    // Score "The capital of France is Paris" against "Paris"
    let score = scorer.score("The capital of France is Paris", "Paris");

    assert!(
        !score.is_nan(),
        "Scorer should not return NaN for valid inputs"
    );
    assert!(
        (score - 1.0).abs() < f64::EPSILON,
        "Semantic match should return 1.0, got: {score}"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore] // Requires GEMINI_API_KEY
async fn test_llm_judge_scorer_semantic_equivalence() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let scorer = create_test_scorer(&api_key);

    // Semantically equivalent but differently worded
    let score = scorer.score(
        "William Shakespeare wrote Romeo and Juliet",
        "Romeo and Juliet was written by Shakespeare",
    );

    assert!(
        !score.is_nan(),
        "Scorer should not return NaN for valid inputs"
    );
    assert!(
        (score - 1.0).abs() < f64::EPSILON,
        "Semantically equivalent answers should return 1.0, got: {score}"
    );
}

#[test]
fn test_llm_judge_scorer_name() {
    // Unit test - doesn't need API key
    let genai_client = rust_genai::Client::builder("fake-key".to_string()).build();
    let llm = Arc::new(LlmClient::new(genai_client, LlmConfig::default()));
    let scorer = LlmJudgeScorer::new(llm);

    assert_eq!(scorer.name(), "llm_judge");
}
