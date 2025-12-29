//! Integration tests for LlmJudgeAgent
//!
//! These tests require a valid GEMINI_API_KEY environment variable.

use futures_util::StreamExt;
use gemicro_core::{Agent, AgentContext, LlmClient, LlmConfig};
use gemicro_judge::{JudgeConfig, JudgeInput, LlmJudgeAgent};
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
async fn test_llm_judge_correct_answer() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let client = create_test_client(&api_key);
    let context = AgentContext::new(client);

    let judge = LlmJudgeAgent::new(JudgeConfig::default());

    // Test: Correct answer
    let input = JudgeInput::new("Paris", "The capital of France is Paris");
    let query = input.to_query();

    let stream = judge.execute(&query, context);
    futures_util::pin_mut!(stream);

    let mut found_result = false;
    while let Some(update) = stream.next().await {
        let update = update.expect("Judge should not fail");
        println!("[{}] {}", update.event_type, update.message);

        if update.event_type == "judge_result" {
            found_result = true;
            let correct = update.data["correct"].as_bool();
            let reasoning = update.data["reasoning"].as_str();

            println!("  Correct: {:?}", correct);
            println!("  Reasoning: {:?}", reasoning);

            assert!(
                correct == Some(true),
                "Paris should be judged correct for 'capital of France', got: {:?}",
                correct
            );
            assert!(reasoning.is_some(), "Should have reasoning");
        }
    }

    assert!(found_result, "Should have received judge_result event");
}

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_llm_judge_incorrect_answer() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let client = create_test_client(&api_key);
    let context = AgentContext::new(client);

    let judge = LlmJudgeAgent::new(JudgeConfig::default());

    // Test: Incorrect answer
    let input = JudgeInput::new("London", "The capital of France is Paris");
    let query = input.to_query();

    let stream = judge.execute(&query, context);
    futures_util::pin_mut!(stream);

    let mut found_result = false;
    while let Some(update) = stream.next().await {
        let update = update.expect("Judge should not fail");

        if update.event_type == "judge_result" {
            found_result = true;
            let correct = update.data["correct"].as_bool();

            println!("London as capital of France - Correct: {:?}", correct);

            assert!(
                correct == Some(false),
                "London should be judged incorrect for 'capital of France', got: {:?}",
                correct
            );
        }
    }

    assert!(found_result, "Should have received judge_result event");
}

#[tokio::test]
#[ignore] // Requires GEMINI_API_KEY
async fn test_llm_judge_semantic_equivalence() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping test: GEMINI_API_KEY not set");
        return;
    };

    let client = create_test_client(&api_key);
    let context = AgentContext::new(client);

    let judge = LlmJudgeAgent::new(JudgeConfig::default());

    // Test: Semantically equivalent but differently worded
    let input = JudgeInput::new(
        "William Shakespeare wrote Romeo and Juliet",
        "Romeo and Juliet was written by Shakespeare",
    );
    let query = input.to_query();

    let stream = judge.execute(&query, context);
    futures_util::pin_mut!(stream);

    let mut found_result = false;
    while let Some(update) = stream.next().await {
        let update = update.expect("Judge should not fail");

        if update.event_type == "judge_result" {
            found_result = true;
            let correct = update.data["correct"].as_bool();

            println!("Semantic equivalence test - Correct: {:?}", correct);

            assert!(
                correct == Some(true),
                "Semantically equivalent answers should be judged correct, got: {:?}",
                correct
            );
        }
    }

    assert!(found_result, "Should have received judge_result event");
}
