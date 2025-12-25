//! Integration tests for EvalHarness retry logic.
//!
//! These tests use mock agents to verify retry behavior without needing
//! actual LLM calls.

use gemicro_core::{
    Agent, AgentContext, AgentError, AgentStream, AgentUpdate, LlmError, ResultMetadata,
};
use gemicro_eval::{
    Dataset, DatasetError, EvalConfig, EvalHarness, EvalProgress, EvalQuestion, Scorers,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A mock agent that succeeds after N failures.
struct MockAgent {
    name: String,
    fail_count: AtomicUsize,
    max_failures: usize,
}

impl MockAgent {
    fn new(max_failures: usize) -> Self {
        Self {
            name: "mock_agent".to_string(),
            fail_count: AtomicUsize::new(0),
            max_failures,
        }
    }

    /// Agent that always succeeds
    fn always_succeeds() -> Self {
        Self::new(0)
    }

    /// Agent that fails once then succeeds
    fn fails_once() -> Self {
        Self::new(1)
    }

    /// Agent that always fails
    fn always_fails() -> Self {
        Self::new(usize::MAX)
    }
}

impl Agent for MockAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "Mock agent for testing"
    }

    fn execute(&self, _query: &str, _context: AgentContext) -> AgentStream<'_> {
        let current_failures = self.fail_count.fetch_add(1, Ordering::SeqCst);
        let max_failures = self.max_failures;

        Box::pin(async_stream::try_stream! {
            if current_failures < max_failures {
                Err(AgentError::Llm(LlmError::Other("Mock failure".to_string())))?;
            } else {
                yield AgentUpdate::final_result(
                    "Mock answer".to_string(),
                    ResultMetadata {
                        total_tokens: 0,
                        tokens_unavailable_count: 0,
                        duration_ms: 0,
                        sub_queries_succeeded: 0,
                        sub_queries_failed: 0,
                    },
                );
            }
        })
    }
}

/// A mock dataset with a fixed set of questions.
struct MockDataset {
    questions: Vec<EvalQuestion>,
}

impl MockDataset {
    fn new(count: usize) -> Self {
        let questions = (0..count)
            .map(|i| EvalQuestion {
                id: format!("q{}", i),
                question: format!("Question {}?", i),
                ground_truth: "Mock answer".to_string(),
            })
            .collect();
        Self { questions }
    }
}

impl Dataset for MockDataset {
    fn name(&self) -> &str {
        "mock_dataset"
    }

    async fn load(&self, sample_size: Option<usize>) -> Result<Vec<EvalQuestion>, DatasetError> {
        let mut questions = self.questions.clone();
        if let Some(size) = sample_size {
            questions.truncate(size);
        }
        Ok(questions)
    }
}

/// Helper to create a mock LLM client (the agent doesn't actually use it).
fn mock_llm() -> gemicro_core::LlmClient {
    let genai_client = rust_genai::Client::builder("fake-key".to_string()).build();
    gemicro_core::LlmClient::new(genai_client, gemicro_core::LlmConfig::default())
}

#[tokio::test]
async fn test_harness_success_no_retries() {
    let agent = MockAgent::always_succeeds();
    let dataset = MockDataset::new(3);
    let harness = EvalHarness::new(EvalConfig::new().with_max_retries(0));

    let summary = harness
        .evaluate(&agent, &dataset, None, Scorers::default(), mock_llm())
        .await
        .unwrap();

    assert_eq!(summary.succeeded, 3);
    assert_eq!(summary.failed, 0);
    assert_eq!(summary.total_questions, 3);
}

#[tokio::test]
async fn test_harness_retry_success() {
    let agent = MockAgent::fails_once();
    let dataset = MockDataset::new(1);
    let harness = EvalHarness::new(EvalConfig::new().with_max_retries(1));

    let summary = harness
        .evaluate(&agent, &dataset, None, Scorers::default(), mock_llm())
        .await
        .unwrap();

    // Should succeed after 1 retry
    assert_eq!(summary.succeeded, 1);
    assert_eq!(summary.failed, 0);

    // Check that retry was recorded
    assert_eq!(summary.results[0].retries, 1);
}

#[tokio::test]
async fn test_harness_all_retries_exhausted() {
    let agent = MockAgent::always_fails();
    let dataset = MockDataset::new(1);
    let harness = EvalHarness::new(EvalConfig::new().with_max_retries(2));

    let summary = harness
        .evaluate(&agent, &dataset, None, Scorers::default(), mock_llm())
        .await
        .unwrap();

    // Should fail after exhausting retries
    assert_eq!(summary.succeeded, 0);
    assert_eq!(summary.failed, 1);

    // Check that all retries were attempted
    assert_eq!(summary.results[0].retries, 2);
    assert!(summary.results[0].error.is_some());
}

#[tokio::test]
async fn test_harness_progress_callback() {
    let agent = MockAgent::always_succeeds();
    let dataset = MockDataset::new(3);
    let harness = EvalHarness::new(EvalConfig::new().with_concurrency(1));

    let progress_events = Arc::new(std::sync::Mutex::new(Vec::new()));
    let events_clone = progress_events.clone();

    let summary = harness
        .evaluate_with_progress(
            &agent,
            &dataset,
            None,
            Scorers::default(),
            mock_llm(),
            move |progress| {
                events_clone.lock().unwrap().push(progress);
            },
        )
        .await
        .unwrap();

    let events = progress_events.lock().unwrap();

    // Should have 1 Started + 3 QuestionCompleted events
    assert_eq!(events.len(), 4);

    // First event should be Started
    match &events[0] {
        EvalProgress::Started { total } => assert_eq!(*total, 3),
        _ => panic!("Expected Started event"),
    }

    // Last event should be QuestionCompleted with completed == total
    match &events[3] {
        EvalProgress::QuestionCompleted {
            completed, total, ..
        } => {
            assert_eq!(*completed, 3);
            assert_eq!(*total, 3);
        }
        _ => panic!("Expected QuestionCompleted event"),
    }

    assert_eq!(summary.total_questions, 3);
}

#[tokio::test]
async fn test_harness_empty_dataset() {
    let agent = MockAgent::always_succeeds();
    let dataset = MockDataset::new(0);
    let harness = EvalHarness::default();

    let summary = harness
        .evaluate(&agent, &dataset, None, Scorers::default(), mock_llm())
        .await
        .unwrap();

    assert_eq!(summary.total_questions, 0);
    assert_eq!(summary.succeeded, 0);
    assert_eq!(summary.failed, 0);
}

#[tokio::test]
async fn test_harness_sample_size() {
    let agent = MockAgent::always_succeeds();
    let dataset = MockDataset::new(10);
    let harness = EvalHarness::default();

    let summary = harness
        .evaluate(&agent, &dataset, Some(3), Scorers::default(), mock_llm())
        .await
        .unwrap();

    assert_eq!(summary.total_questions, 3);
}
