//! Evaluation harness for batch execution.
//!
//! The [`EvalHarness`] orchestrates running an agent against a dataset,
//! managing concurrency, retries, and aggregating results.

use crate::dataset::{Dataset, DatasetError};
use crate::results::{EvalQuestion, EvalResult, EvalSummary};
use crate::scorer::Scorers;
use futures_util::stream::{self, StreamExt};
use gemicro_core::{Agent, AgentContext, AgentError, LlmClient};
use gemicro_runner::AgentRunner;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

/// Errors that can occur during evaluation.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EvalError {
    /// Failed to load dataset
    #[error("Dataset error: {0}")]
    Dataset(#[from] DatasetError),

    /// All questions failed
    #[error("All {0} questions failed during evaluation")]
    AllFailed(usize),
}

/// Progress events emitted during evaluation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum EvalProgress {
    /// Dataset loaded, evaluation starting.
    Started {
        /// Total number of questions to evaluate.
        total: usize,
    },
    /// A question evaluation completed (success or failure).
    QuestionCompleted {
        /// Number of questions completed so far.
        completed: usize,
        /// Total number of questions.
        total: usize,
        /// Whether this question succeeded.
        success: bool,
    },
}

/// Configuration for the evaluation harness.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EvalConfig {
    /// Maximum number of concurrent evaluations (default: 5)
    pub concurrency: usize,

    /// Maximum retry attempts for failed questions (default: 1).
    ///
    /// This is the number of *additional* attempts after the initial try.
    /// For example, `max_retries = 1` means each question gets up to 2 total
    /// attempts (1 initial + 1 retry). `max_retries = 0` means no retries.
    pub max_retries: usize,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            concurrency: 5,
            max_retries: 1,
        }
    }
}

impl EvalConfig {
    /// Create a new configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the concurrency limit.
    #[must_use]
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1); // At least 1
        self
    }

    /// Set the maximum retry count.
    ///
    /// This is the number of *additional* attempts after the initial try.
    /// For example, `max_retries = 1` means each question gets up to 2 total
    /// attempts (1 initial + 1 retry).
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }
}

/// Evaluation harness for benchmarking agents.
///
/// Manages batch execution with configurable concurrency and retries,
/// scoring results against ground truth using multiple metrics.
///
/// # Example
///
/// ```no_run
/// use gemicro_eval::{EvalHarness, EvalConfig, HotpotQA, Scorers};
/// use gemicro_core::{LlmClient, LlmConfig};
/// use gemicro_deep_research::{DeepResearchAgent, ResearchConfig};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create agent and LLM client
/// let agent = DeepResearchAgent::new(ResearchConfig::default())?;
/// let genai_client = rust_genai::Client::builder("api-key".to_string()).build();
/// let llm = LlmClient::new(genai_client, LlmConfig::default());
///
/// // Create harness
/// let harness = EvalHarness::new(EvalConfig::default());
///
/// // Run evaluation
/// let dataset = HotpotQA::new()?;
/// let summary = harness
///     .evaluate(&agent, &dataset, Some(10), Scorers::default(), llm)
///     .await?;
///
/// summary.print_summary();
/// # Ok(())
/// # }
/// ```
pub struct EvalHarness {
    config: EvalConfig,
    runner: AgentRunner,
}

impl EvalHarness {
    /// Create a new evaluation harness.
    pub fn new(config: EvalConfig) -> Self {
        Self {
            config,
            runner: AgentRunner::new(),
        }
    }

    /// Run evaluation against a dataset.
    ///
    /// # Arguments
    ///
    /// * `agent` - The agent to evaluate
    /// * `dataset` - The dataset to use
    /// * `sample_size` - Optional limit on questions to evaluate
    /// * `scorers` - Scoring metrics to apply
    /// * `llm` - LLM client for agent execution
    ///
    /// # Returns
    ///
    /// An `EvalSummary` with aggregated results and per-question details.
    pub async fn evaluate<D>(
        &self,
        agent: &dyn Agent,
        dataset: &D,
        sample_size: Option<usize>,
        scorers: Scorers,
        llm: LlmClient,
    ) -> Result<EvalSummary, EvalError>
    where
        D: Dataset,
    {
        let start_time = Instant::now();

        // Load questions
        let questions = dataset.load(sample_size).await?;
        let total_questions = questions.len();

        if questions.is_empty() {
            return Ok(EvalSummary::from_results(
                dataset.name().to_string(),
                agent.name().to_string(),
                vec![],
                start_time.elapsed(),
            ));
        }

        log::info!(
            "Evaluating {} questions with concurrency {}",
            total_questions,
            self.config.concurrency
        );

        // Shared state
        let llm = Arc::new(llm);
        let scorers = Arc::new(scorers);

        // Process questions with bounded concurrency
        let results: Vec<EvalResult> = stream::iter(questions)
            .map(|question| {
                let llm = llm.clone();
                let scorers = scorers.clone();
                let max_retries = self.config.max_retries;
                let runner = self.runner;

                async move {
                    evaluate_question(agent, question, &scorers, &llm, &runner, max_retries).await
                }
            })
            .buffer_unordered(self.config.concurrency)
            .collect()
            .await;

        let summary = EvalSummary::from_results(
            dataset.name().to_string(),
            agent.name().to_string(),
            results,
            start_time.elapsed(),
        );

        Ok(summary)
    }

    /// Run evaluation with progress callbacks.
    ///
    /// Same as [`evaluate`](Self::evaluate), but calls the provided callback
    /// with progress events as the evaluation proceeds.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gemicro_eval::{EvalHarness, EvalConfig, EvalProgress, HotpotQA, Scorers};
    /// use gemicro_core::{LlmClient, LlmConfig};
    /// use gemicro_deep_research::{DeepResearchAgent, ResearchConfig};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let agent = DeepResearchAgent::new(ResearchConfig::default())?;
    /// let genai_client = rust_genai::Client::builder("api-key".to_string()).build();
    /// let llm = LlmClient::new(genai_client, LlmConfig::default());
    /// let harness = EvalHarness::new(EvalConfig::default());
    /// let dataset = HotpotQA::new()?;
    ///
    /// let summary = harness
    ///     .evaluate_with_progress(&agent, &dataset, Some(10), Scorers::default(), llm, |progress| {
    ///         match progress {
    ///             EvalProgress::Started { total } => println!("Starting {} questions", total),
    ///             EvalProgress::QuestionCompleted { completed, total, success } => {
    ///                 println!("[{}/{}] {}", completed, total, if success { "✓" } else { "✗" });
    ///             }
    ///             _ => {} // Handle future variants
    ///         }
    ///     })
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn evaluate_with_progress<D, F>(
        &self,
        agent: &dyn Agent,
        dataset: &D,
        sample_size: Option<usize>,
        scorers: Scorers,
        llm: LlmClient,
        on_progress: F,
    ) -> Result<EvalSummary, EvalError>
    where
        D: Dataset,
        F: Fn(EvalProgress) + Send + Sync,
    {
        let start_time = Instant::now();

        // Load questions
        let questions = dataset.load(sample_size).await?;
        let total_questions = questions.len();

        if questions.is_empty() {
            return Ok(EvalSummary::from_results(
                dataset.name().to_string(),
                agent.name().to_string(),
                vec![],
                start_time.elapsed(),
            ));
        }

        // Emit started event
        on_progress(EvalProgress::Started {
            total: total_questions,
        });

        log::info!(
            "Evaluating {} questions with concurrency {}",
            total_questions,
            self.config.concurrency
        );

        // Shared state
        let llm = Arc::new(llm);
        let scorers = Arc::new(scorers);
        let completed = Arc::new(AtomicUsize::new(0));
        let on_progress = Arc::new(on_progress);

        // Process questions with bounded concurrency
        let results: Vec<EvalResult> = stream::iter(questions)
            .map(|question| {
                let llm = llm.clone();
                let scorers = scorers.clone();
                let max_retries = self.config.max_retries;
                let runner = self.runner;
                let completed = completed.clone();
                let on_progress = on_progress.clone();

                async move {
                    let result =
                        evaluate_question(agent, question, &scorers, &llm, &runner, max_retries)
                            .await;
                    let count = completed.fetch_add(1, Ordering::SeqCst) + 1;
                    on_progress(EvalProgress::QuestionCompleted {
                        completed: count,
                        total: total_questions,
                        success: result.is_success(),
                    });
                    result
                }
            })
            .buffer_unordered(self.config.concurrency)
            .collect()
            .await;

        let summary = EvalSummary::from_results(
            dataset.name().to_string(),
            agent.name().to_string(),
            results,
            start_time.elapsed(),
        );

        Ok(summary)
    }
}

impl Default for EvalHarness {
    fn default() -> Self {
        Self::new(EvalConfig::default())
    }
}

/// Evaluate a single question with retries.
async fn evaluate_question(
    agent: &dyn Agent,
    question: EvalQuestion,
    scorers: &Scorers,
    llm: &Arc<LlmClient>,
    runner: &AgentRunner,
    max_retries: usize,
) -> EvalResult {
    let mut last_error = String::new();
    let mut retries = 0;

    while retries <= max_retries {
        let context = AgentContext::from_arc(Arc::clone(llm));

        match runner
            .execute_with_tracking(agent, &question.question, context, |_, _| {})
            .await
        {
            Ok(metrics) => {
                if let Some(answer) = &metrics.final_answer {
                    let scores = scorers.score_all(answer, &question.ground_truth);
                    return EvalResult::success(
                        &question,
                        answer.clone(),
                        scores,
                        metrics,
                        retries,
                    );
                } else {
                    // Agent completed but no answer produced
                    last_error = "Agent completed but produced no answer".to_string();
                }
            }
            Err(e) => {
                last_error = format_agent_error(&e);
                log::warn!(
                    "Question {} attempt {}/{} failed: {}",
                    question.id,
                    retries + 1,
                    max_retries + 1,
                    last_error
                );
            }
        }

        retries += 1;
    }

    EvalResult::failure(&question, last_error, retries.saturating_sub(1))
}

/// Format an AgentError for logging.
fn format_agent_error(e: &AgentError) -> String {
    match e {
        AgentError::Timeout {
            elapsed_ms,
            timeout_ms,
            phase,
        } => {
            format!(
                "Timeout in {} phase ({}ms / {}ms limit)",
                phase, elapsed_ms, timeout_ms
            )
        }
        AgentError::AllSubQueriesFailed => "All sub-queries failed".to_string(),
        AgentError::Llm(llm_err) => format!("LLM error: {}", llm_err),
        AgentError::InvalidConfig(msg) => format!("Invalid config: {}", msg),
        _ => format!("{}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_config_default() {
        let config = EvalConfig::default();
        assert_eq!(config.concurrency, 5);
        assert_eq!(config.max_retries, 1);
    }

    #[test]
    fn test_eval_config_builder() {
        let config = EvalConfig::new().with_concurrency(10).with_max_retries(3);

        assert_eq!(config.concurrency, 10);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_eval_config_min_concurrency() {
        let config = EvalConfig::new().with_concurrency(0);
        assert_eq!(config.concurrency, 1); // Minimum is 1
    }

    #[test]
    fn test_harness_default() {
        let harness = EvalHarness::default();
        assert_eq!(harness.config.concurrency, 5);
    }

    // Integration tests would require a mock agent, which is complex.
    // See gemicro-runner for the mock agent pattern if needed.
}
