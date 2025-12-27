//! Scoring metrics for evaluation.
//!
//! Provides the [`Scorer`] trait and built-in implementations for common metrics.

/// Trait for evaluation scorers.
///
/// A scorer compares a predicted answer against the ground truth
/// and returns a score between 0.0 and 1.0.
///
/// # Example
///
/// ```
/// use gemicro_eval::Scorer;
///
/// struct LengthRatioScorer;
///
/// impl Scorer for LengthRatioScorer {
///     fn name(&self) -> &str {
///         "length_ratio"
///     }
///
///     fn score(&self, predicted: &str, ground_truth: &str) -> f64 {
///         let pred_len = predicted.chars().count() as f64;
///         let truth_len = ground_truth.chars().count() as f64;
///         if truth_len == 0.0 {
///             if pred_len == 0.0 { 1.0 } else { 0.0 }
///         } else {
///             (pred_len / truth_len).min(1.0)
///         }
///     }
/// }
/// ```
pub trait Scorer: Send + Sync {
    /// The name of this scorer (used in reports).
    fn name(&self) -> &str;

    /// Score the predicted answer against the ground truth.
    ///
    /// Returns a value between 0.0 (no match) and 1.0 (perfect match).
    fn score(&self, predicted: &str, ground_truth: &str) -> f64;
}

/// Contains scorer.
///
/// Returns 1.0 if the normalized ground truth is contained within
/// the normalized predicted answer, 0.0 otherwise.
///
/// This is useful when the agent may provide additional context
/// around the correct answer.
///
/// # Example
///
/// ```
/// use gemicro_eval::{Scorer, Contains};
///
/// let scorer = Contains;
///
/// // Ground truth is contained in prediction
/// assert_eq!(scorer.score("The capital of France is Paris.", "Paris"), 1.0);
///
/// // Ground truth is not contained
/// assert_eq!(scorer.score("London is a city", "Paris"), 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Contains;

impl Scorer for Contains {
    fn name(&self) -> &str {
        "contains"
    }

    fn score(&self, predicted: &str, ground_truth: &str) -> f64 {
        let pred_normalized = normalize(predicted);
        let truth_normalized = normalize(ground_truth);

        if pred_normalized.contains(&truth_normalized) {
            1.0
        } else {
            0.0
        }
    }
}

/// Normalize text for comparison.
///
/// - Lowercase
/// - Trim whitespace
/// - Collapse multiple spaces into single space
fn normalize(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// A collection of scorers for batch evaluation.
///
/// # Example
///
/// ```
/// use gemicro_eval::{Scorers, Contains};
///
/// let scorers = Scorers::default(); // Contains only
///
/// // Or custom set
/// let scorers = Scorers::new(vec![
///     Box::new(Contains),
/// ]);
/// ```
pub struct Scorers {
    scorers: Vec<Box<dyn Scorer>>,
}

impl Scorers {
    /// Create a new collection of scorers.
    pub fn new(scorers: Vec<Box<dyn Scorer>>) -> Self {
        Self { scorers }
    }

    /// Add a scorer to the collection.
    pub fn add(&mut self, scorer: impl Scorer + 'static) {
        self.scorers.push(Box::new(scorer));
    }

    /// Score a prediction against ground truth using all scorers.
    ///
    /// Returns a map of scorer_name -> score.
    pub fn score_all(
        &self,
        predicted: &str,
        ground_truth: &str,
    ) -> std::collections::HashMap<String, f64> {
        self.scorers
            .iter()
            .map(|s| (s.name().to_string(), s.score(predicted, ground_truth)))
            .collect()
    }

    /// Get the names of all scorers.
    pub fn names(&self) -> Vec<&str> {
        self.scorers.iter().map(|s| s.name()).collect()
    }
}

impl Default for Scorers {
    /// Default scorers: Contains only.
    fn default() -> Self {
        Self::new(vec![Box::new(Contains)])
    }
}

/// LLM-as-judge scorer using semantic comparison.
///
/// Uses an LLM to evaluate whether the predicted answer is semantically
/// correct compared to the ground truth. More flexible than exact matching
/// but slower and requires API calls.
///
/// # Example
///
/// ```no_run
/// use gemicro_eval::{Scorer, LlmJudgeScorer};
/// use gemicro_core::{LlmClient, LlmConfig};
/// use std::sync::Arc;
///
/// let genai = rust_genai::Client::builder("api-key".to_string()).build();
/// let llm = Arc::new(LlmClient::new(genai, LlmConfig::default()));
/// let scorer = LlmJudgeScorer::new(llm);
///
/// // Score is 1.0 if LLM judges semantically correct, 0.0 otherwise
/// // let score = scorer.score("The capital is Paris", "Paris");
/// ```
pub struct LlmJudgeScorer {
    llm: std::sync::Arc<gemicro_core::LlmClient>,
}

impl LlmJudgeScorer {
    /// Create a new LLM judge scorer with the given client.
    pub fn new(llm: std::sync::Arc<gemicro_core::LlmClient>) -> Self {
        Self { llm }
    }
}

impl Scorer for LlmJudgeScorer {
    fn name(&self) -> &str {
        "llm_judge"
    }

    fn score(&self, predicted: &str, ground_truth: &str) -> f64 {
        use crate::judge::{JudgeConfig, JudgeInput, LlmJudgeAgent};
        use futures_util::StreamExt;
        use gemicro_core::{Agent, AgentContext};

        // Create judge agent and input
        let agent = LlmJudgeAgent::new(JudgeConfig::default());
        let input = JudgeInput::new(predicted, ground_truth);
        let context = AgentContext::from_arc(self.llm.clone());

        // Run async agent from sync context using block_in_place
        // This is safe because we're in a tokio multi-threaded runtime
        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let stream = agent.execute(&input.to_query(), context);
                futures_util::pin_mut!(stream);

                while let Some(update) = stream.next().await {
                    if let Ok(update) = update {
                        if update.event_type == "judge_result" {
                            if let Some(correct) =
                                update.data.get("correct").and_then(|v| v.as_bool())
                            {
                                return if correct { 1.0 } else { 0.0 };
                            }
                        }
                    }
                }
                0.0 // Default to incorrect if no result
            })
        });

        result
    }
}

// Note: LlmJudgeScorer is Send but not Sync due to internal async machinery.
// However, the Scorer trait requires Send + Sync. Since we use block_in_place
// which handles the synchronization, this is safe.
unsafe impl Sync for LlmJudgeScorer {}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn test_normalize() {
        assert_eq!(normalize("  Hello   World  "), "hello world");
        assert_eq!(normalize("UPPERCASE"), "uppercase");
        assert_eq!(normalize(""), "");
    }

    // Parameterized Contains tests
    #[rstest]
    #[case::exact("Paris", "Paris", 1.0)]
    #[case::substring("The capital of France is Paris.", "Paris", 1.0)]
    #[case::case_insensitive("PARIS is great", "paris", 1.0)]
    #[case::not_found("London is great", "Paris", 0.0)]
    #[case::empty_truth("anything", "", 1.0)] // empty truth always contained
    #[case::empty_pred("", "Paris", 0.0)]
    fn test_contains(#[case] pred: &str, #[case] truth: &str, #[case] expected: f64) {
        assert_eq!(Contains.score(pred, truth), expected);
    }

    // Scorers collection tests
    #[test]
    fn test_scorers_default() {
        let scorers = Scorers::default();
        let names = scorers.names();
        assert!(names.contains(&"contains"));
        assert_eq!(names.len(), 1);
    }

    #[test]
    fn test_scorers_score_all() {
        let scorers = Scorers::default();
        let scores = scorers.score_all("Paris", "Paris");

        assert_eq!(scores.get("contains"), Some(&1.0));
    }

    #[test]
    fn test_scorers_custom() {
        let scorers = Scorers::new(vec![Box::new(Contains)]);
        let names = scorers.names();
        assert_eq!(names.len(), 1);
        assert!(names.contains(&"contains"));
    }
}
