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

/// Critique-based scorer using semantic comparison.
///
/// Uses the CritiqueAgent to evaluate whether the predicted answer is semantically
/// correct compared to the ground truth. More flexible than exact matching
/// but slower and requires API calls.
///
/// # Returns
///
/// - Score between 0.0 and 1.0 based on the critique verdict:
///   - `Pass`: 1.0
///   - `PassWithWarnings`: 0.75
///   - `NeedsRevision`: 0.25
///   - `Reject`: 0.0
/// - `NaN` if the evaluation itself failed (LLM error, timeout, etc.)
///
/// Use [`f64::is_nan()`] to check for evaluation failures.
///
/// # Panics
///
/// Panics if called outside a tokio multi-threaded runtime. This scorer
/// must be used within the evaluation harness or another async context.
///
/// # Example
///
/// ```no_run
/// use gemicro_eval::{Scorer, CritiqueScorer};
/// use gemicro_core::{LlmClient, LlmConfig};
/// use std::sync::Arc;
///
/// // Must be called within a tokio runtime
/// # let genai = genai_rs::Client::builder("api-key".to_string()).build().unwrap();
/// let llm = Arc::new(LlmClient::new(genai, LlmConfig::default()));
/// let scorer = CritiqueScorer::new(llm);
///
/// // Returns score based on verdict, or NaN if evaluation failed
/// // let score = scorer.score("The capital is Paris", "Paris");
/// // if score.is_nan() { /* handle evaluation failure */ }
/// ```
pub struct CritiqueScorer {
    llm: std::sync::Arc<gemicro_core::LlmClient>,
}

impl CritiqueScorer {
    /// Create a new critique scorer with the given client.
    pub fn new(llm: std::sync::Arc<gemicro_core::LlmClient>) -> Self {
        Self { llm }
    }
}

impl Scorer for CritiqueScorer {
    fn name(&self) -> &str {
        "critique"
    }

    fn score(&self, predicted: &str, ground_truth: &str) -> f64 {
        use gemicro_core::AgentContext;
        use gemicro_critique_agent::{
            CritiqueAgent, CritiqueAgentConfig, CritiqueCriteria, CritiqueInput,
        };

        // Handle empty predicted strings: an empty answer is always incorrect (score 0.0).
        // CritiqueAgent validates that content is non-empty and would return an error,
        // but for evaluation purposes, empty predictions should score 0.0, not NaN.
        if predicted.trim().is_empty() {
            return 0.0;
        }

        // Create critique agent with ground truth criteria
        let agent =
            CritiqueAgent::new(CritiqueAgentConfig::default()).expect("default config is valid");
        let input = CritiqueInput::new(predicted).with_criteria(CritiqueCriteria::GroundTruth {
            expected: ground_truth.into(),
        });
        let context = AgentContext::from_arc(self.llm.clone());

        // Run async agent from sync context using block_in_place.
        // Panics if called outside a tokio multi-threaded runtime.
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                match agent.critique(&input, context).await {
                    Ok(output) => output.to_score(),
                    Err(e) => {
                        log::error!("Critique failed: {:?}", e);
                        f64::NAN
                    }
                }
            })
        })
    }
}

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
