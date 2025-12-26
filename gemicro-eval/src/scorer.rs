//! Scoring metrics for evaluation.
//!
//! Provides the [`Scorer`] trait and built-in implementations for common metrics.

use std::collections::HashSet;

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

/// Exact match scorer (case-insensitive).
///
/// Returns 1.0 if the normalized predicted answer exactly matches
/// the normalized ground truth, 0.0 otherwise.
///
/// Normalization includes:
/// - Lowercasing
/// - Trimming whitespace
/// - Collapsing multiple spaces
///
/// # Example
///
/// ```
/// use gemicro_eval::{Scorer, ExactMatch};
///
/// let scorer = ExactMatch;
/// assert_eq!(scorer.score("Paris", "paris"), 1.0);
/// assert_eq!(scorer.score("London", "Paris"), 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct ExactMatch;

impl Scorer for ExactMatch {
    fn name(&self) -> &str {
        "exact_match"
    }

    fn score(&self, predicted: &str, ground_truth: &str) -> f64 {
        if normalize(predicted) == normalize(ground_truth) {
            1.0
        } else {
            0.0
        }
    }
}

/// F1 score at the word level (set-based).
///
/// Computes the harmonic mean of precision and recall based on unique word overlap.
/// Words are extracted after normalization (lowercase, split on whitespace).
/// Duplicate words are counted only once (set-based, not bag-of-words).
///
/// # Example
///
/// ```
/// use gemicro_eval::{Scorer, F1Score};
///
/// let scorer = F1Score;
///
/// // Perfect match
/// assert_eq!(scorer.score("the cat sat", "the cat sat"), 1.0);
///
/// // Partial overlap: {cat, dog} vs {cat, sat}
/// // overlap = 1, pred_size = 2, truth_size = 2
/// // Precision = 1/2, Recall = 1/2, F1 = 0.5
/// let score = scorer.score("cat dog", "cat sat");
/// assert!((score - 0.5).abs() < 0.01);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct F1Score;

impl Scorer for F1Score {
    fn name(&self) -> &str {
        "f1"
    }

    fn score(&self, predicted: &str, ground_truth: &str) -> f64 {
        let pred_words = words(predicted);
        let truth_words = words(ground_truth);

        if pred_words.is_empty() && truth_words.is_empty() {
            return 1.0;
        }
        if pred_words.is_empty() || truth_words.is_empty() {
            return 0.0;
        }

        let pred_set: HashSet<_> = pred_words.iter().collect();
        let truth_set: HashSet<_> = truth_words.iter().collect();

        let overlap = pred_set.intersection(&truth_set).count() as f64;

        // Use set sizes for consistency (pure set-based F1)
        let precision = overlap / pred_set.len() as f64;
        let recall = overlap / truth_set.len() as f64;

        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }
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

/// Extract words from text.
fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|w| {
            // Remove common punctuation from word boundaries
            w.trim_matches(|c: char| c.is_ascii_punctuation())
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// A collection of scorers for batch evaluation.
///
/// # Example
///
/// ```
/// use gemicro_eval::{Scorers, ExactMatch, F1Score, Contains};
///
/// let scorers = Scorers::default(); // EM, F1, Contains
///
/// // Or custom set
/// let scorers = Scorers::new(vec![
///     Box::new(ExactMatch),
///     Box::new(F1Score),
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
    /// Default scorers: ExactMatch, F1Score, Contains.
    fn default() -> Self {
        Self::new(vec![
            Box::new(ExactMatch),
            Box::new(F1Score),
            Box::new(Contains),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        assert_eq!(normalize("  Hello   World  "), "hello world");
        assert_eq!(normalize("UPPERCASE"), "uppercase");
        assert_eq!(normalize(""), "");
    }

    #[test]
    fn test_words() {
        assert_eq!(words("Hello, World!"), vec!["hello", "world"]);
        assert_eq!(words("  one   two  three  "), vec!["one", "two", "three"]);
        assert_eq!(words(""), Vec::<String>::new());
    }

    // ExactMatch tests
    #[test]
    fn test_exact_match_identical() {
        let scorer = ExactMatch;
        assert_eq!(scorer.score("Paris", "Paris"), 1.0);
    }

    #[test]
    fn test_exact_match_case_insensitive() {
        let scorer = ExactMatch;
        assert_eq!(scorer.score("PARIS", "paris"), 1.0);
        assert_eq!(scorer.score("PaRiS", "paris"), 1.0);
    }

    #[test]
    fn test_exact_match_whitespace() {
        let scorer = ExactMatch;
        assert_eq!(scorer.score("  Paris  ", "Paris"), 1.0);
        assert_eq!(scorer.score("New   York", "new york"), 1.0);
    }

    #[test]
    fn test_exact_match_different() {
        let scorer = ExactMatch;
        assert_eq!(scorer.score("London", "Paris"), 0.0);
    }

    #[test]
    fn test_exact_match_empty() {
        let scorer = ExactMatch;
        assert_eq!(scorer.score("", ""), 1.0);
        assert_eq!(scorer.score("Paris", ""), 0.0);
        assert_eq!(scorer.score("", "Paris"), 0.0);
    }

    // F1 tests
    #[test]
    fn test_f1_identical() {
        let scorer = F1Score;
        assert_eq!(scorer.score("the cat sat", "the cat sat"), 1.0);
    }

    #[test]
    fn test_f1_partial_overlap() {
        let scorer = F1Score;
        // pred: [cat, dog], truth: [cat, sat]
        // overlap: [cat] = 1
        // precision = 1/2, recall = 1/2, F1 = 0.5
        let score = scorer.score("cat dog", "cat sat");
        assert!((score - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_f1_no_overlap() {
        let scorer = F1Score;
        assert_eq!(scorer.score("apple banana", "cat dog"), 0.0);
    }

    #[test]
    fn test_f1_empty() {
        let scorer = F1Score;
        assert_eq!(scorer.score("", ""), 1.0);
        assert_eq!(scorer.score("word", ""), 0.0);
        assert_eq!(scorer.score("", "word"), 0.0);
    }

    #[test]
    fn test_f1_case_insensitive() {
        let scorer = F1Score;
        assert_eq!(scorer.score("THE CAT", "the cat"), 1.0);
    }

    // Contains tests
    #[test]
    fn test_contains_exact() {
        let scorer = Contains;
        assert_eq!(scorer.score("Paris", "Paris"), 1.0);
    }

    #[test]
    fn test_contains_substring() {
        let scorer = Contains;
        assert_eq!(
            scorer.score("The capital of France is Paris.", "Paris"),
            1.0
        );
    }

    #[test]
    fn test_contains_case_insensitive() {
        let scorer = Contains;
        assert_eq!(scorer.score("PARIS is great", "paris"), 1.0);
    }

    #[test]
    fn test_contains_not_found() {
        let scorer = Contains;
        assert_eq!(scorer.score("London is great", "Paris"), 0.0);
    }

    #[test]
    fn test_contains_empty() {
        let scorer = Contains;
        // Empty ground truth is always contained
        assert_eq!(scorer.score("anything", ""), 1.0);
        assert_eq!(scorer.score("", "Paris"), 0.0);
    }

    // Scorers collection tests
    #[test]
    fn test_scorers_default() {
        let scorers = Scorers::default();
        let names = scorers.names();
        assert!(names.contains(&"exact_match"));
        assert!(names.contains(&"f1"));
        assert!(names.contains(&"contains"));
    }

    #[test]
    fn test_scorers_score_all() {
        let scorers = Scorers::default();
        let scores = scorers.score_all("Paris", "Paris");

        assert_eq!(scores.get("exact_match"), Some(&1.0));
        assert_eq!(scores.get("f1"), Some(&1.0));
        assert_eq!(scores.get("contains"), Some(&1.0));
    }

    #[test]
    fn test_scorers_custom() {
        let scorers = Scorers::new(vec![Box::new(ExactMatch)]);
        let names = scorers.names();
        assert_eq!(names.len(), 1);
        assert!(names.contains(&"exact_match"));
    }
}
