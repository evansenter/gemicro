//! Evaluation results and summary types.
//!
//! This module contains the output types for evaluation runs,
//! designed for JSON serialization and programmatic consumption.

use gemicro_runner::ExecutionMetrics;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single question for evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalQuestion {
    /// Unique identifier for the question
    pub id: String,
    /// The question text to ask the agent
    pub question: String,
    /// The expected answer (ground truth)
    pub ground_truth: String,
}

/// Result of evaluating a single question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    /// Question identifier
    pub question_id: String,

    /// The question that was asked
    pub question: String,

    /// Expected answer
    pub ground_truth: String,

    /// Agent's predicted answer (None if agent failed)
    pub predicted: Option<String>,

    /// Scores from each scorer (scorer_name -> score)
    pub scores: HashMap<String, f64>,

    /// Execution metrics from the agent run
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<ExecutionMetrics>,

    /// Number of retry attempts made
    pub retries: usize,

    /// Error message if the question failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl EvalResult {
    /// Create a successful result.
    pub fn success(
        question: &EvalQuestion,
        predicted: String,
        scores: HashMap<String, f64>,
        metrics: ExecutionMetrics,
        retries: usize,
    ) -> Self {
        Self {
            question_id: question.id.clone(),
            question: question.question.clone(),
            ground_truth: question.ground_truth.clone(),
            predicted: Some(predicted),
            scores,
            metrics: Some(metrics),
            retries,
            error: None,
        }
    }

    /// Create a failed result.
    pub fn failure(question: &EvalQuestion, error: String, retries: usize) -> Self {
        Self {
            question_id: question.id.clone(),
            question: question.question.clone(),
            ground_truth: question.ground_truth.clone(),
            predicted: None,
            scores: HashMap::new(),
            metrics: None,
            retries,
            error: Some(error),
        }
    }

    /// Whether this result represents a successful evaluation.
    pub fn is_success(&self) -> bool {
        self.error.is_none() && self.predicted.is_some()
    }
}

/// Summary of an entire evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSummary {
    /// Name of the dataset used
    pub dataset_name: String,

    /// Name of the agent evaluated
    pub agent_name: String,

    /// Total number of questions evaluated
    pub total_questions: usize,

    /// Number of questions that succeeded
    pub succeeded: usize,

    /// Number of questions that failed
    pub failed: usize,

    /// Average score for each scorer across successful evaluations
    pub average_scores: HashMap<String, f64>,

    /// Individual results for each question
    pub results: Vec<EvalResult>,

    /// Total tokens used across all evaluations
    pub total_tokens: u32,

    /// Total duration of the evaluation
    #[serde(with = "duration_serde")]
    pub total_duration: std::time::Duration,
}

impl EvalSummary {
    /// Create a summary from evaluation results.
    pub fn from_results(
        dataset_name: String,
        agent_name: String,
        results: Vec<EvalResult>,
        total_duration: std::time::Duration,
    ) -> Self {
        let total_questions = results.len();
        let succeeded = results.iter().filter(|r| r.is_success()).count();
        let failed = total_questions - succeeded;

        // Calculate average scores
        let mut score_sums: HashMap<String, (f64, usize)> = HashMap::new();
        for result in results.iter().filter(|r| r.is_success()) {
            for (scorer_name, score) in &result.scores {
                let entry = score_sums.entry(scorer_name.clone()).or_insert((0.0, 0));
                entry.0 += score;
                entry.1 += 1;
            }
        }

        let average_scores: HashMap<String, f64> = score_sums
            .into_iter()
            .map(|(name, (sum, count))| (name, if count > 0 { sum / count as f64 } else { 0.0 }))
            .collect();

        // Sum total tokens
        let total_tokens: u32 = results
            .iter()
            .filter_map(|r| r.metrics.as_ref())
            .map(|m| m.total_tokens)
            .sum();

        Self {
            dataset_name,
            agent_name,
            total_questions,
            succeeded,
            failed,
            average_scores,
            results,
            total_tokens,
            total_duration,
        }
    }

    /// Get the average score for a specific scorer.
    ///
    /// Returns `None` if the scorer wasn't used or no questions succeeded.
    pub fn avg_score(&self, scorer_name: &str) -> Option<f64> {
        self.average_scores.get(scorer_name).copied()
    }

    /// Format the summary as a string.
    ///
    /// Returns the summary formatted for display. Use this when you need
    /// the summary as a string (e.g., for logging, testing, or custom output).
    pub fn format_summary(&self) -> String {
        use std::fmt::Write;

        let mut output = String::new();
        writeln!(output).unwrap();
        writeln!(output, "=== Evaluation Summary ===").unwrap();
        writeln!(output, "Dataset: {}", self.dataset_name).unwrap();
        writeln!(output, "Agent: {}", self.agent_name).unwrap();
        writeln!(output).unwrap();
        writeln!(
            output,
            "Questions: {} total, {} succeeded, {} failed",
            self.total_questions, self.succeeded, self.failed
        )
        .unwrap();
        writeln!(
            output,
            "Success rate: {:.1}%",
            if self.total_questions > 0 {
                (self.succeeded as f64 / self.total_questions as f64) * 100.0
            } else {
                0.0
            }
        )
        .unwrap();
        writeln!(output).unwrap();

        if !self.average_scores.is_empty() {
            writeln!(output, "Scores:").unwrap();
            let mut scores: Vec<_> = self.average_scores.iter().collect();
            scores.sort_by(|a, b| a.0.cmp(b.0));
            for (scorer, avg) in scores {
                writeln!(output, "  {}: {:.3}", scorer, avg).unwrap();
            }
            writeln!(output).unwrap();
        }

        writeln!(output, "Tokens: {}", self.total_tokens).unwrap();
        write!(
            output,
            "Duration: {:.1}s",
            self.total_duration.as_secs_f64()
        )
        .unwrap();
        output
    }

    /// Print a summary to stdout.
    ///
    /// Convenience wrapper around [`format_summary`](Self::format_summary).
    pub fn print_summary(&self) {
        println!("{}", self.format_summary());
    }

    /// Write the summary to a JSON file.
    pub fn write_json(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Recalculate average scores from results.
    ///
    /// Call this after manually adding scores to individual results
    /// (e.g., after running `LlmJudgeAgent` on each result).
    pub fn recalculate_averages(&mut self) {
        let mut score_sums: HashMap<String, (f64, usize)> = HashMap::new();
        for result in self.results.iter().filter(|r| r.is_success()) {
            for (scorer_name, score) in &result.scores {
                let entry = score_sums.entry(scorer_name.clone()).or_insert((0.0, 0));
                entry.0 += score;
                entry.1 += 1;
            }
        }

        self.average_scores = score_sums
            .into_iter()
            .map(|(name, (sum, count))| (name, if count > 0 { sum / count as f64 } else { 0.0 }))
            .collect();
    }
}

/// Custom serde for Duration to serialize as seconds (f64).
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error> {
        duration.as_secs_f64().serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Duration, D::Error> {
        let secs = f64::deserialize(deserializer)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_question() -> EvalQuestion {
        EvalQuestion {
            id: "q1".to_string(),
            question: "What is 2+2?".to_string(),
            ground_truth: "4".to_string(),
        }
    }

    #[test]
    fn test_eval_result_success() {
        let question = sample_question();
        let mut scores = HashMap::new();
        scores.insert("contains".to_string(), 1.0);

        let result =
            EvalResult::success(&question, "4".to_string(), scores, create_mock_metrics(), 0);

        assert!(result.is_success());
        assert_eq!(result.predicted, Some("4".to_string()));
        assert!(result.error.is_none());
    }

    #[test]
    fn test_eval_result_failure() {
        let question = sample_question();
        let result = EvalResult::failure(&question, "Timeout".to_string(), 1);

        assert!(!result.is_success());
        assert!(result.predicted.is_none());
        assert_eq!(result.error, Some("Timeout".to_string()));
    }

    #[test]
    fn test_eval_summary_from_results() {
        let question = sample_question();
        let mut scores = HashMap::new();
        scores.insert("contains".to_string(), 1.0);
        scores.insert("llm_judge".to_string(), 0.8);

        let results = vec![
            EvalResult::success(
                &question,
                "4".to_string(),
                scores.clone(),
                create_mock_metrics(),
                0,
            ),
            EvalResult::failure(&question, "Error".to_string(), 1),
        ];

        let summary = EvalSummary::from_results(
            "test_dataset".to_string(),
            "test_agent".to_string(),
            results,
            std::time::Duration::from_secs(10),
        );

        assert_eq!(summary.total_questions, 2);
        assert_eq!(summary.succeeded, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.average_scores.get("contains"), Some(&1.0));
        assert_eq!(summary.average_scores.get("llm_judge"), Some(&0.8));
    }

    #[test]
    fn test_eval_summary_serialization() {
        let summary = EvalSummary::from_results(
            "test".to_string(),
            "agent".to_string(),
            vec![],
            std::time::Duration::from_secs(5),
        );

        let json = serde_json::to_string(&summary).unwrap();
        let parsed: EvalSummary = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.dataset_name, "test");
        assert_eq!(parsed.total_duration.as_secs(), 5);
    }

    #[test]
    fn test_recalculate_averages_after_adding_scores() {
        let question = sample_question();
        let mut scores = HashMap::new();
        scores.insert("contains".to_string(), 1.0);

        let results = vec![
            EvalResult::success(
                &question,
                "4".to_string(),
                scores.clone(),
                create_mock_metrics(),
                0,
            ),
            EvalResult::success(
                &question,
                "four".to_string(),
                scores.clone(),
                create_mock_metrics(),
                0,
            ),
        ];

        let mut summary = EvalSummary::from_results(
            "test".to_string(),
            "agent".to_string(),
            results,
            std::time::Duration::from_secs(5),
        );

        // Verify initial state
        assert_eq!(summary.avg_score("contains"), Some(1.0));
        assert_eq!(summary.avg_score("llm_judge"), None);

        // Add llm_judge scores after the fact (simulating LlmJudgeAgent)
        summary.results[0]
            .scores
            .insert("llm_judge".to_string(), 1.0);
        summary.results[1]
            .scores
            .insert("llm_judge".to_string(), 0.0);

        // Recalculate averages
        summary.recalculate_averages();

        // Verify new score is included
        assert_eq!(summary.avg_score("llm_judge"), Some(0.5));
        // Existing scores are preserved
        assert_eq!(summary.avg_score("contains"), Some(1.0));
    }

    fn create_mock_metrics() -> ExecutionMetrics {
        use gemicro_runner::Phase;

        ExecutionMetrics {
            total_duration: std::time::Duration::from_secs(1),
            sequential_time: None,
            parallel_speedup: None,
            sub_queries_total: 0,
            sub_queries_succeeded: 0,
            sub_queries_failed: 0,
            sub_query_timings: vec![],
            total_tokens: 100,
            tokens_unavailable_count: 0,
            final_answer: Some("4".to_string()),
            completion_phase: Phase::Complete,
        }
    }

    #[test]
    fn test_format_summary() {
        let question = sample_question();
        let mut scores = HashMap::new();
        scores.insert("contains".to_string(), 1.0);

        let results = vec![EvalResult::success(
            &question,
            "4".to_string(),
            scores,
            create_mock_metrics(),
            0,
        )];

        let summary = EvalSummary::from_results(
            "test_dataset".to_string(),
            "test_agent".to_string(),
            results,
            std::time::Duration::from_secs(10),
        );

        let formatted = summary.format_summary();

        assert!(formatted.contains("=== Evaluation Summary ==="));
        assert!(formatted.contains("Dataset: test_dataset"));
        assert!(formatted.contains("Agent: test_agent"));
        assert!(formatted.contains("Questions: 1 total, 1 succeeded, 0 failed"));
        assert!(formatted.contains("Success rate: 100.0%"));
        assert!(formatted.contains("Scores:"));
        assert!(formatted.contains("contains: 1.000"));
        assert!(formatted.contains("Tokens: 100"));
        assert!(formatted.contains("Duration: 10.0s"));
    }
}
