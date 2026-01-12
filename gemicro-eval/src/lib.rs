//! # Gemicro Eval
//!
//! Evaluation framework for benchmarking gemicro agents against datasets.
//!
//! ## Overview
//!
//! `gemicro-eval` provides tools for systematic evaluation of AI agents:
//!
//! - **Datasets**: Load questions from HotpotQA or custom JSON files
//! - **Scorers**: Measure performance with Contains and LLM-as-judge metrics
//! - **Harness**: Batch execution with configurable concurrency and retries
//! - **Results**: Structured JSON output for analysis
//!
//! ## Architecture
//!
//! ```text
//! gemicro-core (agents, events)
//!     ↓
//! gemicro-runner (execution state, metrics)
//!     ↓
//! gemicro-eval (datasets, scorers, harness)  ← this crate
//! ```
//!
//! ## Quick Start
//!
//! ```no_run
//! use gemicro_eval::{EvalHarness, EvalConfig, HotpotQA, Scorers, Dataset};
//! use gemicro_core::{LlmClient, LlmConfig};
//! use gemicro_deep_research_agent::{DeepResearchAgent, DeepResearchAgentConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create agent
//! let agent = DeepResearchAgent::new(DeepResearchAgentConfig::default())?;
//!
//! // Create LLM client
//! let genai_client = genai_rs::Client::builder("api-key".to_string()).build()?;
//! let llm = LlmClient::new(genai_client, LlmConfig::default());
//!
//! // Load dataset (auto-downloads and caches HotpotQA)
//! let dataset = HotpotQA::new()?;
//!
//! // Run evaluation
//! let harness = EvalHarness::new(EvalConfig::default());
//! let summary = harness
//!     .evaluate(&agent, &dataset, Some(100), Scorers::default(), llm)
//!     .await?;
//!
//! // Output results
//! summary.print_summary();
//! summary.write_json(std::path::Path::new("results.json"))?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Custom Datasets
//!
//! Load from a JSON file with `id`, `question`, and `ground_truth` fields:
//!
//! ```no_run
//! use gemicro_eval::{JsonFileDataset, Dataset};
//! use std::path::PathBuf;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let dataset = JsonFileDataset::new(PathBuf::from("my_questions.json"));
//! let questions = dataset.load(None).await?;
//! println!("Loaded {} questions", questions.len());
//! # Ok(())
//! # }
//! ```
//!
//! ## Custom Scorers
//!
//! Implement the [`Scorer`] trait for custom metrics:
//!
//! ```
//! use gemicro_eval::Scorer;
//!
//! struct LengthScore;
//!
//! impl Scorer for LengthScore {
//!     fn name(&self) -> &str {
//!         "length_ratio"
//!     }
//!
//!     fn score(&self, predicted: &str, ground_truth: &str) -> f64 {
//!         let pred_len = predicted.len() as f64;
//!         let truth_len = ground_truth.len() as f64;
//!         if truth_len == 0.0 { 1.0 } else { (pred_len / truth_len).min(1.0) }
//!     }
//! }
//! ```

pub mod dataset;
pub mod harness;
pub mod results;
pub mod scorer;

// Re-export public API
pub use dataset::{
    Dataset, DatasetError, GSM8KSplit, HotpotQA, JsonFileDataset, TrajectoryDataset, GSM8K,
};
pub use harness::{EvalConfig, EvalError, EvalHarness, EvalProgress};
pub use results::{EvalQuestion, EvalResult, EvalSummary};
pub use scorer::{Contains, CritiqueScorer, Scorer, Scorers};
