//! Evaluation CLI for benchmarking gemicro agents.
//!
//! Run evaluations against HotpotQA or custom JSON datasets.

use clap::Parser;
use gemicro_core::{LlmClient, LlmConfig};
use gemicro_deep_research::{DeepResearchAgent, ResearchConfig};
use gemicro_eval::{
    Contains, CritiqueScorer, Dataset, EvalConfig, EvalHarness, EvalProgress, EvalSummary,
    HotpotQA, JsonFileDataset, Scorers, GSM8K,
};
use gemicro_react::{ReactAgent, ReactConfig};
use gemicro_runner::AgentRegistry;
use gemicro_simple_qa::{SimpleQaAgent, SimpleQaConfig};
use gemicro_tool_agent::{ToolAgent, ToolAgentConfig};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

/// Evaluation CLI for benchmarking gemicro agents.
#[derive(Parser, Debug)]
#[command(name = "gemicro-eval")]
#[command(about = "Run evaluations against HotpotQA or custom datasets")]
#[command(version)]
struct Args {
    /// Dataset to use: "hotpotqa", "gsm8k", or path to a JSON file
    #[arg(long, short = 'd')]
    dataset: String,

    /// Number of samples to evaluate (default: all)
    #[arg(long, short = 's')]
    sample: Option<usize>,

    /// Comma-separated list of scorers: contains, critique
    #[arg(long, default_value = "contains,critique")]
    scorer: String,

    /// Agent to evaluate: deep_research, react, simple_qa, tool_agent
    #[arg(long, short = 'a')]
    agent: String,

    /// Maximum concurrent evaluations
    #[arg(long, default_value = "5")]
    concurrency: usize,

    /// Maximum retry attempts per question
    #[arg(long, default_value = "1")]
    retries: usize,

    /// Output format: table or json
    #[arg(long, short = 'o', default_value = "table")]
    output: String,

    /// Output file path (defaults to stdout for table, required for json)
    #[arg(long)]
    output_file: Option<PathBuf>,

    /// Gemini API key (can also use GEMINI_API_KEY env var)
    #[arg(long, env = "GEMINI_API_KEY")]
    api_key: String,

    /// LLM request timeout in seconds
    #[arg(long, default_value = "60")]
    llm_timeout: u64,

    /// Maximum tokens per LLM request
    #[arg(long, default_value = "4096")]
    max_tokens: u32,

    /// Temperature for LLM generation (0.0-1.0)
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

impl Args {
    /// Validate CLI arguments.
    fn validate(&self) -> Result<(), String> {
        // Validate output format
        if !["table", "json"].contains(&self.output.as_str()) {
            return Err(format!(
                "Invalid output format '{}'. Use 'table' or 'json'.",
                self.output
            ));
        }

        // Validate scorers
        for scorer in self.scorer.split(',') {
            let scorer = scorer.trim();
            if !["contains", "critique"].contains(&scorer) {
                return Err(format!(
                    "Invalid scorer '{}'. Use contains or critique.",
                    scorer
                ));
            }
        }

        // Validate agent
        if !["deep_research", "react", "simple_qa", "tool_agent"].contains(&self.agent.as_str()) {
            return Err(format!(
                "Invalid agent '{}'. Use deep_research, react, simple_qa, or tool_agent.",
                self.agent
            ));
        }

        // Validate concurrency
        if self.concurrency == 0 {
            return Err("concurrency must be greater than 0".to_string());
        }

        // Validate temperature
        if !(0.0..=1.0).contains(&self.temperature) {
            return Err(format!(
                "temperature ({}) must be between 0.0 and 1.0",
                self.temperature
            ));
        }

        Ok(())
    }

    /// Build LlmConfig from CLI arguments.
    fn llm_config(&self) -> LlmConfig {
        LlmConfig::default()
            .with_timeout(Duration::from_secs(self.llm_timeout))
            .with_max_tokens(self.max_tokens)
            .with_temperature(self.temperature)
            .with_max_retries(2)
            .with_retry_base_delay_ms(1000)
    }

    /// Build EvalConfig from CLI arguments.
    fn eval_config(&self) -> EvalConfig {
        EvalConfig::new()
            .with_concurrency(self.concurrency)
            .with_max_retries(self.retries)
    }

    /// Build Scorers from CLI arguments.
    ///
    /// Requires an LlmClient for the critique scorer.
    fn scorers(&self, llm: std::sync::Arc<LlmClient>) -> Scorers {
        let mut scorers = Scorers::new(vec![]);
        for scorer in self.scorer.split(',') {
            match scorer.trim() {
                "contains" => scorers.add(Contains),
                "critique" => scorers.add(CritiqueScorer::new(llm.clone())),
                _ => {} // Already validated
            }
        }
        scorers
    }
}

/// Create the agent registry with all available agents.
fn create_registry() -> AgentRegistry {
    let mut registry = AgentRegistry::new();

    registry.register("deep_research", || {
        Box::new(DeepResearchAgent::new(ResearchConfig::default()).unwrap())
    });

    registry.register("react", || {
        Box::new(ReactAgent::new(ReactConfig::default()).unwrap())
    });

    registry.register("simple_qa", || {
        Box::new(SimpleQaAgent::new(SimpleQaConfig::default()).unwrap())
    });

    registry.register("tool_agent", || {
        Box::new(ToolAgent::new(ToolAgentConfig::default()).unwrap())
    });

    registry
}

/// Run evaluation with progress display.
async fn run_evaluation(args: &Args) -> Result<EvalSummary, String> {
    // Create LLM client for agent execution
    let genai_client = rust_genai::Client::builder(args.api_key.clone())
        .build()
        .map_err(|e| format!("Failed to create Gemini client: {}", e))?;
    let llm = LlmClient::new(genai_client, args.llm_config());

    // Create separate LLM client for scorer (critique needs its own client)
    let scorer_genai_client = rust_genai::Client::builder(args.api_key.clone())
        .build()
        .map_err(|e| format!("Failed to create scorer Gemini client: {}", e))?;
    let scorer_llm = std::sync::Arc::new(LlmClient::new(scorer_genai_client, args.llm_config()));

    // Create agent
    let registry = create_registry();
    let agent = registry
        .get(&args.agent)
        .ok_or_else(|| format!("Agent '{}' not found", args.agent))?;

    // Create harness and scorers
    let harness = EvalHarness::new(args.eval_config());
    let scorers = args.scorers(scorer_llm);

    // Load dataset and run evaluation based on dataset type
    match args.dataset.to_lowercase().as_str() {
        "hotpotqa" => {
            let dataset =
                HotpotQA::new().map_err(|e| format!("Failed to create HotpotQA: {}", e))?;
            run_with_progress(
                &harness,
                agent.as_ref(),
                &dataset,
                args.sample,
                scorers,
                llm,
            )
            .await
        }
        "gsm8k" => {
            let dataset = GSM8K::new().map_err(|e| format!("Failed to create GSM8K: {}", e))?;
            run_with_progress(
                &harness,
                agent.as_ref(),
                &dataset,
                args.sample,
                scorers,
                llm,
            )
            .await
        }
        _ => {
            let path = PathBuf::from(&args.dataset);
            if !path.exists() {
                return Err(format!("Dataset file not found: {}", args.dataset));
            }
            let dataset = JsonFileDataset::new(path);
            run_with_progress(
                &harness,
                agent.as_ref(),
                &dataset,
                args.sample,
                scorers,
                llm,
            )
            .await
        }
    }
}

/// Run evaluation with progress bar.
async fn run_with_progress<D: Dataset>(
    harness: &EvalHarness,
    agent: &dyn gemicro_core::Agent,
    dataset: &D,
    sample_size: Option<usize>,
    scorers: Scorers,
    llm: LlmClient,
) -> Result<EvalSummary, String> {
    let progress_bar = ProgressBar::new(0);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let summary = harness
        .evaluate_with_progress(agent, dataset, sample_size, scorers, llm, |progress| {
            match progress {
                EvalProgress::Started { total } => {
                    progress_bar.set_length(total as u64);
                    progress_bar.set_message("Evaluating...");
                }
                EvalProgress::QuestionCompleted {
                    completed, success, ..
                } => {
                    progress_bar.set_position(completed as u64);
                    if !success {
                        progress_bar.set_message("(some failures)");
                    }
                }
                _ => {} // Handle future variants gracefully
            }
        })
        .await
        .map_err(|e| format!("Evaluation failed: {}", e))?;

    progress_bar.finish_with_message("Complete");
    Ok(summary)
}

/// Output results in the requested format.
fn output_results(summary: &EvalSummary, args: &Args) -> Result<(), String> {
    match args.output.as_str() {
        "table" => {
            summary.print_summary();
            if let Some(path) = &args.output_file {
                summary
                    .write_json(path)
                    .map_err(|e| format!("Failed to write output file: {}", e))?;
                println!("\nDetailed results written to: {}", path.display());
            }
        }
        "json" => {
            let json = serde_json::to_string_pretty(summary)
                .map_err(|e| format!("Failed to serialize results: {}", e))?;

            if let Some(path) = &args.output_file {
                std::fs::write(path, &json)
                    .map_err(|e| format!("Failed to write output file: {}", e))?;
                eprintln!("Results written to: {}", path.display());
            } else {
                println!("{}", json);
            }
        }
        _ => unreachable!(), // Already validated
    }
    Ok(())
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();

    // Initialize logging
    if args.verbose {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    }

    // Validate arguments
    if let Err(e) = args.validate() {
        eprintln!("Error: {}", e);
        return ExitCode::FAILURE;
    }

    // Print configuration
    eprintln!("=== Gemicro Evaluation ===");
    eprintln!("Dataset: {}", args.dataset);
    eprintln!("Agent: {}", args.agent);
    eprintln!("Scorers: {}", args.scorer);
    eprintln!(
        "Sample size: {}",
        args.sample
            .map(|s| s.to_string())
            .unwrap_or_else(|| "all".to_string())
    );
    eprintln!("Concurrency: {}", args.concurrency);
    eprintln!();

    // Run evaluation
    match run_evaluation(&args).await {
        Ok(summary) => {
            if let Err(e) = output_results(&summary, &args) {
                eprintln!("Error: {}", e);
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_args() -> Args {
        Args {
            dataset: "hotpotqa".to_string(),
            sample: Some(10),
            scorer: "contains,critique".to_string(),
            agent: "deep_research".to_string(),
            concurrency: 5,
            retries: 1,
            output: "table".to_string(),
            output_file: None,
            api_key: "test-key".to_string(),
            llm_timeout: 60,
            max_tokens: 4096,
            temperature: 0.7,
            verbose: false,
        }
    }

    #[test]
    fn test_validate_valid_args() {
        let args = test_args();
        assert!(args.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_output() {
        let mut args = test_args();
        args.output = "invalid".to_string();
        assert!(args.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_scorer() {
        let mut args = test_args();
        args.scorer = "invalid".to_string();
        assert!(args.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_agent() {
        let mut args = test_args();
        args.agent = "invalid".to_string();
        assert!(args.validate().is_err());
    }

    #[test]
    fn test_validate_zero_concurrency() {
        let mut args = test_args();
        args.concurrency = 0;
        assert!(args.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_temperature() {
        let mut args = test_args();
        args.temperature = 1.5;
        assert!(args.validate().is_err());
    }

    #[test]
    fn test_scorers_parsing() {
        let mut args = test_args();
        args.scorer = "contains".to_string();

        // Create a dummy LlmClient for the scorer (not used for contains scorer)
        let genai_client = rust_genai::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let llm = std::sync::Arc::new(LlmClient::new(genai_client, args.llm_config()));

        let scorers = args.scorers(llm);
        // Scorers are opaque, but we can verify no panic occurred
        assert!(scorers.score_all("test", "test").len() == 1);
    }

    #[test]
    fn test_registry_has_all_agents() {
        let registry = create_registry();
        assert!(registry.contains("deep_research"));
        assert!(registry.contains("react"));
        assert!(registry.contains("simple_qa"));
        assert!(registry.contains("tool_agent"));
    }

    #[test]
    fn test_llm_config() {
        let args = test_args();
        let config = args.llm_config();

        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.max_tokens, 4096);
        assert_eq!(config.temperature, 0.7);
    }

    #[test]
    fn test_eval_config() {
        let args = test_args();
        let config = args.eval_config();

        assert_eq!(config.concurrency, 5);
        assert_eq!(config.max_retries, 1);
    }
}
