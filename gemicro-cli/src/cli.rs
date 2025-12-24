//! Command-line argument parsing.

use clap::Parser;
use gemicro_core::{LlmConfig, ResearchConfig};
use std::time::Duration;

/// AI agent exploration platform
#[derive(Parser, Debug)]
#[command(name = "gemicro")]
#[command(about = "Deep research agent powered by Gemini", long_about = None)]
#[command(version)]
pub struct Args {
    /// Research query
    pub query: String,

    /// Gemini API key (can also use GEMINI_API_KEY env var)
    #[arg(long, env = "GEMINI_API_KEY")]
    pub api_key: String,

    /// Minimum number of sub-queries to generate
    #[arg(long, default_value = "3")]
    pub min_sub_queries: usize,

    /// Maximum number of sub-queries to generate
    #[arg(long, default_value = "5")]
    pub max_sub_queries: usize,

    /// Total timeout in seconds
    #[arg(long, default_value = "180")]
    pub timeout: u64,

    /// Continue if some sub-queries fail
    #[arg(long, default_value = "true")]
    pub continue_on_failure: bool,

    /// LLM request timeout in seconds
    #[arg(long, default_value = "60")]
    pub llm_timeout: u64,

    /// Maximum tokens per LLM request
    #[arg(long, default_value = "1024")]
    pub max_tokens: u32,

    /// Temperature for LLM generation (0.0-1.0)
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,

    /// Enable verbose logging
    #[arg(short, long)]
    pub verbose: bool,
}

impl Args {
    /// Build LlmConfig from CLI arguments.
    pub fn llm_config(&self) -> LlmConfig {
        LlmConfig {
            timeout: Duration::from_secs(self.llm_timeout),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            max_retries: 2,
            retry_base_delay_ms: 1000,
        }
    }

    /// Build ResearchConfig from CLI arguments.
    pub fn research_config(&self) -> ResearchConfig {
        ResearchConfig {
            min_sub_queries: self.min_sub_queries,
            max_sub_queries: self.max_sub_queries,
            continue_on_partial_failure: self.continue_on_failure,
            total_timeout: Duration::from_secs(self.timeout),
            ..Default::default()
        }
    }
}
