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
    /// Research query (required unless using --interactive)
    pub query: Option<String>,

    /// Interactive REPL mode
    #[arg(short, long)]
    pub interactive: bool,

    /// Gemini API key (can also use GEMINI_API_KEY env var)
    #[arg(long, env = "GEMINI_API_KEY")]
    pub api_key: String,

    /// Minimum number of sub-queries to generate
    #[arg(long, default_value = "3")]
    pub min_sub_queries: usize,

    /// Maximum number of sub-queries to generate
    #[arg(long, default_value = "5")]
    pub max_sub_queries: usize,

    /// Maximum concurrent sub-query executions (0 = unlimited)
    #[arg(long, default_value = "5")]
    pub max_concurrent: usize,

    /// Total timeout in seconds
    #[arg(long, default_value = "180")]
    pub timeout: u64,

    /// Continue if some sub-queries fail
    #[arg(long, default_value_t = true)]
    pub continue_on_failure: bool,

    /// LLM request timeout in seconds
    #[arg(long, default_value = "60")]
    pub llm_timeout: u64,

    /// Maximum tokens per LLM request
    #[arg(long, default_value = "16384")]
    pub max_tokens: u32,

    /// Temperature for LLM generation (0.0-1.0)
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,

    /// Enable verbose logging
    #[arg(short, long)]
    pub verbose: bool,
}

impl Args {
    /// Validate CLI arguments.
    ///
    /// Returns an error if arguments are invalid.
    pub fn validate(&self) -> Result<(), String> {
        // Query is required unless in interactive mode
        if !self.interactive && self.query.is_none() {
            return Err("Query is required (or use --interactive for REPL mode)".to_string());
        }

        // Warn if both interactive and query are provided (query will be ignored)
        if self.interactive && self.query.is_some() {
            eprintln!(
                "⚠️  Warning: Query argument is ignored in interactive mode. \
                 Use single-query mode (without --interactive) to run a query."
            );
        }

        // Bounds validation
        if self.min_sub_queries == 0 {
            return Err("min-sub-queries must be greater than 0".to_string());
        }
        if self.max_sub_queries == 0 {
            return Err("max-sub-queries must be greater than 0".to_string());
        }
        if self.timeout == 0 {
            return Err("timeout must be greater than 0".to_string());
        }
        if self.llm_timeout == 0 {
            return Err("llm-timeout must be greater than 0".to_string());
        }
        if self.max_tokens == 0 {
            return Err("max-tokens must be greater than 0".to_string());
        }

        // Relational validation
        if self.min_sub_queries > self.max_sub_queries {
            return Err(format!(
                "min-sub-queries ({}) cannot be greater than max-sub-queries ({})",
                self.min_sub_queries, self.max_sub_queries
            ));
        }
        if self.llm_timeout >= self.timeout {
            return Err(format!(
                "llm-timeout ({}) must be less than total timeout ({})",
                self.llm_timeout, self.timeout
            ));
        }
        if !(0.0..=1.0).contains(&self.temperature) {
            return Err(format!(
                "temperature ({}) must be between 0.0 and 1.0",
                self.temperature
            ));
        }
        Ok(())
    }

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
            max_concurrent_sub_queries: self.max_concurrent,
            continue_on_partial_failure: self.continue_on_failure,
            total_timeout: Duration::from_secs(self.timeout),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create Args with default values for testing.
    fn test_args() -> Args {
        Args {
            query: Some("test query".to_string()),
            interactive: false,
            api_key: "test-key".to_string(),
            min_sub_queries: 3,
            max_sub_queries: 5,
            max_concurrent: 5,
            timeout: 180,
            continue_on_failure: true,
            llm_timeout: 60,
            max_tokens: 16384,
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
    fn test_validate_min_greater_than_max() {
        let mut args = test_args();
        args.min_sub_queries = 10;
        args.max_sub_queries = 5;

        let result = args.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("min-sub-queries"));
    }

    #[test]
    fn test_validate_min_equals_max() {
        let mut args = test_args();
        args.min_sub_queries = 5;
        args.max_sub_queries = 5;

        assert!(args.validate().is_ok());
    }

    #[test]
    fn test_validate_temperature_too_low() {
        let mut args = test_args();
        args.temperature = -0.1;

        let result = args.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("temperature"));
    }

    #[test]
    fn test_validate_temperature_too_high() {
        let mut args = test_args();
        args.temperature = 1.5;

        let result = args.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("temperature"));
    }

    #[test]
    fn test_validate_temperature_boundary_zero() {
        let mut args = test_args();
        args.temperature = 0.0;

        assert!(args.validate().is_ok());
    }

    #[test]
    fn test_validate_temperature_boundary_one() {
        let mut args = test_args();
        args.temperature = 1.0;

        assert!(args.validate().is_ok());
    }

    #[test]
    fn test_validate_min_sub_queries_zero() {
        let mut args = test_args();
        args.min_sub_queries = 0;

        let result = args.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("min-sub-queries"));
    }

    #[test]
    fn test_validate_max_sub_queries_zero() {
        let mut args = test_args();
        args.max_sub_queries = 0;
        args.min_sub_queries = 0; // Also set min to 0 to avoid triggering that check first

        let result = args.validate();
        assert!(result.is_err());
        // Will fail on min first since it's checked first
        assert!(result.unwrap_err().contains("sub-queries"));
    }

    #[test]
    fn test_validate_timeout_zero() {
        let mut args = test_args();
        args.timeout = 0;

        let result = args.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("timeout"));
    }

    #[test]
    fn test_validate_llm_timeout_zero() {
        let mut args = test_args();
        args.llm_timeout = 0;

        let result = args.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("llm-timeout"));
    }

    #[test]
    fn test_validate_max_tokens_zero() {
        let mut args = test_args();
        args.max_tokens = 0;

        let result = args.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("max-tokens"));
    }

    #[test]
    fn test_validate_llm_timeout_exceeds_total() {
        let mut args = test_args();
        args.llm_timeout = 200;
        args.timeout = 180;

        let result = args.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("llm-timeout"));
        assert!(err.contains("less than"));
    }

    #[test]
    fn test_validate_llm_timeout_equals_total() {
        let mut args = test_args();
        args.llm_timeout = 180;
        args.timeout = 180;

        let result = args.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("llm-timeout"));
    }

    #[test]
    fn test_validate_llm_timeout_valid() {
        let mut args = test_args();
        args.llm_timeout = 60;
        args.timeout = 180;

        assert!(args.validate().is_ok());
    }

    #[test]
    fn test_validate_query_required_without_interactive() {
        let mut args = test_args();
        args.query = None;
        args.interactive = false;

        let result = args.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Query is required"));
    }

    #[test]
    fn test_validate_interactive_without_query() {
        let mut args = test_args();
        args.query = None;
        args.interactive = true;

        assert!(args.validate().is_ok());
    }

    #[test]
    fn test_validate_interactive_with_query() {
        let mut args = test_args();
        args.query = Some("test".to_string());
        args.interactive = true;

        assert!(args.validate().is_ok());
    }
}
