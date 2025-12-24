//! Command-line argument parsing.

use clap::{Parser, ValueEnum};
use gemicro_core::{LlmConfig, ResearchConfig};
use std::time::Duration;

/// Output verbosity mode for controlling truncation of displayed text.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum OutputMode {
    /// Compact output with shorter previews (20/30 chars)
    Compact,
    /// Normal output with balanced previews (40/55 chars) \[default\]
    #[default]
    Normal,
    /// Verbose output with longer previews (80/100 chars)
    Verbose,
}

/// Configuration for output truncation limits.
#[derive(Debug, Clone, Copy)]
pub struct OutputConfig {
    /// Maximum characters for sub-query text display
    pub query_display_chars: usize,
    /// Maximum characters for result previews
    pub preview_chars: usize,
    /// Maximum characters for history previews
    pub history_preview_chars: usize,
}

impl OutputConfig {
    /// Create an OutputConfig from an OutputMode.
    pub fn from_mode(mode: OutputMode) -> Self {
        match mode {
            OutputMode::Compact => Self {
                query_display_chars: 30,
                preview_chars: 20,
                history_preview_chars: 50,
            },
            OutputMode::Normal => Self {
                query_display_chars: 55,
                preview_chars: 40,
                history_preview_chars: 100,
            },
            OutputMode::Verbose => Self {
                query_display_chars: 100,
                preview_chars: 80,
                history_preview_chars: 200,
            },
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self::from_mode(OutputMode::Normal)
    }
}

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

    /// Output verbosity mode (compact, normal, verbose)
    #[arg(long, value_enum, default_value_t = OutputMode::Normal)]
    pub output_mode: OutputMode,

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
    #[arg(long, default_value = "1024")]
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

    /// Build OutputConfig from CLI arguments.
    pub fn output_config(&self) -> OutputConfig {
        OutputConfig::from_mode(self.output_mode)
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
            output_mode: OutputMode::Normal,
            api_key: "test-key".to_string(),
            min_sub_queries: 3,
            max_sub_queries: 5,
            max_concurrent: 5,
            timeout: 180,
            continue_on_failure: true,
            llm_timeout: 60,
            max_tokens: 1024,
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

    #[test]
    fn test_output_config_compact() {
        let config = OutputConfig::from_mode(OutputMode::Compact);
        assert_eq!(config.query_display_chars, 30);
        assert_eq!(config.preview_chars, 20);
        assert_eq!(config.history_preview_chars, 50);
    }

    #[test]
    fn test_output_config_normal() {
        let config = OutputConfig::from_mode(OutputMode::Normal);
        assert_eq!(config.query_display_chars, 55);
        assert_eq!(config.preview_chars, 40);
        assert_eq!(config.history_preview_chars, 100);
    }

    #[test]
    fn test_output_config_verbose() {
        let config = OutputConfig::from_mode(OutputMode::Verbose);
        assert_eq!(config.query_display_chars, 100);
        assert_eq!(config.preview_chars, 80);
        assert_eq!(config.history_preview_chars, 200);
    }

    #[test]
    fn test_output_config_default() {
        let config = OutputConfig::default();
        let normal = OutputConfig::from_mode(OutputMode::Normal);
        assert_eq!(config.query_display_chars, normal.query_display_chars);
        assert_eq!(config.preview_chars, normal.preview_chars);
    }

    #[test]
    fn test_args_output_config() {
        let args = test_args();
        let config = args.output_config();
        assert_eq!(config.query_display_chars, 55); // Normal mode default
    }
}
