//! Command-line argument parsing.

use clap::Parser;
use gemicro_core::LlmConfig;
use std::path::PathBuf;
use std::time::Duration;

/// AI agent exploration platform
#[derive(Parser, Debug)]
#[command(name = "gemicro")]
#[command(about = "Agent exploration platform powered by Gemini", long_about = None)]
#[command(version)]
pub struct Args {
    /// Research query (required unless using --interactive)
    pub query: Option<String>,

    /// Interactive REPL mode
    #[arg(short, long)]
    pub interactive: bool,

    /// Agent to use (e.g., deep_research, developer, prompt_agent)
    #[arg(long)]
    pub agent: String,

    /// Gemini API key (can also use GEMINI_API_KEY env var)
    #[arg(long, env = "GEMINI_API_KEY")]
    pub api_key: String,

    /// Model to use for LLM requests (overrides agent defaults)
    #[arg(long, env = "GEMINI_MODEL")]
    pub model: Option<String>,

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

    /// Use plain text output instead of markdown rendering
    #[arg(long)]
    pub plain: bool,

    /// Event bus URL for real-time coordination (e.g., http://localhost:8765)
    ///
    /// When provided, connects to the event bus via SSE for push-based
    /// event notifications. External events are injected into the agent's
    /// update stream in real-time.
    #[arg(long, env = "GEMICRO_EVENT_BUS_URL")]
    pub event_bus_url: Option<String>,

    /// Restrict file operations to specific paths (whitelist)
    ///
    /// When provided, tools that access the filesystem are restricted to
    /// only the specified paths. Any file operation outside these paths
    /// will be denied. This provides defense-in-depth for agent execution.
    ///
    /// Can be specified multiple times: --sandbox-path /workspace --sandbox-path /tmp
    #[arg(long = "sandbox-path", value_name = "PATH")]
    pub sandbox_paths: Vec<PathBuf>,

    /// Auto-approve all tool confirmations without prompting
    ///
    /// When enabled, tools that normally require confirmation (like bash commands)
    /// will be automatically approved. This is useful for:
    /// - Scripted/piped input where stdin is not a terminal
    /// - Trusted automation contexts
    /// - Testing and development
    ///
    /// SECURITY WARNING: Use with caution - this allows agents to execute
    /// arbitrary bash commands and modify files without human review.
    #[arg(long)]
    pub auto_approve: bool,
}

impl Args {
    /// Validate CLI arguments.
    ///
    /// Returns an error if arguments are invalid.
    pub fn validate(&self) -> Result<(), String> {
        // Warn if both interactive and query are provided (query will be ignored)
        if self.interactive && self.query.is_some() {
            eprintln!(
                "⚠️  Warning: Query argument is ignored in interactive mode. \
                 Use single-query mode (without --interactive) to run a query."
            );
        }

        // Bounds validation
        if self.llm_timeout == 0 {
            return Err("llm-timeout must be greater than 0".to_string());
        }
        if self.max_tokens == 0 {
            return Err("max-tokens must be greater than 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.temperature) {
            return Err(format!(
                "temperature ({}) must be between 0.0 and 1.0",
                self.temperature
            ));
        }

        // Validate sandbox paths
        for path in &self.sandbox_paths {
            if !path.exists() {
                return Err(format!("sandbox path does not exist: {}", path.display()));
            }
            if !path.is_dir() {
                return Err(format!(
                    "sandbox path must be a directory: {}",
                    path.display()
                ));
            }
        }

        Ok(())
    }

    /// Build LlmConfig from CLI arguments.
    pub fn llm_config(&self) -> LlmConfig {
        LlmConfig::default()
            .with_timeout(Duration::from_secs(self.llm_timeout))
            .with_max_tokens(self.max_tokens)
            .with_temperature(self.temperature)
            .with_max_retries(2)
            .with_retry_base_delay_ms(1000)
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
            agent: "deep_research".to_string(),
            api_key: "test-key".to_string(),
            model: None,
            llm_timeout: 60,
            max_tokens: 16384,
            temperature: 0.7,
            verbose: false,
            plain: false,
            event_bus_url: None,
            sandbox_paths: vec![],
            auto_approve: false,
        }
    }

    #[test]
    fn test_validate_valid_args() {
        let args = test_args();
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
    fn test_validate_no_query_no_interactive_is_valid() {
        // No query defaults to interactive mode in main.rs
        let mut args = test_args();
        args.query = None;
        args.interactive = false;

        assert!(args.validate().is_ok());
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
    fn test_validate_sandbox_path_not_exists() {
        let mut args = test_args();
        args.sandbox_paths = vec![PathBuf::from("/nonexistent/path/that/should/not/exist")];

        let result = args.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not exist"));
    }

    #[test]
    fn test_validate_sandbox_path_not_directory() {
        // Use a file that should exist on most systems
        let file_path = std::env::current_exe().expect("should get current exe");
        let mut args = test_args();
        args.sandbox_paths = vec![file_path];

        let result = args.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be a directory"));
    }

    #[test]
    fn test_validate_sandbox_path_valid() {
        let mut args = test_args();
        args.sandbox_paths = vec![PathBuf::from("/tmp")];

        // This may fail on some systems without /tmp
        if std::path::Path::new("/tmp").exists() {
            assert!(args.validate().is_ok());
        }
    }
}
