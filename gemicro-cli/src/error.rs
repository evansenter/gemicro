//! Error formatting utilities for the CLI.
//!
//! Provides consistent error formatting with optional emoji hints
//! for both single-query and REPL modes.

use gemicro_core::AgentError;

/// Configuration for error formatting.
#[derive(Debug, Clone, Copy)]
pub struct ErrorFormatter {
    /// Whether to use emoji prefixes in hints.
    pub use_emoji: bool,
}

impl ErrorFormatter {
    /// Create a formatter with emoji hints (for single-query mode).
    pub fn with_emoji() -> Self {
        Self { use_emoji: true }
    }

    /// Create a formatter without emoji hints (for REPL mode).
    pub fn plain() -> Self {
        Self { use_emoji: false }
    }

    /// Format an AgentError with helpful suggestions.
    ///
    /// Prints the suggestion to stderr and returns an anyhow::Error.
    pub fn format(&self, e: AgentError) -> anyhow::Error {
        let prefix = if self.use_emoji { "💡 " } else { "" };

        let suggestion = match &e {
            AgentError::Timeout { phase, .. } => Some(format!(
                "{}Timeout during {}. Try increasing --timeout or --llm-timeout",
                prefix, phase
            )),
            AgentError::AllSubQueriesFailed => Some(format!(
                "{}All sub-queries failed. Check your API key and network connection",
                prefix
            )),
            AgentError::InvalidConfig(msg) => {
                Some(format!("{}Configuration error: {}", prefix, msg))
            }
            AgentError::Llm(llm_err) => Some(format!("{}LLM error: {}", prefix, llm_err)),
            _ => None,
        };

        let err = anyhow::anyhow!("Agent error: {}", e);
        if let Some(hint) = suggestion {
            if self.use_emoji {
                eprintln!("\n{}", hint);
            } else {
                eprintln!("{}", hint);
            }
        }
        err
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formatter_with_emoji() {
        let formatter = ErrorFormatter::with_emoji();
        assert!(formatter.use_emoji);
    }

    #[test]
    fn test_formatter_plain() {
        let formatter = ErrorFormatter::plain();
        assert!(!formatter.use_emoji);
    }

    #[test]
    fn test_format_timeout_error() {
        let formatter = ErrorFormatter::with_emoji();
        let err = AgentError::Timeout {
            elapsed_ms: 30000,
            timeout_ms: 60000,
            phase: "decomposition".to_string(),
        };
        let result = formatter.format(err);
        assert!(result.to_string().contains("Agent error"));
    }

    #[test]
    fn test_format_all_sub_queries_failed() {
        let formatter = ErrorFormatter::plain();
        let err = AgentError::AllSubQueriesFailed;
        let result = formatter.format(err);
        assert!(result.to_string().contains("Agent error"));
    }

    #[test]
    fn test_format_invalid_config() {
        let formatter = ErrorFormatter::with_emoji();
        let err = AgentError::InvalidConfig("bad config".to_string());
        let result = formatter.format(err);
        assert!(result.to_string().contains("Agent error"));
    }
}
