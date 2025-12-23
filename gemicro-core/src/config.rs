use std::time::Duration;

/// The Gemini model to use for all LLM operations
///
/// Hardcoded to gemini-3-flash-preview as per project requirements.
pub const MODEL: &str = "gemini-3-flash-preview";

/// Top-level configuration for gemicro
///
/// Contains only cross-agent configuration. Agent-specific configuration
/// (like `ResearchConfig`) should be passed directly to agent constructors,
/// not embedded here. This follows the Evergreen soft-typing philosophy:
/// extensibility without protocol modifications.
#[derive(Debug, Clone, Default)]
pub struct GemicroConfig {
    /// Configuration for LLM client (shared across all agents)
    pub llm: LlmConfig,
}

/// Configuration for Deep Research agent
#[derive(Debug, Clone)]
pub struct ResearchConfig {
    /// Maximum number of sub-queries to generate
    ///
    /// Default: 5
    pub max_sub_queries: usize,

    /// Minimum number of sub-queries to generate
    ///
    /// Default: 3
    pub min_sub_queries: usize,

    /// Whether to continue if some sub-queries fail
    ///
    /// If true, synthesis will proceed with partial results.
    /// If false, any sub-query failure aborts the entire research.
    ///
    /// Default: true (continue on partial failure)
    pub continue_on_partial_failure: bool,

    /// Maximum time to wait for all sub-queries to complete
    ///
    /// Default: 60 seconds
    pub total_timeout: Duration,
}

impl ResearchConfig {
    /// Validate the configuration
    ///
    /// Returns an error if:
    /// - `min_sub_queries` is 0
    /// - `max_sub_queries` is 0
    /// - `min_sub_queries > max_sub_queries`
    /// - `total_timeout` is 0
    pub fn validate(&self) -> Result<(), String> {
        if self.min_sub_queries == 0 {
            return Err("min_sub_queries must be greater than 0".to_string());
        }

        if self.max_sub_queries == 0 {
            return Err("max_sub_queries must be greater than 0".to_string());
        }

        if self.min_sub_queries > self.max_sub_queries {
            return Err(format!(
                "min_sub_queries ({}) cannot be greater than max_sub_queries ({})",
                self.min_sub_queries, self.max_sub_queries
            ));
        }

        if self.total_timeout.as_secs() == 0 {
            return Err("total_timeout must be greater than 0".to_string());
        }

        Ok(())
    }
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            max_sub_queries: 5,
            min_sub_queries: 3,
            continue_on_partial_failure: true,
            total_timeout: Duration::from_secs(60),
        }
    }
}

/// Configuration for LLM client
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Maximum tokens per request
    ///
    /// Default: 2048
    pub max_tokens: u32,

    /// Timeout for individual LLM requests
    ///
    /// Default: 30 seconds
    pub timeout: Duration,

    /// Temperature for generation (0.0 - 1.0)
    ///
    /// Lower values make output more deterministic.
    /// Default: 0.7
    pub temperature: f32,

    /// Maximum number of retries on transient failures
    ///
    /// Default: 2
    pub max_retries: u32,

    /// Base delay for exponential backoff (milliseconds)
    ///
    /// Default: 1000ms (1 second)
    pub retry_base_delay_ms: u64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            timeout: Duration::from_secs(30),
            temperature: 0.7,
            max_retries: 2,
            retry_base_delay_ms: 1000,
        }
    }
}

impl LlmConfig {
    /// Get the retry delay for a given attempt number (0-indexed)
    ///
    /// Uses exponential backoff: delay = base_delay * 2^attempt
    ///
    /// The delay is capped at 60 seconds to prevent overflow and
    /// unreasonably long wait times.
    pub fn retry_delay(&self, attempt: u32) -> Duration {
        const MAX_DELAY_MS: u64 = 60_000; // 60 seconds

        let delay_ms = self
            .retry_base_delay_ms
            .saturating_mul(2u64.saturating_pow(attempt))
            .min(MAX_DELAY_MS);

        Duration::from_millis(delay_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_constant() {
        assert_eq!(MODEL, "gemini-3-flash-preview");
    }

    #[test]
    fn test_default_gemicro_config() {
        let config = GemicroConfig::default();
        assert_eq!(config.llm.max_tokens, 2048);
        assert_eq!(config.llm.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_default_research_config() {
        let config = ResearchConfig::default();
        assert_eq!(config.max_sub_queries, 5);
        assert_eq!(config.min_sub_queries, 3);
        assert!(config.continue_on_partial_failure);
        assert_eq!(config.total_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_default_llm_config() {
        let config = LlmConfig::default();
        assert_eq!(config.max_tokens, 2048);
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.max_retries, 2);
        assert_eq!(config.retry_base_delay_ms, 1000);
    }

    #[test]
    fn test_retry_delay() {
        let config = LlmConfig::default();

        // Exponential backoff: 1s, 2s, 4s
        assert_eq!(config.retry_delay(0), Duration::from_millis(1000));
        assert_eq!(config.retry_delay(1), Duration::from_millis(2000));
        assert_eq!(config.retry_delay(2), Duration::from_millis(4000));
    }

    #[test]
    fn test_retry_delay_overflow_protection() {
        let config = LlmConfig::default();

        // Large attempt numbers should be capped at 60 seconds
        assert_eq!(config.retry_delay(10), Duration::from_millis(60_000));
        assert_eq!(config.retry_delay(100), Duration::from_millis(60_000));
        assert_eq!(config.retry_delay(u32::MAX), Duration::from_millis(60_000));
    }

    #[test]
    fn test_custom_config() {
        let mut config = ResearchConfig::default();
        config.max_sub_queries = 10;
        config.continue_on_partial_failure = false;

        assert_eq!(config.max_sub_queries, 10);
        assert!(!config.continue_on_partial_failure);
    }

    #[test]
    fn test_research_config_validation_success() {
        let config = ResearchConfig::default();
        assert!(config.validate().is_ok());

        let mut config = ResearchConfig::default();
        config.min_sub_queries = 5;
        config.max_sub_queries = 5;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_research_config_validation_min_greater_than_max() {
        let mut config = ResearchConfig::default();
        config.min_sub_queries = 10;
        config.max_sub_queries = 5;

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("cannot be greater than"));
    }

    #[test]
    fn test_research_config_validation_zero_max() {
        let mut config = ResearchConfig::default();
        config.max_sub_queries = 0;

        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("max_sub_queries must be greater than 0"));
    }

    #[test]
    fn test_research_config_validation_zero_min() {
        let mut config = ResearchConfig::default();
        config.min_sub_queries = 0;

        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("min_sub_queries must be greater than 0"));
    }

    #[test]
    fn test_research_config_validation_zero_timeout() {
        let mut config = ResearchConfig::default();
        config.total_timeout = Duration::from_secs(0);

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be greater than 0"));
    }
}
