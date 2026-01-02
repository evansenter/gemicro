use std::time::Duration;

/// The Gemini model to use for all LLM operations
///
/// Hardcoded to gemini-3-flash-preview as per project requirements.
pub const MODEL: &str = "gemini-3-flash-preview";

/// Top-level configuration for gemicro
///
/// Contains only cross-agent configuration. Agent-specific configuration
/// should be passed directly to agent constructors in the individual agent crates,
/// not embedded here. This follows the [Evergreen spec](https://github.com/google-deepmind/evergreen-spec)
/// soft-typing philosophy: extensibility without protocol modifications.
///
/// ## Agent Crates
///
/// Each agent type has its own crate with its own configuration:
/// - `gemicro-deep-research`: ResearchConfig, ResearchPrompts
/// - `gemicro-react`: ReactConfig, ReactPrompts
/// - `gemicro-simple-qa`: SimpleQaConfig
/// - `gemicro-tool-agent`: ToolAgentConfig
/// - `gemicro-critique`: CritiqueConfig
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct GemicroConfig {
    /// Configuration for LLM client (shared across all agents)
    pub llm: LlmConfig,
}

/// Configuration for LLM client
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    /// Set the maximum tokens per request.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the timeout for individual LLM requests.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the temperature for generation (0.0 - 1.0).
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the maximum number of retries on transient failures.
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set the base delay for exponential backoff (milliseconds).
    #[must_use]
    pub fn with_retry_base_delay_ms(mut self, delay_ms: u64) -> Self {
        self.retry_base_delay_ms = delay_ms;
        self
    }

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
}
