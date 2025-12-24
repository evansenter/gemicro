use crate::error::AgentError;
use std::time::Duration;

/// The Gemini model to use for all LLM operations
///
/// Hardcoded to gemini-3-flash-preview as per project requirements.
pub const MODEL: &str = "gemini-3-flash-preview";

/// Prompts used by the Deep Research agent
///
/// Contains system instructions and user templates for the three phases:
/// decomposition, sub-query execution, and synthesis.
///
/// # Template Placeholders
///
/// - `decomposition_template`: `{min}`, `{max}`, `{query}`
/// - `synthesis_template`: `{query}`, `{findings}`
///
/// # Example
///
/// ```
/// use gemicro_core::ResearchPrompts;
///
/// let mut prompts = ResearchPrompts::default();
/// prompts.decomposition_system = "You are a medical research expert.".to_string();
///
/// let rendered = prompts.render_decomposition(3, 5, "What causes migraines?");
/// assert!(rendered.contains("What causes migraines?"));
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ResearchPrompts {
    /// System instruction for decomposition phase
    pub decomposition_system: String,

    /// User template for decomposition phase
    ///
    /// Placeholders: `{min}`, `{max}`, `{query}`
    pub decomposition_template: String,

    /// System instruction for sub-query execution phase
    ///
    /// Note: Unlike decomposition and synthesis, sub-queries do not use a template.
    /// The sub-question text is passed directly as the user prompt, while this
    /// system instruction is shared across all parallel sub-query executions.
    pub sub_query_system: String,

    /// System instruction for synthesis phase
    pub synthesis_system: String,

    /// User template for synthesis phase
    ///
    /// Placeholders: `{query}`, `{findings}`
    pub synthesis_template: String,
}

impl ResearchPrompts {
    /// Render the decomposition prompt with placeholders substituted
    ///
    /// Note: `{query}` is replaced last to prevent user input containing
    /// literal `{min}` or `{max}` from being substituted.
    pub fn render_decomposition(&self, min: usize, max: usize, query: &str) -> String {
        self.decomposition_template
            .replace("{min}", &min.to_string())
            .replace("{max}", &max.to_string())
            .replace("{query}", query)
    }

    /// Render the synthesis prompt with placeholders substituted
    ///
    /// Note: `{findings}` is replaced before `{query}` to prevent user input
    /// containing literal `{findings}` from being substituted.
    pub fn render_synthesis(&self, query: &str, findings: &str) -> String {
        self.synthesis_template
            .replace("{findings}", findings)
            .replace("{query}", query)
    }

    /// Validate that all prompts are non-empty
    pub fn validate(&self) -> Result<(), AgentError> {
        if self.decomposition_system.trim().is_empty() {
            return Err(AgentError::InvalidConfig(
                "decomposition_system cannot be empty".to_string(),
            ));
        }
        if self.decomposition_template.trim().is_empty() {
            return Err(AgentError::InvalidConfig(
                "decomposition_template cannot be empty".to_string(),
            ));
        }
        if self.sub_query_system.trim().is_empty() {
            return Err(AgentError::InvalidConfig(
                "sub_query_system cannot be empty".to_string(),
            ));
        }
        if self.synthesis_system.trim().is_empty() {
            return Err(AgentError::InvalidConfig(
                "synthesis_system cannot be empty".to_string(),
            ));
        }
        if self.synthesis_template.trim().is_empty() {
            return Err(AgentError::InvalidConfig(
                "synthesis_template cannot be empty".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for ResearchPrompts {
    fn default() -> Self {
        Self {
            decomposition_system:
                "You are a research query decomposition expert. Return only valid JSON arrays of strings."
                    .to_string(),
            decomposition_template: r#"Decompose this research query into {min}-{max} focused, independent sub-questions.

Query: {query}

Return ONLY a JSON array of strings, no other text. Example:
["What is X?", "How does Y work?", "What are the benefits of Z?"]"#
                .to_string(),
            sub_query_system:
                "You are a research assistant. Provide a focused, informative answer.".to_string(),
            synthesis_system:
                "You are a research synthesis expert. Provide comprehensive, coherent answers."
                    .to_string(),
            synthesis_template: r#"Synthesize these research findings into a comprehensive answer.

Original question: {query}

Research findings:
{findings}

Provide a clear, well-organized answer that integrates all findings. Do not mention the research process or sub-questions."#
                .to_string(),
        }
    }
}

/// Top-level configuration for gemicro
///
/// Contains only cross-agent configuration. Agent-specific configuration
/// (like `ResearchConfig`) should be passed directly to agent constructors,
/// not embedded here. This follows the [Evergreen spec](https://github.com/google-deepmind/evergreen-spec)
/// soft-typing philosophy: extensibility without protocol modifications.
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

    /// Maximum number of sub-queries to execute concurrently
    ///
    /// This limits parallel API calls to prevent rate limiting and control costs.
    /// Set to 0 for unlimited concurrency.
    ///
    /// Default: 5
    pub max_concurrent_sub_queries: usize,

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

    /// Enable Google Search grounding for sub-queries
    ///
    /// When enabled, sub-queries can search the web for real-time information.
    /// This is useful for queries about current events, recent releases, or live data.
    ///
    /// Note: Grounded requests may have different pricing.
    ///
    /// Default: false
    pub use_google_search: bool,

    /// Prompts for the research agent
    ///
    /// Default: Built-in prompts optimized for Gemini
    pub prompts: ResearchPrompts,
}

impl ResearchConfig {
    /// Validate the configuration
    ///
    /// Returns an error if:
    /// - `min_sub_queries` is 0
    /// - `max_sub_queries` is 0
    /// - `min_sub_queries > max_sub_queries`
    /// - `total_timeout` is 0
    /// - Any prompt is empty
    pub fn validate(&self) -> Result<(), AgentError> {
        if self.min_sub_queries == 0 {
            return Err(AgentError::InvalidConfig(
                "min_sub_queries must be greater than 0".to_string(),
            ));
        }

        if self.max_sub_queries == 0 {
            return Err(AgentError::InvalidConfig(
                "max_sub_queries must be greater than 0".to_string(),
            ));
        }

        if self.min_sub_queries > self.max_sub_queries {
            return Err(AgentError::InvalidConfig(format!(
                "min_sub_queries ({}) cannot be greater than max_sub_queries ({})",
                self.min_sub_queries, self.max_sub_queries
            )));
        }

        if self.total_timeout.as_secs() == 0 {
            return Err(AgentError::InvalidConfig(
                "total_timeout must be greater than 0".to_string(),
            ));
        }

        self.prompts.validate()?;

        Ok(())
    }
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            max_sub_queries: 5,
            min_sub_queries: 3,
            max_concurrent_sub_queries: 5,
            continue_on_partial_failure: true,
            total_timeout: Duration::from_secs(60),
            use_google_search: false,
            prompts: ResearchPrompts::default(),
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
        assert_eq!(config.max_concurrent_sub_queries, 5);
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
        let config = ResearchConfig {
            max_sub_queries: 10,
            continue_on_partial_failure: false,
            ..Default::default()
        };

        assert_eq!(config.max_sub_queries, 10);
        assert!(!config.continue_on_partial_failure);
    }

    #[test]
    fn test_research_config_validation_success() {
        let config = ResearchConfig::default();
        assert!(config.validate().is_ok());

        let config = ResearchConfig {
            min_sub_queries: 5,
            max_sub_queries: 5,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_research_config_validation_min_greater_than_max() {
        let config = ResearchConfig {
            min_sub_queries: 10,
            max_sub_queries: 5,
            ..Default::default()
        };

        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("cannot be greater than"));
    }

    #[test]
    fn test_research_config_validation_zero_max() {
        let config = ResearchConfig {
            max_sub_queries: 0,
            ..Default::default()
        };

        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("max_sub_queries must be greater than 0"));
    }

    #[test]
    fn test_research_config_validation_zero_min() {
        let config = ResearchConfig {
            min_sub_queries: 0,
            ..Default::default()
        };

        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("min_sub_queries must be greater than 0"));
    }

    #[test]
    fn test_research_config_validation_zero_timeout() {
        let config = ResearchConfig {
            total_timeout: Duration::from_secs(0),
            ..Default::default()
        };

        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must be greater than 0"));
    }

    // ResearchPrompts tests

    #[test]
    fn test_research_prompts_default() {
        let prompts = ResearchPrompts::default();
        assert!(prompts.validate().is_ok());
        assert!(prompts.decomposition_system.contains("decomposition"));
        assert!(prompts.decomposition_template.contains("{min}"));
        assert!(prompts.decomposition_template.contains("{max}"));
        assert!(prompts.decomposition_template.contains("{query}"));
        assert!(prompts.synthesis_template.contains("{query}"));
        assert!(prompts.synthesis_template.contains("{findings}"));
    }

    #[test]
    fn test_research_prompts_render_decomposition() {
        let prompts = ResearchPrompts::default();
        let rendered = prompts.render_decomposition(3, 5, "What is Rust?");

        assert!(rendered.contains("3-5"));
        assert!(rendered.contains("What is Rust?"));
        assert!(!rendered.contains("{min}"));
        assert!(!rendered.contains("{max}"));
        assert!(!rendered.contains("{query}"));
    }

    #[test]
    fn test_research_prompts_render_synthesis() {
        let prompts = ResearchPrompts::default();
        let rendered = prompts.render_synthesis("Original question", "Finding 1\nFinding 2");

        assert!(rendered.contains("Original question"));
        assert!(rendered.contains("Finding 1\nFinding 2"));
        assert!(!rendered.contains("{query}"));
        assert!(!rendered.contains("{findings}"));
    }

    #[test]
    fn test_research_prompts_validation_empty_decomposition_system() {
        let prompts = ResearchPrompts {
            decomposition_system: "".to_string(),
            ..Default::default()
        };
        assert!(prompts.validate().is_err());
        assert!(prompts
            .validate()
            .unwrap_err()
            .to_string()
            .contains("decomposition_system"));
    }

    #[test]
    fn test_research_prompts_validation_whitespace_only() {
        let prompts = ResearchPrompts {
            sub_query_system: "   ".to_string(),
            ..Default::default()
        };
        assert!(prompts.validate().is_err());
        assert!(prompts
            .validate()
            .unwrap_err()
            .to_string()
            .contains("sub_query_system"));
    }

    #[test]
    fn test_research_config_validates_prompts() {
        let config = ResearchConfig {
            prompts: ResearchPrompts {
                synthesis_system: "".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("synthesis_system"));
    }

    #[test]
    fn test_research_config_with_custom_prompts() {
        let config = ResearchConfig {
            prompts: ResearchPrompts {
                decomposition_system: "Custom system instruction".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };

        assert!(config.validate().is_ok());
        assert_eq!(
            config.prompts.decomposition_system,
            "Custom system instruction"
        );
    }

    #[test]
    fn test_default_research_config_includes_prompts() {
        let config = ResearchConfig::default();
        assert!(config.validate().is_ok());
        assert!(!config.prompts.decomposition_system.is_empty());
        assert!(!config.prompts.decomposition_template.is_empty());
        assert!(!config.prompts.sub_query_system.is_empty());
        assert!(!config.prompts.synthesis_system.is_empty());
        assert!(!config.prompts.synthesis_template.is_empty());
    }

    #[test]
    fn test_render_decomposition_with_placeholder_in_query() {
        let prompts = ResearchPrompts::default();
        let rendered = prompts.render_decomposition(3, 5, "Compare {min} and {max} in Rust");

        // User's literal {min} and {max} should be preserved, not replaced
        assert!(rendered.contains("Compare {min} and {max} in Rust"));
        // Template placeholders should be replaced
        assert!(rendered.contains("3-5"));
    }

    #[test]
    fn test_render_synthesis_with_placeholder_in_query() {
        let prompts = ResearchPrompts::default();
        let rendered = prompts.render_synthesis("What about {findings}?", "Finding 1");

        // User's literal {findings} should be preserved
        assert!(rendered.contains("What about {findings}?"));
        // Template placeholder should be replaced with actual findings
        assert!(rendered.contains("Finding 1"));
    }

    #[test]
    fn test_render_synthesis_findings_containing_query_placeholder() {
        let prompts = ResearchPrompts::default();
        // Findings text contains literal "{query}" - this gets replaced because
        // {findings} is substituted first, then {query} is replaced globally
        let rendered =
            prompts.render_synthesis("What is Rust?", "Research says: {query} is important");

        // The literal {query} in findings IS replaced (this is expected behavior
        // since we do global replacement after findings substitution)
        assert!(rendered.contains("Research says: What is Rust? is important"));
        // Original query is also in its proper location
        assert!(rendered.contains("Original question: What is Rust?"));
    }
}
