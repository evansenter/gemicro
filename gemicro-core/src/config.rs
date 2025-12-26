use crate::error::AgentError;
use std::time::Duration;

/// The Gemini model to use for all LLM operations
///
/// Hardcoded to gemini-3-flash-preview as per project requirements.
pub const MODEL: &str = "gemini-3-flash-preview";

/// Placeholder constants for template validation
mod placeholders {
    /// Required placeholders for decomposition template
    pub const DECOMPOSITION: &[&str] = &["{min}", "{max}", "{query}"];

    /// Required placeholders for synthesis template
    pub const SYNTHESIS: &[&str] = &["{query}", "{findings}"];
}

/// Extracts all placeholders from a template string
///
/// Returns placeholders in the format `{name}`. Handles malformed
/// templates gracefully (unclosed braces are ignored, nested braces
/// restart the search from the inner brace with a warning logged).
///
/// # Examples
///
/// ```text
/// extract_placeholders("Hello {name}!") → ["{name}"]
/// extract_placeholders("{a} and {b}") → ["{a}", "{b}"]
/// extract_placeholders("{unclosed") → [] (unclosed brace ignored)
/// extract_placeholders("{{inner}}") → ["{inner}"] (outer brace ignored, warning logged)
/// extract_placeholders("{out{inner}}") → ["{inner}"] (restarts from inner brace)
/// ```
///
/// # Note
///
/// This function does not support brace escaping. If you need literal braces
/// in your template, they will be treated as placeholder delimiters. There is
/// currently no escape sequence to include literal `{` or `}` characters.
fn extract_placeholders(template: &str) -> Vec<String> {
    use std::collections::HashSet;

    let mut placeholders = Vec::new();
    let mut seen = HashSet::new();
    let mut chars = template.char_indices().peekable();

    while let Some((start, ch)) = chars.next() {
        if ch == '{' {
            // Find closing brace, but restart if we hit another opening brace
            let mut current_start = start;
            let mut end = None;

            for (i, c) in chars.by_ref() {
                if c == '{' {
                    // Restart from this new opening brace
                    log::warn!(
                        "Nested opening brace detected at position {} in template (restarting from inner brace)",
                        i
                    );
                    current_start = i;
                } else if c == '}' {
                    end = Some(i);
                    break;
                }
            }

            if let Some(end_idx) = end {
                let placeholder = &template[current_start..=end_idx];
                if seen.insert(placeholder.to_string()) {
                    placeholders.push(placeholder.to_string());
                }
            }
        }
    }

    placeholders
}

/// Validates that a template contains required placeholders
///
/// Returns a list of missing placeholders. Logs warnings for unrecognized placeholders.
///
/// Uses `extract_placeholders()` for exact matching to avoid substring false positives
/// (e.g., `{minimum}` should not satisfy requirement for `{min}`).
fn validate_template_placeholders(
    template: &str,
    required: &[&str],
    template_name: &str,
) -> Vec<String> {
    let mut missing = Vec::new();
    let found_placeholders = extract_placeholders(template);

    // Check for required placeholders using exact match against extracted placeholders
    for placeholder in required {
        if !found_placeholders.iter().any(|p| p == placeholder) {
            missing.push((*placeholder).to_string());
        }
    }

    // Warn about unrecognized placeholders
    for found in &found_placeholders {
        if !required.contains(&found.as_str()) {
            log::warn!(
                "{} contains unrecognized placeholder '{}' (expected: {})",
                template_name,
                found,
                required.join(", ")
            );
        }
    }

    missing
}

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
/// # Note on Brace Escaping
///
/// Template placeholders do **not** support brace escaping. If your template
/// contains literal `{` or `}` characters, they will be interpreted as
/// placeholder delimiters. Nested braces (e.g., `{{name}}`) will extract
/// the innermost placeholder (`{name}`). There is currently no escape sequence
/// to include literal braces in templates.
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

    /// Validate that all prompts are non-empty and templates contain required placeholders
    ///
    /// Returns an error if:
    /// - Any prompt string is empty or whitespace-only
    /// - `decomposition_template` is missing `{min}`, `{max}`, or `{query}`
    /// - `synthesis_template` is missing `{query}` or `{findings}`
    ///
    /// Collects all validation errors and returns them joined by "; ".
    /// Logs warnings for unrecognized placeholders in templates.
    pub fn validate(&self) -> Result<(), AgentError> {
        let mut errors = Vec::new();

        // Validate non-empty prompts
        if self.decomposition_system.trim().is_empty() {
            errors.push("decomposition_system cannot be empty".to_string());
        }
        if self.decomposition_template.trim().is_empty() {
            errors.push("decomposition_template cannot be empty".to_string());
        }
        if self.sub_query_system.trim().is_empty() {
            errors.push("sub_query_system cannot be empty".to_string());
        }
        if self.synthesis_system.trim().is_empty() {
            errors.push("synthesis_system cannot be empty".to_string());
        }
        if self.synthesis_template.trim().is_empty() {
            errors.push("synthesis_template cannot be empty".to_string());
        }

        // Validate placeholder presence (only if template is non-empty to avoid redundant errors)
        if !self.decomposition_template.trim().is_empty() {
            let missing = validate_template_placeholders(
                &self.decomposition_template,
                placeholders::DECOMPOSITION,
                "decomposition_template",
            );
            if !missing.is_empty() {
                errors.push(format!(
                    "decomposition_template missing required placeholders: {}",
                    missing.join(", ")
                ));
            }
        }

        if !self.synthesis_template.trim().is_empty() {
            let missing = validate_template_placeholders(
                &self.synthesis_template,
                placeholders::SYNTHESIS,
                "synthesis_template",
            );
            if !missing.is_empty() {
                errors.push(format!(
                    "synthesis_template missing required placeholders: {}",
                    missing.join(", ")
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(AgentError::InvalidConfig(errors.join("; ")))
        }
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
    /// - Any prompt is empty or missing required placeholders
    ///
    /// Collects all validation errors and returns them joined by "; ".
    pub fn validate(&self) -> Result<(), AgentError> {
        let mut errors = Vec::new();

        if self.min_sub_queries == 0 {
            errors.push("min_sub_queries must be greater than 0".to_string());
        }

        if self.max_sub_queries == 0 {
            errors.push("max_sub_queries must be greater than 0".to_string());
        }

        if self.min_sub_queries > self.max_sub_queries {
            errors.push(format!(
                "min_sub_queries ({}) cannot be greater than max_sub_queries ({})",
                self.min_sub_queries, self.max_sub_queries
            ));
        }

        if self.total_timeout.as_secs() == 0 {
            errors.push("total_timeout must be greater than 0".to_string());
        }

        // Validate prompts and collect their errors
        if let Err(AgentError::InvalidConfig(prompt_errors)) = self.prompts.validate() {
            errors.push(prompt_errors);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(AgentError::InvalidConfig(errors.join("; ")))
        }
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

// =============================================================================
// ReAct Agent Configuration
// =============================================================================

/// Prompts used by the ReAct agent
///
/// Contains system instructions and templates for the reasoning loop.
///
/// # Template Placeholders
///
/// - `iteration_template`: `{query}`, `{scratchpad}`
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ReactPrompts {
    /// System instruction for the ReAct reasoning loop
    pub system: String,

    /// Template for each iteration
    ///
    /// Placeholders: `{query}`, `{scratchpad}`
    pub iteration_template: String,
}

impl ReactPrompts {
    /// Render the iteration prompt with placeholders substituted
    pub fn render_iteration(&self, query: &str, scratchpad: &str) -> String {
        self.iteration_template
            .replace("{query}", query)
            .replace("{scratchpad}", scratchpad)
    }

    /// Validate that prompts are non-empty and contain required placeholders
    pub fn validate(&self) -> Result<(), AgentError> {
        let mut errors = Vec::new();

        if self.system.trim().is_empty() {
            errors.push("system cannot be empty".to_string());
        }
        if self.iteration_template.trim().is_empty() {
            errors.push("iteration_template cannot be empty".to_string());
        }
        if !self.iteration_template.contains("{query}") {
            errors.push("iteration_template must contain {query}".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(AgentError::InvalidConfig(errors.join("; ")))
        }
    }
}

impl Default for ReactPrompts {
    fn default() -> Self {
        Self {
            system: r#"You are a helpful AI assistant that solves problems step by step using available tools.

Available tools:
- web_search: Search the web for current information. Input: search query string
- calculator: Evaluate mathematical expressions. Input: expression (e.g., "2 + 2 * 3", "sqrt(16)")
- final_answer: Return your final answer to the user. Input: your complete answer

Instructions:
1. Think step by step about what you need to find out
2. Use tools to gather information
3. When you have enough information, use final_answer to respond

Always think carefully before acting. Be concise in your thoughts."#.to_string(),

            iteration_template: r#"Question: {query}

{scratchpad}

Based on the above, what should you do next? Think step by step, then choose a tool."#.to_string(),
        }
    }
}

/// Configuration for ReAct agent
#[derive(Debug, Clone)]
pub struct ReactConfig {
    /// Maximum number of thought-action-observation iterations
    ///
    /// Default: 10
    pub max_iterations: usize,

    /// Available tools for this agent instance
    ///
    /// Default: ["web_search", "calculator"]
    pub available_tools: Vec<String>,

    /// Enable Google Search grounding for web_search tool
    ///
    /// Default: true
    pub use_google_search: bool,

    /// Overall timeout for the entire ReAct loop
    ///
    /// Default: 120 seconds
    pub total_timeout: Duration,

    /// Prompts configuration
    pub prompts: ReactPrompts,
}

impl ReactConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), AgentError> {
        let mut errors = Vec::new();

        if self.max_iterations == 0 {
            errors.push("max_iterations must be greater than 0".to_string());
        }

        if self.total_timeout.as_secs() == 0 {
            errors.push("total_timeout must be greater than 0".to_string());
        }

        if self.available_tools.is_empty() {
            errors.push("available_tools cannot be empty".to_string());
        }

        // Validate prompts
        if let Err(AgentError::InvalidConfig(prompt_errors)) = self.prompts.validate() {
            errors.push(prompt_errors);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(AgentError::InvalidConfig(errors.join("; ")))
        }
    }
}

impl Default for ReactConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            available_tools: vec!["web_search".to_string(), "calculator".to_string()],
            use_google_search: true,
            total_timeout: Duration::from_secs(120),
            prompts: ReactPrompts::default(),
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
    use rstest::rstest;

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

    // Placeholder validation tests

    #[test]
    fn test_extract_placeholders() {
        let placeholders = super::extract_placeholders("Hello {foo} and {bar}!");
        assert_eq!(placeholders.len(), 2);
        assert!(placeholders.contains(&"{foo}".to_string()));
        assert!(placeholders.contains(&"{bar}".to_string()));
    }

    #[test]
    fn test_extract_placeholders_deduplicates() {
        let placeholders = super::extract_placeholders("{foo} and {foo} again");
        assert_eq!(placeholders.len(), 1);
        assert_eq!(placeholders[0], "{foo}");
    }

    #[test]
    fn test_extract_placeholders_handles_unclosed_braces() {
        let placeholders = super::extract_placeholders("Hello {foo and {bar}");
        // Should only extract {bar} since {foo is not closed properly
        assert_eq!(placeholders.len(), 1);
        assert_eq!(placeholders[0], "{bar}");
    }

    #[test]
    fn test_extract_placeholders_empty_template() {
        let placeholders = super::extract_placeholders("");
        assert!(placeholders.is_empty());
    }

    #[test]
    fn test_extract_placeholders_no_placeholders() {
        let placeholders = super::extract_placeholders("Just plain text");
        assert!(placeholders.is_empty());
    }

    #[test]
    fn test_validate_missing_decomposition_placeholders() {
        let prompts = ResearchPrompts {
            decomposition_template: "Query: {query}".to_string(), // Missing {min}, {max}
            ..Default::default()
        };
        let result = prompts.validate();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("decomposition_template"));
        assert!(err.contains("{min}"));
        assert!(err.contains("{max}"));
    }

    #[test]
    fn test_validate_missing_synthesis_placeholders() {
        let prompts = ResearchPrompts {
            synthesis_template: "Answer: here".to_string(), // Missing {query}, {findings}
            ..Default::default()
        };
        let result = prompts.validate();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("synthesis_template"));
        assert!(err.contains("{query}"));
        assert!(err.contains("{findings}"));
    }

    #[test]
    fn test_validate_collects_all_prompt_errors() {
        let prompts = ResearchPrompts {
            decomposition_system: "".to_string(),
            decomposition_template: "Query: {query}".to_string(), // Missing {min}, {max}
            synthesis_template: "Answer".to_string(),             // Missing {query}, {findings}
            ..Default::default()
        };
        let result = prompts.validate();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        // Should contain ALL errors, not just the first
        assert!(err.contains("decomposition_system"));
        assert!(err.contains("decomposition_template"));
        assert!(err.contains("synthesis_template"));
    }

    #[test]
    fn test_research_config_collects_all_errors() {
        let config = ResearchConfig {
            min_sub_queries: 0,
            max_sub_queries: 0,
            total_timeout: Duration::from_secs(0),
            prompts: ResearchPrompts {
                decomposition_template: "Bad template".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        // Should contain all config errors AND prompt errors
        assert!(err.contains("min_sub_queries"));
        assert!(err.contains("max_sub_queries"));
        assert!(err.contains("total_timeout"));
        assert!(err.contains("decomposition_template"));
    }

    #[test]
    fn test_validate_partial_decomposition_placeholders() {
        // Has {query} but missing {min} and {max}
        let prompts = ResearchPrompts {
            decomposition_template: "Research this: {query}".to_string(),
            ..Default::default()
        };
        let result = prompts.validate();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("{min}"));
        assert!(err.contains("{max}"));
        assert!(!err.contains("{query}")); // {query} is present, shouldn't be in error
    }

    #[test]
    fn test_validate_unrecognized_placeholder_does_not_error() {
        // Has all required placeholders plus an extra {custom} one
        // Should pass validation (warning is logged but doesn't cause error)
        let prompts = ResearchPrompts {
            decomposition_template:
                "Decompose into {min}-{max} questions: {query}\n\nCustom: {custom}".to_string(),
            ..Default::default()
        };
        // Should pass - unrecognized placeholders only trigger warnings, not errors
        assert!(prompts.validate().is_ok());
    }

    // Parameterized tests for missing individual placeholders
    #[rstest]
    #[case::missing_min_in_decomposition(
        "decomposition_template",
        "Query: {query}, max: {max}",
        "{min}"
    )]
    #[case::missing_max_in_decomposition(
        "decomposition_template",
        "Query: {query}, min: {min}",
        "{max}"
    )]
    #[case::missing_query_in_decomposition(
        "decomposition_template",
        "Generate {min}-{max} sub-queries",
        "{query}"
    )]
    #[case::missing_query_in_synthesis("synthesis_template", "Findings: {findings}", "{query}")]
    #[case::missing_findings_in_synthesis("synthesis_template", "Query: {query}", "{findings}")]
    fn test_missing_placeholder(
        #[case] field: &str,
        #[case] template: &str,
        #[case] expected_missing: &str,
    ) {
        let prompts = match field {
            "decomposition_template" => ResearchPrompts {
                decomposition_template: template.to_string(),
                ..Default::default()
            },
            "synthesis_template" => ResearchPrompts {
                synthesis_template: template.to_string(),
                ..Default::default()
            },
            _ => panic!("Unknown field: {}", field),
        };
        let result = prompts.validate();
        assert!(result.is_err(), "Expected validation to fail");
        assert!(
            result.unwrap_err().to_string().contains(expected_missing),
            "Error should mention {}",
            expected_missing
        );
    }

    #[test]
    fn test_research_prompts_placeholder_case_sensitive() {
        // Uppercase placeholders should NOT be recognized
        let prompts = ResearchPrompts {
            decomposition_template: "{MIN}-{MAX} sub-queries for {QUERY}".to_string(),
            ..Default::default()
        };
        let result = prompts.validate();
        assert!(
            result.is_err(),
            "Uppercase placeholders should not be recognized"
        );
    }

    #[test]
    fn test_validate_substring_false_positive() {
        // {minimum} should NOT satisfy requirement for {min}
        // {maximum} should NOT satisfy requirement for {max}
        // {query_text} should NOT satisfy requirement for {query}
        let prompts = ResearchPrompts {
            decomposition_template: "Generate {minimum}-{maximum} queries for {query_text}"
                .to_string(),
            ..Default::default()
        };
        let result = prompts.validate();
        assert!(
            result.is_err(),
            "Substring matches should not satisfy placeholder requirements"
        );
        let err = result.unwrap_err().to_string();
        assert!(err.contains("{min}"), "Should report {{min}} as missing");
        assert!(err.contains("{max}"), "Should report {{max}} as missing");
        assert!(
            err.contains("{query}"),
            "Should report {{query}} as missing"
        );
    }

    #[test]
    fn test_extract_placeholders_nested_and_unclosed() {
        // Combination of nested braces and unclosed brace
        // "{outer {inner}" has unclosed outer and nested inner - should extract {inner}
        let placeholders = super::extract_placeholders("{outer {inner}");
        assert_eq!(placeholders.len(), 1);
        assert_eq!(placeholders[0], "{inner}");
    }

    #[test]
    fn test_extract_placeholders_double_nested() {
        // "{{name}}" - double opening brace, should extract {name}
        let placeholders = super::extract_placeholders("Hello {{name}}!");
        assert_eq!(placeholders.len(), 1);
        assert_eq!(placeholders[0], "{name}");
    }

    #[test]
    fn test_extract_placeholders_restarts_from_inner() {
        // "{out{inner}}" - should restart from inner brace and extract {inner}
        let placeholders = super::extract_placeholders("{out{inner}}");
        assert_eq!(placeholders.len(), 1);
        assert_eq!(placeholders[0], "{inner}");
    }
}
