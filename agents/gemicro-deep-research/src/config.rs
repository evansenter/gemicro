//! Configuration for the Deep Research agent.

use gemicro_core::AgentError;
use std::time::Duration;

/// Placeholder constants for template validation
mod placeholders {
    /// Required placeholders for decomposition template
    pub const DECOMPOSITION: &[&str] = &["{min}", "{max}", "{query}"];

    /// Required placeholders for synthesis template
    pub const SYNTHESIS: &[&str] = &["{query}", "{findings}"];
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
/// # Example
///
/// ```
/// use gemicro_deep_research::ResearchPrompts;
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

        // Validate placeholder presence
        if !self.decomposition_template.trim().is_empty() {
            let missing = validate_template_placeholders(
                &self.decomposition_template,
                placeholders::DECOMPOSITION,
            );
            if !missing.is_empty() {
                errors.push(format!(
                    "decomposition_template missing required placeholders: {}",
                    missing.join(", ")
                ));
            }
        }

        if !self.synthesis_template.trim().is_empty() {
            let missing =
                validate_template_placeholders(&self.synthesis_template, placeholders::SYNTHESIS);
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

// ============================================================================
// Helper functions for template validation
// ============================================================================

/// Extracts all placeholders from a template string
fn extract_placeholders(template: &str) -> Vec<String> {
    use std::collections::HashSet;

    let mut placeholders = Vec::new();
    let mut seen = HashSet::new();
    let mut chars = template.char_indices().peekable();

    while let Some((start, ch)) = chars.next() {
        if ch == '{' {
            let mut current_start = start;
            let mut end = None;

            for (i, c) in chars.by_ref() {
                if c == '{' {
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
fn validate_template_placeholders(template: &str, required: &[&str]) -> Vec<String> {
    let mut missing = Vec::new();
    let found_placeholders = extract_placeholders(template);

    for placeholder in required {
        if !found_placeholders.iter().any(|p| p == placeholder) {
            missing.push((*placeholder).to_string());
        }
    }

    missing
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_research_config() {
        let config = ResearchConfig::default();
        assert_eq!(config.max_sub_queries, 5);
        assert_eq!(config.min_sub_queries, 3);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_default_prompts() {
        let prompts = ResearchPrompts::default();
        assert!(prompts.validate().is_ok());
    }

    #[test]
    fn test_render_decomposition() {
        let prompts = ResearchPrompts::default();
        let rendered = prompts.render_decomposition(3, 5, "What is Rust?");
        assert!(rendered.contains("3-5"));
        assert!(rendered.contains("What is Rust?"));
    }

    #[test]
    fn test_render_synthesis() {
        let prompts = ResearchPrompts::default();
        let rendered = prompts.render_synthesis("Original", "Finding 1\nFinding 2");
        assert!(rendered.contains("Original"));
        assert!(rendered.contains("Finding 1"));
    }

    #[test]
    fn test_config_validation_min_greater_than_max() {
        let config = ResearchConfig {
            min_sub_queries: 10,
            max_sub_queries: 5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}
