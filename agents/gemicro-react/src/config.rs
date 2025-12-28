//! Configuration for the ReAct agent.

use gemicro_core::AgentError;
use std::time::Duration;

/// Prompts used by the ReAct agent
///
/// Contains system instructions and templates for the reasoning loop.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_react_config() {
        let config = ReactConfig::default();
        assert_eq!(config.max_iterations, 10);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_default_prompts() {
        let prompts = ReactPrompts::default();
        assert!(prompts.validate().is_ok());
    }

    #[test]
    fn test_render_iteration() {
        let prompts = ReactPrompts::default();
        let rendered = prompts.render_iteration("What is 2+2?", "Thought: I need to calculate");
        assert!(rendered.contains("What is 2+2?"));
        assert!(rendered.contains("Thought: I need to calculate"));
    }

    #[test]
    fn test_config_validation_zero_iterations() {
        let config = ReactConfig {
            max_iterations: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_empty_tools() {
        let config = ReactConfig {
            available_tools: vec![],
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}
