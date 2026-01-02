//! Configuration for DeveloperAgent.

use gemicro_core::{tool::ToolSet, AgentError};
use std::path::PathBuf;
use std::time::Duration;

/// Default timeout for developer agent (10 minutes).
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);

/// Default system prompt for developer agent.
const DEFAULT_SYSTEM_PROMPT: &str = r#"You are a developer agent that helps with software engineering tasks.

You have access to tools for reading files, editing code, running commands, and searching the codebase.

When working on tasks:
1. First understand the codebase structure and existing patterns
2. Make targeted, minimal changes that follow existing conventions
3. Verify your changes work correctly
4. Explain what you did and why

Be precise and careful. Prefer editing existing files over creating new ones.
"#;

/// Configuration for DeveloperAgent.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DeveloperConfig {
    /// Base system prompt (before CLAUDE.md injection).
    pub system_prompt: String,

    /// Which tools to make available.
    pub tool_filter: ToolSet,

    /// Maximum execution time.
    pub timeout: Duration,

    /// Paths to load CLAUDE.md from (in order of precedence).
    /// Default: `["./CLAUDE.md", "~/.claude/CLAUDE.md"]`
    pub claude_md_paths: Vec<PathBuf>,

    /// Maximum iterations (LLM turns) before forcing completion.
    pub max_iterations: usize,
}

impl Default for DeveloperConfig {
    fn default() -> Self {
        Self {
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            tool_filter: ToolSet::All,
            timeout: DEFAULT_TIMEOUT,
            claude_md_paths: vec![
                PathBuf::from("./CLAUDE.md"),
                dirs::home_dir()
                    .map(|h| h.join(".claude/CLAUDE.md"))
                    .unwrap_or_else(|| PathBuf::from("~/.claude/CLAUDE.md")),
            ],
            max_iterations: 50,
        }
    }
}

impl DeveloperConfig {
    /// Create a new config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the system prompt.
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set the tool filter.
    #[must_use]
    pub fn with_tool_filter(mut self, filter: ToolSet) -> Self {
        self.tool_filter = filter;
        self
    }

    /// Set the timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the CLAUDE.md paths.
    #[must_use]
    pub fn with_claude_md_paths(mut self, paths: Vec<PathBuf>) -> Self {
        self.claude_md_paths = paths;
        self
    }

    /// Set the maximum iterations.
    #[must_use]
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), AgentError> {
        if self.system_prompt.trim().is_empty() {
            return Err(AgentError::InvalidConfig(
                "System prompt cannot be empty".into(),
            ));
        }
        if self.timeout.is_zero() {
            return Err(AgentError::InvalidConfig(
                "Timeout must be greater than zero".into(),
            ));
        }
        if self.max_iterations == 0 {
            return Err(AgentError::InvalidConfig(
                "Max iterations must be greater than zero".into(),
            ));
        }
        Ok(())
    }

    /// Build the complete system prompt with CLAUDE.md content.
    ///
    /// Loads CLAUDE.md from configured paths and appends to base prompt.
    pub fn build_system_prompt(&self) -> String {
        let mut prompt = self.system_prompt.clone();

        // Load CLAUDE.md content from first available path
        let mut found = false;
        for path in &self.claude_md_paths {
            match std::fs::read_to_string(path) {
                Ok(content) if !content.trim().is_empty() => {
                    prompt.push_str("\n\n## Project Context (from CLAUDE.md)\n\n");
                    prompt.push_str(&content);
                    log::info!("Loaded CLAUDE.md from: {}", path.display());
                    found = true;
                    break;
                }
                Ok(_) => {
                    log::debug!("CLAUDE.md at {} is empty, skipping", path.display());
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    log::debug!("CLAUDE.md not found at {}", path.display());
                }
                Err(e) => {
                    // Log actual read errors (permissions, etc.) at warn level
                    log::warn!("Failed to read CLAUDE.md from {}: {}", path.display(), e);
                }
            }
        }
        if !found {
            log::debug!(
                "No CLAUDE.md found in configured paths: {:?}",
                self.claude_md_paths
            );
        }

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DeveloperConfig::default();
        assert!(config.validate().is_ok());
        assert!(!config.system_prompt.is_empty());
        assert_eq!(config.timeout, Duration::from_secs(600));
        assert_eq!(config.max_iterations, 50);
    }

    #[test]
    fn test_empty_system_prompt_invalid() {
        let config = DeveloperConfig::default().with_system_prompt("");
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_zero_timeout_invalid() {
        let config = DeveloperConfig::default().with_timeout(Duration::ZERO);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_zero_max_iterations_invalid() {
        let config = DeveloperConfig::default().with_max_iterations(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_builder_pattern() {
        let config = DeveloperConfig::new()
            .with_system_prompt("Custom prompt")
            .with_tool_filter(ToolSet::None)
            .with_timeout(Duration::from_secs(120))
            .with_max_iterations(10);

        assert_eq!(config.system_prompt, "Custom prompt");
        assert!(matches!(config.tool_filter, ToolSet::None));
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.max_iterations, 10);
    }

    #[test]
    fn test_build_system_prompt_without_claude_md() {
        let config = DeveloperConfig::default()
            .with_claude_md_paths(vec![PathBuf::from("/nonexistent/path")]);

        let prompt = config.build_system_prompt();
        // Should just return base prompt when CLAUDE.md not found
        assert_eq!(prompt, config.system_prompt);
    }
}
