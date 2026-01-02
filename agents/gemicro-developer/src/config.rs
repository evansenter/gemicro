//! Configuration for DeveloperAgent.

use gemicro_core::{tool::ToolSet, AgentError};
use std::path::PathBuf;
use std::time::Duration;

/// Default timeout for developer agent (10 minutes).
///
/// This covers the entire multi-turn conversation including all LLM calls
/// and tool executions, not just a single request.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);

/// Default system prompt loaded from `prompts/developer_system.md`.
const DEFAULT_SYSTEM_PROMPT: &str = include_str!("../prompts/developer_system.md");

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

    /// Paths to load developer instructions from (in order of precedence).
    ///
    /// These files (e.g., CLAUDE.md) provide project-specific context that gets
    /// appended to the system prompt. First found file is used.
    ///
    /// Default: `["./CLAUDE.md", "~/.claude/CLAUDE.md"]`
    pub di_paths: Vec<PathBuf>,

    /// Maximum iterations (LLM turns) before forcing completion.
    pub max_iterations: usize,

    /// Whether to batch tool approvals.
    ///
    /// When true, the agent collects all tool calls from an LLM response,
    /// shows a plan summary, and requests a single approval for the batch.
    /// When false, each tool requiring confirmation is approved individually.
    ///
    /// Default: true
    pub approval_batching: bool,
}

impl Default for DeveloperConfig {
    fn default() -> Self {
        Self {
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            tool_filter: ToolSet::All,
            timeout: DEFAULT_TIMEOUT,
            di_paths: vec![
                PathBuf::from("./CLAUDE.md"),
                dirs::home_dir()
                    .map(|h| h.join(".claude/CLAUDE.md"))
                    .unwrap_or_else(|| PathBuf::from("~/.claude/CLAUDE.md")),
            ],
            max_iterations: 50,
            approval_batching: true,
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

    /// Set the developer instruction paths.
    #[must_use]
    pub fn with_di_paths(mut self, paths: Vec<PathBuf>) -> Self {
        self.di_paths = paths;
        self
    }

    /// Set the maximum iterations.
    #[must_use]
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set whether to use approval batching.
    #[must_use]
    pub fn with_approval_batching(mut self, enabled: bool) -> Self {
        self.approval_batching = enabled;
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

    /// Build the complete system prompt with CLAUDE.md content and environment context.
    ///
    /// Loads CLAUDE.md from configured paths and appends to base prompt.
    /// Also injects the current working directory for file path resolution.
    pub fn build_system_prompt(&self) -> String {
        let mut prompt = self.system_prompt.clone();

        // Inject working directory so LLM can construct absolute paths
        if let Ok(cwd) = std::env::current_dir() {
            prompt.push_str(&format!(
                "\n## Environment\n\nWorking directory: {}\n",
                cwd.display()
            ));
        }

        // Load developer instructions from first available path
        let mut found = false;
        for path in &self.di_paths {
            match std::fs::read_to_string(path) {
                Ok(content) if !content.trim().is_empty() => {
                    prompt.push_str("\n\n## Developer Instructions\n\n");
                    prompt.push_str(&content);
                    log::info!("Loaded developer instructions from: {}", path.display());
                    found = true;
                    break;
                }
                Ok(_) => {
                    log::debug!(
                        "Developer instructions at {} empty, skipping",
                        path.display()
                    );
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    log::debug!("Developer instructions not found at {}", path.display());
                }
                Err(e) => {
                    // Log actual read errors (permissions, etc.) at warn level
                    log::warn!(
                        "Failed to read developer instructions from {}: {}",
                        path.display(),
                        e
                    );
                }
            }
        }
        if !found {
            log::debug!(
                "No developer instructions found in configured paths: {:?}",
                self.di_paths
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
    fn test_build_system_prompt_without_di() {
        let config =
            DeveloperConfig::default().with_di_paths(vec![PathBuf::from("/nonexistent/path")]);

        let prompt = config.build_system_prompt();
        // Should contain base prompt and working directory
        assert!(prompt.starts_with(&config.system_prompt));
        assert!(prompt.contains("Working directory:"));
        // Should NOT contain developer instructions section
        assert!(!prompt.contains("## Developer Instructions"));
    }

    #[test]
    fn test_build_system_prompt_includes_working_dir() {
        let config =
            DeveloperConfig::default().with_di_paths(vec![PathBuf::from("/nonexistent/path")]);

        let prompt = config.build_system_prompt();

        // Should include working directory section
        assert!(prompt.contains("## Environment"));
        assert!(prompt.contains("Working directory:"));

        // Working directory should be an absolute path
        let cwd = std::env::current_dir().expect("Should have CWD");
        assert!(prompt.contains(&cwd.display().to_string()));
    }
}
